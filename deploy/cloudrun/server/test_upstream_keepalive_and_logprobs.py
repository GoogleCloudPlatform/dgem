# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loopback HTTP/1.1 keep-alive pooling and lossless pre-softmax logprob diagnostics."""

import http.client
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import math
import os
import socket
import sys
import threading
import time
import types
import unittest
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server_key import load_server  # noqa: E402


class _CountingUpstreamServer(ThreadingHTTPServer):
    """Loopback test server counting TCP socket accepts on 127.0.0.1."""

    daemon_threads = True

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.accepted_connections = 0
        self.requests_served = 0
        self.force_close_next = False
        self.next_status = 200
        self.delay_next_s = 0.0
        self.lock = threading.Lock()

    def get_request(self):
        sock, addr = super().get_request()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self.lock:
            self.accepted_connections += 1
        return sock, addr


class _UpstreamHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 stub answering /health, /v1/chat/completions, and /v1/completions."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        del fmt, args

    def do_GET(self):
        if self.path == "/health":
            payload = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        request_body = json.loads(raw_body.decode("utf-8"))
        with self.server.lock:
            self.server.requests_served += 1
            status = self.server.next_status
            force_close = self.server.force_close_next
            delay_s = self.server.delay_next_s
            self.server.force_close_next = False
            self.server.delay_next_s = 0.0
        if delay_s > 0.0:
            time.sleep(delay_s)
        if status != 200:
            err_bytes = json.dumps({"error": {"message": f"simulated {status}"}}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_bytes)))
            self.end_headers()
            self.wfile.write(err_bytes)
            return
        reply = json.dumps({"choices": [{"index": 0, "echo_model": request_body.get("model")}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        if force_close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(reply)


class UpstreamKeepAliveAndLogprobsTest(unittest.TestCase):
    """Verifies persistent upstream HTTP/1.1 reuse, reconnects, and pre-exp logprobs."""

    def setUp(self):
        self.mod = load_server("")
        self.upstream = _CountingUpstreamServer(("127.0.0.1", 0), _UpstreamHandler)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.addCleanup(self._close_cached_upstream_conn)
        host, port = self.upstream.server_address
        self.mod.ARGS = types.SimpleNamespace(
            upstream=f"http://{host}:{port}",
            model="dgemma",
            constrained=True,
        )
        self.mod._upstream_ready = True

    def _close_cached_upstream_conn(self):
        cached = getattr(self.mod._UPSTREAM_LOCAL, "entry", None)
        if cached is not None:
            try:
                cached[1].close()
            except OSError:
                pass
            self.mod._UPSTREAM_LOCAL.entry = None

    def test_keepalive_reuses_single_socket_sequentially_and_reconnects(self):
        for _ in range(12):
            response = self.mod.upstream_chat({"model": "dgemma"})
            self.assertEqual(response["choices"][0]["echo_model"], "dgemma")
        self.assertEqual(self.upstream.requests_served, 12)
        self.assertEqual(self.upstream.accepted_connections, 1)
        cached_sock = self.mod._UPSTREAM_LOCAL.entry[1].sock
        self.assertNotEqual(cached_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0)

        # Verify _Server.get_request sets TCP_NODELAY on accepted HTTP/1.1 client connections.
        srv = self.mod._Server(("127.0.0.1", 0), self.mod.Handler)
        self.addCleanup(srv.server_close)
        accepted_nodelay = []

        def accept_once():
            accepted_sock, _ = srv.get_request()
            accepted_nodelay.append(accepted_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))
            accepted_sock.close()

        accept_thread = threading.Thread(target=accept_once, daemon=True)
        accept_thread.start()
        with socket.create_connection(srv.server_address, timeout=2.0):
            pass
        accept_thread.join(timeout=2.0)
        self.assertEqual(len(accepted_nodelay), 1)
        self.assertNotEqual(accepted_nodelay[0], 0)

        # Force upstream Connection: close on call 13, then verify call 14 opens cleanly.
        self.upstream.force_close_next = True
        self.mod.upstream_chat({"model": "dgemma"})
        self.mod.upstream_completions({"model": "dgemma"})
        self.assertEqual(self.upstream.requests_served, 14)
        self.assertEqual(self.upstream.accepted_connections, 2)

    def test_upstream_http_errors_raise_urllib_httperror(self):
        self.upstream.next_status = 400
        with self.assertRaises(urllib.error.HTTPError) as ctx400:
            self.mod.upstream_chat({"model": "dgemma"})
        self.assertEqual(ctx400.exception.code, 400)
        self.assertIn("simulated 400", ctx400.exception.read().decode("utf-8"))

        self.upstream.next_status = 503
        with self.assertRaises(urllib.error.HTTPError) as ctx503:
            self.mod.upstream_completions({"model": "dgemma"})
        self.assertEqual(ctx503.exception.code, 503)

    def test_timeout_evicts_half_used_connection_from_thread_local(self):
        self.upstream.delay_next_s = 0.20
        with self.assertRaises(TimeoutError):
            self.mod.upstream_chat({"model": "dgemma"}, timeout=0.04)
        self.assertIsNone(self.mod._UPSTREAM_LOCAL.entry)

        # Subsequent call gets a fresh connection without burning a retry on CannotSendRequest.
        recovered = self.mod.upstream_chat({"model": "dgemma"}, timeout=2.0)
        self.assertEqual(recovered["choices"][0]["echo_model"], "dgemma")

    def test_slot_distribution_preserves_extreme_logprobs_across_single_and_multi_sample_reads(self):
        # At gap = 45 nats, linear probability saturates to 1.0 in float64,
        # while slot_distribution['logprobs'] preserves exact [0.0, -45.0].
        dist_hi = self.mod.slot_distribution({10: -0.001, 20: -45.001}, [10, 20])
        self.assertEqual(dist_hi["probs"][0], 1.0)
        self.assertAlmostEqual(dist_hi["logprobs"][0], 0.0, places=9)
        self.assertAlmostEqual(dist_hi["logprobs"][1], -45.0, places=9)

        # Across 2 saturated draws (-42.0 and -46.0), float64 mean probability is [1.0, 0.0],
        # while logsumexp diagnostic logprobs preserve exact log(mean_p) = -42 + log(1 + exp(-4)) - log(2).
        self.mod.SCAFFOLD = [1, 2]
        self.mod.template_for = lambda schema, head, lead="": ([1, 2, 3], [{"pos": 2, "label_ids": [10, 20]}])
        sat_draws = [
            [{"probs": [1.0, math.exp(-42.0)], "logprobs": [0.0, -42.0],
              "label_mass": 1.0, "entropy": 0.0, "argmax_is_label": True}],
            [{"probs": [1.0, math.exp(-46.0)], "logprobs": [0.0, -46.0],
              "label_mass": 1.0, "entropy": 0.0, "argmax_is_label": True}],
        ]
        self.mod.read_many = lambda *args, **kwargs: (sat_draws, [{"prompt_tokens": 18}, {"prompt_tokens": 18}])
        schema = self.mod.parse_schema({
            "questions": [{"id": "allow", "type": "noul", "instructions": "Safe?"}],
            "samples": 2,
        })
        sat_out, _ = self.mod.decide_group(schema, "sys", "state", 42)
        want_no_lp = -42.0 + math.log(1.0 + math.exp(-4.0)) - math.log(2.0)
        self.assertEqual(sat_out["answers"]["allow"]["noul"], 1.0)
        self.assertAlmostEqual(sat_out["diagnostics"]["questions"]["allow"]["logprobs"]["yes"], 0.0, places=9)
        self.assertAlmostEqual(sat_out["diagnostics"]["questions"]["allow"]["logprobs"]["no"], want_no_lp, places=9)

        # Mixed draws preserve exact log-odds of the arithmetic-mean distribution (argmax-consistent with answers).
        lp0_yes = -math.log(1.0 + math.exp(20.0))
        lp0_no = -math.log(1.0 + math.exp(-20.0))
        lp1_yes = -math.log(1.0 + math.exp(-1.5))
        lp1_no = -math.log(1.0 + math.exp(1.5))
        mixed_draws = [
            [{"probs": [math.exp(lp0_yes), math.exp(lp0_no)], "logprobs": [lp0_yes, lp0_no],
              "label_mass": 1.0, "entropy": 0.0, "argmax_is_label": True}],
            [{"probs": [math.exp(lp1_yes), math.exp(lp1_no)], "logprobs": [lp1_yes, lp1_no],
              "label_mass": 1.0, "entropy": 0.47, "argmax_is_label": True}],
            [{"probs": [math.exp(lp1_yes), math.exp(lp1_no)], "logprobs": [lp1_yes, lp1_no],
              "label_mass": 1.0, "entropy": 0.47, "argmax_is_label": True}],
        ]
        self.mod.read_many = lambda *args, **kwargs: (mixed_draws, [{"prompt_tokens": 18}] * 3)
        schema3 = self.mod.parse_schema({
            "questions": [{"id": "allow", "type": "noul", "instructions": "Safe?"}],
            "samples": 3,
        })
        mixed_out, _ = self.mod.decide_group(schema3, "sys", "state", 42)
        mean_yes = mixed_out["answers"]["allow"]["noul"]
        mean_no = 1.0 - mean_yes
        self.assertGreater(mean_yes, 0.5)
        self.assertEqual(mixed_out["answers"]["allow"]["label"], "yes")
        self.assertAlmostEqual(
            mixed_out["diagnostics"]["questions"]["allow"]["logprobs"]["no"],
            math.log(mean_no / mean_yes),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
