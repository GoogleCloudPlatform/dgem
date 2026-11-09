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
"""/ready and /invoke/ready readiness probe routes vs /health cold-start telemetry on structured_server.py.

  python3 -m unittest deploy/cloudrun/server/test_ready_route.py
"""

import argparse
import http.client
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_real_health_server(keys="svc-key", warmup_path=""):
    os.environ["DGEM_SERVER_KEY"] = keys
    if warmup_path:
        os.environ["DGEM_WARMUP_STATE_FILE"] = warmup_path
    else:
        os.environ.pop("DGEM_WARMUP_STATE_FILE", None)
    for name in ("pybase64", "transformers"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                m = types.ModuleType(name)
                m.AutoTokenizer = None
                if name == "pybase64":
                    import base64 as m  # noqa: F811
                sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(
        f"ss_ready_{abs(hash((keys, warmup_path)))}",
        os.path.join(HERE, "structured_server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.environ.pop("DGEM_SERVER_KEY", None)
    os.environ.pop("DGEM_WARMUP_STATE_FILE", None)
    return mod


def _unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ReadyRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.warmup_path = os.path.join(self.tmp.name, "warmup_state.json")
        self.mod = _load_real_health_server("svc-key", self.warmup_path)

        srv = self.mod._Server(("127.0.0.1", 0), self.mod.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        self.conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        self.addCleanup(self.conn.close)

    def get(self, path):
        self.conn.request("GET", path)
        r = self.conn.getresponse()
        return r.status, json.loads(r.read())

    def post(self, path, body=b"", key="svc-key"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {"content-type": "application/json"}
        if key is not None:
            headers["X-DGem-Key"] = key
        self.conn.request("POST", path, body=data, headers=headers)
        r = self.conn.getresponse()
        return r.status, json.loads(r.read())

    def test_no_upstream_returns_503_on_ready_and_200_on_health(self):
        dead_port = _unused_port()
        self.mod.ARGS = argparse.Namespace(upstream=f"http://127.0.0.1:{dead_port}")
        with open(self.warmup_path, "w") as wf:
            json.dump({"phase": "staging_tmpfs", "bytes_staged_gb": 8.4}, wf)

        for ready_path in ("/ready", "/invoke/ready"):
            status, body = self.get(ready_path)
            self.assertEqual(status, 503)
            self.assertIs(body["vllm_ready"], False)
            self.assertEqual(body["phase"], "staging_tmpfs")
            self.assertEqual(body["bytes_staged_gb"], 8.4)

        for health_path in ("/health", "/invoke/health", "/"):
            status, body = self.get(health_path)
            self.assertEqual(status, 200)
            self.assertIs(body["vllm_ready"], False)
            self.assertEqual(body["phase"], "staging_tmpfs")

    def test_upstream_200_transitions_ready_to_200_and_back_to_503_when_stopped(self):
        state = {"code": 200}

        class StubUpstream(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(state["code"])
                    self.send_header("content-length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                self.send_response(404)
                self.end_headers()

        up = ThreadingHTTPServer(("127.0.0.1", 0), StubUpstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        self.addCleanup(up.server_close)
        self.addCleanup(up.shutdown)

        self.mod.ARGS = argparse.Namespace(upstream=f"http://127.0.0.1:{up.server_address[1]}")

        for path in ("/ready", "/invoke/ready", "/health", "/invoke/health"):
            status, body = self.get(path)
            self.assertEqual(status, 200)
            self.assertIs(body["vllm_ready"], True)
            self.assertEqual(body["phase"], "ready")

        # Simulate entrypoint.sh having written vllm_ready=True + phase=ready to warmup_state.json
        # before the upstream engine crashes or begins returning 503.
        with open(self.warmup_path, "w") as wf:
            json.dump({"phase": "ready", "bytes_staged_gb": 17.53, "vllm_ready": True, "warmed": True}, wf)

        state["code"] = 503
        status, body = self.get("/ready")
        self.assertEqual(status, 503)
        self.assertIs(body["vllm_ready"], False)
        self.assertEqual(body["phase"], "loading_vllm_siglip")
        self.assertIs(body["warmed"], True)

        # Shut down the upstream completely (connection refused, as after an engine OOM exit).
        up.shutdown()
        up.server_close()
        for ready_path in ("/ready", "/invoke/ready"):
            status, body = self.get(ready_path)
            self.assertEqual(status, 503)
            self.assertIs(body["vllm_ready"], False)
            self.assertEqual(body["phase"], "loading_vllm_siglip")

        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertIs(body["vllm_ready"], False)

    def test_post_ready_and_health_require_server_key_and_preserve_keepalive(self):
        dead_port = _unused_port()
        self.mod.ARGS = argparse.Namespace(upstream=f"http://127.0.0.1:{dead_port}")

        # POST /ready and /health require X-DGem-Key when DGEM_SERVER_KEY is set (unlike open GET probes).
        self.assertEqual(self.post("/ready", b"{}", key=None)[0], 401)
        self.assertEqual(self.post("/invoke/ready", b"{}", key="wrong")[0], 401)
        self.assertEqual(self.post("/health", b"{}", key=None)[0], 401)

        # With valid X-DGem-Key, empty body and JSON body both work over the same keep-alive connection.
        for path in ("/ready", "/invoke/ready"):
            st, body = self.post(path, b"")
            self.assertEqual((st, body["vllm_ready"]), (503, False))
            st, body = self.post(path, {"ready": True})
            self.assertEqual((st, body["vllm_ready"]), (503, False))
        for path in ("/health", "/invoke/health"):
            st, body = self.post(path, b"")
            self.assertEqual((st, body["vllm_ready"]), (200, False))
        st, body = self.post("/predict", {"ready": True})
        self.assertEqual((st, body["vllm_ready"]), (503, False))

        class ReadyUpstream(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        up = ThreadingHTTPServer(("127.0.0.1", 0), ReadyUpstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        self.addCleanup(up.server_close)
        self.addCleanup(up.shutdown)
        self.mod.ARGS = argparse.Namespace(upstream=f"http://127.0.0.1:{up.server_address[1]}")

        for path in ("/ready", "/invoke/ready"):
            st, body = self.post(path, b"")
            self.assertEqual((st, body["vllm_ready"]), (200, True))
        st, body = self.post("/predict", {"ready": True})
        self.assertEqual((st, body["vllm_ready"]), (200, True))


if __name__ == "__main__":
    unittest.main()
