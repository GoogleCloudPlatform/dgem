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
"""X-DGem-Key on the structured server: real HTTP over one keep-alive connection, no model needed.

  python3 -m unittest deploy/cloudrun/server/test_server_key.py
"""

import http.client
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def load_server(keys):
    os.environ["DGEM_SERVER_KEY"] = keys
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
    spec = importlib.util.spec_from_file_location(f"ss_key_{abs(hash(keys))}", os.path.join(HERE, "structured_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.environ.pop("DGEM_SERVER_KEY", None)
    # No model: health doesn't need one, and a body that fails validation stops before any upstream call.
    mod._health_info = lambda: {"status": "ok"}
    return mod


class ServerKeyTest(unittest.TestCase):
    def serve(self, keys):
        mod = load_server(keys)
        srv = mod._Server(("127.0.0.1", 0), mod.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        self.addCleanup(conn.close)
        return conn, mod

    def post(self, conn, key=None, body=b'{"messages": []}'):
        headers = {"content-type": "application/json"}
        if key is not None:
            headers["X-DGem-Key"] = key
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        r = conn.getresponse()
        return r.status, json.loads(r.read())

    def test_keys_required_health_open_keepalive(self):
        conn, _ = self.serve("old-key, new-key")
        # one keep-alive connection throughout: a 401 must not desync it
        self.assertEqual(self.post(conn)[0], 401)
        self.assertEqual(self.post(conn, "wrong")[0], 401)
        conn.request("GET", "/health")
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        r.read()
        # both rotation keys pass auth; the empty chat body then fails validation (400), so no model is needed
        self.assertEqual(self.post(conn, "old-key")[0], 400)
        self.assertEqual(self.post(conn, "new-key")[0], 400)
        self.assertEqual(self.post(conn, "")[0], 401)

    def test_no_key_configured_is_open(self):
        conn, mod = self.serve("")
        self.assertEqual(mod.SERVER_KEYS, [])
        self.assertEqual(self.post(conn)[0], 400)

    def test_raw_route_also_needs_key(self):
        conn, _ = self.serve("k")
        conn.request("POST", "/v1/raw/chat/completions", body=b"{}", headers={"content-type": "application/json"})
        r = conn.getresponse()
        self.assertEqual(r.status, 401)
        r.read()

    def test_refused_body_is_drained_keepalive(self):
        conn, _ = self.serve("k")
        # a body larger than one read chunk is read and dropped, so the same connection serves the next request
        self.assertEqual(self.post(conn, "wrong", body=b"x" * 300_000)[0], 401)
        conn.request("GET", "/health")
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        r.read()

    def test_refused_large_body_closes_connection(self):
        conn, mod = self.serve("k")
        # only the headers are sent: a refused body over the limit is not read, the connection is closed instead
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("content-type", "application/json")
        conn.putheader("content-length", str(mod.REFUSED_BODY_LIMIT + 1))
        conn.putheader("X-DGem-Key", "wrong")
        conn.endheaders()
        r = conn.getresponse()
        self.assertEqual(r.status, 401)
        self.assertEqual(r.getheader("connection"), "close")
        r.read()


class EntrypointSelfWarmupKeyTest(unittest.TestCase):
    """Self-warmup in deploy/cloudrun/entrypoint.sh sends the first DGEM_SERVER_KEY entry as X-DGem-Key."""

    def _extract_heredoc(self, tag):
        ep_path = os.path.join(HERE, "..", "entrypoint.sh")
        with open(ep_path) as f:
            src = f.read()
        m = re.search(rf"<<'{tag}'.*?\n(.*?)\n{tag}\n", src, re.DOTALL)
        self.assertIsNotNone(m, f"missing heredoc {tag} in entrypoint.sh")
        return m.group(1)

    def _serve_stub_chat(self, server_keys, api_key=""):
        mod = load_server(server_keys)
        mod.API_KEY = api_key
        seen = []

        def stub_chat(handler_self, req):
            seen.append({
                "x_dgem_key": handler_self.headers.get("x-dgem-key"),
                "authorization": handler_self.headers.get("authorization"),
                "messages": req.get("messages", []),
            })
            return handler_self._json(200, {"choices": [{"message": {"content": "{}"}}]})

        mod.Handler._chat = stub_chat
        srv = mod._Server(("127.0.0.1", 0), mod.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1], seen

    def test_self_warmup_sends_first_server_key_and_records_warmed(self):
        port, seen = self._serve_stub_chat(" primary-key , backup-key ", api_key="local-bearer")
        pywarm = self._extract_heredoc("PYWARM")
        pystate = self._extract_heredoc("PYSTATE")

        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "warmup_state.json")
            with open(state_file, "w") as wf:
                json.dump({"phase": "ready", "vllm_ready": True}, wf)

            env = os.environ.copy()
            env.update({
                "PORT": "9999",  # wrong port on purpose: SS_PORT must win (e.g. ROLE=decision-index)
                "SS_PORT": str(port),
                "DISABLE_MM": "0",
                "DGEM_SERVER_KEY": " primary-key , backup-key ",
                "API_KEY": "local-bearer",
                "DGEM_WARMUP_STATE_FILE": state_file,
            })
            res = subprocess.run([sys.executable, "-c", pywarm], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            # samples=1 text + samples=1 image + samples=4 text = 3 requests
            self.assertEqual(len(seen), 3)
            for req in seen:
                self.assertEqual(req["x_dgem_key"], "primary-key")
                self.assertEqual(req["authorization"], "Bearer local-bearer")

            st_res = subprocess.run(
                [sys.executable, "-c", pystate, str(time.time() - 1.2), "1"],
                env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(st_res.returncode, 0, st_res.stderr)
            with open(state_file) as wf:
                st = json.load(wf)
            self.assertIs(st["warmed"], True)
            self.assertIs(st["vllm_ready"], True)

    def test_self_warmup_401_exits_nonzero_and_records_unwarmed(self):
        port, seen = self._serve_stub_chat("expected-key")
        pywarm = self._extract_heredoc("PYWARM")
        pystate = self._extract_heredoc("PYSTATE")

        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "warmup_state.json")
            env = os.environ.copy()
            env.update({
                "PORT": str(port),
                "SS_PORT": str(port),
                "DISABLE_MM": "1",
                "DGEM_SERVER_KEY": "wrong-key",
                "DGEM_WARMUP_STATE_FILE": state_file,
            })
            res = subprocess.run([sys.executable, "-c", pywarm], env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(res.returncode, 0)
            self.assertEqual(len(seen), 0)
            self.assertIn("401", res.stdout)

            subprocess.run(
                [sys.executable, "-c", pystate, str(time.time()), "0"],
                env=env, capture_output=True, text=True, timeout=10, check=True,
            )
            with open(state_file) as wf:
                st = json.load(wf)
            self.assertIs(st["warmed"], False)


if __name__ == "__main__":
    unittest.main()
