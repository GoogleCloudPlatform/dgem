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
import sys
import threading
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


if __name__ == "__main__":
    unittest.main()
