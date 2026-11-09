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
"""Single-route dispatch for Vertex :rawPredict (private/PSC endpoints): /predict answers health, systemone and chat
by body shape. Real HTTP, no model (the handlers are stubbed to report which route ran).

  python3 -m unittest deploy/cloudrun/server/test_predict_route.py
"""

import http.client
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server_key import load_server  # noqa: E402


class PredictRouteTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_server("svc")
        m = self.mod
        m.Handler._systemone = lambda self, req, images: self._json(200, {"route": "systemone", "images": len(images)})
        m.Handler._chat = lambda self, req: self._json(200, {"route": "chat"})
        self.vllm_ready = True
        m._health_info = lambda: {"route": "health", "status": "ok", "vllm_ready": self.vllm_ready}
        srv = m._Server(("127.0.0.1", 0), m.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        self.addCleanup(self.conn.close)

    def post(self, path, body, ctype="application/json", key="svc"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h = {"content-type": ctype}
        if key:
            h["X-DGem-Key"] = key
        self.conn.request("POST", path, body=data, headers=h)
        r = self.conn.getresponse()
        return r.status, json.loads(r.read())

    def test_dispatch_by_body(self):
        chat = {"messages": [{"role": "system", "content": "{}"}, {"role": "user", "content": "{}"}]}
        so = {"state": "x", "questions": {"q": {"type": "noul"}}}
        for path in ("/predict", "/rawPredict"):
            self.assertEqual(self.post(path, chat)[1]["route"], "chat")
            self.assertEqual(self.post(path, so)[1]["route"], "systemone")
            self.assertEqual(self.post(path, {"health": True}), (200, {"route": "health", "status": "ok", "vllm_ready": True}))
            self.vllm_ready = False
            self.assertEqual(self.post(path, {"ready": True})[0], 503)
            self.assertEqual(self.post(path, {"health": True})[0], 200)
            self.vllm_ready = True
            self.assertEqual(self.post(path, {"ready": True})[0], 200)
        # explicit routes unchanged
        self.assertEqual(self.post("/v1/systemone", so)[1]["route"], "systemone")
        self.assertEqual(self.post("/v1/chat/completions", chat)[1]["route"], "chat")

    def test_multipart_goes_to_systemone(self):
        b = "XyZ"
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n"
                '{"state":"x","questions":{"q":{"type":"noul"}}}\r\n'
                f"--{b}\r\nContent-Disposition: form-data; name=\"i\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n"
                "\x89PNG\r\n"
                f"--{b}--\r\n").encode("latin-1")
        status, out = self.post("/predict", body, f"multipart/form-data; boundary={b}")
        self.assertEqual((status, out["route"], out["images"]), (200, "systemone", 1))

    def test_key_still_required_on_predict(self):
        self.assertEqual(self.post("/predict", {"health": True}, key=None)[0], 401)


if __name__ == "__main__":
    unittest.main()
