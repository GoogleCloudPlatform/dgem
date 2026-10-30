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
"""Model-server errors on the structured server: a vLLM 4xx (e.g. an image URL it can't fetch) is the caller's error
(422 on /v1/systemone, 400 on the OpenAI-shaped chat route), a vLLM 5xx stays 502. Real HTTP, no model.

  python3 -m unittest deploy/cloudrun/server/test_upstream_errors.py
"""

import http.client
import io
import json
import os
import threading
import unittest
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server_key import load_server  # noqa: E402  (same directory; stubs pybase64/transformers)


class UpstreamErrorTest(unittest.TestCase):
    def status_for(self, code, message, route="/v1/chat/completions"):
        mod = load_server("")

        def fake_decide(schema, state, seed):
            raise urllib.error.HTTPError("http://vllm/v1/chat/completions", code, "err", {},
                                         io.BytesIO(json.dumps({"error": {"message": message}}).encode()))

        mod.decide = fake_decide
        schema = {"questions": [{"id": "q", "type": "choice", "options": ["a", "b"]}]}
        if route == "/v1/systemone":
            body = json.dumps({"state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "", "b": ""}}}}).encode()
        else:
            body = json.dumps({"messages": [{"role": "system", "content": json.dumps(schema)},
                                            {"role": "user", "content": "{}"}]}).encode()
        srv = mod._Server(("127.0.0.1", 0), mod.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        self.addCleanup(conn.close)
        conn.request("POST", route, body=body, headers={"content-type": "application/json"})
        r = conn.getresponse()
        return r.status, json.loads(r.read())

    def test_client_error_is_4xx(self):
        # /v1/systemone answers 422; the OpenAI-shaped chat route reports request errors as 400
        status, body = self.status_for(400, "Failed to fetch media", "/v1/systemone")
        self.assertEqual(status, 422)
        self.assertIn("Failed to fetch media", body["error"]["message"])
        status, body = self.status_for(400, "Failed to fetch media")
        self.assertEqual(status, 400)
        self.assertIn("upstream 400: ", body["error"]["message"])
        self.assertIn("Failed to fetch media", body["error"]["message"])

    def test_server_error_stays_502(self):
        self.assertEqual(self.status_for(500, "engine dead")[0], 502)
        self.assertEqual(self.status_for(503, "engine dead", "/v1/systemone")[0], 502)


if __name__ == "__main__":
    unittest.main()
