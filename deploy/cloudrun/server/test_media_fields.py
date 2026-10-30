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
"""Media other than images is refused with 400: DiffusionGemma reads text and images only. Real HTTP, no
model (the decide step is stubbed; refused requests never reach it).

  python3 -m unittest deploy/cloudrun/server/test_media_fields.py
"""

import http.client
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server_key import load_server  # noqa: E402


class MediaFieldsTest(unittest.TestCase):
    def setUp(self):
        m = load_server("")
        self.decided = []
        m.Handler._decide = lambda h, schema, state, seed: (self.decided.append(1), (500, {"error": {"message": "stub"}}))[1]
        srv = m._Server(("127.0.0.1", 0), m.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
        self.addCleanup(self.conn.close)

    def post(self, path, body):
        self.conn.request("POST", path, body=json.dumps(body).encode(), headers={"content-type": "application/json"})
        r = self.conn.getresponse()
        return r.status, json.loads(r.read())

    def test_systemone_media_keys(self):
        base = {"state": "x", "questions": {"q": {"type": "noul"}}}
        for key in ("videos", "video", "audio"):
            status, body = self.post("/v1/systemone", dict(base, **{key: ["data:video/mp4;base64,AAAA"]}))
            self.assertEqual(status, 400, key)
            self.assertIn("text and images only", body["error"]["message"])
        self.assertEqual(self.decided, [])
        # images alone still reach the decision step
        self.post("/v1/systemone", dict(base, images=["data:image/png;base64,AAAA"]))
        self.assertEqual(self.decided, [1])

    def test_chat_content_parts(self):
        schema = {"questions": [{"id": "q", "type": "noul"}]}
        for part in ({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}},
                     {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}):
            status, body = self.post("/v1/chat/completions", {"messages": [
                {"role": "system", "content": json.dumps(schema)},
                {"role": "user", "content": [part, {"type": "text", "text": "{}"}]}]})
            self.assertEqual(status, 400, part["type"])
            self.assertIn("not supported", body["error"]["message"])
        self.assertEqual(self.decided, [])


    def test_chat_image_urls_denied_by_default(self):
        schema = {"questions": [{"id": "q", "type": "noul"}]}
        def chat(url, kind="image_url"):
            part = {"type": "image_url", "image_url": {"url": url}} if kind == "image_url" else {"type": "image", "image": url}
            return self.post("/v1/chat/completions", {"messages": [
                {"role": "system", "content": json.dumps(schema)},
                {"role": "user", "content": [part, {"type": "text", "text": "{}"}]}]})
        for url, kind in (("https://example.com/a.png", "image_url"), ("http://169.254.169.254/x", "image_url"),
                          ("https://example.com/a.png", "image")):
            status, body = chat(url, kind)
            self.assertEqual(status, 400, url)
            self.assertIn("data: URIs", body["error"]["message"])
        self.assertEqual(self.decided, [])
        chat("data:image/png;base64,AAAA")  # data URIs still reach the decision step
        self.assertEqual(self.decided, [1])

    def test_allow_mode(self):
        m = load_server("")
        m.IMAGE_URLS = "allow"
        m.check_content_parts([{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}])


if __name__ == "__main__":
    unittest.main()
