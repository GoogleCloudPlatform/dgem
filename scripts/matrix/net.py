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

"""Auth and HTTP for matrix targets (standard library only).

Tokens, by target URL:
  *.prediction.vertexai.goog / aiplatform.googleapis.com  -> OAuth2 access token
  https://*.run.app (and other https hosts)                -> OIDC ID token, audience = scheme://host
  http://<host> (self-hosted)                              -> none, unless DGEM_MATRIX_TOKEN is set
DGEM_MATRIX_TOKEN always wins (an API key for a self-hosted server's API_KEY, or a pre-minted token).
Sources, in order: GCE/Cloud Run metadata server, then Application Default Credentials
(~/.config/gcloud/application_default_credentials.json, authorized_user refresh token), then gcloud.
"""
import http.client
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

_LOCK = threading.Lock()
_CACHE = {}  # key -> (token, minted_at)
_TTL = 1500
_META = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/"
_META_OK = None


def is_vertex(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host == "prediction.vertexai.goog" or host.endswith(".prediction.vertexai.goog"):
        return True
    if host == "aiplatform.googleapis.com" or host.endswith(".aiplatform.googleapis.com"):
        return True
    if host.endswith(".googleapis.com"):
        sub = host[: -len(".googleapis.com")]
        return sub.endswith("-aiplatform") and "." not in sub
    return False


def _metadata_available():
    global _META_OK
    if _META_OK is None:
        try:
            urllib.request.urlopen(urllib.request.Request(_META + "email", headers={"Metadata-Flavor": "Google"}),
                                   timeout=1).read()
            _META_OK = True
        except Exception:
            _META_OK = False
    return _META_OK


def _adc_refresh():
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json")
    with open(path) as f:
        d = json.load(f)
    if d.get("type") != "authorized_user":
        raise RuntimeError(f"ADC type {d.get('type')!r} not supported here; use the metadata server or gcloud")
    body = urllib.parse.urlencode({"client_id": d["client_id"], "client_secret": d["client_secret"],
                                   "refresh_token": d["refresh_token"], "grant_type": "refresh_token"}).encode()
    with urllib.request.urlopen("https://oauth2.googleapis.com/token", body, timeout=30) as r:
        return json.load(r)


def _mint(kind, audience):
    if _metadata_available():
        if kind == "access":
            raw = urllib.request.urlopen(urllib.request.Request(_META + "token", headers={"Metadata-Flavor": "Google"}),
                                         timeout=10).read()
            return json.loads(raw)["access_token"]
        url = _META + "identity?audience=" + urllib.parse.quote(audience, safe="")
        return urllib.request.urlopen(urllib.request.Request(url, headers={"Metadata-Flavor": "Google"}),
                                      timeout=10).read().decode()
    try:
        r = _adc_refresh()
        tok = r["access_token"] if kind == "access" else r.get("id_token")
        if tok:
            return tok
    except Exception:
        pass
    cmd = ["gcloud", "auth", "print-access-token"] if kind == "access" else ["gcloud", "auth", "print-identity-token"]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()


def token_for(url):
    """Bearer token for a target URL, or None."""
    if os.environ.get("DGEM_MATRIX_TOKEN"):
        return os.environ["DGEM_MATRIX_TOKEN"]
    p = urllib.parse.urlparse(url)
    if p.scheme != "https":
        return None
    kind = "access" if is_vertex(url) else "id"
    aud = f"{p.scheme}://{p.netloc}"
    key = (kind, None if kind == "access" else aud)
    with _LOCK:
        v = _CACHE.get(key)
        if v and time.time() - v[1] < _TTL:
            return v[0]
        tok = _mint(kind, aud)
        _CACHE[key] = (tok, time.time())
        return tok


CONNECT_TIMEOUT = 20.0  # TCP connect + TLS handshake; the read timeout (timeout=) can be much longer


class _HTTPSConnection(http.client.HTTPSConnection):
    """Connect and handshake under CONNECT_TIMEOUT, then switch the socket to the request's read timeout. A plain
    urlopen(timeout=300) lets a wedged TLS handshake hang a worker for minutes per attempt."""

    def __init__(self, *a, read_timeout=None, **kw):
        self._read_timeout = read_timeout
        super().__init__(*a, **kw)

    def connect(self):
        self.timeout = CONNECT_TIMEOUT
        super().connect()
        self.sock.settimeout(self._read_timeout)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, read_timeout):
        super().__init__()
        self._rt = read_timeout

    def https_open(self, req):
        return self.do_open(lambda host, **kw: _HTTPSConnection(host, read_timeout=self._rt, **kw), req)


_CREDENTIAL_HEADERS = ("authorization", "x-dgem-key", "x-dgem-api-key")


def _origin(url):
    try:
        u = urllib.parse.urlparse(url)
        scheme = (u.scheme or "").lower()
        host = (u.hostname or "").lower()
        port = u.port or (443 if scheme == "https" else 80 if scheme == "http" else None)
    except ValueError:
        return None
    if scheme not in ("http", "https") or not host:
        return None
    return (scheme, host, port)


def _same_origin(a, b):
    oa = _origin(a)
    return oa is not None and oa == _origin(b)


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Once a redirect leaves the first request's origin, strip Authorization, X-DGem-Key and X-DGem-Api-Key from that
    request and every later hop, including a redirect back to the first origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        first_url = getattr(req, "_first_url", req.full_url)
        crossed = getattr(req, "_crossed_origin", False) or not _same_origin(first_url, new_req.full_url)
        new_req._first_url = first_url
        new_req._crossed_origin = crossed
        if crossed:
            for d in (new_req.headers, new_req.unredirected_hdrs):
                for k in [k for k in d if k.lower() in _CREDENTIAL_HEADERS]:
                    del d[k]
        return new_req


def _open(req, timeout):
    if req.full_url.startswith("https://"):
        return urllib.request.build_opener(_RedirectHandler(), _HTTPSHandler(timeout)).open(req, timeout=timeout)
    return urllib.request.build_opener(_RedirectHandler()).open(req, timeout=timeout)


def server_key_header():
    """{"X-DGem-Key": key} when DGEM_SERVER_KEY is set (first entry of a rotation list), else {}. Send it only to dgem
    serving targets, never to any other server."""
    key = os.environ.get("DGEM_SERVER_KEY", "").split(",")[0].strip()
    return {"X-DGem-Key": key} if key else {}


def request(url, body=None, method=None, timeout=300, retries=3, raw=False, headers=None):
    """-> (status, parsed JSON or text, wall_ms, headers). Retries 429/5xx and network errors with backoff."""
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    method = method or ("POST" if data is not None else "GET")
    last = None
    for attempt in range(retries + 1):
        hdr = {"Content-Type": "application/json"}
        tok = token_for(url)
        if tok:
            hdr["Authorization"] = f"Bearer {tok}"
        hdr.update(headers or {})
        req = urllib.request.Request(url, data=data, method=method, headers=hdr)
        t0 = time.perf_counter()
        try:
            with _open(req, timeout) as r:
                b = r.read()
                ms = (time.perf_counter() - t0) * 1000
                return r.status, (b if raw else _parse(b)), ms, dict(r.headers)
        except urllib.error.HTTPError as e:
            b = e.read()
            last = (e.code, (b if raw else _parse(b)), (time.perf_counter() - t0) * 1000, dict(e.headers))
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                return last
        except Exception as e:  # network error
            last = (None, {"error": repr(e)[:300]}, (time.perf_counter() - t0) * 1000, {})
            if attempt == retries:
                return last
        time.sleep(2 * 2 ** attempt)
    return last


def _parse(b):
    try:
        return json.loads(b)
    except Exception:
        return {"error": b[:300].decode(errors="replace")}
