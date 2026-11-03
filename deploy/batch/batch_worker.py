#!/usr/bin/env python3
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

"""dgem batch runner. Standard library only, so it runs under any Python in the dgem image.

One runner for every batch platform (see docs/deploy/batch.md):

  job             Cloud Run Jobs task: start the dgem server in-process (entrypoint.sh), take this task's share of
                  the input files (CLOUD_RUN_TASK_INDEX / CLOUD_RUN_TASK_COUNT), write one output file per input file.
  remote          Endpoint fan-out: the same loop against a running dgem service (for example Cloud Run).
  vertex-adapter  Vertex batch prediction: an AIP_HTTP_PORT server translating {"instances": [...]} into concurrent
                  decisions against the in-process dgem server.

The Beam pipeline (dgem_beam.py) imports LocalServer / Decider from here.

Inputs are JSONL rows {"id", <state_key>, "gold"}; outputs are JSONL rows
{"id", "answers": {q: label}, "conf": {q: p}, "H": {q: nats}, "ms", "pt", "gold"} or {"id", "error"}.
A metrics JSON per worker records boot, server-ready, first-decision and end timestamps.
"""
import argparse
import http.server
import json
import math
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROCESS_START = time.time()
LOCAL_PORT = int(os.environ.get("DGEM_LOCAL_PORT", "8090"))


def log(*a):
    print(f"[batch {time.time() - PROCESS_START:8.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------- auth + GCS

class Token:
    """Access token from the metadata server (GCP workloads) or CLOUDSDK_AUTH_ACCESS_TOKEN / gcloud (laptop)."""

    def __init__(self):
        self.value, self.exp, self.lock = None, 0, threading.Lock()

    def get(self):
        with self.lock:
            if not self.value or time.time() > self.exp:
                self.value, ttl = self._fetch()
                self.exp = time.time() + ttl
            return self.value

    @staticmethod
    def _fetch():
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"}), timeout=3))
            return r["access_token"], max(60, r.get("expires_in", 600) - 120)
        except Exception:
            pass
        if os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN"):
            return os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"], 1800
        return subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip(), 1800


TOKEN = Token()


def _gcs_split(uri):
    assert uri.startswith("gs://"), uri
    b, _, o = uri[5:].partition("/")
    return b, o


def _http(req, timeout=120, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except (urllib.error.URLError, socket.timeout, ConnectionError):
            if i == tries - 1:
                raise
        time.sleep(min(30, 2 ** i))


def gcs_list(prefix):
    b, p = _gcs_split(prefix)
    out, tok = [], None
    while True:
        q = {"prefix": p, "fields": "items(name),nextPageToken"}
        if tok:
            q["pageToken"] = tok
        r = json.loads(_http(urllib.request.Request(f"https://storage.googleapis.com/storage/v1/b/{b}/o?" + urllib.parse.urlencode(q),
                                                    headers={"Authorization": "Bearer " + TOKEN.get()})))
        out += [f"gs://{b}/{i['name']}" for i in r.get("items", [])]
        tok = r.get("nextPageToken")
        if not tok:
            return sorted(out)


def gcs_read(uri):
    b, o = _gcs_split(uri)
    return _http(urllib.request.Request(f"https://storage.googleapis.com/storage/v1/b/{b}/o/{urllib.parse.quote(o, safe='')}?alt=media",
                                        headers={"Authorization": "Bearer " + TOKEN.get()}), timeout=300)


def gcs_exists(uri):
    b, o = _gcs_split(uri)
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://storage.googleapis.com/storage/v1/b/{b}/o/{urllib.parse.quote(o, safe='')}",
                                                      headers={"Authorization": "Bearer " + TOKEN.get()}), timeout=30)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def gcs_write(uri, data, content_type="application/json"):
    b, o = _gcs_split(uri)
    _http(urllib.request.Request(f"https://storage.googleapis.com/upload/storage/v1/b/{b}/o?uploadType=media&name={urllib.parse.quote(o, safe='')}",
                                 data=data, method="POST",
                                 headers={"Authorization": "Bearer " + TOKEN.get(), "Content-Type": content_type}), timeout=300)


def read_text(uri):
    return gcs_read(uri).decode() if uri.startswith("gs://") else open(uri).read()


# ---------------------------------------------------------------------------- dgem server in this container

def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"unavailable: {e!r}"


class LocalServer:
    """Starts the image's /entrypoint.sh (vLLM + structured_server.py) on LOCAL_PORT and waits until warmed."""

    def __init__(self, port=LOCAL_PORT, entrypoint="/entrypoint.sh", env=None):
        self.port, self.entrypoint, self.extra_env = port, entrypoint, env or {}
        self.proc, self.t_ready = None, None
        self.url = f"http://127.0.0.1:{port}"

    def start(self):
        env = dict(os.environ)
        # entrypoint.sh prefers AIP_HTTP_PORT over PORT; the adapter owns AIP_HTTP_PORT on Vertex.
        env.pop("AIP_HTTP_PORT", None)
        # Beam's venv must not leak into vLLM's interpreter.
        env["PATH"] = os.environ.get("DGEM_SYSTEM_PATH", "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        # DGEM_SERVER_KEY would make the in-container server require X-DGem-Key, which local callers don't send.
        for k in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "DGEM_SERVER_KEY"):
            env.pop(k, None)
        env.update({"PORT": str(self.port)})
        env.update(self.extra_env)
        log("starting dgem server:", {k: env.get(k) for k in ("DGEM_WEIGHTS_URI", "MODEL", "KV_CACHE_GB", "MAX_SEQS", "MAX_INFLIGHT",
                                                             "GPU_UTIL", "MAX_MODEL_LEN", "DISABLE_MM", "CANVAS", "COPY_TO_SHM")})
        self.proc = subprocess.Popen([self.entrypoint], env=env, stdout=sys.stdout, stderr=sys.stderr)
        return self

    def wait_ready(self, timeout=2400):
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"dgem server exited with {self.proc.returncode}")
            try:
                h = json.load(urllib.request.urlopen(self.url + "/health", timeout=2))
                phase = (h.get("phase"), h.get("vllm_ready"), h.get("warmed"))
                if phase != last:
                    log("health:", phase)
                    last = phase
                if h.get("vllm_ready") and (h.get("warmed") or os.environ.get("SELF_WARMUP") == "0"):
                    self.t_ready = time.time()
                    log(f"dgem server ready after {self.t_ready - t0:.1f}s")
                    return h
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("dgem server not ready")


# ---------------------------------------------------------------------------- decisions

def compact(content):
    env = json.loads(content)
    ans, conf, ent = {}, {}, {}
    for q, a in (env.get("answers") or {}).items():
        if a is None:
            ans[q] = None
            continue
        ans[q] = a.get("choice") or a.get("level") or a.get("label")  # option name, not the letter/digit code
        probs = a.get("probabilities") or {}
        if probs:
            ps = [float(p) for p in probs.values()]
            conf[q] = round(max(ps), 5)
            ent[q] = round(-sum(p * math.log(p) for p in ps if p > 0), 5)
    d = env.get("diagnostics") or {}
    return {"answers": ans, "conf": conf, "H": ent, "ms": round(((d.get("timing") or {}).get("total_ms") or 0), 2),
            "pt": d.get("prompt_tokens")}


SERVER_KEY_HEADER = "X-DGem-Key"


def server_key_from_env():
    """The serving image's service-to-service secret (DGEM_SERVER_KEY on the server): the first entry of a
    comma-separated rotation list, or "" when unset. Services hold it; never put it in a file or give it to a person."""
    return os.environ.get("DGEM_SERVER_KEY", "").split(",")[0].strip()


class Decider:
    def __init__(self, base_url, schema, state_key, auth=None, timeout=300, tries=8, server_key=""):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.server_key = server_key
        self.schema_str = schema if isinstance(schema, str) else json.dumps(schema)
        self.state_key, self.auth, self.timeout, self.tries = state_key, auth, timeout, tries
        self.lock = threading.Lock()
        self.retries = self.errors = self.ok = 0
        self.t_first = None

    def decide(self, row):
        state = {self.state_key: row[self.state_key]}
        body = json.dumps({"model": "dgemma", "messages": [{"role": "system", "content": self.schema_str},
                                                         {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]}).encode()
        err = None
        for i in range(self.tries):
            h = {"content-type": "application/json"}
            if self.auth:
                h["Authorization"] = "Bearer " + self.auth.get()
            if self.server_key:
                h[SERVER_KEY_HEADER] = self.server_key
            try:
                r = json.loads(urllib.request.urlopen(urllib.request.Request(self.url, body, h), timeout=self.timeout).read())
                out = {"id": row.get("id")} | compact(r["choices"][0]["message"]["content"])
                if "gold" in row:
                    out["gold"] = row["gold"]
                with self.lock:
                    self.ok += 1
                    if self.t_first is None:
                        self.t_first = time.time()
                return out
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:200]!r}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:  # connection reset, timeout
                err = repr(e)
            with self.lock:
                self.retries += 1
            time.sleep(min(30, 0.5 * 2 ** i))
        with self.lock:
            self.errors += 1
        return {"id": row.get("id"), "error": err}

    def decide_many(self, rows, pool):
        return list(pool.map(self.decide, rows))


def run_files(decider, files, out_prefix, concurrency, metrics, metrics_uri=None):
    pool = ThreadPoolExecutor(concurrency)
    for f in files:
        dst = out_prefix.rstrip("/") + "/" + os.path.basename(f)
        if gcs_exists(dst):
            log("skip (done):", dst)
            continue
        t0 = time.time()
        # split on "\n" only: comments contain U+2028 and other characters str.splitlines() treats as line breaks
        rows = [json.loads(l) for l in read_text(f).split("\n") if l.strip()]
        res = decider.decide_many(rows, pool)
        gcs_write(dst, ("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n").encode(), "application/x-ndjson")
        dt = time.time() - t0
        metrics["files"].append({"file": os.path.basename(f), "rows": len(rows), "s": round(dt, 2),
                                 "errors": sum(1 for r in res if "error" in r)})
        log(f"{os.path.basename(f)}: {len(rows)} rows in {dt:.1f}s ({len(rows) / dt:.1f}/s), ok={decider.ok} err={decider.errors} retries={decider.retries}")
        if metrics_uri:
            metrics.update(t_first_decision=decider.t_first, ok=decider.ok, errors=decider.errors, retries=decider.retries, t_last=time.time())
            gcs_write(metrics_uri, json.dumps(metrics, indent=1).encode())
    pool.shutdown()


def batch_env():
    keys = ("KV_CACHE_GB", "MAX_SEQS", "MAX_INFLIGHT", "GPU_UTIL", "MAX_MODEL_LEN", "DISABLE_MM", "CANVAS", "DEFAULT_SAMPLES")
    return {k: os.environ[k] for k in keys if k in os.environ}


# ---------------------------------------------------------------------------- modes

def mode_job(a):
    idx = int(os.environ.get("CLOUD_RUN_TASK_INDEX", a.task_index))
    cnt = int(os.environ.get("CLOUD_RUN_TASK_COUNT", a.task_count))
    attempt = int(os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0"))
    metrics = {"mode": "job", "platform": a.platform, "task_index": idx, "task_count": cnt, "attempt": attempt,
               "t_process_start": PROCESS_START, "gpu": gpu_info(), "env": batch_env(), "concurrency": a.concurrency, "files": []}
    files = [f for f in gcs_list(a.input) if f.endswith(".jsonl")][idx::cnt]
    log(f"task {idx}/{cnt}: {len(files)} files")
    srv = LocalServer().start()
    srv.wait_ready()
    metrics["t_server_ready"] = srv.t_ready
    schema = read_text(a.schema)
    d = Decider(srv.url, schema, a.state_key)
    murl = f"{a.output.rstrip('/')}/_metrics/task-{idx:04d}-a{attempt}.json"
    run_files(d, files, a.output, a.concurrency, metrics, murl)
    metrics.update(t_end=time.time(), ok=d.ok, errors=d.errors, retries=d.retries, t_first_decision=d.t_first)
    gcs_write(murl, json.dumps(metrics, indent=1).encode())
    log("done", {k: metrics[k] for k in ("ok", "errors", "retries")})
    srv.proc.terminate()


def mode_remote(a):
    metrics = {"mode": "remote", "platform": a.platform, "url_kind": "cloudrun", "t_process_start": PROCESS_START,
               "concurrency": a.concurrency, "files": []}
    files = [f for f in gcs_list(a.input) if f.endswith(".jsonl")]
    auth = IdToken(a.url) if a.auth == "id" else None
    d = Decider(a.url, read_text(a.schema), a.state_key, auth=auth, server_key=server_key_from_env())
    murl = f"{a.output.rstrip('/')}/_metrics/remote.json"
    run_files(d, files, a.output, a.concurrency, metrics, murl)
    metrics.update(t_end=time.time(), ok=d.ok, errors=d.errors, retries=d.retries, t_first_decision=d.t_first)
    gcs_write(murl, json.dumps(metrics, indent=1).encode())
    log("done", {k: metrics[k] for k in ("ok", "errors", "retries")})


class IdToken:
    """OIDC identity token for calling a private Cloud Run service: metadata server on GCP, otherwise minted from the
    authorized_user ADC refresh token (no gcloud dependency). Refreshed every 30 minutes."""

    def __init__(self, audience=None):
        self.audience, self.v, self.exp, self.lock = audience, None, 0, threading.Lock()

    def get(self):
        with self.lock:
            if not self.v or time.time() > self.exp:
                self.v = self._fetch()
                self.exp = time.time() + 1800
            return self.v

    def _fetch(self):
        try:
            url = ("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience="
                   + urllib.parse.quote(self.audience or ""))
            return urllib.request.urlopen(urllib.request.Request(url, headers={"Metadata-Flavor": "Google"}), timeout=3).read().decode()
        except Exception:
            pass
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        c = json.load(open(path))
        body = urllib.parse.urlencode({"grant_type": "refresh_token", "client_id": c["client_id"], "client_secret": c["client_secret"],
                                       "refresh_token": c["refresh_token"]}).encode()
        return json.loads(urllib.request.urlopen("https://oauth2.googleapis.com/token", body, timeout=30).read())["id_token"]


def mode_vertex_adapter(a):
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    health_route = os.environ.get("AIP_HEALTH_ROUTE", "/health")
    predict_route = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
    state = {"ready": False, "decider": None, "n": 0, "t_first": None}
    pool = ThreadPoolExecutor(a.concurrency)
    log("vertex adapter on", port, health_route, predict_route, "gpu:", gpu_info())

    def boot():
        srv = LocalServer().start()
        srv.wait_ready()
        state["decider"] = Decider(srv.url, read_text(a.schema), a.state_key)
        state["ready"] = True
        log("adapter ready")

    threading.Thread(target=boot, daemon=True).start()

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith(health_route):
                return self._send(200 if state["ready"] else 503, {"ready": state["ready"]})
            self._send(404, {})

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if not state["ready"]:
                return self._send(503, {"error": "loading"})
            inst = body.get("instances") or []
            t0 = time.time()
            preds = state["decider"].decide_many(inst, pool)
            state["n"] += len(inst)
            log(f"predict: {len(inst)} instances in {time.time() - t0:.2f}s (total {state['n']}, err {state['decider'].errors})")
            self._send(200, {"predictions": preds})

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        request_queue_size = 256

    S(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["job", "remote", "vertex-adapter"])
    ap.add_argument("--input", default=os.environ.get("BATCH_INPUT"), help="gs:// prefix of JSONL parts")
    ap.add_argument("--output", default=os.environ.get("BATCH_OUTPUT"), help="gs:// prefix for outputs")
    ap.add_argument("--schema", default=os.environ.get("BATCH_SCHEMA"), help="schema JSON (gs:// or path)")
    ap.add_argument("--state-key", default=os.environ.get("BATCH_STATE_KEY", "comment"))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("BATCH_CONCURRENCY", "64")))
    ap.add_argument("--platform", default=os.environ.get("BATCH_PLATFORM", "unknown"))
    ap.add_argument("--url", default=os.environ.get("BATCH_URL"))
    ap.add_argument("--auth", default="id", choices=["id", "none"])
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--task-count", type=int, default=1)
    a = ap.parse_args()
    {"job": mode_job, "remote": mode_remote, "vertex-adapter": mode_vertex_adapter}[a.mode](a)


if __name__ == "__main__":
    main()
