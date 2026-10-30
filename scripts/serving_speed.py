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

"""Serving speed review: per-mode latency and concurrency sweeps against dgemma endpoints.

Targets are given as name=URL (the /v1/chat/completions URL of a structured server). Vertex AI
endpoints (*.prediction.vertexai.goog) get an OAuth access token; other URLs (Cloud Run) get an
ID token. Both come from Application Default Credentials.

  python3 scripts/serving_speed.py modes --target g4=<url> --target cloudrun=<url> -n 50 -o out.json
  python3 scripts/serving_speed.py sweep --target g4=<url> --workers 1 2 4 8 16 32 --samples 1 4 -o out.json

Every request records wall time, server-reported denoise time (diagnostics.timing.total_ms), reads,
HTTP status and error text, so failures are never silently counted as zero-latency.
"""

import argparse
import base64
import concurrent.futures as cf
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def server_key_header():
    """X-DGem-Key for the serving image when DGEM_SERVER_KEY is set (first entry of a rotation list)."""
    key = os.environ.get("DGEM_SERVER_KEY", "").split(",")[0].strip()
    return {"X-DGem-Key": key} if key else {}


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TICKETS = [
    "Customer reports double billing on annual plan ($1200 instead of $600).",
    "EMERGENCY: Production Kubernetes API cluster returning HTTP 502 across all US-East nodes.",
    "Hi there, how do I update my profile avatar in the settings page? No rush!",
    "Our GDPR Data Processing Addendum requires counter-signature from your legal counsel before Friday.",
    "Can we get SSO SAML metadata XML for our Okta staging environment?",
    "I was charged for 15 seats after downgrading to 10 seats last billing cycle.",
    "Webhook delivery latency spiked from 200ms to 45s over the last hour.",
    "Where can I download last quarter's SOC2 Type II compliance report?",
    "Your mobile app crashes on launch whenever FaceID is enabled on iOS 18.",
    "Please cancel our sandbox subscription at the end of the current month.",
]

Q = [
    {"id": "urgent", "type": "boolean", "instructions": "Does this issue require immediate same-day escalation?"},
    {"id": "team", "type": "choice", "instructions": "Which department owns resolution of this ticket?", "options": [
        {"name": "billing", "description": "Charges, invoices, payment methods, renewals"},
        {"name": "support", "description": "General questions, password resets, account setup"},
        {"name": "engineering", "description": "System outages, 500 errors, bugs, API failures"}]},
    {"id": "sentiment", "type": "score", "instructions": "Customer anger or distress level", "levels": ["calm", "frustrated", "furious"]},
]
MIRROR = [
    {"id": "urgent__rev", "type": "choice", "instructions": Q[0]["instructions"], "options": [
        {"name": "no", "description": "no"}, {"name": "yes", "description": "yes"}]},
    {"id": "team__rev", "type": "choice", "instructions": Q[1]["instructions"], "options": list(reversed(Q[1]["options"]))},
    {"id": "sentiment__rev", "type": "score", "instructions": Q[2]["instructions"], "levels": list(reversed(Q[2]["levels"]))},
]
TAXONOMY = [
    {"id": "department", "type": "choice", "instructions": "Route this request. Choose other_unclassified if none fit.", "options": [
        {"name": "it_helpdesk", "description": "Laptops, VPN, SSO resets"}, {"name": "hr_people_ops", "description": "Payroll, benefits, PTO"},
        {"name": "facilities", "description": "Badges, desks, building"}, {"name": "other_unclassified", "description": "None of the listed departments"}]},
    {"id": "requires_new_category", "type": "boolean", "instructions": "Is a department missing from the taxonomy?"},
]
IMAGE_Q = [
    {"id": "has_button", "type": "boolean", "instructions": "Does the image show a clickable button?"},
    {"id": "dominant_color", "type": "choice", "instructions": "Which color dominates the image?", "options": [
        {"name": "red", "description": "red"}, {"name": "green", "description": "green"}, {"name": "blue", "description": "blue"}, {"name": "white", "description": "white or light"}]},
]


def image_data_uri():
    for cand in ("fixtures/bbox/bbox-t1-03-offgrid-card.png", "fixtures/bbox/bbox-t1-01-centered-button.png"):
        p = os.path.join(REPO, cand)
        if os.path.exists(p):
            return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
    import glob
    p = sorted(glob.glob(os.path.join(REPO, "fixtures/bbox/*.png")))[0]
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


MODES = {
    "1A_samples1": lambda: ({"instructions": "Triage the ticket.", "samples": 1, "think": 0, "questions": Q}, None),
    "1B_samples4": lambda: ({"instructions": "Triage the ticket.", "samples": 4, "think": 0, "questions": Q}, None),
    "1C_auto": lambda: ({"instructions": "Triage the ticket.", "samples": "auto", "think": 0, "questions": Q}, None),
    "1D_mirror_s1": lambda: ({"instructions": "Triage the ticket.", "samples": 1, "think": 0, "questions": Q + MIRROR}, None),
    "1E_mirror_s4": lambda: ({"instructions": "Triage the ticket.", "samples": 4, "think": 0, "questions": Q + MIRROR}, None),
    "1F_think120": lambda: ({"instructions": "Route the request.", "samples": 2, "think": 120, "questions": TAXONOMY}, None),
    "1G_image_s1": lambda: ({"instructions": "Inspect the screenshot.", "samples": 1, "think": 0, "questions": IMAGE_Q}, "image"),
}


class Tokens:
    def __init__(self):
        self.lock = threading.Lock()
        self.at = self.it = None
        self.t = 0

    def get(self):
        with self.lock:
            if time.time() - self.t > 1500:
                d = json.load(open(os.path.expanduser("~/.config/gcloud/application_default_credentials.json")))
                body = urllib.parse.urlencode({"client_id": d["client_id"], "client_secret": d["client_secret"],
                                               "refresh_token": d["refresh_token"], "grant_type": "refresh_token"}).encode()
                r = json.load(urllib.request.urlopen("https://oauth2.googleapis.com/token", body))
                self.at, self.it, self.t = r["access_token"], r.get("id_token"), time.time()
            return self.at, self.it


TOK = Tokens()
IMG = None


def call(url, schema, image, ticket, timeout=300):
    global IMG
    at, it = TOK.get()
    token = at if "prediction.vertexai.goog" in url or "aiplatform.googleapis.com" in url else it
    if image:
        IMG = IMG or image_data_uri()
        user = [{"type": "text", "text": json.dumps({"note": ticket})}, {"type": "image_url", "image_url": {"url": IMG}}]
    else:
        user = json.dumps({"ticket": ticket})
    payload = {"model": "dgemma", "messages": [{"role": "system", "content": json.dumps(schema)}, {"role": "user", "content": user}],
               "logprobs": True, "top_logprobs": 5}
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {token}", **server_key_header()})
    t0 = time.perf_counter()
    rec = {"status": None, "error": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            rec["status"] = r.status
    except urllib.error.HTTPError as e:
        rec.update(status=e.code, error=e.read()[:300].decode(errors="replace"))
    except Exception as e:
        rec.update(error=repr(e)[:300])
    rec["wall_ms"] = (time.perf_counter() - t0) * 1000
    if rec["error"] is None:
        try:
            inner = json.loads(json.loads(raw)["choices"][0]["message"]["content"])
            t = inner.get("diagnostics", {}).get("timing", {})
            rec["denoise_ms"] = float(t.get("total_ms", 0) or 0)
            rec["reads"] = int(t.get("reads", 1) or 1)
        except Exception as e:
            rec["error"] = "parse: " + repr(e)[:200]
    return rec


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def summarize(recs, elapsed=None):
    ok = [r for r in recs if r["error"] is None]
    out = {"n": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
           "error_samples": sorted({f"{r['status']}: {(r['error'] or '')[:120]}" for r in recs if r["error"]})[:3]}
    for key in ("wall_ms", "denoise_ms"):
        xs = [r[key] for r in ok if key in r]
        if xs:
            out[key] = {"mean": round(statistics.mean(xs), 1), "p50": round(pct(xs, 50), 1), "p95": round(pct(xs, 95), 1), "p99": round(pct(xs, 99), 1)}
    if elapsed:
        out["elapsed_s"] = round(elapsed, 2)
        out["throughput_per_s"] = round(len(ok) / elapsed, 2)
    return out


def parse_targets(ts):
    return [tuple(t.split("=", 1)) for t in ts]


def cmd_modes(a):
    res = {"kind": "modes", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": a.n, "targets": {}}
    for name, url in parse_targets(a.target):
        res["targets"][name] = {"url": url, "modes": {}}
        for mode in a.modes:
            schema, img = MODES[mode]()
            for i in range(a.warmup):
                call(url, schema, img, TICKETS[i % len(TICKETS)])
            recs = [call(url, schema, img, TICKETS[i % len(TICKETS)]) for i in range(a.n)]
            s = summarize(recs)
            res["targets"][name]["modes"][mode] = {"summary": s, "requests": recs}
            print(f"{name:10} {mode:14} ok={s['ok']}/{s['n']} denoise p50={s.get('denoise_ms', {}).get('p50')} wall p50={s.get('wall_ms', {}).get('p50')} {s['error_samples'][:1]}", flush=True)
    json.dump(res, open(a.output, "w"), indent=1)


def cmd_sweep(a):
    res = {"kind": "sweep", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": {}}
    for name, url in parse_targets(a.target):
        res["targets"][name] = {"url": url, "cells": {}}
        maxw = a.max_workers.get(name) if a.max_workers else None
        for samples in a.samples:
            schema = {"instructions": "Triage the ticket.", "samples": samples, "think": 0, "questions": Q}
            for w in a.workers:
                if maxw and w > maxw:
                    print(f"{name:10} s={samples} w={w:<3} skipped (cap {maxw})", flush=True)
                    continue
                n = max(a.min_requests, 4 * w)
                t0 = time.perf_counter()
                with cf.ThreadPoolExecutor(w) as ex:
                    recs = list(ex.map(lambda i: call(url, schema, None, TICKETS[i % len(TICKETS)]), range(n)))
                s = summarize(recs, time.perf_counter() - t0)
                res["targets"][name]["cells"][f"s{samples}_w{w}"] = {"samples": samples, "workers": w, "summary": s, "requests": recs}
                print(f"{name:10} s={samples} w={w:<3} ok={s['ok']}/{s['n']} thr={s.get('throughput_per_s')}/s wall p50={s.get('wall_ms', {}).get('p50')} p95={s.get('wall_ms', {}).get('p95')} {s['error_samples'][:1]}", flush=True)
                if s["errors"] > n // 2 and a.stop_on_failure:
                    print(f"{name:10} stopping sweep at w={w}: majority failed", flush=True)
                    break
    json.dump(res, open(a.output, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("modes")
    m.add_argument("--target", action="append", required=True)
    m.add_argument("--modes", nargs="+", default=list(MODES))
    m.add_argument("-n", type=int, default=50)
    m.add_argument("--warmup", type=int, default=3)
    m.add_argument("-o", "--output", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--target", action="append", required=True)
    s.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    s.add_argument("--samples", type=int, nargs="+", default=[1, 4])
    s.add_argument("--min-requests", type=int, default=50)
    s.add_argument("--max-workers", type=lambda v: dict((k, int(x)) for k, x in (p.split(":") for p in v.split(","))), default=None,
                   help="per-target cap, e.g. l4:8")
    s.add_argument("--stop-on-failure", action="store_true")
    s.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    {"modes": cmd_modes, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
