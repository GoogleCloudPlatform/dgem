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

"""Scheduled health check (synthetic probe) for dgem serving endpoints.

For each target it runs the API contract cases from scripts/contract_diff.py (a failure is any case whose HTTP
status differs from the expected one, or a valid case without answers) and a short latency sample, then writes
ONE structured JSON log line per target to stdout. On Cloud Run jobs, Cloud Logging parses that line; the
log-based metrics and alert policies created by scripts/setup_cloud_monitoring.sh read these fields:

  probe_event="dgem.probe"   probe_target   probe_ok (true/false)   probe_failures (list)
  probe_latency_p50_ms   probe_latency_p95_ms   probe_denoise_p50_ms   probe_version

Targets (name=URL, repeatable, or PROBE_TARGETS="name=url,name=url"):
  - a serving base URL (Cloud Run https://<service>.run.app, or a Vertex dedicated .../invoke base): contract + latency
  - gateway=https://<gateway>: one /api/decide through the gateway default route + /health

  python3 scripts/probe.py --target vertex=<invoke-base> --target gateway=https://<gateway> [-n 20]

Exit code 1 if any target failed (the Cloud Run job execution is then marked failed as well).
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import contract_diff as cd  # noqa: E402
import serving_speed as ss  # noqa: E402

# Expected HTTP status per contract case (anything else is a failure).
EXPECTED = {
    "choice_27_error": 400, "unknown_type_error": 400, "empty_questions_error": 400, "invalid_json_error": 400,
}


def token_provider():
    """ADC locally; the metadata server on Cloud Run (no key files)."""
    md = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default"
    try:
        urllib.request.urlopen(urllib.request.Request(md + "/email", headers={"Metadata-Flavor": "Google"}), timeout=1)
    except Exception:
        return None  # not on GCP: serving_speed.Tokens uses ADC

    class MD:
        def get(self, audience=None):
            at = json.load(urllib.request.urlopen(urllib.request.Request(md + "/token", headers={"Metadata-Flavor": "Google"}), timeout=5))["access_token"]
            return at, None

    return MD()


MD = token_provider()


def id_token(aud):
    if MD is None:
        return ss.TOK.get()[1]
    u = f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={aud}"
    return urllib.request.urlopen(urllib.request.Request(u, headers={"Metadata-Flavor": "Google"}), timeout=5).read().decode()


def access_token():
    return (MD.get()[0] if MD else ss.TOK.get()[0])


def bearer(url, audience=None):
    if "prediction.vertexai.goog" in url:
        return access_token()
    base = url.split("/", 3)
    return id_token(audience or (base[0] + "//" + base[2]))


# A gateway protected by IAP (Cloud Run IAP integration) only accepts service-account ID tokens whose audience is
# one of its programmatic OAuth clients (gcloud iap settings get ... accessSettings.oauthSettings.programmaticClients),
# not the service URL. Set PROBE_GATEWAY_AUDIENCE to one of them.
GATEWAY_AUDIENCE = os.environ.get("PROBE_GATEWAY_AUDIENCE") or None


class ProbeTarget(cd.Target):
    def token(self):
        return bearer(self.base)


def probe_serving(name, base, n):
    t = ProbeTarget(name, base)
    failures = []
    st, raw = t.request("/health", None, method="GET")
    version = ""
    try:
        h = json.loads(raw)
        version = h.get("version", "")
        if st != 200 or not h.get("vllm_ready", True):
            failures.append(f"health: {st} vllm_ready={h.get('vllm_ready')}")
    except Exception:
        failures.append(f"health: {st} {raw[:80]!r}")
    for case in cd.CASES:
        try:
            r = cd.run_case(t, case)
        except Exception as e:  # network
            failures.append(f"{case}: {e!r}"[:160])
            continue
        want = EXPECTED.get(case, 200)
        if r["status"] != want:
            failures.append(f"{case}: HTTP {r['status']} (want {want}) {(r.get('error') or '')[:80]}")
        elif want == 200 and not r.get("labels") and case != "systemone":
            failures.append(f"{case}: no answers")
    walls, denoise = [], []
    schema = {"instructions": "Triage the ticket.", "samples": 1, "think": 0, "questions": ss.Q}
    url = base.rstrip("/") + "/v1/chat/completions"
    for i in range(n):
        payload = {"model": "dgemma", "messages": [{"role": "system", "content": json.dumps(schema)},
                                                   {"role": "user", "content": json.dumps({"ticket": ss.TICKETS[i % len(ss.TICKETS)]})}]}
        req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {bearer(base)}"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
            walls.append((time.perf_counter() - t0) * 1000)
            inner = json.loads(body["choices"][0]["message"]["content"])
            denoise.append(float(inner.get("diagnostics", {}).get("timing", {}).get("total_ms") or 0))
        except Exception as e:
            failures.append(f"latency request {i}: {e!r}"[:160])
    return result(name, failures, walls, denoise, version)


def probe_gateway(name, base):
    failures, walls = [], []
    tok = bearer(base, GATEWAY_AUDIENCE)
    try:
        h = json.load(urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + "/health", headers={"Authorization": f"Bearer {tok}"}), timeout=30))
        version = h.get("version", "")
    except Exception as e:
        failures.append(f"health: {e!r}"[:160])
        version = ""
    for i in range(3):
        req = urllib.request.Request(base.rstrip("/") + "/api/decide/support_triage",
                                     json.dumps({"variables": {"ticket": ss.TICKETS[i]}}).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {tok}", "X-DGem-Surface": "probe"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
                used = r.headers.get("X-DGem-Backend-Used")
            walls.append((time.perf_counter() - t0) * 1000)
            if not d.get("answers"):
                failures.append("decide: no answers")
            if used != "vertex":
                failures.append(f"decide: answered by {used} (vertex_first failed over)")
        except urllib.error.HTTPError as e:
            failures.append(f"decide: HTTP {e.code} {e.read()[:80]!r}")
        except Exception as e:
            failures.append(f"decide: {e!r}"[:160])
    return result(name, failures, walls, [], version)


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 1)


def result(name, failures, walls, denoise, version):
    rec = {"severity": "ERROR" if failures else "INFO", "message": f"dgem probe {name}: {'FAIL' if failures else 'ok'}",
           "probe_event": "dgem.probe", "probe_target": name, "probe_ok": not failures, "probe_failures": failures[:10],
           "probe_failure_count": len(failures), "probe_version": version,
           "probe_latency_p50_ms": pct(walls, 0.5), "probe_latency_p95_ms": pct(walls, 0.95),
           "probe_denoise_p50_ms": round(statistics.median(denoise), 1) if denoise else None,
           "probe_requests": len(walls)}
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("-n", type=int, default=int(os.environ.get("PROBE_LATENCY_REQUESTS", "20")))
    a = ap.parse_args()
    targets = a.target or [t for t in os.environ.get("PROBE_TARGETS", "").split(",") if t.strip()]
    if not targets:
        sys.exit("no targets (--target name=url or PROBE_TARGETS)")
    ok = True
    for spec in targets:
        name, url = spec.split("=", 1)
        rec = probe_gateway(name, url) if name.startswith("gateway") else probe_serving(name, url, a.n)
        ok &= rec["probe_ok"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
