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

"""Measure Cloud Run GPU cold start for one service configuration.

Creates a new revision (forcing a fresh instance), then reads the revision's logs and reports:
  deploy_s   `gcloud run services update` wall time (includes image pull and startup probe)
  first_log  seconds from update start to the container's first log line (≈ image pull + boot)
  staged     "range copy ... completed in Xs" (weights copied from Cloud Storage), if any
  vllm_ready seconds from update start to the tmpfs reclaim / vLLM ready line
  warmed     seconds from update start to "[init] Self-warmup complete."
  first_ok   seconds from update start to the first successful decision sent by this probe

  python3 scripts/coldstart_probe.py --service <svc> --project <p> --region <r> --label A1 \
      [--image <uri>] [--no-vpc | --vpc] -o out.jsonl

Uses gcloud (for the update) and the Logging API with the gcloud access token in
$CLOUDSDK_AUTH_ACCESS_TOKEN_FILE or `gcloud auth print-access-token`.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import serving_speed as ss  # noqa: E402


def token():
    f = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE")
    if f and os.path.exists(f):
        return open(f).read().strip()
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def logs(project, service, revision, since):
    flt = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}" '
           f'AND resource.labels.revision_name="{revision}" AND timestamp>="{since}"')
    out, page = [], None
    while True:
        body = {"resourceNames": [f"projects/{project}"], "filter": flt, "orderBy": "timestamp asc", "pageSize": 1000}
        if page:
            body["pageToken"] = page
        req = urllib.request.Request("https://logging.googleapis.com/v2/entries:list", json.dumps(body).encode(),
                                     {"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
        for attempt in range(8):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 7:
                    raise
                time.sleep(10 * (attempt + 1))  # Logging API read quota
        out += d.get("entries", [])
        page = d.get("nextPageToken")
        if not page:
            return out


def ts(s):
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    return dt.datetime.fromisoformat(s + "+00:00").timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--image")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--vpc", action="store_true", help="Direct VPC egress default/default all-traffic")
    g.add_argument("--no-vpc", action="store_true", help="clear VPC network (public egress)")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("-o", "--output")
    a = ap.parse_args()

    cmd = ["gcloud", "run", "services", "update", a.service, f"--project={a.project}", f"--region={a.region}",
           f"--update-env-vars=COLDSTART_LABEL={a.label}-{int(time.time())}", "--format=value(status.latestCreatedRevisionName)"]
    if a.image:
        cmd.append(f"--image={a.image}")
    if a.vpc:
        cmd += ["--network=default", "--subnet=default", "--vpc-egress=all-traffic"]
    if a.no_vpc:
        cmd.append("--clear-network")
    t0 = time.time()
    since = dt.datetime.fromtimestamp(t0 - 5, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # gcloud stops waiting after ~10 minutes (large images take longer to pull), so don't trust its exit
    # code: wait until the newly created revision is the latest ready one.
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode

    def describe(fmt):
        return subprocess.check_output(["gcloud", "run", "services", "describe", a.service, f"--project={a.project}",
                                        f"--region={a.region}", f"--format=value({fmt})"], text=True).strip()
    created = describe("status.latestCreatedRevisionName")
    while describe("status.latestReadyRevisionName") != created and time.time() - t0 < a.timeout:
        time.sleep(15)
    deploy_s = time.time() - t0
    rev = created
    if rc:
        print(f"[{a.label}] gcloud exited {rc} (wait timeout); revision {rev} became ready anyway", flush=True)
    url = subprocess.check_output(["gcloud", "run", "services", "describe", a.service, f"--project={a.project}",
                                   f"--region={a.region}", "--format=value(status.url)"], text=True).strip()
    print(f"[{a.label}] revision {rev} ready after {deploy_s:.0f}s; probing decisions and logs...", flush=True)

    first_ok = None
    rec = {}
    while time.time() - t0 < a.timeout:
        if first_ok is None:
            r = ss.call(url + "/v1/chat/completions", {"instructions": "Triage.", "samples": 1, "think": 0, "questions": ss.Q},
                        None, "charged twice", timeout=60)
            if r.get("status") == 200 and not r.get("error"):
                first_ok = time.time() - t0
        es = logs(a.project, a.service, rev, since)
        text = [(ts(e["timestamp"]), e.get("textPayload") or "") for e in es]
        warmed = next((t for t, x in text if "Self-warmup complete" in x), None)
        if warmed and first_ok:
            first = text[0][0] if text else None  # first container log line ≈ image pulled + started
            staged = next((float(m.group(1)) for _, x in text for m in [re.search(r"range copy.*completed in ([\d.]+)s", x)] if m), None)
            ready = next((t for t, x in text if "Reclaimed" in x or "Application startup complete" in x), None)
            rec = {"label": a.label, "revision": rev, "image": a.image, "vpc": "on" if a.vpc else ("off" if a.no_vpc else "unchanged"),
                   "deploy_s": round(deploy_s, 1), "first_log_s": round(first - t0, 1) if first else None,
                   "staged_s": staged, "vllm_ready_s": round(ready - t0, 1) if ready else None,
                   "warmed_s": round(warmed - t0, 1), "first_ok_s": round(first_ok, 1)}
            break
        time.sleep(30)
    else:
        rec = {"label": a.label, "revision": rev, "error": "timeout"}
    print(json.dumps(rec), flush=True)
    if a.output:
        with open(a.output, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
