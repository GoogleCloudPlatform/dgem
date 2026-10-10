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

"""dgem regression matrix: run a tier of benchmarks against one or more serving targets and report verdicts.

  # Compare a candidate with your current deployment in the same session (serving-image change, release gate)
  scripts/bench_matrix.py run --tier T1 --target base=<current-url> --target new=<candidate-url> --baseline base \
      --label image-abc123

  # Smoke-check one target (every deploy)
  scripts/bench_matrix.py run --tier T0 --target gpu=http://<GPU_HOST>:8080

  # Full matrix incl. frozen multilingual / typed-decisions / option-order sets (~25k requests per target)
  scripts/bench_matrix.py run --tier T2 --target base=<url> --target new=<url> --baseline base --confirm

  scripts/bench_matrix.py run --tier T1 --target x=<url> --plan   # print what would run, send nothing
  scripts/bench_matrix.py report <run_dir>                        # (re)build report.md + summary.json
  scripts/bench_matrix.py fetch                                   # download + verify the pinned datasets
  scripts/bench_matrix.py fetch-receipts <run_dir> --bucket <BUCKET>  # restore receipts listed in receipts.lock.json
  scripts/bench_matrix.py list                                    # tiers and suites

Target URLs: a Vertex dedicated invoke base (https://<ENDPOINT_ID>.<REGION>-<PROJECT_NUMBER>.prediction.vertexai.goog/v1/
projects/<PROJECT>/locations/<REGION>/endpoints/<ENDPOINT_ID>/invoke), a Cloud Run URL (<CLOUD_RUN_URL>, tagged revision
URLs work), or a self-hosted server (http://<GPU_HOST>:8080). Auth: see scripts/matrix/net.py (DGEM_MATRIX_TOKEN for an
API key). The run directory never contains URLs or tokens: receipts are redacted to <kind:name>.

Verdicts come from the candidate-vs-baseline comparison in the same session, judged against the noise floor the run
measures itself. The example matrix (benchmarks/matrix/matrix_example.json) has no reference ranges, so a single
target without --baseline gets INFO results only. Suites using JevBench need `./bin/dgem bench-jev --sync`
first. Docs: docs/operate/regression-matrix.md.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix import cases as caselib  # noqa: E402
from matrix import datasets  # noqa: E402
from matrix import report as reportlib  # noqa: E402
from matrix import runners  # noqa: E402
from matrix.targets import Target  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATRIX = os.path.join(REPO, "benchmarks", "matrix", "matrix_example.json")
MATRIX_DIR = os.path.dirname(MATRIX)
# Files some suites need that are not in the repository, and how to get them (matrix suites list them in "requires").
REQUIRES_HINT = {"benchmarks/jevbench/jevbench_public.jsonl": "./bin/dgem bench-jev --sync"}


def load_matrix(path=MATRIX):
    with open(path) as f:
        return json.load(f)


def runs_for(suite, tier):
    r = suite.get("runs", 1)
    return r.get(tier, 1) if isinstance(r, dict) else r


def git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
    except Exception:
        return ""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def log_event(enabled, **kw):
    """One structured line per event (Cloud Logging parses these on Cloud Run jobs)."""
    if enabled:
        print(json.dumps({"matrix_event": kw.pop("event"), **kw}), flush=True)
    else:
        print("  " + " ".join(f"{k}={v}" for k, v in kw.items()), file=sys.stderr, flush=True)


def find_dgem(arg):
    for c in (arg, os.path.join(REPO, "bin", "dgem"), "dgem"):
        if c and (os.path.exists(c) or subprocess.run(["which", c], capture_output=True).returncode == 0):
            return c
    raise SystemExit("dgem binary not found: build it (make build) or pass --dgem")


def parquet_hint(mx, suites):
    """Why a run can't start without pyarrow: the suites of this run that read parquet datasets. '' when none do, or
    when pyarrow is installed. It is checked before the first request, so a missing install does not surface halfway."""
    need = [s for s in suites if mx["suites"][s].get("cases") in caselib.PARQUET_SUITES]
    if not need or datasets.have_pyarrow():
        return ""
    return (f"pyarrow is not installed, and these suites read parquet datasets: {', '.join(need)}\n"
            "  pip install -r scripts/requirements-matrix.txt\n(or leave those suites out with --skip)")


def cmd_run(a):
    mx = load_matrix(a.matrix or MATRIX)
    tier = a.tier
    if tier not in mx["tiers"]:
        raise SystemExit(f"tier {tier} is not in {os.path.basename(a.matrix or MATRIX)} (tiers: {', '.join(mx['tiers'])})")
    if tier in mx.get("confirm_required", []) and not a.confirm and not a.plan:
        raise SystemExit(f"{tier} sends ~25k requests per target to shared GPUs; re-run with --confirm")
    suites = [s for s in mx["tiers"][tier] if not a.only or s in a.only]
    suites = [s for s in suites if s not in (a.skip or [])]
    specs = a.target or [t for t in os.environ.get("MATRIX_TARGETS", "").split(",") if t]
    if not specs:
        raise SystemExit("give --target name=url (or MATRIX_TARGETS=name=url,name=url)")
    targets = [Target(*t.split("=", 1)) for t in specs]
    if a.baseline and a.baseline not in [t.name for t in targets]:
        raise SystemExit(f"--baseline {a.baseline} is not one of the --target targets")
    names = [t.name for t in targets]
    if len(set(names)) != len(names):
        raise SystemExit("target names must be unique")
    missing = sorted({f for s in suites for f in mx["suites"][s].get("requires", [])
                      if not os.path.exists(os.path.join(REPO, f))})
    hint = ("missing files for this tier:\n" + "\n".join(
        f"  {f}: {REQUIRES_HINT.get(f, 'see the suite description in the matrix file')}" for f in missing)
        + "\n(or leave those suites out with --skip)") if missing else ""
    hint = "\n".join(h for h in (hint, parquet_hint(mx, suites)) if h)
    if a.plan:
        return plan(mx, tier, suites, targets, a, hint)
    if hint:
        raise SystemExit(hint)
    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%d}-{a.label or tier.lower()}"
    run_dir = os.path.abspath(a.out_dir or os.path.join(REPO, "benchmarks", "runs", run_id))
    if os.path.exists(os.path.join(run_dir, "manifest.json")) and not a.resume:
        raise SystemExit(f"{run_dir} exists; choose another --label or pass --resume")
    os.makedirs(run_dir, exist_ok=True)
    needs_dgem = any(mx["suites"][s]["kind"] == "dgem" or mx["suites"][s].get("via") == "adapter" for s in suites)
    dgem_bin = find_dgem(a.dgem) if needs_dgem else None
    man_path = os.path.join(run_dir, "manifest.json")
    man = json.load(open(man_path)) if a.resume and os.path.exists(man_path) else {
        "run_id": run_id, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "notes": a.note or "", "receipts": [],
        "matrix": {"matrix_version": mx["version"], "tier": tier, "suites": suites, "baseline": a.baseline,
                   "targets": [t.redacted() for t in targets],
                   "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}}
    done = {(e["suite"], e["config"], e.get("run", 1), e.get("perm")) for e in man["receipts"]}

    def save():
        with open(man_path, "w") as f:
            json.dump(man, f, indent=2)
            f.write("\n")

    def record(suite, t, run, perm, path, seconds, kind):
        runners.redact_file(path, t)
        man["receipts"].append({"path": os.path.relpath(path, run_dir), "suite": suite, "config": t.name, "run": run,
                                "perm": perm, "kind": kind, "sha256": sha(path), "seconds": round(seconds, 1)})
        save()

    case_cache = {}
    adapters = {}

    def via(spec, t):
        """The target a suite talks to: the target itself, or a local `dgem systemone serve` in front of it."""
        if spec.get("via") != "adapter":
            return t
        if t.name not in adapters:
            adapters[t.name] = runners.Adapter(t, dgem_bin, os.path.join(run_dir, f"adapter__{t.name}.log"))
            import atexit
            atexit.register(adapters[t.name].stop)
        return Target(t.name, adapters[t.name].url)

    def cases_for(name):
        if name not in case_cache:
            case_cache[name] = caselib.SUITES[name]()
        return case_cache[name]

    log_event(a.log_json, event="dgem.matrix.start", run_id=run_id, tier=tier, targets=len(targets),
              suites=",".join(suites))
    max_runs = max([runs_for(mx["suites"][s], tier) for s in suites] + [1])
    for r in range(1, max_runs + 1):
        order = targets if r % 2 else list(reversed(targets))  # alternate target order between runs
        for s in suites:
            spec = mx["suites"][s]
            if r > runs_for(spec, tier) or (spec["kind"] in ("health", "contract", "latency", "adapter_probes", "kit_compat") and r > 1):
                continue
            if spec["kind"] == "latency":
                continue  # after all accuracy work, so it is not measured under the matrix's own load
            for t in order:
                perms = spec.get("permutes", [None])
                for perm in perms:
                    if (s, t.name, r, perm) in done:
                        continue
                    stem = f"{s}{'__' + perm if perm else ''}__{t.name}__r{r}"
                    out = os.path.join(run_dir, stem + ".json")
                    t0 = time.time()
                    try:
                        if spec["kind"] == "health":
                            runners.health(t, out)
                            hb = json.load(open(out))["body"]
                            for tt in man["matrix"]["targets"]:
                                if tt["name"] == t.name:
                                    tt["health"] = {k: hb.get(k) for k in ("version", "revision", "vllm_commit")}
                        elif spec["kind"] == "contract":
                            runners.contract(t, out)
                        elif spec["kind"] == "dgem":
                            runners.dgem(t, spec["args"], out, dgem_bin, workers=a.workers)
                        elif spec["kind"] == "systemone":
                            p = "none" if perm in (None, "none2") else perm
                            mo = None if spec.get("via") == "adapter" else 26
                            runners.systemone(via(spec, t), cases_for(spec["cases"]), out, workers=a.workers * 2,
                                              permute=p, run=r, suite=s, max_options=mo)
                        elif spec["kind"] == "adapter_probes":
                            runners.adapter_probes(via(spec, t), out)
                        elif spec["kind"] == "kit_compat":
                            runners.kit_compat(via(spec, t), out)
                        record(s, t, r, perm, out, time.time() - t0, spec["kind"])
                        log_event(a.log_json, event="dgem.matrix.suite", run_id=run_id, suite=s, target=t.name, run=r,
                                  perm=perm or "", seconds=round(time.time() - t0, 1), status="ok")
                    except Exception as e:
                        import traceback
                        where = traceback.extract_tb(e.__traceback__)[-1]
                        log_event(a.log_json, event="dgem.matrix.suite", run_id=run_id, suite=s, target=t.name, run=r,
                                  perm=perm or "", status="error", error=str(e)[:300],
                                  where=f"{os.path.basename(where.filename)}:{where.lineno}")
                        if a.fail_fast:
                            raise
    for ad in adapters.values():
        ad.stop()
    if "latency" in suites:
        for t in targets:
            if ("latency", t.name, 1, None) in done:
                continue
            out = os.path.join(run_dir, f"latency__{t.name}__r1.json")
            t0 = time.time()
            runners.latency(t, out, n=a.latency_n)
            record("latency", t, 1, None, out, time.time() - t0, "latency")
            log_event(a.log_json, event="dgem.matrix.suite", run_id=run_id, suite="latency", target=t.name, run=1,
                      seconds=round(time.time() - t0, 1), status="ok")
    man["matrix"]["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    save()
    summary = write_report(run_dir, mx)
    for g in summary["gates"]:
        log_event(a.log_json, event="dgem.matrix.gate", run_id=run_id, tier=tier, **g)
    log_event(a.log_json, event="dgem.matrix.done", run_id=run_id, tier=tier,
              overall=json.dumps(summary["overall"]), report=os.path.join(run_dir, "report.md"))
    if a.upload:
        dest = upload(run_dir, a.upload.rstrip("/") + "/" + os.path.basename(run_dir))
        log_event(a.log_json, event="dgem.matrix.uploaded", run_id=run_id, destination=dest)
    print(f"\n{run_dir}/report.md\n" + "\n".join(f"  {n}: {v}" for n, v in summary["overall"].items()), file=sys.stderr)
    return 1 if any(v == "FAIL" for v in summary["overall"].values()) else 0


def plan(mx, tier, suites, targets, a, hint=""):
    """--plan: list the suites, runs and targets a run would execute, without sending a request."""
    print(f"matrix {mx['version']}, tier {tier}, baseline {a.baseline or '(none: single-target mode)'}")
    print("targets: " + ", ".join(f"{t.name} ({t.kind})" for t in targets))
    for s in suites:
        spec = mx["suites"][s]
        n = 1 if spec["kind"] in ("health", "contract", "latency", "adapter_probes", "kit_compat") else runs_for(spec, tier)
        perms = spec.get("permutes")
        print(f"  {s:18s} {spec['kind']:14s} runs {n}{' x ' + str(len(perms)) + ' orders' if perms else ''}"
              f"{' via adapter' if spec.get('via') == 'adapter' else ''}{' [frozen]' if spec.get('frozen') else ''}")
    if not mx.get("reference") and not a.baseline:
        print("note: no --baseline and no reference ranges in the matrix file: per-suite results will be INFO")
    if hint:
        print(hint)
    return 1 if hint else 0


def upload(run_dir, dest):
    """Copy a run directory to gs://bucket/prefix with the Cloud Storage JSON API (standard library only)."""
    import urllib.parse
    from matrix import net
    if not dest.startswith("gs://"):
        raise SystemExit("--upload needs gs://bucket/prefix")
    bucket, _, prefix = dest[5:].partition("/")
    tok = net._mint("access", None) if not os.environ.get("DGEM_MATRIX_TOKEN") else os.environ["DGEM_MATRIX_TOKEN"]
    import urllib.request
    for root, _, files in os.walk(run_dir):
        for fn in files:
            p = os.path.join(root, fn)
            name = f"{prefix}/{os.path.relpath(p, run_dir)}".lstrip("/")
            ctype = "text/markdown" if fn.endswith(".md") else "application/json" if fn.endswith(".json") else "text/plain"
            url = (f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name="
                   + urllib.parse.quote(name, safe=""))
            with open(p, "rb") as f:
                req = urllib.request.Request(url, data=f.read(), method="POST",
                                             headers={"Authorization": f"Bearer {tok}", "Content-Type": ctype})
            urllib.request.urlopen(req, timeout=120).read()
    return f"gs://{bucket}/{prefix}"


def write_report(run_dir, mx):
    md, summary = reportlib.build(run_dir, mx)
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(md)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def run_matrix(run_dir, explicit):
    """The matrix a run was recorded with (matrix_<version>.json next to the default matrix, from the manifest's
    matrix_version), unless --matrix is given explicitly; otherwise the default matrix."""
    if explicit:
        return explicit
    try:
        with open(os.path.join(run_dir, "manifest.json")) as f:
            v = json.load(f)["matrix"]["matrix_version"]
        for p in (os.path.join(MATRIX_DIR, f"matrix_{v}.json"), os.path.join(MATRIX_DIR, f"matrix_{v.split('-')[0]}.json")):
            if os.path.exists(p) and load_matrix(p).get("version") == v:
                return p
        return MATRIX
    except Exception:
        return MATRIX


def cmd_report(a):
    s = write_report(os.path.abspath(a.run_dir), load_matrix(run_matrix(os.path.abspath(a.run_dir), a.matrix_explicit)))
    print(open(os.path.join(a.run_dir, "report.md")).read())
    return 1 if any(v == "FAIL" for v in s["overall"].values()) else 0


def cmd_fetch(a):
    bad = 0
    for repo, file, ok in datasets.fetch(a.repo):
        if ok is not True:
            bad += 1
            print(f"FAIL {repo}/{file}: {ok}")
    print(f"{'all files verified' if not bad else f'{bad} files failed'} (cache {datasets.CACHE})")
    return 1 if bad else 0


def fetch_receipts(run_dir, bucket=None, download=None):
    """Download the receipts listed in <run_dir>/receipts.lock.json from gs://<bucket>/receipts/<run_id>/ and verify
    size and SHA-256. Files already present and matching are skipped. -> number of files written."""
    with open(os.path.join(run_dir, "receipts.lock.json")) as f:
        lock = json.load(f)
    bucket = bucket or os.environ.get("DGEM_MATRIX_BUCKET")
    if not bucket and download is None:
        raise SystemExit("pass --bucket <BUCKET> (or set DGEM_MATRIX_BUCKET): the Cloud Storage bucket holding "
                         "receipts/<run_id>/")
    run_id = lock.get("run_id") or os.path.basename(os.path.normpath(run_dir))
    if download is None:
        import urllib.parse
        import urllib.request
        from matrix import net

        def download(name):
            tok = os.environ.get("DGEM_MATRIX_TOKEN") or net._mint("access", None)
            url = (f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
                   + urllib.parse.quote(f"receipts/{run_id}/{name}", safe="") + "?alt=media")
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            return urllib.request.urlopen(req, timeout=300).read()
    wrote = 0
    for fe in lock["files"]:
        dst = os.path.join(run_dir, fe["path"])
        if os.path.exists(dst) and sha(dst) == fe["sha256"]:
            continue
        data = download(fe["path"])
        if len(data) != fe["bytes"] or hashlib.sha256(data).hexdigest() != fe["sha256"]:
            raise SystemExit(f"{fe['path']}: size or SHA-256 does not match receipts.lock.json")
        with open(dst, "wb") as f:
            f.write(data)
        wrote += 1
    return wrote


def cmd_fetch_receipts(a):
    n = fetch_receipts(os.path.abspath(a.run_dir), a.bucket)
    print(f"{n} receipt(s) restored and verified in {a.run_dir}")
    return 0


def cmd_list(a):
    mx = load_matrix(a.matrix or MATRIX)
    for t, ss in mx["tiers"].items():
        print(f"{t}{' (needs --confirm)' if t in mx.get('confirm_required', []) else ''}: {', '.join(ss)}")
    print()
    for s, spec in mx["suites"].items():
        print(f"  {s:18s} {spec['kind']:14s} {'frozen ' if spec.get('frozen') else '       '}{spec.get('description', '')}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=None,
                    help="matrix file (default: benchmarks/matrix/matrix_example.json; for report, the one the run "
                         "was recorded with when it is next to it)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tier", required=True, help="a tier of the matrix file (T0, T1, T2 in the example matrix)")
    r.add_argument("--target", action="append", help="name=url (repeat; or MATRIX_TARGETS=name=url,...)")
    r.add_argument("--baseline", help="target name to compare the others against (same session)")
    r.add_argument("--label", help="run id suffix (default: tier)")
    r.add_argument("--out-dir", help="run directory (default: <date>-<label> under benchmarks/runs in the repository)")
    r.add_argument("--only", nargs="*", help="run only these suites of the tier")
    r.add_argument("--skip", nargs="*", help="skip these suites")
    r.add_argument("--workers", type=int, default=4, help="workers per dgem harness (systemone suites use 2x)")
    r.add_argument("--latency-n", type=int, default=100)
    r.add_argument("--dgem", help="path to the dgem binary (default bin/dgem)")
    r.add_argument("--confirm", action="store_true", help="required for T2")
    r.add_argument("--resume", action="store_true", help="continue an interrupted run in the same directory")
    r.add_argument("--fail-fast", action="store_true")
    r.add_argument("--plan", action="store_true", help="print the suites, runs and targets, then exit (no requests)")
    r.add_argument("--log-json", action="store_true", help="structured JSON event lines on stdout (scheduled jobs)")
    r.add_argument("--note", help="free text stored in the manifest")
    r.add_argument("--upload", help="gs://bucket/prefix: copy the run directory there when done (scheduled jobs)")
    p = sub.add_parser("report")
    p.add_argument("run_dir")
    f = sub.add_parser("fetch")
    f.add_argument("--repo", nargs="*")
    fr = sub.add_parser("fetch-receipts", help="restore receipts listed in receipts.lock.json from a Cloud Storage bucket")
    fr.add_argument("run_dir")
    fr.add_argument("--bucket", help="bucket name, objects at receipts/<run_id>/<file> (default $DGEM_MATRIX_BUCKET)")
    sub.add_parser("list")
    a = ap.parse_args()
    a.matrix_explicit = a.matrix
    a.matrix = a.matrix or MATRIX
    sys.exit({"run": cmd_run, "report": cmd_report, "fetch": cmd_fetch, "fetch-receipts": cmd_fetch_receipts,
              "list": cmd_list}[a.cmd](a))


if __name__ == "__main__":
    main()
