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

"""Suite runners. Each writes one receipt file and returns its path.

Receipt kinds:
  health     {"kind": "health", "target", "status", "body"}
  contract   {"kind": "contract", "target", "cases": {case: summary}}       (cases from scripts/contract_diff.py)
  dgem       the native receipt written by `dgem bench-* -o`
  systemone  {"kind": "systemone", "suite", "target", "permute", "run", "cases": [row, ...]}
  latency    {"kind": "latency", "target", "modes": {...}, "sweep": {...}}
"""
import concurrent.futures as cf
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.parse

from . import cases as caselib
from . import net

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.dirname(SCRIPTS)


def _round(x):
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, dict):
        pr = x.get("probabilities")
        if isinstance(pr, dict) and all(isinstance(v, (int, float)) for v in pr.values()):  # drop options < 1e-5
            x = {**x, "probabilities": {k: v for k, v in pr.items() if v >= 1e-5}}
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _dump(path, obj):
    """Compact JSON, floats to 5 decimals, one row per line for per-item receipts (receipts get committed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = _round(obj)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        if isinstance(obj.get("cases"), list):
            head = {k: v for k, v in obj.items() if k != "cases"}
            f.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "cases": [\n')
            f.write(",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in obj["cases"]))
            f.write("\n]}\n")
        else:
            json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- health
def health(target, out):
    st, body = target.health()
    ok = st == 200 and isinstance(body, dict) and body.get("status") == "ok" and body.get("vllm_ready", True) is not False
    keep = ("status", "server", "version", "revision", "vllm_commit", "vllm_ready", "warmed", "phase")
    body = {k: body.get(k) for k in keep if k in body} if isinstance(body, dict) else {"error": str(body)[:200]}
    return _dump(out, {"kind": "health", "target": target.name, "target_kind": target.kind, "status": st, "ok": ok,
                       "body": body})


# ---------------------------------------------------------------- contract (scripts/contract_diff.py cases)
def contract(target, out):
    sys.path.insert(0, SCRIPTS)
    import contract_diff as cd  # noqa: E402
    res = {}
    with cf.ThreadPoolExecutor(6) as ex:
        futs = {name: ex.submit(cd.run_case, target, name) for name in cd.CASES}
        for name, f in futs.items():
            try:
                res[name] = f.result()
            except Exception as e:
                res[name] = {"status": None, "error": repr(e)[:200]}
            if name in cd.EXPECT:
                res[name]["expected_labels"] = cd.EXPECT[name]
                res[name]["wrong"] = res[name].get("labels") != cd.EXPECT[name]
    return _dump(out, {"kind": "contract", "target": target.name, "cases": res})


# ---------------------------------------------------------------- dgem CLI harnesses
def dgem(target, args, out, dgem_bin, workers=4, timeout_s=3600):
    """Run `dgem <args...> -u <url> [-k token] --http-retries 3 [-w N] -o out`."""
    if args[0] == "bench-vision":
        ensure_vision_fixtures()
    cmd = [dgem_bin, *args, "-u", target.cli_url, "--http-retries", "3", "-o", out]
    if workers and args[0] not in ("bench-bbox",):
        cmd += ["-w", str(workers)]
    tok = net.token_for(target.base)
    if tok:
        cmd += ["-k", tok]
    env = {k: v for k, v in os.environ.items() if not k.startswith("DGEM_VERTEX") and k != "DGEM_URL"}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = out[:-5] + ".log"
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_s)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"{' '.join(args)} on {target.name} failed (exit {p.returncode}); see {log}")
    for p in (log, out):  # tokens and endpoint URLs never land in a receipt or log
        redact_file(p, target, tok)
    return out, time.time() - t0


def ensure_vision_fixtures():
    """The vision suites read the seeded synthetic sweep (fixtures/bbox_sweep/, gitignored). Generate it when missing."""
    first = os.path.join(REPO, "fixtures", "bbox_sweep", "geo-1x1-grid-00.png")
    if os.path.exists(first):
        return
    try:
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "generate_bbox_sweep.py")], cwd=REPO, check=True,
                       stdout=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError("vision suites need fixtures/bbox_sweep/: pip install -r scripts/requirements-vision.txt && "
                           f"python3 scripts/generate_bbox_sweep.py ({e})")


def redact_file(path, target, tok=None):
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    new = txt
    if tok:
        new = new.replace(tok, "<TOKEN>")
    host = urllib.parse.urlparse(target.base).netloc
    for s in (target.base, host):
        if s:
            new = new.replace(s, f"<{target.kind}:{target.name}>")
    if new != txt:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)


# ---------------------------------------------------------------- /v1/systemone suites
def _run_case(target, case, perm, max_options=26):
    single = len(case["qs"]) == 1
    meta = []
    for q in case["qs"]:
        shown = (q.get("shown_order") or list(q["criteria"].keys()))[0] if q["type"] == "choice" else None
        meta.append({"id": case["id"] if single else f"{case['id']}/{q['qid']}", "case": case["id"], "qid": q["qid"],
                     "suite": case["suite"], "subset": case["subset"], "tier": case.get("tier"), "perm": perm,
                     "type": q["type"], "K": len(q["labels"]), "expected": q["expected"], "shown_first": shown})
    if max_options and any(q["type"] == "choice" and len(q["labels"]) > max_options for q in case["qs"]):
        return [{**m, "status": "n/a"} for m in meta]
    st, resp, ms, hdr = target.systemone(caselib.body(case))
    if st != 200:
        return [{**m, "status": st, "wall_ms": round(ms, 1), "error": str(resp)[:300]} for m in meta]
    answers = resp.get("answers") or {}
    timing = (resp.get("diagnostics") or {}).get("timing") or {}
    rows = []
    for q, m in zip(case["qs"], meta):
        d = caselib.distribution(q, answers.get(q["qid"]))
        if d is None:
            rows.append({**m, "status": "bad_answer", "error": json.dumps(answers.get(q["qid"]))[:300]})
            continue
        rows.append({**m, "status": 200, "wall_ms": round(ms, 1), "server_ms": timing.get("total_ms", resp.get("latency_ms")),
                     "probabilities": d, **caselib.score(q, d)})
    return rows


def systemone(target, cases, out, workers=8, permute="none", run=1, suite=None, max_options=26):
    """max_options: the raw server takes at most 26 options per choice; None when the target is the adapter."""
    rng = random.Random(99)  # one Random(99) per suite pass, one shuffle per case
    todo = [caselib.permute(c, permute, rng) for c in cases]
    results = [None] * len(todo)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_run_case, target, c, permute, max_options): i for i, c in enumerate(todo)}
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:
                c = todo[i]
                results[i] = [{"id": c["id"], "suite": c["suite"], "status": "exception", "error": repr(e)[:300]}]
    rows = [r for rs in results for r in rs]
    return _dump(out, {"kind": "systemone", "suite": suite or (cases[0]["suite"] if cases else ""), "target": target.name,
                       "permute": permute, "run": run, "cases": rows})


# ---------------------------------------------------------------- Decision Index adapter (`dgem systemone serve`)
class Adapter:
    """A local `dgem systemone serve` in front of a target: the adapter a Decision Index run uses (bracket routing for
    > 26 options, slot batching, 422 capacity refusals). The matrix tests the adapter code of this checkout's dgem
    binary against the deployed model. Started once per target and stopped at the end of the run."""

    def __init__(self, target, dgem_bin, log_path):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        cmd = [dgem_bin, "systemone", "serve", "--host", "127.0.0.1", "--port", str(self.port),
               "--upstream", target.cli_url, "--temperature", "1.0", "--http-retries", "3"]
        if target.options.get("layout"):
            cmd += ["--prompt-layout", target.options["layout"]]
        tok = net.token_for(target.base)
        if tok:
            cmd += ["-k", tok]
        env = {k: v for k, v in os.environ.items() if not k.startswith("DGEM_VERTEX") and k not in ("DGEM_URL", "API_KEY")}
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{self.port}"
        for _ in range(60):
            if self.proc.poll() is not None:
                raise RuntimeError(f"systemone adapter exited ({self.proc.returncode}); see {log_path}")
            st, _b, _ms, _h = net.request(self.url + "/health", timeout=5, retries=0)
            if st == 200:
                return
            time.sleep(0.5)
        raise RuntimeError("systemone adapter did not become healthy")

    def stop(self):
        if self.log.closed:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def adapter_probes(adapter_target, out):
    rows = []
    for p in caselib.di_probes():
        st, body, ms, _ = adapter_target.systemone(p["body"], timeout=600)
        exp, ok, notes, info = p["expect"], True, [], {}
        if "status" in exp and st != exp["status"]:
            ok = False
            notes.append(f"HTTP {st}: {str(body)[:160]}")
        if "status_in" in exp and st not in exp["status_in"]:
            ok = False
            notes.append(f"HTTP {st} (want {exp['status_in']}): {str(body)[:160]}")
        if st == 422 and exp.get("marker_if_422"):
            txt = json.dumps(body).lower()
            hit = [m for m in caselib.KIT_MARKERS if m in txt]
            info["marker"] = hit[0] if hit else None
            if not hit:
                ok = False
                notes.append("422 without a Decision Index kit marker (the kit would record an error, not unsupported)")
        if st == 200 and isinstance(body, dict):
            ans = body.get("answers") or {}
            for qid, want in {**(exp.get("answers") or {}), **(exp.get("answers_if_200") or {})}.items():
                a = ans.get(qid) or {}
                got = a.get("choice")
                if got is None and a.get("noul") is not None:
                    got = "yes" if a["noul"] >= 0.5 else "no"
                if got != want:
                    ok = False
                    notes.append(f"{qid}: {got} != {want}")
            if exp.get("complete"):
                qid0 = next(iter(exp.get("answers") or {"intent": None}))
                pr = (ans.get(qid0) or {}).get("probabilities") or {}
                tot = sum(pr.values())
                info.update(keys=len(pr), prob_sum=tot, top_p=max(pr.values()) if pr else None)
                if len(pr) != exp["complete"] or abs(tot - 1) > 0.01 or any(v != v or v < 0 for v in pr.values()):
                    ok = False
                    notes.append(f"probabilities: {len(pr)} keys (want {exp['complete']}), sum {tot:.4f}")
        rows.append({"name": p["name"], "status": st, "ok": ok, "wall_ms": round(ms, 1), "notes": notes,
                     "info_only": bool(p.get("info_only")), **info})
    return _dump(out, {"kind": "adapter_probes", "target": adapter_target.name, "cases": rows})


def kit_compat(adapter_target, out):
    """Optional: the Decision Index kit's compatibility pass through the adapter. Needs DGEM_DI_KIT_DIR (a kit
    checkout with its .venv) and DGEM_DI_COMPAT_ROWS (compat rows built from a rebuilt suite). Skipped otherwise."""
    kit, rows_path = os.environ.get("DGEM_DI_KIT_DIR"), os.environ.get("DGEM_DI_COMPAT_ROWS")
    if not (kit and rows_path and os.path.exists(rows_path)):
        return _dump(out, {"kind": "kit_compat", "target": adapter_target.name, "skipped": True,
                           "reason": "set DGEM_DI_KIT_DIR and DGEM_DI_COMPAT_ROWS to run the kit compatibility pass"})
    import tempfile
    import collections
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([os.path.join(kit, ".venv", "bin", "python"), "-m", "decision_index", "run", "--engine", "http",
                        "--option", f"base_url={adapter_target.base}", "--option", "model=dgem", "--option", "timeout=900",
                        "--rows", rows_path, "--out", tmp, "--fresh"], cwd=kit, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7200)
        counts, non_ok = collections.Counter(), collections.Counter()
        res = os.path.join(tmp, "results.jsonl")
        if os.path.exists(res):
            with open(res) as f:
                for line in f:
                    r = json.loads(line)
                    counts[r.get("status")] += 1
                    if r.get("status") != "ok":
                        non_ok[f"{r.get('dataset')}: {(r.get('error') or '')[:90]}"] += 1
    return _dump(out, {"kind": "kit_compat", "target": adapter_target.name, "skipped": False, "counts": dict(counts),
                       "non_ok": dict(non_ok.most_common(40))})


# ---------------------------------------------------------------- latency (keep-alive, /v1/systemone)
_TRIAGE = {
    "urgent": {"type": "noul", "instructions": "Does this issue require immediate same-day escalation?"},
    "team": {"type": "choice", "instructions": "Which department owns resolution of this ticket?",
             "criteria": {"billing": "Charges, invoices, payment methods, renewals",
                          "support": "General questions, password resets, account setup",
                          "engineering": "System outages, 500 errors, bugs, API failures"}},
    "sentiment": {"type": "score", "instructions": "Customer anger or distress level",
                  "criteria": ["calm", "frustrated", "furious"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
}
_TICKET = {"ticket": "We were charged twice for our annual renewal and the API returns 500 on every invoice call. "
                     "Refund the duplicate today or we cancel."}


JEVBENCH = os.path.join(REPO, "benchmarks", "jevbench", "jevbench_public.jsonl")


def _long_state():
    """The longest JevBench choice item, or None when the dataset has not been synced (`dgem bench-jev --sync`)."""
    if not os.path.exists(JEVBENCH):
        return None
    with open(JEVBENCH) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    r = max((r for r in rows if r["question"]["type"] == "choice"), key=lambda r: len(json.dumps(r["state"])))
    return r["state"], {"decision": r["question"]}


def latency_payloads():
    q5 = dict(_TRIAGE)
    out = {
        "q1": {"state": _TICKET, "questions": {"team": _TRIAGE["team"]}, "samples": 1},
        "q5": {"state": _TICKET, "questions": q5, "samples": 1},
        "q10": {"state": _TICKET, "questions": {**q5, **{k + "_b": v for k, v in q5.items()}}, "samples": 1},
    }
    long = _long_state()
    if long:  # long_q1 is skipped (for every target alike) when JevBench is not synced
        out["long_q1"] = {"state": long[0], "questions": long[1], "samples": 1}
    out["q5_s4"] = {"state": _TICKET, "questions": q5, "samples": 4}
    return out


class _KeepAlive:
    def __init__(self, base):
        u = urllib.parse.urlparse(base)
        self.https, self.host, self.prefix, self.base = u.scheme == "https", u.netloc, u.path.rstrip("/"), base
        self.local = threading.local()

    def _conn(self):
        c = getattr(self.local, "c", None)
        if c is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            c = self.local.c = cls(self.host, timeout=300)
        return c

    def call(self, payload):
        data = json.dumps({"model": "dgemma", "seed": 42, **payload}).encode()
        hdr = {"Content-Type": "application/json", "Connection": "keep-alive"}
        tok = net.token_for(self.base)
        if tok:
            hdr["Authorization"] = f"Bearer {tok}"
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                c = self._conn()
                c.request("POST", self.prefix + "/v1/systemone", body=data, headers=hdr)
                r = c.getresponse()
                raw = r.read()
                rec = {"status": r.status, "wall_ms": (time.perf_counter() - t0) * 1000}
                if r.status == 200:
                    j = json.loads(raw)
                    t = (j.get("diagnostics") or {}).get("timing") or {}
                    rec["server_ms"] = t.get("total_ms", j.get("latency_ms"))
                else:
                    rec["error"] = raw[:200].decode(errors="replace")
                return rec
            except Exception as e:
                self.local.c = None
                if attempt:
                    return {"status": None, "wall_ms": (time.perf_counter() - t0) * 1000, "error": repr(e)[:200]}


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def _summ(recs, elapsed=None):
    ok = [r for r in recs if r["status"] == 200]
    w = [r["wall_ms"] for r in ok]
    s = [r.get("server_ms") for r in ok]
    out = {"n": len(recs), "ok": len(ok), "errors": len(recs) - len(ok), "wall_p50": _pct(w, 50),
           "wall_p90": _pct(w, 90), "wall_p99": _pct(w, 99), "server_p50": _pct(s, 50)}
    if elapsed:
        out["rps"] = len(ok) / elapsed
    errs = [r.get("error") for r in recs if r["status"] != 200][:3]
    if errs:
        out["sample_errors"] = errs
    return out


def latency(target, out, n=100, warmup=5, sweep_workers=(16, 32), sweep_modes=("q1", "q5"), min_requests=256):
    P = latency_payloads()
    ka = _KeepAlive(target.base)
    res = {"kind": "latency", "target": target.name, "modes": {}, "sweep": {}}
    for m, payload in P.items():
        for _ in range(warmup):
            ka.call(payload)
        res["modes"][m] = _summ([ka.call(payload) for _ in range(n)])
    for m in sweep_modes:
        for w in sweep_workers:
            k = max(min_requests, w * 8)
            for _ in range(min(w, 8)):
                ka.call(P[m])
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(w) as ex:
                recs = list(ex.map(lambda _: ka.call(P[m]), range(k)))
            res["sweep"][f"{m}/w{w}"] = _summ(recs, time.perf_counter() - t0)
    return _dump(out, res)
