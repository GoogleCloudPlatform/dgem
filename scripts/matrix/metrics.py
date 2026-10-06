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

"""Per-item normalization of every receipt kind, and the metrics the matrix report uses.

Normalized row: {"id", "actual", "accurate", "confidence", "probabilities" (optional {label: p}),
                 "expected", "wall_ms", ...}
"""
import hashlib
import json
import math

EPS = 1e-6
_YN = {"true": "yes", "false": "no", "yes": "yes", "no": "no"}


def load(path):
    with open(path) as f:
        return json.load(f)


def rows(receipt):
    """Normalized, successful rows of any per-item receipt (systemone, bench-jev, bench-calibration, bench-intents)."""
    d = receipt
    if d.get("kind") == "systemone":
        return [r for r in d["cases"] if r.get("status") == 200]
    if d.get("receipt_kind") == "bench-vision":  # one row per (image, aspect); ids are "<case>#<aspect>"
        return [_row(c["id"], c["expected"], c["actual"], c["accurate"], c.get("confidence"), c.get("top_probabilities"),
                     c.get("wall_time_ms")) for c in d.get("items") or []]
    out = []
    if isinstance(d.get("results"), list) and d["results"] and "intent_accurate" in d["results"][0]:
        for c in d["results"]:
            out.append(_row(c["id"], c.get("expected_intent"), c.get("actual_intent"), c.get("intent_accurate"),
                            c.get("confidence"), c.get("top_probabilities"), c.get("wall_time_ms")))
        return out
    for c in d.get("cases") or []:
        if c.get("error"):
            continue
        boolean = c.get("question_type") == "noul" or str(c.get("metric", "")).endswith("-check")
        exp, act = str(c.get("expected", "")), str(c.get("actual", ""))
        tp = c.get("top_probabilities") or None
        if boolean:
            exp, act = _YN.get(exp.lower(), exp), _YN.get(act.lower(), act)
            if tp:
                tp = {_YN.get(k.lower(), k): v for k, v in tp.items()}
        out.append(_row(c["id"], exp, act, c.get("accurate"), c.get("confidence"), tp, c.get("wall_time_ms")))
    return out


def refusal_reason(status, error):
    """Why an item went unanswered: "context" (prompt longer than the served context), "capacity" (the server's
    shape limits: options, questions, canvas; HTTP 422 or a schema 400), "na" (skipped by the matrix, e.g. >26 options
    on the raw server), or "error" (anything else: network, 5xx, timeouts)."""
    e = str(error or "").lower()
    if status == "n/a":
        return "na"
    if "maximum context length" in e or "context length" in e or "prompt is too long" in e:
        return "context"
    if status == 422 or "at most" in e or "unsupported" in e or "canvas" in e or "schema:" in e:
        return "capacity"
    return "error"


def refusals(receipt):
    """{reason: count} over unanswered items of one receipt (see refusal_reason)."""
    out = {}
    d = receipt
    if d.get("kind") == "systemone":
        items = [(r.get("status"), r.get("error")) for r in d.get("cases") or [] if r.get("status") != 200]
    else:
        res = d.get("results")
        if isinstance(res, list) and res and "intent_accurate" in res[0]:
            items = [(None, r["error"]) for r in res if r.get("error")]
        else:
            items = [(None, c["error"]) for c in d.get("cases") or [] if c.get("error")]
    for st, err in items:
        k = refusal_reason(st, err)
        out[k] = out.get(k, 0) + 1
    return out


def attempted(receipt):
    """(answered, total) item counts, for coverage. Unanswered = refused (HTTP 4xx, e.g. 422 capacity), errors, n/a.
    The Decision Index scores unanswered items as wrong, so coverage belongs next to accuracy."""
    d = receipt
    if d.get("kind") == "systemone":
        cs = d.get("cases") or []
        return sum(1 for r in cs if r.get("status") == 200), len(cs)
    res = d.get("results")
    if isinstance(res, list) and res and "intent_accurate" in res[0]:
        return sum(1 for r in res if not r.get("error")), len(res)
    cs = d.get("cases") or []
    return sum(1 for c in cs if not c.get("error")), len(cs)


def macro_f1(rs):
    """Macro-averaged F1 over gold labels (labels namespaced by question id for multi-question suites). The Decision
    Index and published decision-model results report intent benchmarks (BANKING77, CLINC150) as macro-F1."""
    def key(r, lab):
        return f"{r['qid']}:{lab}" if r.get("qid") else str(lab)
    tp, fp, fn = {}, {}, {}
    for r in rs:
        g, p = key(r, r["expected"]), key(r, r["actual"])
        if g == p:
            tp[g] = tp.get(g, 0) + 1
        else:
            fn[g] = fn.get(g, 0) + 1
            fp[p] = fp.get(p, 0) + 1
    labels = set(tp) | set(fn)  # gold labels only: a label never in the gold set has no recall to average
    if not labels:
        return None
    f1s = []
    for lab in labels:
        t, f_p, f_n = tp.get(lab, 0), fp.get(lab, 0), fn.get(lab, 0)
        f1s.append(2 * t / (2 * t + f_p + f_n) if t else 0.0)
    return sum(f1s) / len(f1s)


def case_exact(rs):
    """Share of multi-question cases with every field right (rows carry `case`). None for single-question suites.
    Field accuracy can hide this: with many fields per case, a high field accuracy can still mean few cases fully right."""
    by = {}
    for r in rs:
        c = r.get("case")
        if c is None or r.get("qid") is None:
            return None
        by.setdefault(c, []).append(bool(r["accurate"]))
    if not by or all(len(v) == 1 for v in by.values()):
        return None
    return sum(all(v) for v in by.values()) / len(by)


def reliability(rs, bins=10):
    """Reliability diagram bins: share of items, mean confidence and accuracy per confidence bin."""
    rs = [r for r in rs if r.get("confidence") is not None]
    n = len(rs)
    out = []
    for b in range(bins):
        sel = [r for r in rs if min(int(float(r["confidence"]) * bins), bins - 1) == b]
        out.append({"lo": b / bins, "hi": (b + 1) / bins, "count": len(sel),
                    "share": (len(sel) / n) if n else 0.0,
                    "conf": (sum(float(r["confidence"]) for r in sel) / len(sel)) if sel else None,
                    "acc": (sum(1 for r in sel if r["accurate"]) / len(sel)) if sel else None})
    return out


def _row(i, exp, act, acc, conf, tp, ms):
    r = {"id": i, "expected": exp, "actual": act, "accurate": bool(acc), "confidence": conf, "wall_ms": ms}
    if tp:
        s = sum(tp.values())
        if s > 0:
            p = {k: v / s for k, v in tp.items()}
            r["probabilities"] = p
            r["confidence"] = max(p.values())
            if exp in p:
                r["nll"] = -math.log(max(p[exp], EPS))
                r["brier"] = sum((v - (1.0 if k == exp else 0.0)) ** 2 for k, v in p.items())
    return r


# ---------------------------------------------------------------- metrics
def ece(confs, hits, bins=10):
    n = len(confs)
    if not n:
        return None
    tot = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confs) if min(int(c * bins), bins - 1) == b]
        if idx:
            tot += len(idx) / n * abs(sum(hits[i] for i in idx) / len(idx) - sum(confs[i] for i in idx) / len(idx))
    return tot


def auroc(scores, hits):
    pos = sum(1 for h in hits if h)
    neg = len(hits) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r for r, h in zip(ranks, hits) if h)
    return (rp - pos * (pos + 1) / 2) / (pos * neg)


def summary(rs):
    rs = [r for r in rs if r.get("confidence") is not None]
    n = len(rs)
    if not n:
        return {"n": 0}
    hits = [1 if r["accurate"] else 0 for r in rs]
    confs = [float(r["confidence"]) for r in rs]
    out = {"n": n, "correct": sum(hits), "accuracy": sum(hits) / n, "ece10": ece(confs, hits),
           "auroc": auroc(confs, hits), "mean_conf": sum(confs) / n, "macro_f1": macro_f1(rs),
           "case_exact": case_exact(rs)}
    for k in ("brier", "nll", "soft_acc", "soft_brier", "abs_err_ev"):
        v = [r[k] for r in rs if k in r]
        if v:
            out[k] = sum(v) / len(v)
    w = sorted(r["wall_ms"] for r in rs if r.get("wall_ms") is not None)
    if w:
        out["wall_p50"] = w[len(w) // 2]
    return out


def agreement(a, b):
    """Share of shared ids with the same answer."""
    A = {r["id"]: r["actual"] for r in a}
    B = {r["id"]: r["actual"] for r in b}
    ids = set(A) & set(B)
    return (sum(A[i] == B[i] for i in ids) / len(ids)) if ids else None


def mcnemar(pairs):
    """pairs: list of (rows_a, rows_b) aligned runs. Exact two-sided test on pooled discordant counts."""
    b = c = 0
    for ra, rb in pairs:
        A = {r["id"]: r["accurate"] for r in ra}
        B = {r["id"]: r["accurate"] for r in rb}
        for i in set(A) & set(B):
            b += A[i] and not B[i]
            c += B[i] and not A[i]
    d = b + c
    if not d:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) / 2 ** d)
    return {"a_only": b, "b_only": c, "p": p}


# ---------------------------------------------------------------- held-out temperature (T-cal)
GRID = [round(0.5 + 0.05 * i, 2) for i in range(91)]  # 0.50 .. 5.00


def _temper(p, T):
    z = {k: math.log(max(v, EPS)) / T for k, v in p.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def _nll(rs, T):
    return sum(-math.log(max(_temper(r["probabilities"], T).get(r["expected"], 0.0), EPS)) for r in rs) / max(1, len(rs))


def _fold(r, k):
    return int(hashlib.sha1(str(r.get("case", r["id"])).encode()).hexdigest(), 16) % k


def heldout_temperature(rs, folds=5):
    """5-fold CV global temperature (folds by case). -> {"ece_raw", "ece_heldout", "nll_raw", "nll_heldout", "T"}."""
    rs = [r for r in rs if r.get("probabilities") and r.get("expected") in r["probabilities"]]
    if len(rs) < 20:
        return None
    out_c, out_h, Ts = [], [], []
    for f in range(folds):
        train = [r for r in rs if _fold(r, folds) != f]
        test = [r for r in rs if _fold(r, folds) == f]
        T = min(GRID, key=lambda t: (_nll(train, t), abs(t - 1)))
        Ts.append(T)
        for r in test:
            p = _temper(r["probabilities"], T)
            top = max(p, key=p.get)
            out_c.append(p[top])
            out_h.append(1 if top == r["expected"] else 0)
    raw_c = [max(r["probabilities"].values()) for r in rs]
    raw_h = [1 if max(r["probabilities"], key=r["probabilities"].get) == r["expected"] else 0 for r in rs]
    return {"n": len(rs), "ece_raw": ece(raw_c, raw_h), "ece_heldout": ece(out_c, out_h), "T": Ts,
            "nll_raw": _nll(rs, 1.0), "nll_heldout": sum(_nll([r for r in rs if _fold(r, folds) == f], Ts[f]) *
                                                          sum(1 for r in rs if _fold(r, folds) == f)
                                                          for f in range(folds)) / len(rs)}


def flip_rate(base, other):
    A = {r["id"]: r["actual"] for r in base}
    B = {r["id"]: r["actual"] for r in other}
    ids = set(A) & set(B)
    return (sum(A[i] != B[i] for i in ids) / len(ids)) if ids else None
