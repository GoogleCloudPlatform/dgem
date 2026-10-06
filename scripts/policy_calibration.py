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

"""Calibration report for a decision policy, from a dataset-runner receipt.

Reads the JSONL receipt written by the `run_dgem_dataset.py` runner in docs/policies/datasets.md (one row per
item, with per-question predicted/expected/confidence/hesitation) and reports, per question and overall:

  accuracy                 share of answers that match the label
  Brier (top answer)       mean (confidence - correct)^2 of the chosen answer; lower is better
  ECE (10 bins)            mean gap between confidence and accuracy, weighted by bin size; lower is better
  reliability table        confidence bins with their observed accuracy
  hesitation thresholds    for each cut-off: share of answers kept automatically and their accuracy,
                           i.e. what you get if everything above the cut-off goes to a person or a larger model

  python3 scripts/policy_calibration.py results_receipt.jsonl [--question reason] [--json]

With fewer than ~100 labelled answers per question, treat ECE and the threshold table as rough guides.
"""
import argparse
import json
import math
import sys

THRESHOLDS = [0.05, 0.10, 0.16, 0.25, 0.35, 0.50, 1.01]


def load(path, question=None):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for qid, s in (row.get("slots") or {}).items():
            if question and qid != question:
                continue
            if s.get("correct") is None:
                continue  # no label for this question
            conf = float(s.get("confidence") or 0.0)
            hes = s.get("hesitation")
            if hes is None:
                hes = 1.0 - conf  # older receipts without hesitation: rough stand-in
            out.append({"q": qid, "correct": bool(s["correct"]), "conf": conf, "hes": float(hes)})
    return out


def report(items):
    n = len(items)
    if not n:
        return None
    acc = sum(i["correct"] for i in items) / n
    brier = sum((i["conf"] - i["correct"]) ** 2 for i in items) / n
    bins = []
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        sel = [i for i in items if lo <= i["conf"] < hi or (b == 9 and i["conf"] == 1.0)]
        if sel:
            c = sum(i["conf"] for i in sel) / len(sel)
            a = sum(i["correct"] for i in sel) / len(sel)
            ece += len(sel) / n * abs(c - a)
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel), "mean_conf": round(c, 3), "accuracy": round(a, 3)})
    gates = []
    for t in THRESHOLDS:
        kept = [i for i in items if i["hes"] < t]
        gates.append({"hesitation_below": t, "kept_share": round(len(kept) / n, 3),
                      "kept_accuracy": round(sum(i["correct"] for i in kept) / len(kept), 3) if kept else None,
                      "escalated": n - len(kept)})
    return {"n": n, "accuracy": round(acc, 3), "brier_top": round(brier, 4), "ece10": round(ece, 4),
            "reliability": bins, "gates": gates}


def show(name, r):
    print(f"\n== {name}: n={r['n']}  accuracy={r['accuracy']:.1%}  Brier(top)={r['brier_top']:.4f}  ECE(10 bins)={r['ece10']:.4f}")
    print("  confidence bin   n   mean conf  accuracy")
    for b in r["reliability"]:
        print(f"  {b['bin']:>12s} {b['n']:4d}   {b['mean_conf']:8.3f}  {b['accuracy']:8.3f}")
    print("  keep if hesitation <   kept   accuracy of kept   escalated")
    for g in r["gates"]:
        t = "any" if g["hesitation_below"] > 1 else f"{g['hesitation_below']:.0%}"
        acc = "—" if g["kept_accuracy"] is None else f"{g['kept_accuracy']:.1%}"
        print(f"  {t:>20s} {g['kept_share']:7.1%} {acc:>18s} {g['escalated']:11d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("receipt")
    ap.add_argument("--question", help="only this question id")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    items = load(a.receipt, a.question)
    if not items:
        sys.exit("no labelled answers found (does the dataset have expected values?)")
    out = {"overall": report(items)}
    for q in sorted({i["q"] for i in items}):
        out[q] = report([i for i in items if i["q"] == q])
    if a.json:
        print(json.dumps(out, indent=1))
        return
    for name, r in out.items():
        show(name, r)


if __name__ == "__main__":
    main()
