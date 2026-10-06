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

"""Matrix report: verdicts per gate, per-suite tables and the measured noise floor -> report.md + summary.json.

Two modes:
  baseline mode (a run has a baseline target): every other target is a candidate compared with the baseline
                measured in the same session.
  single-target mode: the target is compared with the reference ranges in the matrix file ("reference"), when the
                matrix file has them; otherwise its results are reported as INFO. The example matrix has no reference
                ranges: compare a candidate against your own baseline target in the same session.
Verdicts: PASS, REVIEW (look before promoting), FAIL, INFO (reported, not gated).
"""
import json
import os
import statistics
from collections import defaultdict

from . import metrics as M

RANK = {"PASS": 0, "INFO": 0, "REVIEW": 1, "FAIL": 2}
# summary.json contract; see benchmarks/matrix/summary.schema.json. Bump on breaking changes only.
SUMMARY_SCHEMA = "dgem.matrix.summary/v2"


def _f(x, d=3):
    if x is None:
        return "—"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def load_run(run_dir):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        man = json.load(f)
    by = defaultdict(lambda: defaultdict(list))  # suite -> target -> [(entry, receipt)]
    missing = [e["path"] for e in man["receipts"] if not os.path.exists(os.path.join(run_dir, e["path"]))]
    if missing:
        locked = set()
        lock = os.path.join(run_dir, "receipts.lock.json")
        if os.path.exists(lock):
            with open(lock) as f:
                locked = {x["path"] for x in json.load(f)["files"]}
        stored = [m for m in missing if m in locked]
        if stored:
            raise SystemExit(f"{len(stored)} receipt(s) of {os.path.basename(run_dir)} are stored in the matrix bucket "
                             f"(receipts.lock.json): run `scripts/bench_matrix.py fetch-receipts {run_dir}` first")
        raise SystemExit(f"missing receipts in {run_dir}: {', '.join(missing[:5])}")
    for e in man["receipts"]:
        by[e["suite"]][e["config"]].append((e, M.load(os.path.join(run_dir, e["path"]))))
    for s in by.values():
        for lst in s.values():
            lst.sort(key=lambda x: (x[0].get("run", 1), x[0].get("perm") or ""))
    return man, by


def _within(runs):
    vals = [M.agreement(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _cross(ra, rb):
    vals = [M.agreement(a, b) for a in ra for b in rb]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _avg(xs, k):
    v = [x[k] for x in xs if x.get(k) is not None]
    return statistics.mean(v) if v else None


def _suite_summary(entries, runs_rows):
    """Machine-readable per-suite, per-target block for summary.json: every run, means, coverage, held-out
    calibration (first run) and reliability bins (all runs pooled)."""
    runs = []
    for (e, receipt), rr in zip(entries, runs_rows):
        x = M.summary(rr)
        if not x.get("n"):
            continue
        answered, total = M.attempted(receipt)
        x.update({"run": e.get("run", 1), "answered": answered, "attempted": total,
                  "coverage": (answered / total) if total else None, "refusals": M.refusals(receipt)})
        runs.append(x)
    if not runs:
        return None
    pooled = [r for rr in runs_rows for r in rr]
    ho = M.heldout_temperature(runs_rows[0])
    return {
        "runs": runs,
        "accuracy": _avg(runs, "accuracy"), "accuracy_range": [min(x["accuracy"] for x in runs), max(x["accuracy"] for x in runs)],
        "coverage": _avg(runs, "coverage"),
        "refusals": {k: sum(x["refusals"].get(k, 0) for x in runs) for k in sorted({k for x in runs for k in x["refusals"]})},
        "macro_f1": _avg(runs, "macro_f1"), "case_exact": _avg(runs, "case_exact"),
        "ece10": _avg(runs, "ece10"),
        "brier": _avg(runs, "brier"), "nll": _avg(runs, "nll"), "auroc": _avg(runs, "auroc"),
        "soft_acc": _avg(runs, "soft_acc"), "score_mae": _avg(runs, "abs_err_ev"),
        "wall_p50": _avg(runs, "wall_p50"), "mean_conf": _avg(runs, "mean_conf"),
        "noise_within": _within(runs_rows) if len(runs_rows) > 1 else None,
        "heldout": ({"ece_raw": ho["ece_raw"], "ece_heldout": ho["ece_heldout"], "nll_raw": ho["nll_raw"],
                     "nll_heldout": ho["nll_heldout"], "temperatures": ho["T"], "n": ho["n"]} if ho else None),
        "reliability": M.reliability(pooled),
    }


def _noise_table(rows, acc_suites, names, noise):
    multi = [s for s in acc_suites if any(len(rows[s].get(n, [])) > 1 for n in names)]
    out = ["## Measured noise floor (answer agreement between repeated identical runs)", "",
           "| target | " + " | ".join(multi) + " | mean |", "|---|" + "---|" * (len(multi) + 1)]
    for n in names:
        out.append(f"| {n} | " + " | ".join(_f(_within(rows[s].get(n, []))) for s in multi) + f" | {_f(noise.get(n))} |")
    return out + [""]


LEGACY_ACCURACY = ("calibration", "jev_native", "jev_systemone", "intents_banking77", "intents_clinc150", "massive_spot",
                   "di_wide", "di_catchall", "rag_dev", "massive", "xnli", "typed")


def accuracy_suites(matrix):
    """Suites scored per item (accuracy, calibration, coverage, noise floor), in matrix order: those marked
    `"accuracy": true` in the matrix file. Matrix files without the flag fall back to LEGACY_ACCURACY."""
    flagged = [s for s, spec in (matrix.get("suites") or {}).items() if spec.get("accuracy")]
    return flagged or list(LEGACY_ACCURACY)


def build(run_dir, matrix):
    man, by = load_run(run_dir)
    mx = man["matrix"]
    th = matrix["thresholds"]
    ref = (matrix.get("reference") or {}).get("suites", {})
    names = [t["name"] for t in mx["targets"]]
    base = mx.get("baseline")
    cands = [n for n in names if n != base]
    has_ref = bool(ref)
    gates = []  # (gate, target, verdict, detail)
    L = [f"# Regression matrix report: {man['run_id']}", "",
         f"- Matrix **{mx['matrix_version']}**, tier **{mx['tier']}**, started {mx.get('started')}, finished {mx.get('finished')}",
         f"- Repository commit `{man.get('git_commit', '')[:10]}`",
         f"- Mode: {'baseline `' + base + '` measured in the same session' if base else 'single target, compared with the reference ranges' if has_ref else 'single target, no baseline and no reference ranges (results are INFO)'}",
         "", "| target | kind | version | revision | vLLM commit |", "|---|---|---|---|---|"]
    for t in mx["targets"]:
        h = t.get("health") or {}
        tag = " (baseline)" if t["name"] == base else ""
        L.append(f"| {t['name']}{tag} | {t['kind']} | {h.get('version', '—')} | "
                 f"{h.get('revision', '—')} | {str(h.get('vllm_commit', '—'))[:10]} |")
    L.append("")

    # ---------------- health
    for n, lst in by.get("health", {}).items():
        ok = all(r.get("ok") for _, r in lst)
        gates.append(("health", n, "PASS" if ok else "FAIL", "ready" if ok else f"status {lst[0][1].get('status')}"))

    # ---------------- contract
    con = {n: lst[0][1]["cases"] for n, lst in by.get("contract", {}).items()}
    for n in cands:
        if n not in con:
            continue
        wrong = sorted(k for k, v in con[n].items() if v.get("wrong"))
        diffs = []
        if base and base in con:
            diffs = sorted(k for k in con[n] if k in con[base] and con[n][k].get("status") != con[base][k].get("status"))
        v = "FAIL" if wrong or diffs else "PASS"
        gates.append(("contract", n, v, (f"wrong: {', '.join(wrong)}; " if wrong else "") +
                      (f"status differs: {', '.join(diffs)}" if diffs else "") or "all cases as expected"))

    # ---------------- per-item accuracy suites
    acc_suites = [s for s in accuracy_suites(matrix) if s in by]
    rows = {s: {n: [M.rows(r) for _, r in lst] for n, lst in by[s].items()} for s in acc_suites}
    noise = {}
    for n in names:
        ws = [_within(rows[s][n]) for s in acc_suites if n in rows[s] and len(rows[s][n]) > 1]
        ws = [w for w in ws if w is not None]
        noise[n] = statistics.mean(ws) if ws else None
    L += ["## Suites", "",
          "Coverage = answered / attempted items (refusals such as HTTP 422 and errors count as unanswered, as the Decision "
          "Index scores them). Accuracy, ECE and the other metrics are over answered items. Unanswered reasons: context "
          "(prompt longer than the served context), capacity (server shape limits), na (skipped by the matrix), error. "
          "The coverage gate reviews any suite that answers fewer items than the baseline or reference.", "",
          "| suite | target | runs: correct / n | coverage | unanswered (all runs) | mean accuracy | macro-F1 | ECE10 | Brier | AUROC | wall p50 ms |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    suite_summ = {}
    for s in acc_suites:
        for n in names:
            if n not in rows[s]:
                continue
            ss = _suite_summary(by[s][n], rows[s][n])
            if not ss:
                continue
            suite_summ.setdefault(s, {})[n] = ss
            ok = ss["runs"]
            L.append(f"| {s} | {n} | {', '.join(str(x['correct']) for x in ok)} / {ok[0]['n']} | {_f(ss['coverage'])} | "
                     f"{', '.join(f'{k} {v}' for k, v in ss['refusals'].items()) or '—'} | {ss['accuracy']:.3f} | {_f(ss['macro_f1'])} | {_f(ss['ece10'])} | {_f(ss['brier'])} | "
                     f"{_f(ss['auroc'])} | {_f(ss['wall_p50'], 0)} |")
    L.append("")

    for s in acc_suites:
        for n in cands:
            if n not in rows[s] or not rows[s][n]:
                continue
            cand = rows[s][n]
            if not any(cand):
                gates.append((s, n, "FAIL", "no answered items (all refused, n/a or errors)"))
                continue
            mean_c = statistics.mean([M.summary(r)["accuracy"] for r in cand if r])
            if s == "massive_spot":
                bylang = defaultdict(list)
                for r in cand[0]:
                    bylang[r["id"].split("-")[1]].append(r["accurate"])
                low = {lg: sum(v) / len(v) for lg, v in bylang.items() if sum(v) / len(v) < th["multilingual_min_accuracy"]}
                gates.append((s, n, "FAIL" if low else "PASS",
                              ", ".join(f"{lg} {sum(v)}/{len(v)}" for lg, v in sorted(bylang.items()))))
                continue
            if base and base in rows[s]:
                bl = rows[s][base]
                if not any(bl):
                    gates.append((s, n, "REVIEW", "baseline has no answered items"))
                    continue
                mean_b = statistics.mean([M.summary(r)["accuracy"] for r in bl if r])
                within = [w for w in (_within(cand), _within(bl)) if w is not None]
                floor = statistics.mean(within) if within else statistics.mean([v for v in noise.values() if v] or [0.94])
                cross = _cross(cand, bl)
                mc = M.mcnemar(list(zip(cand, bl)))
                fails = []
                # allowance = fixed margin + sampling error of an agreement rate on n items (matters for small suites)
                n_items = len(cand[0]) or 1
                allow = th["agreement_margin_pp"] / 100 + 2 * (max(floor * (1 - floor), 0.0) / n_items) ** 0.5
                if cross is not None and cross < floor - allow:
                    fails.append(f"agreement {cross:.3f} < noise floor {floor:.3f} - {allow:.3f}")
                if mc["p"] < th["mcnemar_p"]:
                    fails.append(f"McNemar p={mc['p']:.2g} ({mc['a_only']} vs {mc['b_only']})")
                v = "PASS" if not fails else ("REVIEW" if len(fails) == 1 else "FAIL")
                frozen = " [frozen set]" if s in ("massive", "xnli", "typed") else ""
                gates.append((s, n, v, f"acc {mean_c:.3f} vs {mean_b:.3f}; agreement {_f(cross)} (floor {_f(floor)}, allowance {allow:.3f}); "
                                       f"McNemar p={mc['p']:.2g}{frozen}" + (" — " + "; ".join(fails) if fails else "")))
            elif s in ref and "accuracy" in ref[s]:
                lo, hi = ref[s]["accuracy"]
                slack = th.get("reference_slack_items_frac", 0.02)
                v = "PASS" if mean_c >= lo - 1e-4 else ("REVIEW" if mean_c >= lo - slack else "FAIL")  # 1e-4: ranges are rounded
                gates.append((s, n, v, f"acc {mean_c:.3f}; reference {lo:.3f}–{hi:.3f} ({matrix['reference'].get('image', 'reference')})"))
            else:
                gates.append((s, n, "INFO", f"acc {mean_c:.3f}; no baseline or reference"))

    # ---------------- coverage: a candidate must answer as many items as the baseline (or the reference). Accuracy is
    # over answered items, so an item that starts being refused (e.g. a prompt that now exceeds the served context)
    # would otherwise drop out silently.
    for n in cands:
        worse, info = [], []
        for s in acc_suites:
            c = suite_summ.get(s, {}).get(n)
            if not c or c.get("coverage") is None:
                continue
            why = ", ".join(f"{k} {v}" for k, v in c["refusals"].items() if k != "na")
            if base and base in suite_summ.get(s, {}):
                b = suite_summ[s][base]
                ca, ba = _avg(c["runs"], "answered"), _avg(b["runs"], "answered")
                if ca is not None and ba is not None and ca < ba - 0.5:
                    worse.append(f"{s} {ca:.0f} vs {ba:.0f} answered ({why or 'no reason recorded'})")
            elif s in ref and ref[s].get("coverage") is not None:
                if c["coverage"] < ref[s]["coverage"] - 1e-4:
                    worse.append(f"{s} coverage {c['coverage']:.4f} < reference {ref[s]['coverage']:.4f} ({why or 'no reason recorded'})")
            elif c["coverage"] < 1 and why:
                info.append(f"{s} {c['coverage']:.4f} ({why})")
        if worse:
            gates.append(("coverage", n, "REVIEW", "fewer answered items: " + "; ".join(worse)))
        else:
            gates.append(("coverage", n, "PASS" if not info else "INFO",
                          "no coverage loss" + ("; unanswered: " + "; ".join(info) if info else "")))

    # ---------------- noise floor
    if any(len(rows[s].get(n, [])) > 1 for s in acc_suites for n in names):
        L += _noise_table(rows, acc_suites, names, noise)

    # ---------------- held-out calibration (T-cal)
    L += ["## Calibration: raw vs held-out temperature (5-fold, folds by case; first run)", "",
          "| suite | target | n | ECE raw | ECE held-out | NLL raw | NLL held-out | T per fold |", "|---|---|---|---|---|---|---|---|"]
    for s in acc_suites:
        for n in names:
            if rows[s].get(n):
                h = M.heldout_temperature(rows[s][n][0])
                if h:
                    L.append(f"| {s} | {n} | {h['n']} | {_f(h['ece_raw'])} | {_f(h['ece_heldout'])} | {_f(h['nll_raw'])} | "
                             f"{_f(h['nll_heldout'])} | {', '.join(f'{t:.2f}' for t in h['T'])} |")
    L.append("")

    # ---------------- multilingual per language
    for s in ("massive", "xnli"):
        if s not in rows:
            continue
        L += [f"## {s} per language (accuracy / mean confidence)", "", "| lang | " + " | ".join(rows[s]) + " |",
              "|---|" + "---|" * len(rows[s])]
        per = {}
        for n, rr in rows[s].items():
            g = defaultdict(list)
            for r in rr[0]:
                g[r["id"].split("-")[1]].append(r)
            per[n] = {lg: (sum(x["accurate"] for x in v) / len(v), sum(x["confidence"] for x in v) / len(v)) for lg, v in g.items()}
        for lg in sorted(next(iter(per.values()))):
            L.append(f"| {lg} | " + " | ".join(f"{per[n][lg][0]:.2f} / {per[n][lg][1]:.2f}" if lg in per[n] else "—" for n in per) + " |")
        L.append("| **macro** | " + " | ".join(f"**{statistics.mean(a for a, _ in per[n].values()):.3f}**" for n in per) + " |")
        L.append("")

    # ---------------- typed-decisions soft metrics
    if "typed" in rows:
        L += ["## typed-decisions (soft metrics vs the teacher distribution)", "",
              "| target | accuracy | soft acc | soft Brier | score MAE |", "|---|---|---|---|---|"]
        for n, rr in rows["typed"].items():
            x = M.summary(rr[0])
            L.append(f"| {n} | {_f(x.get('accuracy'))} | {_f(x.get('soft_acc'))} | {_f(x.get('soft_brier'))} | {_f(x.get('abs_err_ev'))} |")
        L.append("")

    # ---------------- option order
    order_summ = defaultdict(dict)
    if "order" in by:
        L += ["## Option order (flip = answer changes; net = minus the identical-repeat flip)", "",
              "| target | subset | n | identity flip | random flip (net) | reverse flip (net) | picks first shown / gold first shown |",
              "|---|---|---|---|---|---|---|"]
        net_by = defaultdict(dict)
        for n, lst in by["order"].items():
            perm = {e.get("perm"): M.rows(r) for e, r in lst}
            for sub in sorted({r["suite"] for r in perm.get("none", [])}):
                P = {k: [r for r in v if r["suite"] == sub] for k, v in perm.items()}
                fi = M.flip_rate(P["none"], P.get("none2", [])) or 0.0
                fr = M.flip_rate(P["none"], P.get("random", []))
                fv = M.flip_rate(P["none"], P.get("reverse", []))
                first = statistics.mean([r["actual"] == r.get("shown_first") for r in P["none"]])
                gfirst = statistics.mean([r["expected"] == r.get("shown_first") for r in P["none"]])
                net_by[n][sub] = (fr or 0) - fi
                order_summ[n][sub] = {"n": len(P["none"]), "identity_flip": fi, "random_flip": fr, "reverse_flip": fv,
                                      "net_random": (fr or 0) - fi, "net_reverse": (fv or 0) - fi,
                                      "first_shown_pick": first, "first_shown_gold": gfirst}
                L.append(f"| {n} | {sub} | {len(P['none'])} | {fi:.3f} | {_f(fr)} ({(fr or 0) - fi:+.3f}) | "
                         f"{_f(fv)} ({(fv or 0) - fi:+.3f}) | {first:.3f} / {gfirst:.3f} |")
        L.append("")
        for n in cands:
            if base and base in net_by and n in net_by:
                worse = [f"{s} {net_by[n][s] - net_by[base][s]:+.3f}" for s in net_by[n] if s in net_by[base]
                         and net_by[n][s] - net_by[base][s] > th["order_net_flip_delta_review"]]
                gates.append(("order", n, "REVIEW" if worse else "PASS", "net flip vs baseline: " + (", ".join(worse) or "within tolerance")))
            elif n in net_by:
                gates.append(("order", n, "INFO", ", ".join(f"{s} net {v:+.3f}" for s, v in net_by[n].items())))

    # ---------------- adapter-weakness diagnostics (INFO): catch-all over-selection, yes/no 'no' bias
    for n in names:
        if "di_catchall" in rows and rows["di_catchall"].get(n):
            rr = [r for run in rows["di_catchall"][n] for r in run]
            ins = [r for r in rr if r.get("subset") == "in"]
            oo = [r for r in rr if r.get("subset") == "oos"]
            ca = next((r["expected"] for r in oo), None)
            if ins and ca:
                gates.append(("di_catchall_oos_rate", n, "INFO",
                              f"in-scope answered as the catch-all: {sum(r['actual'] == ca for r in ins) / len(ins):.1%}; "
                              f"out-of-scope recall {sum(r['accurate'] for r in oo) / max(1, len(oo)):.1%} (pooled runs)"))
        if "rag_dev" in rows and rows["rag_dev"].get(n):
            rr = [r for run in rows["rag_dev"][n] for r in run]
            tp = sum(1 for r in rr if r["actual"] == "yes" and r["expected"] == "yes")
            fp = sum(1 for r in rr if r["actual"] == "yes" and r["expected"] == "no")
            fn = sum(1 for r in rr if r["actual"] == "no" and r["expected"] == "yes")
            pos = tp + fn
            flag_f1 = 2 * pos / (2 * pos + (len(rr) - pos)) if rr else 0
            gates.append(("rag_dev_yes_bias", n, "INFO",
                          f"hallucinated-class F1 {2 * tp / max(1, 2 * tp + fp + fn):.3f} (always-flag {flag_f1:.3f}), "
                          f"recall {tp / max(1, pos):.1%}, predicted yes {(tp + fp) / max(1, len(rr)):.1%} vs gold {pos / max(1, len(rr)):.1%}"))

    # ---------------- Decision Index adapter probes + kit compatibility pass
    probes_summ = {}
    if "di_probes" in by:
        L += ["## Decision Index adapter probes (`dgem systemone serve` from this checkout)", "",
              "| target | probe | HTTP | ok | detail |", "|---|---|---|---|---|"]
        for n, lst in by["di_probes"].items():
            cs = lst[0][1]["cases"]
            probes_summ[n] = cs
            for c in cs:
                det = "; ".join(c.get("notes") or []) or ", ".join(f"{k}={_f(c[k])}" for k in ("keys", "prob_sum", "top_p", "marker") if k in c)
                L.append(f"| {n} | {c['name']} | {c['status']} | {'yes' if c['ok'] else '**no**'} | {det} |")
            gated = [c for c in cs if not c.get("info_only")]
            bad = [c["name"] for c in gated if not c["ok"]]
            gates.append(("di_probes", n, "FAIL" if bad else "PASS",
                          f"failed: {', '.join(bad)}" if bad else f"all {len(gated)} gated probes as expected"))
            for c in cs:
                if c.get("info_only"):
                    gates.append((f"di_probe_{c['name']}", n, "INFO",
                                  ("as expected" if c["ok"] else "; ".join(c.get("notes") or [])) + " (known adapter gap; not gated)"))
            tops = [c["top_p"] for c in cs if c.get("top_p") is not None]
            if tops:
                gates.append(("di_confidence_cap", n, "INFO",
                              f"top probability on unambiguous wide-option probes: {min(tops):.3f}–{max(tops):.3f} "
                              f"(a flat ceiling across K means bracket fusion is capping confidence)"))
        L.append("")
    if "di_kit_compat" in by:
        for n, lst in by["di_kit_compat"].items():
            r = lst[0][1]
            if r.get("skipped"):
                gates.append(("di_kit_compat", n, "INFO", "skipped: " + r.get("reason", "")))
                continue
            c = r.get("counts") or {}
            errs = c.get("error", 0)
            gates.append(("di_kit_compat", n, "FAIL" if errs else "PASS",
                          f"ok {c.get('ok', 0)}, unsupported {c.get('unsupported', 0)}, error {errs}"))
            probes_summ.setdefault(n, [])

    # ---------------- summary-only suites
    summary_only = {}
    for s, keys in (("bbox", ("acc_at_50_expectation_pct", "mean_expectation_iou")),
                    ("decision_index", ("headline_decision_index", "ece_10bin"))):
        if s in by:
            for n, lst in by[s].items():
                r = lst[0][1]
                r = r.get("dgem_wide_canvas_report", r)
                summary_only.setdefault(s, {})[n] = {k: r.get(k) for k in keys}
                gates.append((s, n, "INFO", ", ".join(f"{k}={_f(r.get(k))}" for k in keys)))

    # ---------------- latency
    if "latency" in by:
        lat = {n: lst[0][1] for n, lst in by["latency"].items()}
        L += ["## Latency (/v1/systemone, keep-alive, seed 42)", "", "| target | mode | ok/n | wall p50 | wall p90 | wall p99 | server p50 | rps |",
              "|---|---|---|---|---|---|---|---|"]
        for n, d in lat.items():
            for m, x in {**d["modes"], **{"sweep " + k: v for k, v in d["sweep"].items()}}.items():
                L.append(f"| {n} | {m} | {x['ok']}/{x['n']} | {_f(x['wall_p50'], 1)} | {_f(x['wall_p90'], 1)} | {_f(x['wall_p99'], 1)} | "
                         f"{_f(x['server_p50'], 1)} | {_f(x.get('rps'), 1)} |")
        L.append("")
        for n in cands:
            if n not in lat:
                continue
            d = lat[n]
            errs = sum(x["errors"] for x in list(d["modes"].values()) + list(d["sweep"].values()))
            notes, v = [], "PASS"
            if errs:
                v, notes = "FAIL", [f"{errs} errors"]
            if base and base in lat:
                for m, x in d["modes"].items():
                    b = lat[base]["modes"].get(m)
                    if b and b["wall_p50"] and x["wall_p50"]:
                        r = x["wall_p50"] / b["wall_p50"]
                        if r > th["latency_p50_ratio_fail"]:
                            v, notes = "FAIL", notes + [f"{m} p50 x{r:.2f}"]
                        elif r > th["latency_p50_ratio_review"]:
                            v = "REVIEW" if v == "PASS" else v
                            notes.append(f"{m} p50 x{r:.2f}")
                for k, x in d["sweep"].items():
                    b = lat[base]["sweep"].get(k)
                    if b and b.get("rps") and x.get("rps") and x["rps"] / b["rps"] < th["throughput_ratio_min"]:
                        v = "REVIEW" if v == "PASS" else v
                        notes.append(f"{k} throughput x{x['rps'] / b['rps']:.2f}")
            gates.append(("latency", n, v, "; ".join(notes) or "within tolerance"))

    overall = {}
    for g, n, v, _ in gates:
        if RANK[v] > RANK[overall.get(n, "PASS")]:
            overall[n] = v
    head = ["## Verdicts", "", "| gate | target | verdict | detail |", "|---|---|---|---|"]
    head += [f"| {g} | {n} | **{v}** | {d} |" for g, n, v, d in gates]
    head += ["", ("Overall: " + ", ".join(f"**{n}: {overall.get(n, 'PASS')}**" for n in cands)) if cands else
             "Overall: no candidate in this run (only the baseline was measured)", ""]
    i = L.index("## Suites")
    L = L[:i] + head + L[i:]
    lat_summ = {n: {"modes": lst[0][1].get("modes"), "sweep": lst[0][1].get("sweep")}
                for n, lst in by.get("latency", {}).items()}
    summary = {"schema": SUMMARY_SCHEMA, "run_id": man["run_id"], "created": man.get("created"),
               "git_commit": man.get("git_commit"), "matrix_version": mx["matrix_version"], "tier": mx["tier"],
               "baseline": base, "started": mx.get("started"), "finished": mx.get("finished"),
               "targets": [{"name": t["name"], "kind": t.get("kind"), "health": t.get("health") or {}} for t in mx["targets"]],
               "overall": {n: overall.get(n, "PASS") for n in cands}, "noise_floor": noise,
               "gates": [{"gate": g, "target": n, "verdict": v, "detail": d} for g, n, v, d in gates],
               "suites": suite_summ,
               "order": {n: dict(v) for n, v in order_summ.items()},
               "latency": lat_summ, "summary_only": summary_only, "di_probes": probes_summ}
    return "\n".join(L) + "\n", summary
