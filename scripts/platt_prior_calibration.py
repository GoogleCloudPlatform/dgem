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
"""Out-of-fold temperature, Platt prior-bias, and safety-margin calibration."""

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import sys
from typing import TypedDict


class CalibrationRow(TypedDict):
    """Single labeled question evaluation extracted from a dataset receipt."""

    item_id: str
    question_id: str
    gold_option: str
    logprobs: dict[str, float]
    case_sd: float


def _logsumexp(values: Sequence[float]) -> float:
    """Compute log(sum(exp(values))) without floating-point overflow."""
    if not values:
        return float("-inf")
    peak = max(values)
    if math.isinf(peak):
        return peak
    return peak + math.log(sum(math.exp(val - peak) for val in values))


def softmax_with_calibration(
    logprobs: Mapping[str, float],
    temperature: float,
    option_bias: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Apply z'_k = z_k / T + b_k and return normalized option probabilities.

    Args:
      logprobs: Uncalibrated per-option log-probabilities or log-odds.
      temperature: Positive scaling temperature in (0, 20].
      option_bias: Optional additive log-odds offset per option name.

    Returns:
      Normalized probability dictionary over the keys of logprobs.
    """
    bias_map = option_bias or {}
    scaled = {
        option: (score / temperature) + bias_map.get(option, 0.0)
        for option, score in logprobs.items()
    }
    norm = _logsumexp(list(scaled.values()))
    return {option: math.exp(score - norm) for option, score in scaled.items()}


def load_calibration_rows(
    receipt_path: str,
    question_filter: str | None = None,
) -> list[CalibrationRow]:
    """Load labeled per-question log-odds rows from a JSONL evaluation receipt.

    Args:
      receipt_path: Path to a JSONL receipt file.
      question_filter: Optional question ID to filter rows.

    Returns:
      List of CalibrationRow records containing lossless or reconstructed logprobs.
    """
    rows: list[CalibrationRow] = []
    with open(receipt_path, encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            item_id = str(record.get("id", f"row_{line_index}"))
            diag_questions = (record.get("diagnostics") or {}).get("questions") or {}
            slots = record.get("slots") or {}
            for qid, slot_data in slots.items():
                if question_filter and qid != question_filter:
                    continue
                expected = slot_data.get("expected")
                if expected is None:
                    continue
                gold_option = (
                    ("yes" if expected else "no")
                    if isinstance(expected, bool)
                    else str(expected)
                )
                diag_lp = (diag_questions.get(qid) or {}).get("logprobs")
                probs = slot_data.get("probabilities") or {}
                logprobs: dict[str, float] = {}
                if isinstance(diag_lp, dict) and diag_lp:
                    logprobs = {str(k): float(v) for k, v in diag_lp.items()}
                elif isinstance(probs, dict) and probs:
                    logprobs = {
                        str(k): math.log(max(float(v), 1e-12))
                        for k, v in probs.items()
                    }
                elif slot_data.get("predicted") is not None and slot_data.get("confidence") is not None:
                    predicted = str(slot_data["predicted"])
                    conf = min(max(float(slot_data["confidence"]), 1e-12), 1.0 - 1e-12)
                    alt = (
                        ("no" if predicted == "yes" else "yes")
                        if gold_option in ("yes", "no")
                        else (gold_option if gold_option != predicted else "_other")
                    )
                    logprobs = {predicted: math.log(conf), alt: math.log(1.0 - conf)}
                if not logprobs or gold_option not in logprobs:
                    continue
                case_sd = float(slot_data.get("stderr") or slot_data.get("case_sd") or 0.0)
                rows.append(
                    CalibrationRow(
                        item_id=item_id,
                        question_id=qid,
                        gold_option=gold_option,
                        logprobs=logprobs,
                        case_sd=case_sd,
                    )
                )
    return rows


def evaluate_predictions(
    rows: Sequence[CalibrationRow],
    probabilities_per_row: Sequence[Mapping[str, float]],
    positive_option: str | None = None,
) -> dict[str, float]:
    """Compute accuracy, NLL, top-label Brier, 10-bin ECE, and binary AUROC.

    Args:
      rows: Labeled evaluation rows.
      probabilities_per_row: Predicted option distributions aligned with rows.
      positive_option: Optional positive class name for AUROC computation.

    Returns:
      Dictionary of scalar calibration and ranking metrics.
    """
    count = len(rows)
    if count == 0:
        return {"n": 0, "accuracy": 0.0, "nll": 0.0, "brier_top": 0.0, "ece10": 0.0}
    correct_sum = 0.0
    nll_sum = 0.0
    brier_sum = 0.0
    conf_correct_pairs: list[tuple[float, float]] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for row, probs in zip(rows, probabilities_per_row):
        winner = max(probs, key=lambda key: probs[key])
        top_conf = probs[winner]
        is_correct = 1.0 if winner == row["gold_option"] else 0.0
        correct_sum += is_correct
        brier_sum += (top_conf - is_correct) ** 2
        gold_prob = max(probs.get(row["gold_option"], 1e-15), 1e-15)
        nll_sum -= math.log(gold_prob)
        conf_correct_pairs.append((top_conf, is_correct))
        if positive_option is not None and positive_option in probs:
            # Rank in log-odds space so crowd-near-1 scores retain ordering.
            pos_p = min(max(probs[positive_option], 1e-15), 1.0 - 1e-15)
            score = math.log(pos_p / (1.0 - pos_p))
            if row["gold_option"] == positive_option:
                pos_scores.append(score)
            else:
                neg_scores.append(score)

    ece10 = 0.0
    for bin_idx in range(10):
        lo, hi = bin_idx / 10.0, (bin_idx + 1) / 10.0
        bucket = [
            pair
            for pair in conf_correct_pairs
            if lo <= pair[0] < hi or (bin_idx == 9 and pair[0] == 1.0)
        ]
        if bucket:
            mean_conf = sum(pair[0] for pair in bucket) / len(bucket)
            mean_acc = sum(pair[1] for pair in bucket) / len(bucket)
            ece10 += (len(bucket) / count) * abs(mean_conf - mean_acc)

    metrics: dict[str, float] = {
        "n": float(count),
        "accuracy": round(correct_sum / count, 4),
        "nll": round(nll_sum / count, 4),
        "brier_top": round(brier_sum / count, 4),
        "ece10": round(ece10, 4),
    }
    if pos_scores and neg_scores:
        wins = 0.0
        for pos_val in pos_scores:
            for neg_val in neg_scores:
                if pos_val > neg_val:
                    wins += 1.0
                elif pos_val == neg_val:
                    wins += 0.5
        metrics["auroc"] = round(wins / (len(pos_scores) * len(neg_scores)), 4)
    return metrics


def fit_temperature_and_bias(
    train_rows: Sequence[CalibrationRow],
    fit_bias: bool = True,
) -> tuple[float, dict[str, float]]:
    """Fit NLL-minimizing temperature T* and optional per-option log-odds bias b*.

    Args:
      train_rows: Labeled calibration rows for a single question.
      fit_bias: Whether to fit additive per-option log-odds offsets (Platt / vector bias).

    Returns:
      Tuple of (best_temperature, option_bias_dict) bounded within valid schema limits.
    """
    if not train_rows:
        return 1.0, {}
    options = sorted({opt for row in train_rows for opt in row["logprobs"]})
    anchor_option = options[-1]

    def loss_for(temp: float, bias_map: Mapping[str, float]) -> float:
        total = 0.0
        reg = 1e-4 * sum(val * val for val in bias_map.values())
        for row in train_rows:
            probs = softmax_with_calibration(row["logprobs"], temp, bias_map)
            total -= math.log(max(probs.get(row["gold_option"], 1e-15), 1e-15))
        return (total / len(train_rows)) + reg

    # Coarse-to-fine grid search over T in [0.25, 20.0] + coordinate / gradient polish on b.
    coarse_temps = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.5, 6.5, 9.5, 14.0, 20.0]
    best_temp = 1.0
    best_bias: dict[str, float] = {}
    best_loss = float("inf")

    for candidate_temp in coarse_temps:
        bias_vec = {opt: 0.0 for opt in options if opt != anchor_option}
        if fit_bias:
            step_size = 0.8
            for _ in range(45):
                grad = {opt: 0.0 for opt in bias_vec}
                for row in train_rows:
                    probs = softmax_with_calibration(row["logprobs"], candidate_temp, bias_vec)
                    for opt in bias_vec:
                        indicator = 1.0 if row["gold_option"] == opt else 0.0
                        grad[opt] += (probs.get(opt, 0.0) - indicator) / len(train_rows)
                for opt in bias_vec:
                    updated = bias_vec[opt] - step_size * (grad[opt] + 2e-4 * bias_vec[opt])
                    bias_vec[opt] = min(max(updated, -20.0), 20.0)
                step_size *= 0.96
        candidate_loss = loss_for(candidate_temp, bias_vec)
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_temp = candidate_temp
            best_bias = dict(bias_vec)

    # Local refinement around best_temp.
    for delta in (-0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.6):
        refined_temp = min(max(best_temp + delta, 0.1), 20.0)
        bias_vec = dict(best_bias)
        if fit_bias:
            for _ in range(25):
                grad = {opt: 0.0 for opt in bias_vec}
                for row in train_rows:
                    probs = softmax_with_calibration(row["logprobs"], refined_temp, bias_vec)
                    for opt in bias_vec:
                        indicator = 1.0 if row["gold_option"] == opt else 0.0
                        grad[opt] += (probs.get(opt, 0.0) - indicator) / len(train_rows)
                for opt in bias_vec:
                    updated = bias_vec[opt] - 0.35 * (grad[opt] + 2e-4 * bias_vec[opt])
                    bias_vec[opt] = min(max(updated, -20.0), 20.0)
        candidate_loss = loss_for(refined_temp, bias_vec)
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_temp = refined_temp
            best_bias = dict(bias_vec)

    cleaned_bias = {
        opt: round(val, 4) for opt, val in best_bias.items() if abs(val) >= 1e-4
    }
    return round(best_temp, 4), cleaned_bias


def cross_validate_question(
    rows: Sequence[CalibrationRow],
    num_folds: int = 5,
    safety_option: str | None = None,
    safety_sigma_margin: float = 2.0,
) -> dict[str, object]:
    """Run K-fold out-of-fold evaluation comparing raw, temperature, and Platt scaling.

    Args:
      rows: Labeled rows for a single question.
      num_folds: Number of deterministic folds (minimum 2 when len(rows) >= 2).
      safety_option: Optional high-risk action label (for example "allow") that must
        incur zero false positives out-of-fold.
      safety_sigma_margin: Multiplier on per-case SD added above the worst fold negative.

    Returns:
      Summary dictionary with out-of-fold metrics, full-set policy snippet, and safety gate.
    """
    total = len(rows)
    effective_folds = max(2, min(num_folds, total)) if total >= 2 else 1
    options = sorted({opt for row in rows for opt in row["logprobs"]})
    positive_option = "yes" if "yes" in options else (safety_option or options[0])

    raw_probs = [softmax_with_calibration(row["logprobs"], 1.0) for row in rows]
    oof_temp_probs: list[dict[str, float]] = [{} for _ in range(total)]
    oof_platt_probs: list[dict[str, float]] = [{} for _ in range(total)]
    oof_safety_decisions = 0
    oof_unsafe_allows = 0

    for fold in range(effective_folds):
        if effective_folds == 1:
            train_idx = list(range(total))
            test_idx = list(range(total))
        else:
            test_idx = [idx for idx in range(total) if idx % effective_folds == fold]
            train_idx = [idx for idx in range(total) if idx % effective_folds != fold]
        train_rows = [rows[idx] for idx in train_idx]

        fold_temp_only, _ = fit_temperature_and_bias(train_rows, fit_bias=False)
        fold_platt_temp, fold_platt_bias = fit_temperature_and_bias(train_rows, fit_bias=True)

        # Calibrate asymmetric zero-false-allow threshold on train_rows if requested.
        allow_threshold = 1.01
        if safety_option is not None:
            neg_train_peaks = [
                softmax_with_calibration(row["logprobs"], fold_platt_temp, fold_platt_bias).get(
                    safety_option, 0.0
                )
                + safety_sigma_margin * row["case_sd"]
                for row in train_rows
                if row["gold_option"] != safety_option
            ]
            if neg_train_peaks:
                allow_threshold = min(1.01, max(neg_train_peaks) + 0.01)

        for idx in test_idx:
            oof_temp_probs[idx] = softmax_with_calibration(rows[idx]["logprobs"], fold_temp_only)
            platt_dist = softmax_with_calibration(
                rows[idx]["logprobs"], fold_platt_temp, fold_platt_bias
            )
            oof_platt_probs[idx] = platt_dist
            if safety_option is not None:
                safe_score = platt_dist.get(safety_option, 0.0) - (
                    safety_sigma_margin * rows[idx]["case_sd"]
                )
                if safe_score >= allow_threshold:
                    oof_safety_decisions += 1
                    if rows[idx]["gold_option"] != safety_option:
                        oof_unsafe_allows += 1

    full_temp, full_bias = fit_temperature_and_bias(rows, fit_bias=True)
    temp_only_full, _ = fit_temperature_and_bias(rows, fit_bias=False)
    qid = rows[0]["question_id"]
    result: dict[str, object] = {
        "question_id": qid,
        "folds": effective_folds,
        "raw": evaluate_predictions(rows, raw_probs, positive_option),
        "oof_temperature_only": evaluate_predictions(rows, oof_temp_probs, positive_option),
        "oof_platt_prior_bias": evaluate_predictions(rows, oof_platt_probs, positive_option),
        "fitted_temperature_only": temp_only_full,
        "fitted_platt": {
            "temperature": full_temp,
            "prior_bias": {qid: full_bias},
        },
    }
    if safety_option is not None:
        result["safety_gate"] = {
            "safety_option": safety_option,
            "sigma_margin": safety_sigma_margin,
            "oof_auto_decided_share": round(oof_safety_decisions / total, 4),
            "oof_unsafe_allows": oof_unsafe_allows,
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for out-of-fold policy temperature and prior-bias fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", help="Path to JSONL evaluation receipt.")
    parser.add_argument("--question", help="Evaluate only this question ID.")
    parser.add_argument("--folds", type=int, default=5, help="Out-of-fold CV count.")
    parser.add_argument(
        "--safety-option",
        help="Action label requiring zero out-of-fold false positives (for example allow).",
    )
    parser.add_argument(
        "--safety-sigma",
        type=float,
        default=2.0,
        help="Safety margin multiple of per-case SD.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    rows = load_calibration_rows(args.receipt, args.question)
    if not rows:
        sys.exit("no labeled rows with probabilities or logprobs found in receipt")

    grouped: dict[str, list[CalibrationRow]] = {}
    for row in rows:
        grouped.setdefault(row["question_id"], []).append(row)

    reports = {
        qid: cross_validate_question(
            qrows,
            num_folds=args.folds,
            safety_option=args.safety_option,
            safety_sigma_margin=args.safety_sigma,
        )
        for qid, qrows in sorted(grouped.items())
    }
    if args.json:
        print(json.dumps(reports, indent=2))
        return 0

    for qid, summary in reports.items():
        raw = summary["raw"]
        temp_m = summary["oof_temperature_only"]
        platt_m = summary["oof_platt_prior_bias"]
        fitted = summary["fitted_platt"]
        print(f"\n== Question {qid} (n={int(raw['n'])}, {summary['folds']}-fold OOF) ==")
        print(
            f"  Raw            acc={raw['accuracy']:.1%}  ECE={raw['ece10']:.4f}  "
            f"Brier={raw['brier_top']:.4f}  NLL={raw['nll']:.4f}"
        )
        print(
            f"  Temp-only OOF  acc={temp_m['accuracy']:.1%}  ECE={temp_m['ece10']:.4f}  "
            f"Brier={temp_m['brier_top']:.4f}  NLL={temp_m['nll']:.4f}  "
            f"(T*={summary['fitted_temperature_only']})"
        )
        print(
            f"  Platt+Temp OOF acc={platt_m['accuracy']:.1%}  ECE={platt_m['ece10']:.4f}  "
            f"Brier={platt_m['brier_top']:.4f}  NLL={platt_m['nll']:.4f}"
        )
        print(f"  Policy snippet: {json.dumps(fitted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
