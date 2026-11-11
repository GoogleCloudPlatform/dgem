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

"""Calibration report for a decision policy from a dataset-runner receipt.

Reads the JSONL receipt written by the `run_dgem_dataset.py` runner in
docs/policies/datasets.md (one row per item, with per-question predicted,
expected, confidence, hesitation, and optional probabilities) and reports, per
question and overall:

  accuracy                 share of answers that match the label
  Brier (top answer)       mean (confidence - correct)^2 of the chosen answer
  ECE (10 bins)            mean gap between confidence and accuracy
  reliability table        confidence bins with their observed accuracy
  hesitation thresholds    share of answers kept automatically and accuracy
  conformal sets           optional 5-fold out-of-sample split conformal sets

Usage:
  python3 scripts/policy_calibration.py results_receipt.jsonl [--question reason]
    [--conformal {lac,aps,raps}] [--conformal-alpha 0.10]
    [--conformal-abstain-labels abstain,insufficient_evidence] [--json]
"""

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import pathlib
import sys
from typing import TypedDict

THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.16, 0.25, 0.35, 0.50, 1.01)
CONFORMAL_FOLDS: int = 5
CONFORMAL_TOLERANCE: float = 1e-12
DEFAULT_RAPS_K_REG: int = 1
DEFAULT_RAPS_LAMBDA_REG: float = 0.01
DEFAULT_ABSTAIN_LABELS: frozenset[str] = frozenset()


class ReceiptSlotItem(TypedDict):
  item_id: str
  q: str
  correct: bool
  conf: float
  hes: float
  expected: str | None
  probabilities: dict[str, float] | None


class ConformalSummary(TypedDict):
  method: str
  alpha: float
  folds: int
  evaluated_n: int
  empirical_coverage: float
  mean_set_size: float
  singleton_share: float
  escalated_share: float


def compute_fold_index(item_id: str, num_folds: int = CONFORMAL_FOLDS) -> int:
  """Assigns a deterministic SHA-1 fold index matching dgem bench-calibration.

  Args:
    item_id: Stable identifier for the dataset row and question slot.
    num_folds: Total number of cross-validation folds.

  Returns:
    Integer fold index in `[0, num_folds - 1]`.

  Raises:
    ValueError: If `num_folds` is less than 1.
  """
  if num_folds < 1:
    raise ValueError(f"num_folds must be >= 1, got {num_folds}")
  digest_hex = hashlib.sha1(item_id.encode("utf-8")).hexdigest()
  return int(digest_hex, 16) % num_folds


def resolve_gold_probability_key(
    probabilities: Mapping[str, float],
    expected_label: str,
) -> str | None:
  """Resolves an expected label against a slot probability dictionary.

  Args:
    probabilities: Mapping from option or label name to probability mass.
    expected_label: Ground-truth label recorded in the receipt.

  Returns:
    Matching key inside `probabilities`, or `None` when absent.

  Raises:
    ValueError: Never raised; non-numeric labels skip float comparison cleanly.
  """
  if not probabilities:
    return None
  if expected_label in probabilities:
    return expected_label
  trimmed_expected = expected_label.strip()
  if trimmed_expected in probabilities:
    return trimmed_expected
  for candidate_key in probabilities:
    if candidate_key.strip().lower() == trimmed_expected.lower():
      return candidate_key

  lowered = trimmed_expected.lower()
  if lowered in ("true", "yes", "1"):
    for alias in ("yes", "true", "1"):
      for candidate_key in probabilities:
        if candidate_key.strip().lower() == alias:
          return candidate_key
  elif lowered in ("false", "no", "0"):
    for alias in ("no", "false", "0"):
      for candidate_key in probabilities:
        if candidate_key.strip().lower() == alias:
          return candidate_key

  try:
    expected_float = float(trimmed_expected)
  except ValueError:
    return None
  for candidate_key in probabilities:
    try:
      if abs(float(candidate_key.strip()) - expected_float) <= 1e-9:
        return candidate_key
    except ValueError:
      continue
  return None


def compute_option_nonconformity_scores(
    probabilities: Mapping[str, float],
    method: str,
    k_reg: int = DEFAULT_RAPS_K_REG,
    lambda_reg: float = DEFAULT_RAPS_LAMBDA_REG,
) -> dict[str, float]:
  """Computes deterministic LAC, APS, or RAPS nonconformity scores per option.

  Args:
    probabilities: Non-empty mapping from option name to non-negative mass.
    method: Conformal score family (`"lac"`, `"aps"`, or `"raps"`).
    k_reg: Unpenalized top-rank cutoff for RAPS (`>= 0`).
    lambda_reg: Per-rank size penalty weight for RAPS (`>= 0`).

  Returns:
    Dictionary mapping each option name to its nonconformity score.

  Raises:
    ValueError: If `probabilities` is empty/invalid or `method` is unknown.
  """
  normalized_method = method.strip().lower()
  if normalized_method not in ("lac", "aps", "raps"):
    raise ValueError(f"unknown conformal method: {method!r}")
  if not probabilities:
    raise ValueError("probabilities must be non-empty")

  total_mass = 0.0
  cleaned_pairs: list[tuple[str, float]] = []
  for option_name in sorted(probabilities):
    prob_val = float(probabilities[option_name])
    if not math.isfinite(prob_val) or prob_val < 0.0:
      raise ValueError(f"invalid probability {prob_val} for {option_name!r}")
    cleaned_pairs.append((option_name, prob_val))
    total_mass += prob_val
  if total_mass <= 0.0 or not math.isfinite(total_mass):
    raise ValueError("probabilities must sum to a positive finite number")

  ordered_options = sorted(
      [(name, prob_val / total_mass) for name, prob_val in cleaned_pairs],
      key=lambda pair: (-pair[1], pair[0]),
  )
  if normalized_method == "lac":
    return {name: 1.0 - prob_val for name, prob_val in ordered_options}

  scores: dict[str, float] = {}
  cumulative_higher_mass = 0.0
  strict_rank_offset = 0
  cursor = 0
  while cursor < len(ordered_options):
    group_prob = ordered_options[cursor][1]
    group_end = cursor + 1
    while (
        group_end < len(ordered_options)
        and abs(ordered_options[group_end][1] - group_prob)
        <= CONFORMAL_TOLERANCE
    ):
      group_end += 1
    tie_group = ordered_options[cursor:group_end]
    group_start_rank = strict_rank_offset + 1
    rank_penalty = 0.0
    if normalized_method == "raps" and group_start_rank > k_reg:
      rank_penalty = lambda_reg * float(group_start_rank - k_reg)
    group_score = cumulative_higher_mass + rank_penalty
    group_mass = 0.0
    for option_name, prob_val in tie_group:
      scores[option_name] = group_score
      group_mass += prob_val
    cumulative_higher_mass += group_mass
    strict_rank_offset += len(tie_group)
    cursor = group_end
  return scores


def calibrate_conformal_quantile(
    calibration_scores: Sequence[float],
    alpha: float,
) -> float:
  """Computes the finite-sample split conformal quantile at level `1 - alpha`.

  Args:
    calibration_scores: Non-empty sequence of nonconformity scores on held-in
      calibration examples.
    alpha: Target miscoverage rate strictly inside `(0, 1)`.

  Returns:
    Order-statistic threshold `q_hat` (`math.inf` when `(n + 1)(1 - alpha) > n`).

  Raises:
    ValueError: If `calibration_scores` is empty or `alpha` is outside `(0, 1)`.
  """
  num_cal = len(calibration_scores)
  if num_cal == 0:
    raise ValueError("calibration_scores must be non-empty")
  if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
    raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
  sorted_scores = sorted(calibration_scores)
  order_index = max(
      1,
      math.ceil(float(num_cal + 1) * (1.0 - alpha) - CONFORMAL_TOLERANCE),
  )
  if order_index > num_cal:
    return math.inf
  return sorted_scores[order_index - 1]


def evaluate_kfold_conformal(
    items: Sequence[ReceiptSlotItem],
    method: str,
    alpha: float,
    abstain_labels: Sequence[str] | None = None,
) -> ConformalSummary | None:
  """Evaluates 5-fold out-of-sample split conformal prediction sets.

  Args:
    items: Loaded receipt slot items for one question or across all questions.
    method: Conformal score family (`"lac"`, `"aps"`, or `"raps"`).
    alpha: Target miscoverage rate strictly inside `(0, 1)`.
    abstain_labels: Optional sequence of option labels treated as abstention
      triggers when included in a prediction set (empty by default).

  Returns:
    Aggregated `ConformalSummary` when probability maps and labels are present,
    or `None` when fewer than 2 probability-bearing items exist.

  Raises:
    ValueError: If `method` is unknown or `alpha` lies outside `(0, 1)`.
  """
  normalized_method = method.strip().lower()
  if normalized_method not in ("lac", "aps", "raps"):
    raise ValueError(f"unknown conformal method: {method!r}")
  if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
    raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")

  prepared_entries: list[tuple[int, dict[str, float], dict[str, float], str]] = []
  for item in items:
    prob_map = item.get("probabilities")
    expected_label = item.get("expected")
    if not prob_map or expected_label is None:
      continue
    gold_key = resolve_gold_probability_key(prob_map, expected_label)
    if gold_key is None:
      continue
    score_map = compute_option_nonconformity_scores(prob_map, normalized_method)
    fold_idx = compute_fold_index(item["item_id"], CONFORMAL_FOLDS)
    prepared_entries.append((fold_idx, prob_map, score_map, gold_key))

  if len(prepared_entries) < 2:
    return None

  fold_quantiles: list[float] = []
  for fold_idx in range(CONFORMAL_FOLDS):
    train_scores = [
        score_map[gold_key]
        for entry_fold, _, score_map, gold_key in prepared_entries
        if entry_fold != fold_idx
    ]
    if not train_scores:
      train_scores = [
          score_map[gold_key] for _, _, score_map, gold_key in prepared_entries
      ]
    fold_quantiles.append(calibrate_conformal_quantile(train_scores, alpha))

  covered_count = 0
  singleton_count = 0
  escalated_count = 0
  total_set_size = 0

  abstain_lookup = (
      DEFAULT_ABSTAIN_LABELS
      if abstain_labels is None
      else frozenset(
          label.strip().lower() for label in abstain_labels if label.strip()
      )
  )

  for entry_fold, _, score_map, gold_key in prepared_entries:
    threshold_q = fold_quantiles[entry_fold]
    prediction_set = [
        opt_name
        for opt_name, score_val in score_map.items()
        if math.isinf(threshold_q)
        or score_val <= threshold_q + CONFORMAL_TOLERANCE
    ]
    if gold_key in prediction_set:
      covered_count += 1
    set_size = len(prediction_set)
    total_set_size += set_size
    is_singleton = set_size == 1
    contains_abstain = any(
        opt_name.strip().lower() in abstain_lookup
        for opt_name in prediction_set
    )
    if is_singleton:
      singleton_count += 1
    if not is_singleton or contains_abstain:
      escalated_count += 1

  evaluated_total = len(prepared_entries)
  return {
      "method": normalized_method,
      "alpha": round(alpha, 4),
      "folds": CONFORMAL_FOLDS,
      "evaluated_n": evaluated_total,
      "empirical_coverage": round(covered_count / evaluated_total, 4),
      "mean_set_size": round(total_set_size / evaluated_total, 3),
      "singleton_share": round(singleton_count / evaluated_total, 4),
      "escalated_share": round(escalated_count / evaluated_total, 4),
  }


def load(
    receipt_path: str | pathlib.Path,
    question: str | None = None,
) -> list[ReceiptSlotItem]:
  """Loads labelled slot evaluations from a JSONL dataset-runner receipt.

  Args:
    receipt_path: Path to the JSONL receipt file.
    question: Optional question ID filter.

  Returns:
    List of `ReceiptSlotItem` records containing confidence, hesitation, and
    optional probability distributions.

  Raises:
    OSError: If `receipt_path` cannot be opened.
    json.JSONDecodeError: If a non-empty line is not valid JSON.
  """
  loaded_items: list[ReceiptSlotItem] = []
  with open(receipt_path, encoding="utf-8") as receipt_file:
    for line_index, raw_line in enumerate(receipt_file):
      if not raw_line.strip():
        continue
      row_payload = json.loads(raw_line)
      row_id = str(row_payload.get("id") or f"row-{line_index}")
      slots_map = row_payload.get("slots") or {}
      for question_id, slot_data in slots_map.items():
        if question and question_id != question:
          continue
        if slot_data.get("correct") is None:
          continue
        confidence = float(slot_data.get("confidence") or 0.0)
        hesitation = slot_data.get("hesitation")
        if hesitation is None:
          hesitation = 1.0 - confidence
        raw_probs = slot_data.get("probabilities")
        parsed_probs: dict[str, float] | None = None
        if isinstance(raw_probs, dict) and raw_probs:
          parsed_probs = {
              str(opt_key): float(opt_val)
              for opt_key, opt_val in raw_probs.items()
          }
        expected_raw = slot_data.get("expected")
        expected_str = None if expected_raw is None else str(expected_raw)
        loaded_items.append({
            "item_id": f"{row_id}:{question_id}",
            "q": str(question_id),
            "correct": bool(slot_data["correct"]),
            "conf": confidence,
            "hes": float(hesitation),
            "expected": expected_str,
            "probabilities": parsed_probs,
        })
  return loaded_items


def report(
    items: Sequence[ReceiptSlotItem],
    conformal_method: str | None = None,
    conformal_alpha: float = 0.10,
    conformal_abstain_labels: Sequence[str] | None = None,
) -> dict[str, object] | None:
  """Builds accuracy, Brier, 10-bin ECE, gate, and optional conformal summaries.

  Args:
    items: Sequence of loaded slot items.
    conformal_method: Optional split conformal method (`"lac"`, `"aps"`,
      `"raps"`).
    conformal_alpha: Target miscoverage rate in `(0, 1)` when `conformal_method`
      is enabled.
    conformal_abstain_labels: Optional sequence of option labels treated as
      abstention triggers when included in a conformal prediction set.

  Returns:
    Summary dictionary for JSON or console rendering, or `None` if `items` is
    empty.

  Raises:
    ValueError: If `conformal_method` or `conformal_alpha` is invalid.
  """
  total_items = len(items)
  if not total_items:
    return None
  accuracy = sum(item["correct"] for item in items) / total_items
  brier_top = (
      sum((item["conf"] - item["correct"]) ** 2 for item in items) / total_items
  )
  reliability_bins: list[dict[str, object]] = []
  ece_10bin = 0.0
  for bin_index in range(10):
    lower_bound = bin_index / 10
    upper_bound = (bin_index + 1) / 10
    selected_items = [
        item
        for item in items
        if lower_bound <= item["conf"] < upper_bound
        or (bin_index == 9 and item["conf"] == 1.0)
    ]
    if selected_items:
      mean_conf = sum(item["conf"] for item in selected_items) / len(
          selected_items
      )
      mean_acc = sum(item["correct"] for item in selected_items) / len(
          selected_items
      )
      ece_10bin += len(selected_items) / total_items * abs(mean_conf - mean_acc)
      reliability_bins.append({
          "bin": f"{lower_bound:.1f}-{upper_bound:.1f}",
          "n": len(selected_items),
          "mean_conf": round(mean_conf, 3),
          "accuracy": round(mean_acc, 3),
      })

  hesitation_gates: list[dict[str, object]] = []
  for threshold_value in THRESHOLDS:
    kept_items = [item for item in items if item["hes"] < threshold_value]
    kept_accuracy = (
        round(
            sum(item["correct"] for item in kept_items) / len(kept_items),
            3,
        )
        if kept_items
        else None
    )
    hesitation_gates.append({
        "hesitation_below": threshold_value,
        "kept_share": round(len(kept_items) / total_items, 3),
        "kept_accuracy": kept_accuracy,
        "escalated": total_items - len(kept_items),
    })

  summary_payload: dict[str, object] = {
      "n": total_items,
      "accuracy": round(accuracy, 3),
      "brier_top": round(brier_top, 4),
      "ece10": round(ece_10bin, 4),
      "reliability": reliability_bins,
      "gates": hesitation_gates,
  }
  if conformal_method:
    summary_payload["conformal"] = evaluate_kfold_conformal(
        items,
        conformal_method,
        conformal_alpha,
        abstain_labels=conformal_abstain_labels,
    )
  return summary_payload


def show(section_name: str, summary_report: Mapping[str, object]) -> None:
  """Prints a human-readable calibration table for one question or overall.

  Args:
    section_name: Display heading (`"overall"` or question ID).
    summary_report: Dictionary returned by `report(...)`.

  Returns:
    None.

  Raises:
    KeyError: If required summary keys are missing from `summary_report`.
  """
  print(
      f"\n== {section_name}: n={summary_report['n']}"
      f"  accuracy={summary_report['accuracy']:.1%}"
      f"  Brier(top)={summary_report['brier_top']:.4f}"
      f"  ECE(10 bins)={summary_report['ece10']:.4f}"
  )
  print("  confidence bin   n   mean conf  accuracy")
  reliability_rows = summary_report["reliability"]
  if isinstance(reliability_rows, list):
    for bin_row in reliability_rows:
      print(
          f"  {bin_row['bin']:>12s} {bin_row['n']:4d}"
          f"   {bin_row['mean_conf']:8.3f}  {bin_row['accuracy']:8.3f}"
      )
  print("  keep if hesitation <   kept   accuracy of kept   escalated")
  gate_rows = summary_report["gates"]
  if isinstance(gate_rows, list):
    for gate_row in gate_rows:
      threshold_label = (
          "any"
          if gate_row["hesitation_below"] > 1
          else f"{gate_row['hesitation_below']:.0%}"
      )
      kept_acc_label = (
          "-"
          if gate_row["kept_accuracy"] is None
          else f"{gate_row['kept_accuracy']:.1%}"
      )
      print(
          f"  {threshold_label:>20s} {gate_row['kept_share']:7.1%}"
          f" {kept_acc_label:>18s} {gate_row['escalated']:11d}"
      )
  conformal_block = summary_report.get("conformal")
  if isinstance(conformal_block, dict):
    print(
        f"  conformal ({conformal_block['method']}, alpha={conformal_block['alpha']:.2f},"
        f" {conformal_block['folds']}-fold n={conformal_block['evaluated_n']}):"
        f" coverage={conformal_block['empirical_coverage']:.1%}"
        f"  mean_size={conformal_block['mean_set_size']:.2f}"
        f"  singleton={conformal_block['singleton_share']:.1%}"
        f"  escalated={conformal_block['escalated_share']:.1%}"
    )


def main(argv: Sequence[str] | None = None) -> None:
  """Parses CLI arguments and prints JSON or tabular policy calibration metrics.

  Args:
    argv: Optional argument sequence overriding `sys.argv[1:]`.

  Returns:
    None.

  Raises:
    SystemExit: If no labelled answers match or CLI arguments are invalid.
  """
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("receipt")
  parser.add_argument("--question", help="only this question id")
  parser.add_argument(
      "--conformal",
      choices=("lac", "aps", "raps"),
      default=None,
      help="compute 5-fold out-of-sample split conformal prediction sets",
  )
  parser.add_argument(
      "--conformal-alpha",
      type=float,
      default=0.10,
      help="target miscoverage rate alpha in (0, 1) for --conformal",
  )
  parser.add_argument(
      "--conformal-abstain-labels",
      default="",
      help=(
          "comma-separated option labels treated as abstention triggers in "
          "--conformal sets (empty by default)"
      ),
  )
  parser.add_argument("--json", action="store_true")
  parsed_args = parser.parse_args(argv)

  if parsed_args.conformal and not (0.0 < parsed_args.conformal_alpha < 1.0):
    sys.exit("--conformal-alpha must lie strictly between 0 and 1")

  abstain_labels = tuple(
      part.strip()
      for part in parsed_args.conformal_abstain_labels.split(",")
      if part.strip()
  )

  loaded_items = load(parsed_args.receipt, parsed_args.question)
  if not loaded_items:
    sys.exit("no labelled answers found (does the dataset have expected values?)")

  output_reports: dict[str, dict[str, object]] = {}
  overall_summary = report(
      loaded_items,
      conformal_method=parsed_args.conformal,
      conformal_alpha=parsed_args.conformal_alpha,
      conformal_abstain_labels=abstain_labels,
  )
  if overall_summary is not None:
    output_reports["overall"] = overall_summary
  for question_id in sorted({item["q"] for item in loaded_items}):
    question_items = [item for item in loaded_items if item["q"] == question_id]
    question_summary = report(
        question_items,
        conformal_method=parsed_args.conformal,
        conformal_alpha=parsed_args.conformal_alpha,
        conformal_abstain_labels=abstain_labels,
    )
    if question_summary is not None:
      output_reports[question_id] = question_summary

  if parsed_args.json:
    print(json.dumps(output_reports, indent=1))
    return
  for section_name, section_summary in output_reports.items():
    show(section_name, section_summary)


if __name__ == "__main__":
  main()
