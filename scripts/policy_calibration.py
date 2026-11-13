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
  matched escalation       optional cross-fitted head-to-head risk comparison

Usage:
  python3 scripts/policy_calibration.py results_receipt.jsonl [--question reason]
    [--conformal {lac,aps,raps}] [--conformal-alpha 0.10]
    [--oof-temperature] [--matched-escalation-rates 0.10,0.20,0.30,0.40]
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
MATCHED_POLICIES: tuple[str, ...] = (
    "raw_h_norm",
    "lac",
    "raps",
    "oof_temp_lac",
    "oof_temp_raps",
)


class ReceiptSlotItem(TypedDict):
  item_id: str
  q: str
  correct: bool
  conf: float
  hes: float
  expected: str | None
  probabilities: dict[str, float] | None


class ConformalSummary(TypedDict, total=False):
  method: str
  alpha: float
  folds: int
  evaluated_n: int
  empirical_coverage: float
  mean_set_size: float
  singleton_share: float
  escalated_share: float
  oof_temperature: bool
  mean_temperature: float


def _probability_desc_sort_key(pair: tuple[str, float]) -> tuple[float, str]:
  """Sort `(option_name, probability)` descending by probability then name."""
  return (-pair[1], pair[0])


def _normalized_ordered_options(
    probabilities: Mapping[str, float],
) -> list[tuple[str, float]]:
  """Validate, normalize, and sort option probabilities in descending order."""
  if not probabilities:
    raise ValueError("probabilities must be non-empty")

  total_mass = 0.0
  cleaned_pairs: list[tuple[str, float]] = []
  for option_name in sorted(probabilities):
    raw_value = probabilities[option_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
      raise ValueError(f"non-numeric probability for {option_name!r}")
    prob_val = float(raw_value)
    if not math.isfinite(prob_val) or prob_val < 0.0:
      raise ValueError(f"invalid probability {prob_val} for {option_name!r}")
    cleaned_pairs.append((option_name, prob_val))
    total_mass += prob_val
  if total_mass <= 0.0 or not math.isfinite(total_mass):
    raise ValueError("probabilities must sum to a positive finite number")

  normalized_pairs = [
      (option_name, prob_val / total_mass)
      for option_name, prob_val in cleaned_pairs
  ]
  normalized_pairs.sort(key=_probability_desc_sort_key)
  return normalized_pairs


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


def _tie_jitter(item_id: str, policy_tag: str) -> float:
  """Computes deterministic `1e-12` tie-breaking jitter from SHA-1."""
  digest = hashlib.sha1(f"{item_id}:{policy_tag}".encode("utf-8")).digest()
  top_bits = int.from_bytes(digest[:8], "big") >> 11
  return 1e-12 * (float(top_bits) / float(1 << 53))


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


def apply_temperature(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
  """Scales discrete option probabilities by post-hoc temperature `p^(1 / T)`.

  Args:
    probabilities: Non-empty mapping from option name to non-negative mass.
    temperature: Positive scalar temperature in `(0.0, 20.0]`.

  Returns:
    Normalized probability dictionary scaled at `temperature`.

  Raises:
    ValueError: If `probabilities` or `temperature` is invalid.
  """
  if (
      isinstance(temperature, bool)
      or not isinstance(temperature, (int, float))
      or not math.isfinite(float(temperature))
      or float(temperature) <= 0.0
      or float(temperature) > 20.0
  ):
    raise ValueError(f"temperature must lie in (0, 20.0], got {temperature!r}")

  temperature_scalar = float(temperature)
  ordered_options = _normalized_ordered_options(probabilities)
  if abs(temperature_scalar - 1.0) <= CONFORMAL_TOLERANCE:
    return dict(ordered_options)

  scaled_logits: list[tuple[str, float]] = []
  max_logit = -math.inf
  for option_name, prob_val in ordered_options:
    logit_val = (
        -30.0
        if prob_val <= 1e-13
        else math.log(prob_val) / temperature_scalar
    )
    scaled_logits.append((option_name, logit_val))
    if logit_val > max_logit:
      max_logit = logit_val

  exp_pairs = [
      (option_name, math.exp(logit_val - max_logit))
      for option_name, logit_val in scaled_logits
  ]
  partition_sum = sum(weight for _, weight in exp_pairs)
  return {
      option_name: weight / partition_sum for option_name, weight in exp_pairs
  }


def fit_temperature_nll(
    probabilities_list: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
    min_temp: float = 0.05,
    max_temp: float = 10.0,
) -> float:
  """Fits scalar temperature `T` minimizing multiclass cross-entropy NLL.

  Args:
    probabilities_list: Non-empty sequence of option probability mappings.
    gold_labels: Expected option labels aligned with `probabilities_list`.
    min_temp: Positive lower bound on fitted temperature.
    max_temp: Upper bound on fitted temperature (`<= 20.0`).

  Returns:
    Fitted scalar temperature in `[min_temp, max_temp]`.

  Raises:
    ValueError: If inputs are empty, mismatched, or missing gold labels.
  """
  if not probabilities_list:
    raise ValueError("probabilities_list must be non-empty")
  if len(probabilities_list) != len(gold_labels):
    raise ValueError("probabilities_list and gold_labels length mismatch")
  if min_temp <= 0.0 or max_temp <= min_temp or max_temp > 20.0:
    raise ValueError("invalid [min_temp, max_temp] bounds")

  prepared_logits: list[tuple[list[float], int]] = []
  for row_index, prob_map in enumerate(probabilities_list):
    ordered_options = _normalized_ordered_options(prob_map)
    resolved_gold = resolve_gold_probability_key(prob_map, gold_labels[row_index])
    if resolved_gold is None:
      raise ValueError(
          f"gold label {gold_labels[row_index]!r} missing at index {row_index}"
      )
    row_logits: list[float] = []
    gold_index = -1
    for opt_index, (option_name, prob_val) in enumerate(ordered_options):
      row_logits.append(-30.0 if prob_val <= 1e-13 else math.log(prob_val))
      if option_name == resolved_gold:
        gold_index = opt_index
    prepared_logits.append((row_logits, gold_index))

  def _mean_nll_at_inv_temp(inv_temp: float) -> float:
    total_nll = 0.0
    for row_logits, gold_index in prepared_logits:
      scaled_row = [logit_val * inv_temp for logit_val in row_logits]
      max_val = max(scaled_row)
      log_sum_exp = max_val + math.log(
          sum(math.exp(logit_val - max_val) for logit_val in scaled_row)
      )
      total_nll += log_sum_exp - scaled_row[gold_index]
    return total_nll / float(len(prepared_logits))

  beta_low = 1.0 / max_temp
  beta_high = 1.0 / min_temp
  golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0

  left_probe = beta_high - golden_ratio * (beta_high - beta_low)
  right_probe = beta_low + golden_ratio * (beta_high - beta_low)
  left_loss = _mean_nll_at_inv_temp(left_probe)
  right_loss = _mean_nll_at_inv_temp(right_probe)

  for _ in range(60):
    if abs(beta_high - beta_low) <= 1e-8:
      break
    if left_loss < right_loss:
      beta_high = right_probe
      right_probe = left_probe
      right_loss = left_loss
      left_probe = beta_high - golden_ratio * (beta_high - beta_low)
      left_loss = _mean_nll_at_inv_temp(left_probe)
    else:
      beta_low = left_probe
      left_probe = right_probe
      left_loss = right_loss
      right_probe = beta_low + golden_ratio * (beta_high - beta_low)
      right_loss = _mean_nll_at_inv_temp(right_probe)

  optimal_beta = 0.5 * (beta_low + beta_high)
  return max(min_temp, min(max_temp, 1.0 / optimal_beta))


def compute_normalized_entropy(probabilities: Mapping[str, float]) -> float:
  """Computes cardinality-normalized entropy `H(p) / ln(K)` in `[0.0, 1.0]`.

  Args:
    probabilities: Non-empty mapping from option name to non-negative mass.

  Returns:
    Normalized Shannon entropy in `[0.0, 1.0]` (`0.0` when `K <= 1`).

  Raises:
    ValueError: If `probabilities` is empty or contains invalid values.
  """
  ordered_options = _normalized_ordered_options(probabilities)
  if len(ordered_options) <= 1:
    return 0.0
  entropy_nats = 0.0
  for _, prob_val in ordered_options:
    if prob_val > 0.0:
      entropy_nats -= prob_val * math.log(prob_val)
  return max(0.0, min(1.0, entropy_nats / math.log(float(len(ordered_options)))))


def _compute_raps_uncertainty_score(
    probabilities: Mapping[str, float],
    item_id: str,
) -> float:
  """Computes continuous RAPS competitor-aware uncertainty score."""
  ordered_options = _normalized_ordered_options(probabilities)
  top_prob = ordered_options[0][1]
  second_prob = ordered_options[1][1] if len(ordered_options) > 1 else 0.0
  third_prob = ordered_options[2][1] if len(ordered_options) > 2 else 0.0
  return (
      (1.0 - top_prob)
      + 0.35 * second_prob
      + 0.12 * third_prob
      + _tie_jitter(item_id, "raps")
  )


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
  ordered_options = _normalized_ordered_options(probabilities)
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
  if (
      isinstance(alpha, bool)
      or not math.isfinite(alpha)
      or alpha <= 0.0
      or alpha >= 1.0
  ):
    raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")
  sorted_scores = sorted(calibration_scores)
  order_index = max(
      1,
      math.ceil(float(num_cal + 1) * (1.0 - alpha) - CONFORMAL_TOLERANCE),
  )
  if order_index > num_cal:
    return math.inf
  return sorted_scores[order_index - 1]


def solve_matched_escalation_threshold(
    validation_scores: Sequence[float],
    target_escalation_rate: float,
) -> float:
  """Solves the `D_val` cutoff escalating `target_escalation_rate` of items.

  Args:
    validation_scores: Finite uncertainty scores on `D_val` (`score > tau`
      escalates).
    target_escalation_rate: Target escalation fraction strictly in `(0, 1)`.

  Returns:
    Scalar threshold `tau` matching `target_escalation_rate` on `D_val`.

  Raises:
    ValueError: If `validation_scores` is empty/non-finite or
      `target_escalation_rate` lies outside `(0, 1)`.
  """
  if not validation_scores:
    raise ValueError("validation_scores must be non-empty")
  if (
      isinstance(target_escalation_rate, bool)
      or not math.isfinite(target_escalation_rate)
      or target_escalation_rate <= 0.0
      or target_escalation_rate >= 1.0
  ):
    raise ValueError(
        "target_escalation_rate must lie strictly in (0, 1), got "
        f"{target_escalation_rate!r}"
    )

  cleaned_scores: list[float] = []
  for raw_score in validation_scores:
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
      raise ValueError(f"non-numeric validation score {raw_score!r}")
    numeric_score = float(raw_score)
    if not math.isfinite(numeric_score):
      raise ValueError(f"non-finite validation score {numeric_score!r}")
    cleaned_scores.append(numeric_score)

  ordered_scores = sorted(cleaned_scores)
  keep_fraction = 1.0 - target_escalation_rate
  cutoff_index = min(
      len(ordered_scores) - 1,
      max(0, math.ceil(keep_fraction * len(ordered_scores)) - 1),
  )
  return ordered_scores[cutoff_index]


def parse_escalation_rates(raw_rates: str) -> tuple[float, ...]:
  """Parses a comma-separated list of target escalation rates in `(0, 1)`.

  Args:
    raw_rates: Comma-separated float string (for example `"0.10,0.20,0.30"`).

  Returns:
    Sorted tuple of unique target escalation fractions.

  Raises:
    ValueError: If any entry is malformed or outside `(0, 1)`.
  """
  trimmed = raw_rates.strip()
  if not trimmed:
    return ()
  parsed_values: list[float] = []
  for token in trimmed.split(","):
    part = token.strip()
    if not part:
      continue
    rate_val = float(part)
    if not math.isfinite(rate_val) or rate_val <= 0.0 or rate_val >= 1.0:
      raise ValueError(
          f"escalation rate must lie strictly in (0, 1), got {part!r}"
      )
    parsed_values.append(round(rate_val, 6))
  if not parsed_values:
    raise ValueError("no valid escalation rates found")
  return tuple(sorted(set(parsed_values)))


def _fit_nested_oof_calibration_probs(
    calibration_entries: Sequence[tuple[int, str, dict[str, float], str]],
) -> tuple[float, list[dict[str, float]]]:
  """Fits global calibration temperature and nested OOF-scaled probabilities.

  Ensures temperature scaling on `D_cal` is fitted out-of-fold prior to
  computing conformal calibration scores `s(x, y)`.

  Args:
    calibration_entries: Sequence of `(fold_idx, item_id, prob_map, gold_key)`
      strictly inside `D_cal`.

  Returns:
    Tuple `(eval_temperature, oof_scaled_probs)` aligned with
    `calibration_entries`.
  """
  if not calibration_entries:
    return 1.0, []
  all_probs = [prob_map for _, _, prob_map, _ in calibration_entries]
  all_golds = [gold_key for _, _, _, gold_key in calibration_entries]
  eval_temperature = fit_temperature_nll(all_probs, all_golds)

  distinct_folds = sorted({fold_id for fold_id, _, _, _ in calibration_entries})
  if len(distinct_folds) < 2:
    scaled_all = [
        apply_temperature(prob_map, eval_temperature) for prob_map in all_probs
    ]
    return eval_temperature, scaled_all

  subfold_temperatures: dict[int, float] = {}
  for held_fold in distinct_folds:
    sub_probs = [
        prob_map
        for fold_id, _, prob_map, _ in calibration_entries
        if fold_id != held_fold
    ]
    sub_golds = [
        gold_key
        for fold_id, _, _, gold_key in calibration_entries
        if fold_id != held_fold
    ]
    subfold_temperatures[held_fold] = (
        fit_temperature_nll(sub_probs, sub_golds)
        if sub_probs
        else eval_temperature
    )

  oof_scaled_probs = [
      apply_temperature(prob_map, subfold_temperatures[fold_id])
      for fold_id, _, prob_map, _ in calibration_entries
  ]
  return eval_temperature, oof_scaled_probs


def evaluate_kfold_conformal(
    items: Sequence[ReceiptSlotItem],
    method: str,
    alpha: float,
    abstain_labels: Sequence[str] | None = None,
    oof_temperature: bool = False,
) -> ConformalSummary | None:
  """Evaluates 5-fold out-of-sample split conformal prediction sets.

  Args:
    items: Loaded receipt slot items for one question or across all questions.
    method: Conformal score family (`"lac"`, `"aps"`, or `"raps"`).
    alpha: Target miscoverage rate strictly inside `(0, 1)`.
    abstain_labels: Optional sequence of option labels treated as abstention
      triggers when included in a prediction set (empty by default).
    oof_temperature: When `True`, fits nested out-of-fold NLL temperature
      scaling on `D_cal` prior to conformal quantile fitting.

  Returns:
    Aggregated `ConformalSummary` when probability maps and labels are present,
    or `None` when fewer than 2 probability-bearing items exist.

  Raises:
    ValueError: If `method` is unknown or `alpha` lies outside `(0, 1)`.
  """
  normalized_method = method.strip().lower()
  if normalized_method not in ("lac", "aps", "raps"):
    raise ValueError(f"unknown conformal method: {method!r}")
  if (
      isinstance(alpha, bool)
      or not math.isfinite(alpha)
      or alpha <= 0.0
      or alpha >= 1.0
  ):
    raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}")

  raw_entries: list[tuple[int, str, dict[str, float], str]] = []
  for item in items:
    prob_map = item.get("probabilities")
    expected_label = item.get("expected")
    if not prob_map or expected_label is None:
      continue
    gold_key = resolve_gold_probability_key(prob_map, expected_label)
    if gold_key is None:
      continue
    _normalized_ordered_options(prob_map)
    fold_idx = compute_fold_index(item["item_id"], CONFORMAL_FOLDS)
    raw_entries.append((fold_idx, item["item_id"], prob_map, gold_key))

  if len(raw_entries) < 2:
    return None

  fold_quantiles: list[float] = []
  fold_temperatures: list[float] = []
  for fold_idx in range(CONFORMAL_FOLDS):
    cal_entries = [
        entry for entry in raw_entries if entry[0] != fold_idx
    ]
    if not cal_entries:
      cal_entries = list(raw_entries)

    if oof_temperature:
      eval_temp, cal_probs = _fit_nested_oof_calibration_probs(cal_entries)
    else:
      eval_temp = 1.0
      cal_probs = [entry[2] for entry in cal_entries]

    train_scores = [
        compute_option_nonconformity_scores(
            cal_probs[idx], normalized_method
        )[cal_entries[idx][3]]
        for idx in range(len(cal_entries))
    ]
    fold_quantiles.append(calibrate_conformal_quantile(train_scores, alpha))
    fold_temperatures.append(eval_temp)

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

  for entry_fold, _, prob_map, gold_key in raw_entries:
    eval_probs = (
        apply_temperature(prob_map, fold_temperatures[entry_fold])
        if oof_temperature
        else prob_map
    )
    score_map = compute_option_nonconformity_scores(
        eval_probs, normalized_method
    )
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

  evaluated_total = len(raw_entries)
  summary: ConformalSummary = {
      "method": normalized_method,
      "alpha": round(alpha, 4),
      "folds": CONFORMAL_FOLDS,
      "evaluated_n": evaluated_total,
      "empirical_coverage": round(covered_count / evaluated_total, 4),
      "mean_set_size": round(total_set_size / evaluated_total, 3),
      "singleton_share": round(singleton_count / evaluated_total, 4),
      "escalated_share": round(escalated_count / evaluated_total, 4),
  }
  if oof_temperature:
    summary["oof_temperature"] = True
    summary["mean_temperature"] = round(
        sum(fold_temperatures) / float(len(fold_temperatures)), 4
    )
  return summary


def evaluate_matched_escalation_sweep(
    items: Sequence[ReceiptSlotItem],
    target_rates: Sequence[float],
    conformal_alpha: float = 0.10,
) -> list[dict[str, object]] | None:
  """Compares `H_norm`, `LAC`, and `RAPS` at matched `D_val` escalation rates.

  Rotates across the 5 SHA-1 folds: for each held-out `D_test` fold, partitions
  held-in folds into `D_cal` (OOF NLL temperature + conformal quantile fitting)
  and `D_val` (matched-escalation threshold solving via
  `solve_matched_escalation_threshold`).

  Args:
    items: Loaded receipt slot items.
    target_rates: Sequence of target escalation rates strictly in `(0, 1)`.
    conformal_alpha: Miscoverage rate in `(0, 1)` for Stage-2 shortlist sets.

  Returns:
    List of per-rate comparison dictionaries, or `None` when fewer than 5
    probability-bearing items exist or `target_rates` is empty.

  Raises:
    ValueError: If `target_rates` or `conformal_alpha` contains invalid values.
  """
  if not target_rates:
    return None
  for rate_val in target_rates:
    if (
        isinstance(rate_val, bool)
        or not math.isfinite(rate_val)
        or rate_val <= 0.0
        or rate_val >= 1.0
    ):
      raise ValueError(
          f"target escalation rate must lie in (0, 1), got {rate_val!r}"
      )
  if (
      isinstance(conformal_alpha, bool)
      or not math.isfinite(conformal_alpha)
      or conformal_alpha <= 0.0
      or conformal_alpha >= 1.0
  ):
    raise ValueError(
        f"conformal_alpha must lie in (0, 1), got {conformal_alpha!r}"
    )

  raw_entries: list[tuple[int, str, dict[str, float], str]] = []
  for item in items:
    prob_map = item.get("probabilities")
    expected_label = item.get("expected")
    if not prob_map or expected_label is None:
      continue
    gold_key = resolve_gold_probability_key(prob_map, expected_label)
    if gold_key is None:
      continue
    _normalized_ordered_options(prob_map)
    fold_idx = compute_fold_index(item["item_id"], CONFORMAL_FOLDS)
    raw_entries.append((fold_idx, item["item_id"], prob_map, gold_key))

  if len(raw_entries) < CONFORMAL_FOLDS:
    return None

  accumulators: dict[float, dict[str, list[tuple[bool, bool, int]]]] = {
      rate_val: {policy_name: [] for policy_name in MATCHED_POLICIES}
      for rate_val in target_rates
  }

  for test_fold in range(CONFORMAL_FOLDS):
    test_entries = [entry for entry in raw_entries if entry[0] == test_fold]
    if not test_entries:
      continue
    held_in = [entry for entry in raw_entries if entry[0] != test_fold]
    if not held_in:
      held_in = list(raw_entries)

    val_fold = (test_fold + 1) % CONFORMAL_FOLDS
    val_entries = [entry for entry in held_in if entry[0] == val_fold]
    cal_entries = [entry for entry in held_in if entry[0] != val_fold]
    if not cal_entries or not val_entries:
      cal_entries = held_in
      val_entries = held_in

    oof_temp, cal_oof_probs = _fit_nested_oof_calibration_probs(cal_entries)

    raw_lac_cal = [
        compute_option_nonconformity_scores(entry[2], "lac")[entry[3]]
        for entry in cal_entries
    ]
    raw_raps_cal = [
        compute_option_nonconformity_scores(entry[2], "raps")[entry[3]]
        for entry in cal_entries
    ]
    oof_lac_cal = [
        compute_option_nonconformity_scores(cal_oof_probs[idx], "lac")[
            cal_entries[idx][3]
        ]
        for idx in range(len(cal_entries))
    ]
    oof_raps_cal = [
        compute_option_nonconformity_scores(cal_oof_probs[idx], "raps")[
            cal_entries[idx][3]
        ]
        for idx in range(len(cal_entries))
    ]

    q_raw_lac = calibrate_conformal_quantile(raw_lac_cal, conformal_alpha)
    q_raw_raps = calibrate_conformal_quantile(raw_raps_cal, conformal_alpha)
    q_oof_lac = calibrate_conformal_quantile(oof_lac_cal, conformal_alpha)
    q_oof_raps = calibrate_conformal_quantile(oof_raps_cal, conformal_alpha)

    def _score_for_policy(
        policy_name: str,
        item_id: str,
        prob_map: Mapping[str, float],
    ) -> tuple[float, int]:
      ordered_raw = _normalized_ordered_options(prob_map)
      num_options = len(ordered_raw)
      if policy_name == "raw_h_norm":
        return (
            compute_normalized_entropy(prob_map) + _tie_jitter(item_id, "h"),
            num_options,
        )
      if policy_name == "lac":
        lac_map = compute_option_nonconformity_scores(prob_map, "lac")
        set_count = sum(
            1
            for val in lac_map.values()
            if math.isinf(q_raw_lac) or val <= q_raw_lac + CONFORMAL_TOLERANCE
        )
        return (
            (1.0 - ordered_raw[0][1]) + _tie_jitter(item_id, "lac"),
            min(num_options, max(2, set_count)),
        )
      if policy_name == "raps":
        raps_map = compute_option_nonconformity_scores(prob_map, "raps")
        set_count = sum(
            1
            for val in raps_map.values()
            if math.isinf(q_raw_raps) or val <= q_raw_raps + CONFORMAL_TOLERANCE
        )
        return (
            _compute_raps_uncertainty_score(prob_map, item_id),
            min(num_options, max(2, set_count)),
        )
      scaled_map = apply_temperature(prob_map, oof_temp)
      ordered_scaled = _normalized_ordered_options(scaled_map)
      if policy_name == "oof_temp_lac":
        oof_lac_map = compute_option_nonconformity_scores(scaled_map, "lac")
        set_count = sum(
            1
            for val in oof_lac_map.values()
            if math.isinf(q_oof_lac) or val <= q_oof_lac + CONFORMAL_TOLERANCE
        )
        return (
            (1.0 - ordered_scaled[0][1]) + _tie_jitter(item_id, "oof_lac"),
            min(num_options, max(2, set_count)),
        )
      oof_raps_map = compute_option_nonconformity_scores(scaled_map, "raps")
      set_count = sum(
          1
          for val in oof_raps_map.values()
          if math.isinf(q_oof_raps) or val <= q_oof_raps + CONFORMAL_TOLERANCE
      )
      return (
          _compute_raps_uncertainty_score(scaled_map, item_id),
          min(num_options, max(2, set_count)),
      )

    val_scores_by_policy: dict[str, list[float]] = {
        policy_name: [
            _score_for_policy(policy_name, item_id, prob_map)[0]
            for _, item_id, prob_map, _ in val_entries
        ]
        for policy_name in MATCHED_POLICIES
    }

    for rate_val in target_rates:
      for policy_name in MATCHED_POLICIES:
        threshold_tau = solve_matched_escalation_threshold(
            val_scores_by_policy[policy_name], rate_val
        )
        for _, item_id, prob_map, gold_key in test_entries:
          ordered_raw = _normalized_ordered_options(prob_map)
          top_correct = ordered_raw[0][0] == gold_key
          item_score, shortlist_k = _score_for_policy(
              policy_name, item_id, prob_map
          )
          is_escalated = item_score > threshold_tau
          accumulators[rate_val][policy_name].append(
              (is_escalated, top_correct, shortlist_k)
          )

  sweep_rows: list[dict[str, object]] = []
  for rate_val in target_rates:
    policies_block: dict[str, dict[str, object]] = {}
    for policy_name in MATCHED_POLICIES:
      outcomes = accumulators[rate_val][policy_name]
      total_n = len(outcomes)
      if total_n == 0:
        continue
      kept_outcomes = [row for row in outcomes if not row[0]]
      escalated_outcomes = [row for row in outcomes if row[0]]
      kept_acc: float | None = None
      kept_err: float | None = None
      if kept_outcomes:
        correct_kept = sum(1 for _, is_corr, _ in kept_outcomes if is_corr)
        kept_acc = round(correct_kept / float(len(kept_outcomes)), 4)
        kept_err = round(1.0 - (correct_kept / float(len(kept_outcomes))), 4)
      mean_candidates = (
          round(
              sum(cand_k for _, _, cand_k in escalated_outcomes)
              / float(len(escalated_outcomes)),
              2,
          )
          if escalated_outcomes
          else 0.0
      )
      policies_block[policy_name] = {
          "evaluated_n": total_n,
          "empirical_escalation_share": round(
              len(escalated_outcomes) / float(total_n), 4
          ),
          "kept_share": round(len(kept_outcomes) / float(total_n), 4),
          "kept_accuracy": kept_acc,
          "kept_error_rate": kept_err,
          "mean_stage2_candidates": mean_candidates,
      }
    sweep_rows.append({
        "target_escalation_rate": round(rate_val, 4),
        "policies": policies_block,
    })
  return sweep_rows


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
    oof_temperature: bool = False,
    matched_escalation_rates: Sequence[float] | None = None,
) -> dict[str, object] | None:
  """Builds accuracy, Brier, ECE, gate, conformal, and matched-budget summaries.

  Args:
    items: Sequence of loaded slot items.
    conformal_method: Optional split conformal method (`"lac"`, `"aps"`,
      `"raps"`).
    conformal_alpha: Target miscoverage rate in `(0, 1)` when `conformal_method`
      or `matched_escalation_rates` is enabled.
    conformal_abstain_labels: Optional sequence of option labels treated as
      abstention triggers when included in a conformal prediction set.
    oof_temperature: When `True`, applies 5-fold out-of-fold NLL temperature
      scaling on `D_cal` prior to conformal quantile fitting.
    matched_escalation_rates: Optional sequence of target escalation rates in
      `(0, 1)` for head-to-head `H_norm` vs `LAC` vs `RAPS` comparison.

  Returns:
    Summary dictionary for JSON or console rendering, or `None` if `items` is
    empty.

  Raises:
    ValueError: If conformal or matched-escalation arguments are invalid.
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
        oof_temperature=oof_temperature,
    )
  if matched_escalation_rates:
    summary_payload["matched_escalation"] = evaluate_matched_escalation_sweep(
        items,
        matched_escalation_rates,
        conformal_alpha=conformal_alpha,
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
    temp_suffix = (
        f"  T_oof={conformal_block['mean_temperature']:.3f}"
        if conformal_block.get("oof_temperature")
        else ""
    )
    print(
        f"  conformal ({conformal_block['method']}, alpha={conformal_block['alpha']:.2f},"
        f" {conformal_block['folds']}-fold n={conformal_block['evaluated_n']}):"
        f" coverage={conformal_block['empirical_coverage']:.1%}"
        f"  mean_size={conformal_block['mean_set_size']:.2f}"
        f"  singleton={conformal_block['singleton_share']:.1%}"
        f"  escalated={conformal_block['escalated_share']:.1%}"
        f"{temp_suffix}"
    )
  matched_rows = summary_report.get("matched_escalation")
  if isinstance(matched_rows, list):
    for matched_entry in matched_rows:
      target_rate = matched_entry["target_escalation_rate"]
      print(f"  matched escalation (target={target_rate:.0%}):")
      policies_map = matched_entry.get("policies")
      if isinstance(policies_map, dict):
        for policy_name, metrics in policies_map.items():
          kept_err_str = (
              "-"
              if metrics["kept_error_rate"] is None
              else f"{metrics['kept_error_rate']:.2%}"
          )
          print(
              f"    {policy_name:>14s}: esc={metrics['empirical_escalation_share']:.1%}"
              f"  kept_err={kept_err_str:>7s}"
              f"  stage2_k={metrics['mean_stage2_candidates']:5.2f}"
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
      "--oof-temperature",
      action="store_true",
      help=(
          "fit 5-fold out-of-fold NLL temperature scaling on D_cal prior to "
          "conformal quantile fitting"
      ),
  )
  parser.add_argument(
      "--matched-escalation-rates",
      default="",
      help=(
          "comma-separated target escalation rates in (0, 1) (for example "
          "0.10,0.20,0.30,0.40) solved on D_val across H_norm, LAC, and RAPS"
      ),
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

  if (parsed_args.conformal or parsed_args.matched_escalation_rates) and not (
      0.0 < parsed_args.conformal_alpha < 1.0
  ):
    sys.exit("--conformal-alpha must lie strictly between 0 and 1")

  try:
    escalation_rates = parse_escalation_rates(
        parsed_args.matched_escalation_rates
    )
  except ValueError as rate_error:
    sys.exit(f"invalid --matched-escalation-rates: {rate_error}")

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
      oof_temperature=parsed_args.oof_temperature,
      matched_escalation_rates=escalation_rates,
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
        oof_temperature=parsed_args.oof_temperature,
        matched_escalation_rates=escalation_rates,
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
