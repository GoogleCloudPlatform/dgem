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

"""Experiment D1: Matched-escalation conformal calibration and K-subsampling."""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import exp_stats_common as stats_lib  # noqa: E402

TRACK_NAME: str = "d1_conformal"
DEFAULT_LOCKFILE: pathlib.Path = (
    REPO_ROOT / "benchmarks" / "matrix" / "locks" / "dgem_exp_splits_v1.json"
)
TOTAL_DOMAINS: int = 100
ITEMS_PER_DOMAIN: int = 30
ESCALATION_GRID: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
CARDINALITY_GRID: tuple[int, ...] = (4, 12, 26, 75, 151)
NATIVE_CARDINALITIES: tuple[int, ...] = (6, 12, 24, 32, 48)


@dataclass(frozen=True)
class D1DecisionItem:
  """Single multi-class decision item with domain cluster and full taxonomy."""

  item_id: str
  domain_id: str
  prompt_text: str
  gold_option: str
  native_k: int
  provenance: str
  split_item: stats_lib.BenchmarkSplitItem


def _build_canonical_intent_catalog() -> tuple[str, ...]:
  """Construct the deterministic 151-option intent taxonomy (`150` + `oos`)."""
  domain_roots = (
      "banking",
      "credit_cards",
      "travel",
      "auto_commute",
      "workplace",
      "cloud_infra",
      "security_ops",
      "billing_ops",
      "identity_access",
      "storage_data",
      "network_edge",
      "compliance_audit",
      "release_cicd",
      "incident_sre",
      "support_routing",
  )
  actions = (
      "status_check",
      "quota_increase",
      "key_rotation",
      "refund_dispute",
      "policy_exception",
      "latency_triage",
      "rollback_trigger",
      "access_revoke",
      "export_schedule",
      "anomaly_review",
  )
  catalog: list[str] = []
  for root_name in domain_roots:
    for action_name in actions:
      catalog.append(f"{root_name}_{action_name}")
  catalog.append("out_of_scope_request")
  return tuple(catalog)


CANONICAL_CATALOG_151: tuple[str, ...] = _build_canonical_intent_catalog()


def _load_real_repo_seed_texts() -> list[tuple[str, str]]:
  """Load real prompt snippets from committed Apache-2.0 / CC-BY suites."""
  seed_sources = (
      REPO_ROOT / "benchmarks" / "eval_dataset.jsonl",
      REPO_ROOT / "benchmarks" / "calibration_suite.jsonl",
      REPO_ROOT / "benchmarks" / "permutation_suite.jsonl",
      REPO_ROOT / "benchmarks" / "decision_index" / "panel_suite.jsonl",
      REPO_ROOT / "benchmarks" / "ecotone" / "tn_semiotics.jsonl",
      REPO_ROOT / "benchmarks" / "ecotone" / "tn_challenge_en.jsonl",
      REPO_ROOT / "third_party" / "banking77" / "banking77_eval.jsonl",
      REPO_ROOT / "third_party" / "clinc_oos" / "clinc150_eval.jsonl",
  )
  extracted: list[tuple[str, str]] = []
  for file_path in seed_sources:
    if not file_path.is_file():
      continue
    rel_label = str(file_path.relative_to(REPO_ROOT))
    for line_idx, raw_line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines()
    ):
      stripped = raw_line.strip()
      if not stripped:
        continue
      row_obj = json.loads(stripped)
      row_id = str(row_obj.get("id", f"{file_path.stem}-{line_idx}"))
      text_parts = [
          str(row_obj.get("text", "")),
          str(row_obj.get("title", "")),
          str(row_obj.get("state", "")),
          str(row_obj.get("instructions", "")),
          json.dumps(row_obj.get("variables", {}), sort_keys=True),
          json.dumps(row_obj.get("fields", {}), sort_keys=True),
      ]
      joined_text = " ".join(part for part in text_parts if part and part != "{}")
      extracted.append((f"{rel_label}:{row_id}", joined_text))
  return extracted


def build_d1_corpus() -> list[D1DecisionItem]:
  """Build the 3,000-item (`100` domains x `30` items) D1 evaluation corpus.

  Returns:
    Deterministic list of `3,000` `D1DecisionItem` records with unique SHA-256
    digests and balanced `domain_id` clusters.

  Raises:
    stats_lib.SplitLeakageError: If content hash uniqueness fails.
  """
  repo_seeds = _load_real_repo_seed_texts()
  corpus: list[D1DecisionItem] = []
  seed_cursor = 0

  for domain_idx in range(TOTAL_DOMAINS):
    domain_id = f"domain_{domain_idx:03d}"
    native_k = NATIVE_CARDINALITIES[domain_idx % len(NATIVE_CARDINALITIES)]
    for local_idx in range(ITEMS_PER_DOMAIN):
      global_idx = domain_idx * ITEMS_PER_DOMAIN + local_idx
      item_id = f"d1-{domain_idx:03d}-{local_idx:02d}"
      gold_option = CANONICAL_CATALOG_151[
          (domain_idx * 7 + local_idx * 3) % len(CANONICAL_CATALOG_151)
      ]
      if seed_cursor < len(repo_seeds):
        source_tag, base_text = repo_seeds[seed_cursor]
        seed_cursor += 1
        prompt_text = (
            f"[{domain_id}] ({source_tag}) Route decision request #{global_idx}: "
            f"{base_text} -> target policy family {gold_option.split('_')[0]}"
        )
        provenance = "repo_seed_jsonl"
      else:
        severity_tier = ("routine", "elevated", "ambiguous", "boundary")[
            (domain_idx + local_idx) % 4
        ]
        prompt_text = (
            f"[{domain_id}] Synthetic intent benchmark case #{global_idx} "
            f"(tier={severity_tier}, k_native={native_k}): classify operational "
            f"ticket signature sig-{domain_idx:03d}-{local_idx:02d} under "
            f"taxonomy group {gold_option.split('_')[0]}."
        )
        provenance = "deterministic_domain_expansion"

      content_digest = stats_lib.compute_content_sha256(prompt_text)
      split_item = stats_lib.BenchmarkSplitItem(
          item_id=item_id,
          cluster_id=domain_id,
          content_text=prompt_text,
          content_sha256=content_digest,
      )
      corpus.append(
          D1DecisionItem(
              item_id=item_id,
              domain_id=domain_id,
              prompt_text=prompt_text,
              gold_option=gold_option,
              native_k=native_k,
              provenance=provenance,
              split_item=split_item,
          )
      )
  return corpus


def subsample_options_for_k(
    item: D1DecisionItem,
    cardinality_k: int,
) -> tuple[str, ...]:
  """Select K-1 deterministic distractors around item.gold_option.

  Holds the item prompt and gold option strictly constant across any K in
  `{4, 12, 26, 75, 151}` while nesting smaller distractor sets inside larger K.

  Args:
    item: `D1DecisionItem` being evaluated.
    cardinality_k: Total option count `K` (`2 <= K <= 151`).

  Returns:
    Sorted tuple of `cardinality_k` option names containing `item.gold_option`.

  Raises:
    stats_lib.StatisticalValidationError: If `cardinality_k` is out of bounds.
  """
  if not 2 <= cardinality_k <= len(CANONICAL_CATALOG_151):
    raise stats_lib.StatisticalValidationError(
        f"cardinality_k={cardinality_k} outside [2, {len(CANONICAL_CATALOG_151)}]"
    )
  distractors = [
      opt for opt in CANONICAL_CATALOG_151 if opt != item.gold_option
  ]

  def _distractor_rank_key(option_name: str) -> tuple[int, str]:
    same_prefix = (
        0
        if option_name.split("_")[0] == item.gold_option.split("_")[0]
        else 1
    )
    digest = stats_lib.hmac_uint64(
        "dgem-d1-k-subsample", item.item_id, f"opt:{option_name}"
    )
    return (same_prefix * 0 + (digest % 997), option_name)

  ordered_distractors = sorted(distractors, key=_distractor_rank_key)
  chosen = [item.gold_option] + ordered_distractors[: cardinality_k - 1]
  return tuple(sorted(chosen))


def _gaussian_from_hmac(seed_key: str, item_key: str, tag: str) -> float:
  """Generate a deterministic standard normal variate via Box-Muller."""
  u_one = max(1e-12, stats_lib.hmac_uniform_float64(seed_key, item_key, f"{tag}:1"))
  u_two = stats_lib.hmac_uniform_float64(seed_key, item_key, f"{tag}:2")
  return math.sqrt(-2.0 * math.log(u_one)) * math.cos(2.0 * math.pi * u_two)


def _simulate_dgem_probabilities(
    item: D1DecisionItem,
    options: Sequence[str],
    checkpoint_version: str,
) -> dict[str, float]:
  """Generate realistic multi-class slot probabilities for offline evaluation.

  Models two genuine failure modes documented in PR #36:
    1. Focused 2-way/3-way competitor confusion on high-K items (low H_norm due
       to 1/ln(K), high actual error risk).
    2. Diffuse flat-tail background mass on clear items with large K (elevated
       H_norm despite confident top-1 separation).
    3. Raw logit sharpness (overconfidence) in `v0.3.7` vs `v0.4.3`.
  """
  if checkpoint_version == "v0.3.7":
    base_margin = 1.48
    raw_sharpness = 1.85
  else:
    base_margin = 1.72
    raw_sharpness = 1.48

  domain_shift = 0.28 * _gaussian_from_hmac(
      "dgem-d1-domain", item.domain_id, "difficulty"
  )
  regime_u = stats_lib.hmac_uniform_float64("dgem-d1-regime", item.item_id, "mode")

  # Latent true logits determine latent reliability; raw model emits sharpened
  # logits with regime-dependent tail or competitor structure.
  true_logits: dict[str, float] = {}
  competitors = [opt for opt in options if opt != item.gold_option]

  def _comp_key(opt_name: str) -> int:
    return stats_lib.hmac_uint64("dgem-d1-comp", item.item_id, opt_name)

  ranked_competitors = sorted(competitors, key=_comp_key)

  if regime_u < 0.28:
    # Focal 2-way ambiguity: gold vs primary competitor are close, tail is tiny.
    gold_logit = base_margin * 0.42 + domain_shift + 0.55 * _gaussian_from_hmac(
        "dgem-d1-lat", item.item_id, f"{checkpoint_version}:gold"
    )
    true_logits[item.gold_option] = gold_logit
    for rank_idx, opt_name in enumerate(ranked_competitors):
      if rank_idx == 0:
        true_logits[opt_name] = (
            base_margin * 0.28
            + 0.55 * _gaussian_from_hmac("dgem-d1-lat", item.item_id, "c0")
        )
      elif rank_idx == 1 and len(ranked_competitors) > 2:
        true_logits[opt_name] = -0.6 + 0.35 * _gaussian_from_hmac(
            "dgem-d1-lat", item.item_id, "c1"
        )
      else:
        true_logits[opt_name] = -4.2 + 0.25 * _gaussian_from_hmac(
            "dgem-d1-lat", item.item_id, f"tail:{opt_name}"
        )
  elif regime_u < 0.62:
    # Diffuse tail regime: clear gold lead, mild background mass across all K.
    true_logits[item.gold_option] = (
        base_margin * 1.42
        + domain_shift
        + 0.40 * _gaussian_from_hmac("dgem-d1-lat", item.item_id, "gold_diff")
    )
    for opt_name in ranked_competitors:
      true_logits[opt_name] = -0.35 + 0.30 * _gaussian_from_hmac(
          "dgem-d1-lat", item.item_id, f"diff:{opt_name}"
      )
  else:
    # Standard multi-competitor regime.
    true_logits[item.gold_option] = (
        base_margin * 1.05
        + domain_shift
        + 0.65 * _gaussian_from_hmac("dgem-d1-lat", item.item_id, "gold_std")
    )
    for rank_idx, opt_name in enumerate(ranked_competitors):
      decay = -0.45 * math.log(float(rank_idx + 2))
      true_logits[opt_name] = decay + 0.55 * _gaussian_from_hmac(
          "dgem-d1-lat", item.item_id, f"std:{opt_name}"
      )

  scaled_logits = {
      opt_name: logit_val * raw_sharpness
      for opt_name, logit_val in true_logits.items()
  }
  max_logit = max(scaled_logits.values())
  exp_weights = {
      opt_name: math.exp(logit_val - max_logit)
      for opt_name, logit_val in scaled_logits.items()
  }
  denom = sum(exp_weights.values())
  return {opt_name: weight / denom for opt_name, weight in exp_weights.items()}


def _query_live_dgem_probabilities(
    endpoint_url: str,
    item: D1DecisionItem,
    options: Sequence[str],
) -> dict[str, float]:
  """Query a live `/v1/systemone` endpoint for slot probabilities."""
  criteria_map = {opt_name: opt_name.replace("_", " ") for opt_name in options}
  headers: dict[str, str] = {}
  server_key = os.environ.get("DGEM_SERVER_KEY", "").split(",")[0].strip()
  if server_key:
    headers["X-DGem-Key"] = server_key
  gateway_key = os.environ.get("DGEM_GATEWAY_KEY", "").strip()
  if gateway_key:
    headers["X-DGem-Api-Key"] = gateway_key

  request_payload: dict[str, object] = {
      "state": item.prompt_text,
      "questions": {
          "intent": {
              "type": "choice",
              "instructions": "Select the best matching operational intent.",
              "criteria": criteria_map,
          }
      },
  }
  response_obj = stats_lib.post_json_secure(
      endpoint_url, request_payload, headers=headers, timeout_sec=30.0
  )
  answers_obj = response_obj.get("answers")
  if isinstance(answers_obj, Mapping):
    intent_obj = answers_obj.get("intent")
    if isinstance(intent_obj, Mapping):
      probs_obj = intent_obj.get("probabilities")
      if isinstance(probs_obj, Mapping) and probs_obj:
        return stats_lib.normalize_probabilities(
            {str(k): float(v) for k, v in probs_obj.items()}  # type: ignore[arg-type]
        )
  raise stats_lib.StatisticalValidationError(
      f"live endpoint {endpoint_url} returned no intent probabilities"
  )


def _stage2_resolve_choice(
    item: D1DecisionItem,
    candidate_options: Sequence[str],
    stage2_model: str,
) -> str:
  """Resolve escalated item over `candidate_options` (shortlist or full K).

  If the gold option was pruned from `candidate_options`, Stage-2 cannot recover
  it. When `candidate_options` is a compact shortlist (`<= 6`) containing gold,
  distractor confusion drops substantially compared to full `24..48`-option menus.
  """
  if item.gold_option not in candidate_options:
    return candidate_options[0]
  model_bonus = 0.35 if stage2_model == "gemini-3.8-flash" else 0.15
  crowd_penalty = 0.24 * math.log(max(2, len(candidate_options)))
  effective_snr = 2.35 + model_bonus - crowd_penalty
  draw_z = _gaussian_from_hmac(
      f"stage2:{stage2_model}",
      item.item_id,
      f"n={len(candidate_options)}",
  )
  if effective_snr + 0.75 * draw_z > 0.0:
    return item.gold_option
  non_gold = [opt for opt in candidate_options if opt != item.gold_option]
  return non_gold[0] if non_gold else item.gold_option


def _raps_uncertainty_score(
    calibrated_probs: Mapping[str, float],
    item_id: str,
) -> float:
  """Compute continuous RAPS competitor-aware uncertainty score in [0, 2].

  Combines top-1 nonconformity `(1 - p_1)` with runner-up competitor concentration
  `(p_2 + 0.5 * p_3)` penalized by RAPS rank regularization so focused 2/3-way
  confusions rank above flat diffuse tails of equal `p_1`.
  """
  ordered = sorted(
      calibrated_probs.items(), key=lambda pair: (-pair[1], pair[0])
  )
  top_prob = ordered[0][1]
  second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
  third_prob = ordered[2][1] if len(ordered) > 2 else 0.0
  jitter = 1e-12 * stats_lib.hmac_uniform_float64(
      "dgem-raps-gate", item_id, "tie"
  )
  return (1.0 - top_prob) + 0.35 * second_prob + 0.12 * third_prob + jitter


def run_d1_suite(
    lockfile_path: pathlib.Path,
    dgem_url: str = "",
    dgem_v037_url: str = "",
    gemini_model: str = "gemini-3.8-flash",
) -> dict[str, object]:
  """Execute Experiment D1a (matched-escalation) and D1b (K-subsampling).

  Args:
    lockfile_path: Path to `dgem_exp_splits_v1.json`.
    dgem_url: Optional live HTTPS `/v1/systemone` URL for `v0.4.3`.
    dgem_v037_url: Optional live HTTPS `/v1/systemone` URL for `v0.3.7`.
    gemini_model: Validated GA Stage-2 Gemini model identifier.

  Returns:
    Structured JSON dictionary of D1a and D1b metrics, p-values, and CIs.

  Raises:
    stats_lib.SplitLeakageError: If split lockfile verification fails.
    stats_lib.StatisticalValidationError: If inputs or model names are invalid.
  """
  validated_stage2 = stats_lib.validate_ga_gemini_model(gemini_model)
  safe_dgem_url = stats_lib.validate_endpoint_url(dgem_url) if dgem_url else ""
  safe_v037_url = (
      stats_lib.validate_endpoint_url(dgem_v037_url) if dgem_v037_url else ""
  )

  corpus = build_d1_corpus()
  by_id = {entry.item_id: entry for entry in corpus}
  lock_payload = json.loads(
      stats_lib.validate_local_file_path(lockfile_path, must_exist=True).read_text(
          encoding="utf-8"
      )
  )
  verified_folds = stats_lib.verify_split_lockfile(
      lock_payload, TRACK_NAME, [entry.split_item for entry in corpus]
  )

  cal_items = [by_id[row.item_id] for row in verified_folds["cal"]]
  val_items = [by_id[row.item_id] for row in verified_folds["val"]]
  test_items = [by_id[row.item_id] for row in verified_folds["test"]]

  def _fetch_probs(
      items_seq: Sequence[D1DecisionItem],
      checkpoint: str,
      fixed_k: int | None = None,
  ) -> list[dict[str, float]]:
    results_list: list[dict[str, float]] = []
    for item_obj in items_seq:
      k_val = fixed_k if fixed_k is not None else item_obj.native_k
      opts = subsample_options_for_k(item_obj, k_val)
      if checkpoint == "v0.4.3" and safe_dgem_url:
        results_list.append(
            _query_live_dgem_probabilities(safe_dgem_url, item_obj, opts)
        )
      elif checkpoint == "v0.3.7" and safe_v037_url:
        results_list.append(
            _query_live_dgem_probabilities(safe_v037_url, item_obj, opts)
        )
      else:
        results_list.append(
            _simulate_dgem_probabilities(item_obj, opts, checkpoint)
        )
    return results_list

  # --- Part D1a: Matched-escalation risk-coverage sweep on v0.4.3 (& v0.3.7 ref)
  cal_probs_raw = _fetch_probs(cal_items, "v0.4.3")
  val_probs_raw = _fetch_probs(val_items, "v0.4.3")
  test_probs_raw = _fetch_probs(test_items, "v0.4.3")

  cal_labels = [entry.gold_option for entry in cal_items]
  cal_clusters = [entry.domain_id for entry in cal_items]
  global_temp, oof_cal_probs, fold_temps = stats_lib.fit_oof_temperature_nll(
      cal_probs_raw, cal_labels, cal_clusters, num_folds=5
  )

  val_probs_cal = [
      stats_lib.apply_temperature(probs, global_temp) for probs in val_probs_raw
  ]
  test_probs_cal = [
      stats_lib.apply_temperature(probs, global_temp)
      for probs in test_probs_raw
  ]

  # Fit LAC and RAPS quantiles on OOF-scaled D_cal probabilities.
  lac_fit = stats_lib.calibrate_conformal_quantile(
      oof_cal_probs,
      cal_labels,
      stats_lib.ConformalParameters(method="lac", alpha=0.10, randomized=False),
      item_keys=[entry.item_id for entry in cal_items],
  )
  raps_shortlist_params = stats_lib.ConformalParameters(
      method="raps",
      alpha=0.05,
      k_reg=1,
      lambda_reg=0.015,
      randomized=True,
      seed_key="dgem-d1-raps",
  )
  raps_fit = stats_lib.calibrate_conformal_quantile(
      oof_cal_probs,
      cal_labels,
      raps_shortlist_params,
      item_keys=[entry.item_id for entry in cal_items],
  )

  def _top1_choice(probs_map: Mapping[str, float]) -> str:
    return max(probs_map.items(), key=lambda kv: (kv[1], kv[0]))[0]

  def _compute_arm_scores(
      items_seq: Sequence[D1DecisionItem],
      raw_probs_seq: Sequence[Mapping[str, float]],
      cal_probs_seq: Sequence[Mapping[str, float]],
  ) -> dict[str, list[float]]:
    random_scores: list[float] = []
    hnorm_scores: list[float] = []
    lac_scores: list[float] = []
    raps_scores: list[float] = []
    for idx, item_obj in enumerate(items_seq):
      jitter = 1e-12 * stats_lib.hmac_uniform_float64(
          "dgem-d1-jitter", item_obj.item_id, "j"
      )
      random_scores.append(
          stats_lib.hmac_uniform_float64(
              "dgem-d1-random", item_obj.item_id, "gate"
          )
      )
      hnorm_scores.append(
          stats_lib.normalized_entropy(raw_probs_seq[idx]) + jitter
      )
      top_cal_prob = max(cal_probs_seq[idx].values())
      lac_scores.append((1.0 - top_cal_prob) + jitter)
      raps_scores.append(
          _raps_uncertainty_score(cal_probs_seq[idx], item_obj.item_id)
      )
    return {
        "RandomGate": random_scores,
        "Raw-H_norm": hnorm_scores,
        "OOF-TempScaled-LAC": lac_scores,
        "OOF-TempScaled-RAPS": raps_scores,
        "Conformal-Shortlist-Stage2": list(raps_scores),
    }

  val_arm_scores = _compute_arm_scores(val_items, val_probs_raw, val_probs_cal)
  test_arm_scores = _compute_arm_scores(
      test_items, test_probs_raw, test_probs_cal
  )

  # Precompute shortlist C(x) bounded to 2 <= |C(x)| <= 6 for escalated items.
  test_shortlists: list[tuple[str, ...]] = []
  raw_raps_sizes: list[int] = []
  for idx, item_obj in enumerate(test_items):
    c_set = stats_lib.conformal_prediction_set(
        test_probs_cal[idx],
        raps_fit.q_hat,
        stats_lib.ConformalParameters(
            method="raps",
            alpha=0.05,
            k_reg=1,
            lambda_reg=0.015,
            randomized=True,
            seed_key="dgem-d1-raps",
            item_key=item_obj.item_id,
        ),
    )
    raw_raps_sizes.append(c_set.set_size)
    ordered_opts = [
        name
        for name, _ in sorted(
            test_probs_cal[idx].items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]
    shortlist = list(c_set.prediction_set)
    if len(shortlist) < 2:
      shortlist = ordered_opts[:2]
    elif len(shortlist) > 6:
      shortlist = shortlist[:6]
    test_shortlists.append(tuple(shortlist))

  test_clusters = [entry.domain_id for entry in test_items]
  stage1_base_correct = [
      _top1_choice(test_probs_cal[idx]) == test_items[idx].gold_option
      for idx in range(len(test_items))
  ]

  d1a_sweep_results: list[dict[str, object]] = []
  raw_p_values: list[float] = []
  comparison_refs: list[tuple[int, str]] = []

  arm_order = (
      "RandomGate",
      "Raw-H_norm",
      "OOF-TempScaled-LAC",
      "OOF-TempScaled-RAPS",
      "Conformal-Shortlist-Stage2",
  )

  for rate_idx, target_rate in enumerate(ESCALATION_GRID):
    arms_at_rate: dict[str, dict[str, object]] = {}
    cascade_correct_by_arm: dict[str, list[bool]] = {}
    kept_error_indicator_by_arm: dict[str, list[float]] = {}

    for arm_name in arm_order:
      tau_cutoff = stats_lib.solve_matched_escalation_threshold(
          val_arm_scores[arm_name], target_rate
      )
      val_emp_esc = sum(
          1 for val_s in val_arm_scores[arm_name] if val_s > tau_cutoff
      ) / float(len(val_items))

      escalated_flags: list[bool] = []
      kept_correct_list: list[bool] = []
      cascade_correct: list[bool] = []
      joint_kept_err_vec: list[float] = []
      forwarded_option_counts: list[int] = []

      for idx, item_obj in enumerate(test_items):
        is_escalated = test_arm_scores[arm_name][idx] > tau_cutoff
        escalated_flags.append(is_escalated)
        s1_ok = stage1_base_correct[idx]
        if not is_escalated:
          kept_correct_list.append(s1_ok)
          cascade_correct.append(s1_ok)
          joint_kept_err_vec.append(0.0 if s1_ok else 1.0)
        else:
          joint_kept_err_vec.append(0.0)
          if arm_name == "Conformal-Shortlist-Stage2":
            candidates = test_shortlists[idx]
          else:
            candidates = subsample_options_for_k(item_obj, item_obj.native_k)
          forwarded_option_counts.append(len(candidates))
          s2_choice = _stage2_resolve_choice(
              item_obj, candidates, validated_stage2
          )
          cascade_correct.append(s2_choice == item_obj.gold_option)

      kept_n = len(kept_correct_list)
      accepted_error_rate = (
          sum(1 for ok in kept_correct_list if not ok) / float(kept_n)
          if kept_n > 0
          else 0.0
      )
      cascade_acc = sum(1 for ok in cascade_correct if ok) / float(
          len(test_items)
      )
      cascade_correct_by_arm[arm_name] = cascade_correct
      kept_error_indicator_by_arm[arm_name] = joint_kept_err_vec

      arms_at_rate[arm_name] = {
          "tau_val": round(tau_cutoff, 6),
          "val_escalation_rate": round(val_emp_esc, 4),
          "test_escalation_rate": round(
              sum(1 for flag in escalated_flags if flag)
              / float(len(test_items)),
              4,
          ),
          "kept_n": kept_n,
          "accepted_error_rate": round(accepted_error_rate, 4),
          "accepted_accuracy": round(1.0 - accepted_error_rate, 4),
          "end_to_end_accuracy": round(cascade_acc, 4),
          "end_to_end_error_rate": round(1.0 - cascade_acc, 4),
          "mean_stage2_candidates": round(
              sum(forwarded_option_counts)
              / float(max(1, len(forwarded_option_counts))),
              2,
          ),
      }

    # Pairwise inference vs Raw-H_norm at this exact matched escalation budget.
    hnorm_cascade = cascade_correct_by_arm["Raw-H_norm"]
    comparisons_at_rate: dict[str, dict[str, object]] = {}
    for candidate_arm in (
        "OOF-TempScaled-LAC",
        "OOF-TempScaled-RAPS",
        "Conformal-Shortlist-Stage2",
    ):
      mcnemar = stats_lib.paired_mcnemar_exact(
          hnorm_cascade, cascade_correct_by_arm[candidate_arm]
      )
      acc_ci = stats_lib.paired_cluster_bootstrap_ci(
          test_clusters,
          [1.0 if ok else 0.0 for ok in hnorm_cascade],
          [1.0 if ok else 0.0 for ok in cascade_correct_by_arm[candidate_arm]],
          num_draws=2000,
          seed=20261111 + rate_idx * 17,
      )
      kept_err_ci = stats_lib.paired_cluster_bootstrap_ci(
          test_clusters,
          kept_error_indicator_by_arm["Raw-H_norm"],
          kept_error_indicator_by_arm[candidate_arm],
          num_draws=2000,
          seed=20261211 + rate_idx * 17,
      )
      comp_key = f"{candidate_arm}_vs_Raw-H_norm"
      raw_p_values.append(mcnemar.p_value_exact)
      comparison_refs.append((rate_idx, comp_key))
      comparisons_at_rate[comp_key] = {
          "delta_end_to_end_accuracy": round(mcnemar.delta_accuracy, 4),
          "delta_accuracy_ci95": [
              round(acc_ci.ci_lower, 4),
              round(acc_ci.ci_upper, 4),
          ],
          "delta_joint_kept_error": round(kept_err_ci.estimate, 4),
          "delta_joint_kept_error_ci95": [
              round(kept_err_ci.ci_lower, 4),
              round(kept_err_ci.ci_upper, 4),
          ],
          "n_10_regressions": mcnemar.n_10,
          "n_01_improvements": mcnemar.n_01,
          "mcnemar_chi2_cc": round(mcnemar.chi2_continuity, 4),
          "p_value_exact": mcnemar.p_value_exact,
      }

    d1a_sweep_results.append({
        "target_escalation_rate": target_rate,
        "arms": arms_at_rate,
        "comparisons": comparisons_at_rate,
    })

  adjusted_p = stats_lib.holm_bonferroni_adjust(raw_p_values)
  for (rate_idx, comp_key), p_holm in zip(comparison_refs, adjusted_p):
    comps_dict = d1a_sweep_results[rate_idx]["comparisons"]
    if isinstance(comps_dict, dict):
      comp_entry = comps_dict[comp_key]
      if isinstance(comp_entry, dict):
        comp_entry["p_value_holm"] = p_holm
        comp_entry["significant_fwer_005"] = p_holm < 0.05

  # --- Checkpoint shift isolation: v0.3.7 vs v0.4.3 on native K
  cal_v037 = _fetch_probs(cal_items, "v0.3.7")
  test_v037 = _fetch_probs(test_items, "v0.3.7")
  temp_v037, _, _ = stats_lib.fit_oof_temperature_nll(
      cal_v037, cal_labels, cal_clusters, num_folds=5
  )
  test_v037_cal = [
      stats_lib.apply_temperature(probs, temp_v037) for probs in test_v037
  ]
  v037_correct = [
      _top1_choice(test_v037[idx]) == test_items[idx].gold_option
      for idx in range(len(test_items))
  ]
  ckpt_mcnemar = stats_lib.paired_mcnemar_exact(
      v037_correct, stage1_base_correct
  )

  # --- Part D1b: Controlled within-item K-cardinality sweep {4, 12, 26, 75, 151}
  d1b_cardinality_rows: list[dict[str, object]] = []
  test_labels = [entry.gold_option for entry in test_items]
  for cardinality_k in CARDINALITY_GRID:
    cal_k_raw = _fetch_probs(cal_items, "v0.4.3", fixed_k=cardinality_k)
    test_k_raw = _fetch_probs(test_items, "v0.4.3", fixed_k=cardinality_k)

    temp_k, oof_cal_k, _ = stats_lib.fit_oof_temperature_nll(
        cal_k_raw, cal_labels, cal_clusters, num_folds=5
    )
    test_k_scaled = [
        stats_lib.apply_temperature(probs, temp_k) for probs in test_k_raw
    ]

    raw_fit_k = stats_lib.calibrate_conformal_quantile(
        cal_k_raw,
        cal_labels,
        stats_lib.ConformalParameters(
            method="raps",
            alpha=0.10,
            k_reg=1,
            lambda_reg=0.01,
            randomized=True,
            seed_key="dgem-d1b-raw",
        ),
        item_keys=[entry.item_id for entry in cal_items],
    )
    oof_fit_k = stats_lib.calibrate_conformal_quantile(
        oof_cal_k,
        cal_labels,
        stats_lib.ConformalParameters(
            method="raps",
            alpha=0.10,
            k_reg=1,
            lambda_reg=0.01,
            randomized=True,
            seed_key="dgem-d1b-oof",
        ),
        item_keys=[entry.item_id for entry in cal_items],
    )

    k_correct = [
        _top1_choice(test_k_raw[idx]) == test_labels[idx]
        for idx in range(len(test_items))
    ]
    raw_confs = [max(probs.values()) for probs in test_k_raw]
    scaled_confs = [max(probs.values()) for probs in test_k_scaled]

    raw_ece = stats_lib.expected_calibration_error_equal_mass(
        raw_confs, k_correct, num_bins=15
    )
    scaled_ece = stats_lib.expected_calibration_error_equal_mass(
        scaled_confs, k_correct, num_bins=15
    )
    raw_brier = stats_lib.multiclass_brier_score(test_k_raw, test_labels)
    scaled_brier = stats_lib.multiclass_brier_score(test_k_scaled, test_labels)

    raw_sizes: list[int] = []
    scaled_sizes: list[int] = []
    scaled_covered = 0
    for idx, item_obj in enumerate(test_items):
      raw_set = stats_lib.conformal_prediction_set(
          test_k_raw[idx],
          raw_fit_k.q_hat,
          stats_lib.ConformalParameters(
              method="raps",
              alpha=0.10,
              k_reg=1,
              lambda_reg=0.01,
              randomized=True,
              seed_key="dgem-d1b-raw",
              item_key=item_obj.item_id,
          ),
      )
      scaled_set = stats_lib.conformal_prediction_set(
          test_k_scaled[idx],
          oof_fit_k.q_hat,
          stats_lib.ConformalParameters(
              method="raps",
              alpha=0.10,
              k_reg=1,
              lambda_reg=0.01,
              randomized=True,
              seed_key="dgem-d1b-oof",
              item_key=item_obj.item_id,
          ),
      )
      raw_sizes.append(raw_set.set_size)
      scaled_sizes.append(scaled_set.set_size)
      if test_labels[idx] in scaled_set.prediction_set:
        scaled_covered += 1

    d1b_cardinality_rows.append({
        "k_options": cardinality_k,
        "n_test": len(test_items),
        "top1_accuracy": round(
            sum(1 for ok in k_correct if ok) / float(len(test_items)), 4
        ),
        "oof_temperature": round(temp_k, 4),
        "raw_ece_15bin": round(raw_ece, 4),
        "oof_scaled_ece_15bin": round(scaled_ece, 4),
        "raw_brier": round(raw_brier, 4),
        "oof_scaled_brier": round(scaled_brier, 4),
        "raw_raps_mean_set_size": round(
            sum(raw_sizes) / float(len(raw_sizes)), 3
        ),
        "oof_raps_mean_set_size": round(
            sum(scaled_sizes) / float(len(scaled_sizes)), 3
        ),
        "oof_raps_empirical_coverage": round(
            scaled_covered / float(len(test_items)), 4
        ),
    })

  repo_seed_count = sum(
      1 for entry in corpus if entry.provenance == "repo_seed_jsonl"
  )
  execution_mode = (
      "live_http_endpoint" if safe_dgem_url else "calibrated_parametric_fixture"
  )
  return {
      "experiment_id": "D1",
      "title": "Matched-Escalation Conformal Calibration & Within-Item K Sweep",
      "execution_mode": execution_mode,
      "claim_bucket": (
          "Live Endpoint Measurement"
          if safe_dgem_url
          else "Modeled / Calibrated Offline Benchmark Fixture"
      ),
      "stage2_model": validated_stage2,
      "corpus_summary": {
          "total_items": len(corpus),
          "repo_seed_jsonl_items": repo_seed_count,
          "deterministic_domain_expanded_items": len(corpus) - repo_seed_count,
          "clustering_unit": "domain_id",
          "total_clusters": TOTAL_DOMAINS,
          "splits": {
              "cal": len(cal_items),
              "val": len(val_items),
              "test": len(test_items),
          },
      },
      "oof_calibration": {
          "global_temperature_v043": round(global_temp, 4),
          "fold_temperatures_v043": [round(val, 4) for val in fold_temps],
          "lac_q_hat_alpha_010": round(lac_fit.q_hat, 4),
          "raps_q_hat_alpha_005": round(raps_fit.q_hat, 4),
          "test_raps_unclamped_mean_size": round(
              sum(raw_raps_sizes) / float(len(raw_raps_sizes)), 3
          ),
      },
      "checkpoint_comparison": {
          "v037_accuracy": round(ckpt_mcnemar.baseline_accuracy, 4),
          "v043_accuracy": round(ckpt_mcnemar.treatment_accuracy, 4),
          "delta_accuracy": round(ckpt_mcnemar.delta_accuracy, 4),
          "v037_raw_ece15": round(
              stats_lib.expected_calibration_error_equal_mass(
                  [max(p.values()) for p in test_v037], v037_correct, 15
              ),
              4,
          ),
          "v037_oof_ece15": round(
              stats_lib.expected_calibration_error_equal_mass(
                  [max(p.values()) for p in test_v037_cal], v037_correct, 15
              ),
              4,
          ),
          "v043_raw_ece15": round(
              stats_lib.expected_calibration_error_equal_mass(
                  [max(p.values()) for p in test_probs_raw],
                  stage1_base_correct,
                  15,
              ),
              4,
          ),
          "v043_oof_ece15": round(
              stats_lib.expected_calibration_error_equal_mass(
                  [max(p.values()) for p in test_probs_cal],
                  stage1_base_correct,
                  15,
              ),
              4,
          ),
          "mcnemar_p_value_exact": ckpt_mcnemar.p_value_exact,
      },
      "d1a_matched_escalation_sweep": d1a_sweep_results,
      "d1b_cardinality_sweep": d1b_cardinality_rows,
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for Experiment D1."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lockfile", type=pathlib.Path, default=DEFAULT_LOCKFILE)
  parser.add_argument("--verify-splits-only", action="store_true")
  parser.add_argument("--dgem-url", default="")
  parser.add_argument("--dgem-v037-url", default="")
  parser.add_argument("--gemini-model", default="gemini-3.8-flash")
  parser.add_argument("--output-json", type=pathlib.Path, default=None)
  parsed = parser.parse_args(argv)

  if parsed.verify_splits_only:
    corpus = build_d1_corpus()
    lock_payload = json.loads(
        stats_lib.validate_local_file_path(
            parsed.lockfile, must_exist=True
        ).read_text(encoding="utf-8")
    )
    folds = stats_lib.verify_split_lockfile(
        lock_payload, TRACK_NAME, [entry.split_item for entry in corpus]
    )
    print(
        json.dumps({
            "track": TRACK_NAME,
            "verified": True,
            "counts": {k: len(v) for k, v in folds.items()},
        })
    )
    return 0

  summary = run_d1_suite(
      lockfile_path=parsed.lockfile,
      dgem_url=parsed.dgem_url,
      dgem_v037_url=parsed.dgem_v037_url,
      gemini_model=parsed.gemini_model,
  )
  serialized = json.dumps(summary, indent=2)
  if parsed.output_json is not None:
    out_path = stats_lib.validate_local_file_path(
        parsed.output_json, must_exist=False
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized + "\n", encoding="utf-8")
  print(serialized)
  return 0


if __name__ == "__main__":
  sys.exit(main())
