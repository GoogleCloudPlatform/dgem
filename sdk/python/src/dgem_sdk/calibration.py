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

"""Post-hoc temperature scaling, cosine-prior fusion, and conformal sets."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import hmac
import math
import struct

from dgem_sdk.types import ConformalCalibrationResult
from dgem_sdk.types import ConformalConfig
from dgem_sdk.types import ConformalSetResult
from dgem_sdk.types import DGemValidationError
from dgem_sdk.types import QuestionAnswer

DEFAULT_COSINE_PRIOR_TEMPERATURE = 0.07
DEFAULT_CONFORMAL_SEED_KEY = "dgem-conformal-v1"
CONFORMAL_SCORE_TOLERANCE = 1e-12
MAX_TEMPERATURE = 20.0

DEFAULT_ABSTAIN_LABELS: tuple[str, ...] = (
    "0",
    "abstain",
    "insufficient_evidence",
    "none",
    "none_of_the_above",
    "none_visible",
    "oos",
    "out_of_scope",
    "uncertain",
    "unknown",
)


def _normalize_probabilities(
    probabilities: Mapping[str, float],
) -> list[tuple[str, float]]:
  """Validate, normalize, and sort (option, prob) descending by prob then key."""
  if not probabilities:
    raise DGemValidationError("probability map must be non-empty")
  sorted_keys = sorted(str(key) for key in probabilities.keys())
  total_mass = 0.0
  raw_items: list[tuple[str, float]] = []
  for key in sorted_keys:
    value = probabilities[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise DGemValidationError(f"probability for {key!r} must be numeric")
    prob = float(value)
    if not math.isfinite(prob) or prob < 0.0:
      raise DGemValidationError(
          f"probability for {key!r} must be finite and >= 0 (got {prob})"
      )
    raw_items.append((key, prob))
    total_mass += prob
  if not math.isfinite(total_mass) or total_mass <= 0.0:
    raise DGemValidationError("probability map total mass must be > 0")
  normalized = [(key, prob / total_mass) for key, prob in raw_items]
  normalized.sort(key=lambda item: (-item[1], item[0]))
  return normalized


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
  """Compute Shannon entropy H(p) in nats from an option probability mapping.

  Args:
    probabilities: Mapping of option names to non-negative probability weights.

  Returns:
    Shannon entropy in nats (`>= 0.0`).

  Raises:
    DGemValidationError: If `probabilities` is empty, negative, or non-finite.
  """
  ordered = _normalize_probabilities(probabilities)
  entropy_nats = 0.0
  for _, prob in ordered:
    if prob > 0.0:
      entropy_nats -= prob * math.log(prob)
  return max(0.0, entropy_nats)


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
  """Compute cardinality-normalized hesitation H(p) / ln(K) in [0.0, 1.0].

  Args:
    probabilities: Mapping of option names to non-negative probability weights.

  Returns:
    Dimensionless hesitation score in `[0.0, 1.0]`.

  Raises:
    DGemValidationError: If `probabilities` is empty, negative, or non-finite.
  """
  ordered = _normalize_probabilities(probabilities)
  if len(ordered) <= 1:
    return 0.0
  entropy_nats = 0.0
  for _, prob in ordered:
    if prob > 0.0:
      entropy_nats -= prob * math.log(prob)
  return max(0.0, min(1.0, entropy_nats / math.log(len(ordered))))


def apply_temperature(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
  """Scale option probabilities by post-hoc temperature p_k^(1/T).

  Args:
    probabilities: Mapping of option names to probability mass.
    temperature: Post-hoc temperature strictly inside `(0.0, 20.0]`.

  Returns:
    Normalized temperature-scaled probability dictionary.

  Raises:
    DGemValidationError: If `temperature` is outside `(0.0, 20.0]` or probs invalid.
  """
  if (
      isinstance(temperature, bool)
      or not isinstance(temperature, (int, float))
      or not math.isfinite(float(temperature))
      or float(temperature) <= 0.0
      or float(temperature) > MAX_TEMPERATURE
  ):
    raise DGemValidationError(
        f"temperature must lie in (0, {MAX_TEMPERATURE}] (got {temperature!r})"
    )
  temp = float(temperature)
  ordered = _normalize_probabilities(probabilities)
  if abs(temp - 1.0) <= 1e-12:
    return dict(ordered)

  log_scaled: list[tuple[str, float]] = []
  max_logit = -math.inf
  for name, prob in ordered:
    logit = -1e9 if prob <= 0.0 else math.log(prob) / temp
    log_scaled.append((name, logit))
    if logit > max_logit:
      max_logit = logit

  exp_items = [(name, math.exp(logit - max_logit)) for name, logit in log_scaled]
  normalizer = sum(val for _, val in exp_items)
  return {name: val / normalizer for name, val in exp_items}


def apply_answer_temperature(
    answer: QuestionAnswer,
    temperature: float,
) -> QuestionAnswer:
  """Return a copy of `QuestionAnswer` calibrated with post-hoc temperature `T`.

  Args:
    answer: Original question answer returned by `dgem`.
    temperature: Fitted per-policy temperature in `(0.0, 20.0]`.

  Returns:
    Updated `QuestionAnswer` with rescaled probabilities, confidence, and entropy.

  Raises:
    DGemValidationError: If `temperature` is invalid.
  """
  if not answer.probabilities:
    return answer
  scaled_probs = apply_temperature(answer.probabilities, temperature)
  new_entropy = shannon_entropy(scaled_probs)
  winner_key = max(scaled_probs, key=lambda key: (scaled_probs[key], key))
  new_confidence = scaled_probs.get(answer.choice or answer.label, scaled_probs[winner_key])
  new_noul = answer.noul
  if answer.display_type == "boolean":
    for yes_key in ("yes", "true", "1"):
      if yes_key in scaled_probs:
        new_noul = scaled_probs[yes_key]
        break
  new_score = answer.score
  if answer.display_type == "score":
    expected_score = 0.0
    numeric_keys = True
    for key, prob in scaled_probs.items():
      try:
        expected_score += float(key) * prob
      except ValueError:
        numeric_keys = False
        break
    if numeric_keys:
      new_score = expected_score
  return replace(
      answer,
      probabilities=scaled_probs,
      confidence=new_confidence,
      entropy=new_entropy,
      noul=new_noul,
      score=new_score,
  )


def normalize_cosine_prior_logits(
    cosine_scores: Mapping[str, float],
    temperature: float = DEFAULT_COSINE_PRIOR_TEMPERATURE,
) -> dict[str, float]:
  """Convert embedding cosine similarities into log-softmax `prior_logits`.

  Matches `decisionindex.NormalizeCosinePriorLogits` in Go: non-finite values
  are skipped and `temperature <= 0` defaults to `0.07`.

  Args:
    cosine_scores: Mapping of option names to cosine similarities in `[-1, 1]`.
    temperature: Bi-encoder InfoNCE temperature (`0.07` when `<= 0` or non-finite).

  Returns:
    Log-probability dictionary suitable for `SystemOneQuestion.prior_logits`.
  """
  if not cosine_scores:
    return {}
  tau = float(temperature)
  if not math.isfinite(tau) or tau <= 0.0:
    tau = DEFAULT_COSINE_PRIOR_TEMPERATURE

  scaled: dict[str, float] = {}
  max_scaled = -math.inf
  for key, raw_score in cosine_scores.items():
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
      continue
    score = float(raw_score)
    if not math.isfinite(score):
      continue
    val = score / tau
    scaled[str(key)] = val
    if val > max_scaled:
      max_scaled = val
  if not scaled:
    return {}
  log_partition = max_scaled + math.log(
      sum(math.exp(val - max_scaled) for val in scaled.values())
  )
  return {key: val - log_partition for key, val in scaled.items()}


def fuse_prior_probabilities(
    probabilities: Mapping[str, float],
    prior_logits: Mapping[str, float],
    prior_weight: float = 0.2,
) -> dict[str, float]:
  """Fuse diffusion probabilities with external log-prior logits in log space.

  Computes `p_fused(k) proportional to p_dgem(k) * exp(prior_weight * l_k)`.

  Args:
    probabilities: Diffusion slot probability dictionary.
    prior_logits: External log-prior dictionary keyed by option name.
    prior_weight: Non-negative log-linear weight (`0.0` leaves `probabilities` unchanged).

  Returns:
    Normalized posterior probability dictionary.

  Raises:
    DGemValidationError: If `prior_weight < 0` or `probabilities` is invalid.
  """
  if not math.isfinite(prior_weight) or prior_weight < 0.0:
    raise DGemValidationError(
        f"prior_weight must be finite and >= 0 (got {prior_weight})"
    )
  ordered = _normalize_probabilities(probabilities)
  if prior_weight == 0.0 or not prior_logits:
    return dict(ordered)

  combined_logits: dict[str, float] = {}
  max_logit = -math.inf
  for name, prob in ordered:
    base_log = -30.0 if prob <= 0.0 else math.log(prob)
    prior_val = float(prior_logits.get(name, 0.0))
    if not math.isfinite(prior_val):
      prior_val = 0.0
    fused = base_log + prior_weight * prior_val
    combined_logits[name] = fused
    if fused > max_logit:
      max_logit = fused

  exp_scores = {
      name: math.exp(val - max_logit) for name, val in combined_logits.items()
  }
  normalizer = sum(exp_scores.values())
  return {name: score / normalizer for name, score in exp_scores.items()}


def hmac_uint64(seed_key: str, item_key: str, tag: str) -> int:
  """Compute the exact big-endian uint64 HMAC-SHA256 digest used by Go `client`."""
  effective_key = seed_key if seed_key else DEFAULT_CONFORMAL_SEED_KEY
  mac = hmac.new(effective_key.encode("utf-8"), digestmod=hashlib.sha256)
  mac.update(item_key.encode("utf-8"))
  mac.update(b"\x00")
  mac.update(tag.encode("utf-8"))
  return int(struct.unpack(">Q", mac.digest()[:8])[0])


def hmac_uniform_float64(seed_key: str, item_key: str, tag: str) -> float:
  """Draw a deterministic Uniform[0, 1) float64 matching Go `hmacUniformFloat64`."""
  top_53 = hmac_uint64(seed_key, item_key, tag) >> 11
  return float(top_53) * (1.0 / (1 << 53))


def _validate_conformal_config(
    config: ConformalConfig,
) -> tuple[str, int, float]:
  """Validate method, k_reg, and lambda_reg on a ConformalConfig."""
  raw_method = config.method.strip().lower()
  if raw_method in ("", "raps"):
    method = "raps"
  elif raw_method in ("lac", "aps"):
    method = raw_method
  else:
    raise DGemValidationError(f"unknown conformal method: {config.method!r}")

  if config.k_reg < 0:
    raise DGemValidationError(f"k_reg must be >= 0 (got {config.k_reg})")
  if not math.isfinite(config.lambda_reg) or config.lambda_reg < 0.0:
    raise DGemValidationError(
        f"lambda_reg must be finite and >= 0 (got {config.lambda_reg})"
    )
  k_reg = config.k_reg
  lambda_reg = config.lambda_reg
  if raw_method == "" and k_reg == 0 and lambda_reg == 0.0:
    k_reg = 1
    lambda_reg = 0.01
  return method, k_reg, lambda_reg


def _partition_tie_groups(
    ordered: Sequence[tuple[str, float]],
) -> list[list[tuple[str, float]]]:
  """Partition sorted `(name, prob)` tuples into probability tie groups."""
  groups: list[list[tuple[str, float]]] = []
  index = 0
  total = len(ordered)
  while index < total:
    anchor_prob = ordered[index][1]
    end = index + 1
    while (
        end < total
        and abs(ordered[end][1] - anchor_prob) <= CONFORMAL_SCORE_TOLERANCE
    ):
      end += 1
    groups.append(list(ordered[index:end]))
    index = end
  return groups


def _compute_scores_for_item(
    ordered: Sequence[tuple[str, float]],
    method: str,
    k_reg: int,
    lambda_reg: float,
    config: ConformalConfig,
    item_key: str,
) -> dict[str, float]:
  """Compute LAC, APS, or RAPS nonconformity scores for one probability vector."""
  if not config.randomized:
    if method == "lac":
      return {name: 1.0 - prob for name, prob in ordered}
    scores: dict[str, float] = {}
    cum_strictly_greater = 0.0
    strict_count = 0
    for group in _partition_tie_groups(ordered):
      group_start_rank = strict_count + 1
      penalty = 0.0
      if method == "raps" and group_start_rank > k_reg:
        penalty = lambda_reg * float(group_start_rank - k_reg)
      group_score = cum_strictly_greater + penalty
      group_mass = 0.0
      for name, prob in group:
        scores[name] = group_score
        group_mass += prob
      cum_strictly_greater += group_mass
      strict_count += len(group)
    return scores

  uniform_draw = hmac_uniform_float64(config.seed_key, item_key, "u")
  flattened: list[tuple[str, float]] = []
  for group in _partition_tie_groups(ordered):
    if len(group) <= 1:
      flattened.extend(group)
      continue
    shuffled_group = sorted(
        group,
        key=lambda item: (
            hmac_uint64(config.seed_key, item_key, f"tie:{item[0]}"),
            item[0],
        ),
    )
    flattened.extend(shuffled_group)

  rand_scores: dict[str, float] = {}
  cum_before = 0.0
  for idx, (name, prob) in enumerate(flattened):
    rank_one = idx + 1
    if method == "lac":
      rand_scores[name] = 1.0 - prob
    elif method == "aps":
      rand_scores[name] = cum_before + uniform_draw * prob
    else:
      penalty = (
          lambda_reg * float(rank_one - k_reg) if rank_one > k_reg else 0.0
      )
      rand_scores[name] = cum_before + uniform_draw * prob + penalty
    cum_before += prob
  return rand_scores


def resolve_gold_probability_key(
    probabilities: Mapping[str, float],
    expected_label: str,
) -> str | None:
  """Resolve a gold label against probability keys (including yes/no aliases).

  Args:
    probabilities: Question probability mapping.
    expected_label: Gold label string from evaluation data.

  Returns:
    Matching dictionary key in `probabilities`, or `None` if absent.
  """
  if not probabilities:
    return None
  if expected_label in probabilities:
    return expected_label
  trimmed = expected_label.strip()
  if trimmed in probabilities:
    return trimmed
  lowered = trimmed.lower()
  for key in probabilities:
    if key.strip().lower() == lowered:
      return key
  if lowered in ("true", "yes", "1"):
    for alias in ("yes", "true", "1"):
      for key in probabilities:
        if key.strip().lower() == alias:
          return key
  elif lowered in ("false", "no", "0"):
    for alias in ("no", "false", "0"):
      for key in probabilities:
        if key.strip().lower() == alias:
          return key
  try:
    expected_num = float(trimmed)
  except ValueError:
    return None
  for key in probabilities:
    try:
      if abs(expected_num - float(key.strip())) <= 1e-9:
        return key
    except ValueError:
      continue
  return None


def calibrate_conformal_quantile(
    calibration_probabilities: Sequence[Mapping[str, float]],
    calibration_labels: Sequence[str],
    alpha: float = 0.10,
    config: ConformalConfig | None = None,
    item_keys: Sequence[str] | None = None,
) -> ConformalCalibrationResult:
  """Fit the finite-sample split-conformal quantile q_hat on calibration rows.

  Order statistic index matches Go `CalibrateConformalQuantileWithKeys`:
  `k = ceil((n + 1) * (1 - alpha) - 1e-12)`.

  Args:
    calibration_probabilities: Sequence of per-item probability maps.
    calibration_labels: Sequence of gold labels aligned with `calibration_probabilities`.
    alpha: Target miscoverage rate strictly inside `(0.0, 1.0)`.
    config: Optional `ConformalConfig` (`raps` default).
    item_keys: Optional per-item keys used when `config.randomized` is True.

  Returns:
    `ConformalCalibrationResult` containing `q_hat` and empirical coverage.

  Raises:
    DGemValidationError: If inputs are empty, mismatched, or out of range.
  """
  effective_config = config or ConformalConfig(alpha=alpha)
  if not calibration_probabilities:
    raise DGemValidationError("calibration_probabilities must be non-empty")
  if len(calibration_probabilities) != len(calibration_labels):
    raise DGemValidationError(
        "calibration_probabilities and calibration_labels length mismatch"
    )
  if item_keys is not None and len(item_keys) != len(calibration_probabilities):
    raise DGemValidationError("item_keys length must match calibration rows")
  if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
    raise DGemValidationError(f"alpha must lie strictly in (0, 1) (got {alpha})")

  method, k_reg, lambda_reg = _validate_conformal_config(effective_config)
  num_items = len(calibration_probabilities)
  raw_scores: list[float] = []

  for idx in range(num_items):
    probs = calibration_probabilities[idx]
    ordered = _normalize_probabilities(probs)
    resolved = resolve_gold_probability_key(probs, calibration_labels[idx])
    if resolved is None:
      raise DGemValidationError(
          f"gold label {calibration_labels[idx]!r} missing at index {idx}"
      )
    key = (
        item_keys[idx]
        if item_keys is not None and item_keys[idx]
        else f"cal:{idx}"
    )
    score_map = _compute_scores_for_item(
        ordered, method, k_reg, lambda_reg, effective_config, key
    )
    raw_scores.append(score_map[resolved])

  sorted_scores = sorted(raw_scores)
  order_index = max(
      1,
      int(
          math.ceil(
              float(num_items + 1) * (1.0 - alpha) - CONFORMAL_SCORE_TOLERANCE
          )
      ),
  )
  if order_index > num_items:
    order_index = num_items + 1
    q_hat = math.inf
  else:
    q_hat = sorted_scores[order_index - 1]

  covered = sum(
      1
      for score in raw_scores
      if math.isinf(q_hat) or score <= q_hat + CONFORMAL_SCORE_TOLERANCE
  )
  return ConformalCalibrationResult(
      method=method,
      alpha=alpha,
      num_calibration=num_items,
      order_index=order_index,
      q_hat=q_hat,
      empirical_coverage=float(covered) / float(num_items),
      sorted_scores=tuple(sorted_scores),
  )


def conformal_prediction_set(
    probabilities: Mapping[str, float],
    q_hat: float,
    config: ConformalConfig | None = None,
) -> ConformalSetResult:
  """Construct the calibrated prediction set C_alpha(x) = {y : s(x, y) <= q_hat}.

  Args:
    probabilities: Option probability mapping for one question slot.
    q_hat: Calibrated quantile threshold (`>= 0.0`, or `+inf` for trivial cover).
    config: Optional `ConformalConfig` (`raps` default).

  Returns:
    `ConformalSetResult` with `should_escalate = (set_size != 1 or contains_abstain)`.

  Raises:
    DGemValidationError: If `q_hat`, `probabilities`, or `config` is invalid.
  """
  effective_config = config or ConformalConfig()
  method, k_reg, lambda_reg = _validate_conformal_config(effective_config)
  if math.isnan(q_hat) or q_hat == -math.inf or q_hat < 0.0:
    raise DGemValidationError(f"q_hat must be >= 0 and non-NaN (got {q_hat})")

  ordered = _normalize_probabilities(probabilities)
  scores = _compute_scores_for_item(
      ordered,
      method,
      k_reg,
      lambda_reg,
      effective_config,
      effective_config.item_key,
  )
  abstain_source = (
      effective_config.abstain_labels
      if effective_config.abstain_labels is not None
      else DEFAULT_ABSTAIN_LABELS
  )
  abstain_lookup = {
      label.strip().lower() for label in abstain_source if label.strip()
  }

  prediction_members: list[str] = []
  contains_abstain = False
  for name, _ in ordered:
    if math.isinf(q_hat) or scores[name] <= q_hat + CONFORMAL_SCORE_TOLERANCE:
      prediction_members.append(name)
      if name.strip().lower() in abstain_lookup:
        contains_abstain = True

  top_option, top_probability = ordered[0]
  singleton = len(prediction_members) == 1
  return ConformalSetResult(
      prediction_set=tuple(prediction_members),
      scores=scores,
      q_hat=q_hat,
      top_option=top_option,
      top_probability=top_probability,
      set_size=len(prediction_members),
      singleton=singleton,
      contains_abstain=contains_abstain,
      should_escalate=(not singleton) or contains_abstain,
  )
