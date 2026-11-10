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

"""Typed data models and exceptions for the dgem Python SDK."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]

CLEAR_HESITATION_CEILING = 0.16
SOMEWHAT_UNSURE_CEILING = 0.50


class DGemError(Exception):
  """Base exception raised by the dgem Python SDK."""


class DGemValidationError(DGemError):
  """Raised when caller inputs or calibration parameters fail validation."""


class DGemTransportError(DGemError):
  """Raised when a network or timeout failure prevents receiving a response."""


class DGemHTTPError(DGemError):
  """Raised when the gateway or serving endpoint returns a non-2xx status."""

  def __init__(
      self,
      status_code: int,
      message: str,
      response_body: str = "",
  ) -> None:
    """Initialize an HTTP status error.

    Args:
      status_code: HTTP status code returned by the remote server.
      message: Concise error summary.
      response_body: Raw response payload body when available.
    """
    super().__init__(f"HTTP {status_code}: {message}")
    self.status_code = status_code
    self.response_body = response_body


def _hesitation_from_probabilities(
    probabilities: Mapping[str, float],
    entropy_nats: float,
) -> float:
  """Compute cardinality-normalized entropy H / ln(K) clamped to [0.0, 1.0]."""
  option_count = len(probabilities)
  if option_count <= 1:
    return 0.0
  computed_entropy = entropy_nats
  if computed_entropy <= 0.0:
    computed_entropy = 0.0
    for probability in probabilities.values():
      if probability > 0.0 and math.isfinite(probability):
        computed_entropy -= probability * math.log(probability)
  normalized = computed_entropy / math.log(option_count)
  return max(0.0, min(1.0, normalized))


def classify_hesitation_band(hesitation: float) -> str:
  """Classify normalized hesitation into dgem's three operating bands.

  Args:
    hesitation: Normalized entropy in [0.0, 1.0].

  Returns:
    Band identifier (`"clear"`, `"somewhat_unsure"`, or `"very_unsure"`).
  """
  if hesitation < CLEAR_HESITATION_CEILING:
    return "clear"
  if hesitation <= SOMEWHAT_UNSURE_CEILING:
    return "somewhat_unsure"
  return "very_unsure"


@dataclass(frozen=True)
class QuestionAnswer:
  """Evaluated single-question decision result returned by dgem."""

  question_type: str
  label: str
  confidence: float
  probabilities: dict[str, float] = field(default_factory=dict)
  logprob: float = 0.0
  entropy: float = 0.0
  stderr: float = 0.0
  agreement: float = 1.0
  choice: str = ""
  score: float = 0.0
  level: str = ""
  noul: float = 0.0
  prior_logits: dict[str, float] = field(default_factory=dict)
  prior_fused: bool = False
  raw_entropy: float = 0.0
  evidence: str = ""

  @property
  def display_type(self) -> str:
    """Return canonical question type, normalizing `noul` to `boolean`."""
    normalized = self.question_type.strip().lower()
    if normalized in ("noul", "bool", "boolean"):
      return "boolean"
    return normalized

  @property
  def value(self) -> str | bool | float:
    """Return the primary resolved value for boolean, choice, or score slots."""
    kind = self.display_type
    if kind == "boolean":
      return self.label.strip().lower() in ("yes", "true", "1")
    if kind == "choice":
      return self.choice or self.label
    if kind == "score":
      return self.score
    return self.choice or self.level or self.label

  @property
  def hesitation(self) -> float:
    """Return cardinality-normalized hesitation H / ln(K) in [0.0, 1.0]."""
    return _hesitation_from_probabilities(self.probabilities, self.entropy)

  @property
  def hesitation_band(self) -> str:
    """Return `"clear"`, `"somewhat_unsure"`, or `"very_unsure"`."""
    return classify_hesitation_band(self.hesitation)

  @classmethod
  def from_dict(cls, payload: Mapping[str, object]) -> "QuestionAnswer":
    """Deserialize a QuestionAnswer from a JSON mapping with safe defaults.

    Args:
      payload: Decoded JSON dictionary representing one question answer.

    Returns:
      Populated `QuestionAnswer` instance with non-null probability maps.
    """
    raw_probs = payload.get("probabilities")
    probabilities: dict[str, float] = {}
    if isinstance(raw_probs, Mapping):
      for key, val in raw_probs.items():
        if isinstance(val, (int, float)):
          probabilities[str(key)] = float(val)

    raw_priors = payload.get("prior_logits")
    prior_logits: dict[str, float] = {}
    if isinstance(raw_priors, Mapping):
      for key, val in raw_priors.items():
        if isinstance(val, (int, float)):
          prior_logits[str(key)] = float(val)

    def _float_field(field_name: str, default: float = 0.0) -> float:
      raw_val = payload.get(field_name)
      if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        return float(raw_val)
      return default

    return cls(
        question_type=str(payload.get("type", "")),
        label=str(payload.get("label", "")),
        confidence=_float_field("confidence"),
        probabilities=probabilities,
        logprob=_float_field("logprob"),
        entropy=_float_field("entropy"),
        stderr=_float_field("stderr"),
        agreement=_float_field("agreement", 1.0),
        choice=str(payload.get("choice", "")),
        score=_float_field("score"),
        level=str(payload.get("level", "")),
        noul=_float_field("noul"),
        prior_logits=prior_logits,
        prior_fused=bool(payload.get("prior_fused", False)),
        raw_entropy=_float_field("raw_entropy"),
        evidence=str(payload.get("evidence", "")),
    )


@dataclass(frozen=True)
class StructuredDecisionResponse:
  """Structured decision payload returned by `/v1/systemone` or serving engine."""

  answers: dict[str, QuestionAnswer] = field(default_factory=dict)
  diagnostics: dict[str, object] = field(default_factory=dict)

  @property
  def max_hesitation(self) -> float:
    """Return the maximum normalized hesitation across all answered questions."""
    if not self.answers:
      return 0.0
    return max(answer.hesitation for answer in self.answers.values())

  @classmethod
  def from_dict(
      cls,
      payload: Mapping[str, object],
  ) -> "StructuredDecisionResponse":
    """Parse a `/v1/systemone` or structured decision JSON dictionary.

    Args:
      payload: Decoded JSON response mapping.

    Returns:
      Parsed `StructuredDecisionResponse` with initialized maps.
    """
    raw_answers = payload.get("answers")
    answers: dict[str, QuestionAnswer] = {}
    if isinstance(raw_answers, Mapping):
      for question_id, raw_item in raw_answers.items():
        if isinstance(raw_item, Mapping):
          answers[str(question_id)] = QuestionAnswer.from_dict(raw_item)
    raw_diagnostics = payload.get("diagnostics")
    diagnostics = (
        dict(raw_diagnostics) if isinstance(raw_diagnostics, Mapping) else {}
    )
    return cls(answers=answers, diagnostics=diagnostics)


@dataclass(frozen=True)
class GatewayDecideResponse:
  """Typed response returned by `POST /api/decide` on the dgem gateway."""

  template: str
  answers: dict[str, QuestionAnswer]
  diagnostics: dict[str, object] = field(default_factory=dict)
  max_entropy: float = 0.0
  wall_time_ms: int = 0
  gpu_forward_ms: int = 0
  cold_start_wait_ms: int = 0
  model: str = ""
  backend_target: str = ""
  backend_used: str = ""
  trace_id: str = ""
  cascade: dict[str, object] | None = None

  @property
  def max_hesitation(self) -> float:
    """Return the maximum normalized hesitation across all answered questions."""
    if not self.answers:
      return 0.0
    return max(answer.hesitation for answer in self.answers.values())

  @classmethod
  def from_dict(
      cls,
      payload: Mapping[str, object],
      backend_used_header: str = "",
  ) -> "GatewayDecideResponse":
    """Deserialize a `POST /api/decide` gateway JSON response.

    Args:
      payload: Decoded JSON response mapping.
      backend_used_header: Value of `X-DGem-Backend-Used` HTTP header if set.

    Returns:
      Populated `GatewayDecideResponse`.
    """
    structured = StructuredDecisionResponse.from_dict(payload)
    raw_cascade = payload.get("cascade")
    cascade_dict = (
        dict(raw_cascade) if isinstance(raw_cascade, Mapping) else None
    )

    def _int_field(field_name: str) -> int:
      raw_val = payload.get(field_name)
      if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        return int(raw_val)
      return 0

    raw_max_entropy = payload.get("max_entropy")
    max_entropy = (
        float(raw_max_entropy)
        if isinstance(raw_max_entropy, (int, float))
        and not isinstance(raw_max_entropy, bool)
        else 0.0
    )
    backend_target = str(payload.get("backend_target", ""))
    return cls(
        template=str(payload.get("template", "")),
        answers=structured.answers,
        diagnostics=structured.diagnostics,
        max_entropy=max_entropy,
        wall_time_ms=_int_field("wall_time_ms"),
        gpu_forward_ms=_int_field("gpu_forward_ms"),
        cold_start_wait_ms=_int_field("cold_start_wait_ms"),
        model=str(payload.get("model", "")),
        backend_target=backend_target,
        backend_used=backend_used_header or backend_target,
        trace_id=str(payload.get("trace_id", "")),
        cascade=cascade_dict,
    )


@dataclass(frozen=True)
class SystemOneQuestion:
  """Question definition for `POST /v1/systemone` requests."""

  question_type: str
  instructions: str
  criteria: Mapping[str, str] | Sequence[str] | None = None
  prior_logits: Mapping[str, float] | None = None
  evidence_prompt: str = ""
  evidence_criteria: Mapping[str, str] | None = None

  def to_dict(self) -> dict[str, object]:
    """Serialize the question into `/v1/systemone` wire JSON format.

    Returns:
      Dictionary matching `decisionindex.SystemOneQuestion`.
    """
    wire: dict[str, object] = {
        "type": self.question_type,
        "instructions": self.instructions,
    }
    if self.criteria is not None:
      if isinstance(self.criteria, Mapping):
        wire["criteria"] = dict(self.criteria)
      else:
        wire["criteria"] = list(self.criteria)
    if self.prior_logits:
      wire["prior_logits"] = {
          str(key): float(val) for key, val in self.prior_logits.items()
      }
    if self.evidence_prompt:
      wire["evidence_prompt"] = self.evidence_prompt
    if self.evidence_criteria:
      wire["evidence_criteria"] = dict(self.evidence_criteria)
    return wire


@dataclass(frozen=True)
class ConformalConfig:
  """Stateless split-conformal configuration matching `pkg/client/conformal.go`."""

  method: str = "raps"
  alpha: float = 0.10
  k_reg: int = 1
  lambda_reg: float = 0.01
  randomized: bool = False
  seed_key: str = "dgem-conformal-v1"
  item_key: str = ""
  abstain_labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ConformalCalibrationResult:
  """Fitted split-conformal quantile and empirical coverage summary."""

  method: str
  alpha: float
  num_calibration: int
  order_index: int
  q_hat: float
  empirical_coverage: float
  sorted_scores: tuple[float, ...]


@dataclass(frozen=True)
class ConformalSetResult:
  """Evaluated prediction set and escalation decision for one question slot."""

  prediction_set: tuple[str, ...]
  scores: dict[str, float]
  q_hat: float
  top_option: str
  top_probability: float
  set_size: int
  singleton: bool
  contains_abstain: bool
  should_escalate: bool
