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

"""Zero-dependency Python client SDK for DiffusionGemma (dgem)."""

from dgem_sdk.adapters import create_adk_before_tool_guard
from dgem_sdk.adapters import create_langgraph_decision_node
from dgem_sdk.calibration import apply_answer_temperature
from dgem_sdk.calibration import apply_temperature
from dgem_sdk.calibration import calibrate_conformal_quantile
from dgem_sdk.calibration import conformal_prediction_set
from dgem_sdk.calibration import fuse_prior_probabilities
from dgem_sdk.calibration import hmac_uint64
from dgem_sdk.calibration import hmac_uniform_float64
from dgem_sdk.calibration import normalize_cosine_prior_logits
from dgem_sdk.calibration import normalized_entropy
from dgem_sdk.calibration import resolve_gold_probability_key
from dgem_sdk.calibration import shannon_entropy
from dgem_sdk.client import DGemClient
from dgem_sdk.types import CLEAR_HESITATION_CEILING
from dgem_sdk.types import SOMEWHAT_UNSURE_CEILING
from dgem_sdk.types import ConformalCalibrationResult
from dgem_sdk.types import ConformalConfig
from dgem_sdk.types import ConformalSetResult
from dgem_sdk.types import DGemError
from dgem_sdk.types import DGemHTTPError
from dgem_sdk.types import DGemTransportError
from dgem_sdk.types import DGemValidationError
from dgem_sdk.types import GatewayDecideResponse
from dgem_sdk.types import QuestionAnswer
from dgem_sdk.types import StructuredDecisionResponse
from dgem_sdk.types import SystemOneQuestion
from dgem_sdk.types import classify_hesitation_band

__all__ = [
    "CLEAR_HESITATION_CEILING",
    "SOMEWHAT_UNSURE_CEILING",
    "ConformalCalibrationResult",
    "ConformalConfig",
    "ConformalSetResult",
    "DGemClient",
    "DGemError",
    "DGemHTTPError",
    "DGemTransportError",
    "DGemValidationError",
    "GatewayDecideResponse",
    "QuestionAnswer",
    "StructuredDecisionResponse",
    "SystemOneQuestion",
    "apply_answer_temperature",
    "apply_temperature",
    "calibrate_conformal_quantile",
    "classify_hesitation_band",
    "conformal_prediction_set",
    "create_adk_before_tool_guard",
    "create_langgraph_decision_node",
    "fuse_prior_probabilities",
    "hmac_uint64",
    "hmac_uniform_float64",
    "normalize_cosine_prior_logits",
    "normalized_entropy",
    "resolve_gold_probability_key",
    "shannon_entropy",
]
