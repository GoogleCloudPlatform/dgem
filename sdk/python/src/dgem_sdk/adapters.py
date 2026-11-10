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

"""Framework-agnostic agent adapters for Google ADK (Python) and LangGraph."""

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Protocol

from dgem_sdk.calibration import apply_answer_temperature
from dgem_sdk.calibration import conformal_prediction_set
from dgem_sdk.client import DGemClient
from dgem_sdk.types import CLEAR_HESITATION_CEILING
from dgem_sdk.types import ConformalConfig
from dgem_sdk.types import DGemError


class SupportsToolName(Protocol):
  """Structural protocol matching `google.adk.tools.BaseTool`."""

  @property
  def name(self) -> str:
    """Return the registered tool identifier."""


def _sanitize_untrusted_text(raw_text: str) -> str:
  """Neutralize embedded `<untrusted_data>` closing tags in tool arguments."""
  return raw_text.replace("</untrusted_data>", "<\\/untrusted_data>")


def create_adk_before_tool_guard(
    client: DGemClient,
    *,
    template: str = "tool_safety_gate",
    hazard_slot: str = "action_hazard",
    origin_slot: str = "intent_origin",
    deny_hazards: Sequence[str] = ("destructive", "credential_exfiltration"),
    deny_origins: Sequence[str] = ("untrusted_prompt_injection",),
    hesitation_ceiling: float = CLEAR_HESITATION_CEILING,
    temperature: float | None = None,
    conformal_q_hat: float | None = None,
    conformal_config: ConformalConfig | None = None,
    fail_closed: bool = True,
) -> Callable[
    [SupportsToolName | str, Mapping[str, object], object | None],
    dict[str, object] | None,
]:
  """Build a Google ADK `before_tool_callback` backed by a dgem safety policy.

  In Google ADK (`LlmAgent(before_tool_callback=...)`), returning `None` allows
  the tool call to execute, whereas returning a `dict` short-circuits tool
  execution and passes that dictionary back to the model as the tool response.

  Args:
    client: Configured `DGemClient` pointing at a `dgem serve` gateway.
    template: Safety policy template name (default `"tool_safety_gate"`).
    hazard_slot: Question ID evaluating operational hazard level.
    origin_slot: Question ID evaluating user vs. prompt-injection origin.
    deny_hazards: Hazard choice values that immediately block execution.
    deny_origins: Origin choice values that immediately block execution.
    hesitation_ceiling: Maximum normalized hesitation (`0.16` default) allowed
      for automatic approval without human confirmation.
    temperature: Optional post-hoc policy temperature applied before gating.
    conformal_q_hat: Optional calibrated conformal threshold; escalates whenever
      `should_escalate` is True on `hazard_slot`.
    conformal_config: Optional `ConformalConfig` paired with `conformal_q_hat`.
    fail_closed: When True (default), transport or HTTP errors block execution.

  Returns:
    Callback `(tool, args, tool_context) -> dict | None` compatible with ADK.
  """
  blocked_hazards = {item.strip().lower() for item in deny_hazards}
  blocked_origins = {item.strip().lower() for item in deny_origins}

  def _before_tool_callback(
      tool: SupportsToolName | str,
      args: Mapping[str, object],
      tool_context: object | None = None,
  ) -> dict[str, object] | None:
    del tool_context
    tool_name = tool if isinstance(tool, str) else getattr(tool, "name", "tool")
    serialized_args = _sanitize_untrusted_text(
        json.dumps(dict(args), sort_keys=True, default=str)
    )
    try:
      decision = client.decide(
          template,
          variables={
              "tool_name": str(tool_name),
              "tool_args": f"<untrusted_data>{serialized_args}</untrusted_data>",
          },
      )
    except DGemError as error:
      if fail_closed:
        return {
            "status": "blocked",
            "decision": "ask_human",
            "reason": f"dgem safety gate unavailable: {error}",
        }
      return None

    answers = dict(decision.answers)
    if temperature is not None:
      answers = {
          qid: apply_answer_temperature(ans, temperature)
          for qid, ans in answers.items()
      }

    origin_answer = answers.get(origin_slot)
    if origin_answer is not None:
      origin_choice = str(origin_answer.value).strip().lower()
      if origin_choice in blocked_origins:
        return {
            "status": "blocked",
            "decision": "deny",
            "reason": f"blocked intent origin: {origin_choice}",
            "hesitation": origin_answer.hesitation,
        }

    hazard_answer = answers.get(hazard_slot)
    if hazard_answer is not None:
      hazard_choice = str(hazard_answer.value).strip().lower()
      if hazard_choice in blocked_hazards:
        return {
            "status": "blocked",
            "decision": "deny",
            "reason": f"blocked action hazard: {hazard_choice}",
            "hesitation": hazard_answer.hesitation,
        }
      if conformal_q_hat is not None and hazard_answer.probabilities:
        conformal = conformal_prediction_set(
            hazard_answer.probabilities,
            conformal_q_hat,
            conformal_config,
        )
        if conformal.should_escalate:
          return {
              "status": "approval_required",
              "decision": "ask_human",
              "reason": "conformal prediction set is non-singleton",
              "prediction_set": list(conformal.prediction_set),
          }

    max_hesitation = (
        max(ans.hesitation for ans in answers.values()) if answers else 1.0
    )
    if max_hesitation >= hesitation_ceiling:
      return {
          "status": "approval_required",
          "decision": "ask_human",
          "reason": (
              f"hesitation {max_hesitation:.3f} >= ceiling"
              f" {hesitation_ceiling:.3f}"
          ),
          "hesitation": max_hesitation,
      }
    return None

  return _before_tool_callback


def create_langgraph_decision_node(
    client: DGemClient,
    *,
    template: str,
    input_keys: Sequence[str],
    route_question_id: str,
    output_key: str = "dgem_decision",
    hesitation_ceiling: float = CLEAR_HESITATION_CEILING,
    escalate_route: str = "escalate_stage2",
    temperature: float | None = None,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
  """Create a LangGraph state-update node that routes on a dgem policy slot.

  Args:
    client: Configured `DGemClient` pointing at a `dgem serve` gateway.
    template: Policy `.json.tmpl` identifier to evaluate.
    input_keys: Graph state keys forwarded as template `variables`.
    route_question_id: Question ID whose choice selects `next_route`.
    output_key: Graph state key written by the node.
    hesitation_ceiling: Normalized hesitation threshold triggering escalation.
    escalate_route: Route label emitted when hesitation meets `hesitation_ceiling`.
    temperature: Optional post-hoc temperature applied before computing hesitation.

  Returns:
    Callable `(state: Mapping[str, object]) -> dict[str, object]` for LangGraph.
  """

  def _decision_node(state: Mapping[str, object]) -> dict[str, object]:
    variables = {key: state.get(key, "") for key in input_keys}
    response = client.decide(template, variables=variables)
    answers = dict(response.answers)
    if temperature is not None:
      answers = {
          qid: apply_answer_temperature(ans, temperature)
          for qid, ans in answers.items()
      }
    primary = answers.get(route_question_id)
    if primary is None:
      selected_route = escalate_route
      hesitation = 1.0
    elif primary.hesitation >= hesitation_ceiling:
      selected_route = escalate_route
      hesitation = primary.hesitation
    else:
      selected_route = str(primary.value)
      hesitation = primary.hesitation

    return {
        output_key: {
            "route": selected_route,
            "hesitation": hesitation,
            "escalated": selected_route == escalate_route,
            "answers": {
                qid: {
                    "value": ans.value,
                    "confidence": ans.confidence,
                    "hesitation": ans.hesitation,
                    "probabilities": dict(ans.probabilities),
                }
                for qid, ans in answers.items()
            },
            "wall_time_ms": response.wall_time_ms,
        }
    }

  return _decision_node
