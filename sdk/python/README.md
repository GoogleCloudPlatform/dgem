# `dgem-sdk` (Python)

Zero-dependency Python client, post-hoc calibration primitives, split-conformal prediction sets (`LAC`, `APS`, `RAPS`), cosine-prior fusion, and **Google ADK** / **LangGraph** adapters for `dgem`.

## Install locally

```bash
pip install -e sdk/python
```

## Quick start

```python
from dgem_sdk import (
    ConformalConfig,
    DGemClient,
    SystemOneQuestion,
    apply_temperature,
    conformal_prediction_set,
    normalize_cosine_prior_logits,
)

client = DGemClient("http://127.0.0.1:8090")

# 1. Evaluate a .json.tmpl policy via POST /api/decide
decision = client.decide(
    "support_triage",
    variables={"ticket": "I was billed $500 twice for my annual renewal!"},
)
team = decision.answers["team"]
print(team.choice, team.confidence, team.hesitation, team.hesitation_band)

# 2. Evaluate ad-hoc questions via POST /v1/systemone with cosine-prior fusion
priors = normalize_cosine_prior_logits({"billing": 0.84, "sre": 0.41, "account": 0.29})
sys1 = client.decide_systemone(
    "Database cluster unreachable in us-central1",
    questions={
        "queue": SystemOneQuestion(
            question_type="choice",
            instructions="Which engineering queue owns this incident?",
            criteria={
                "billing": "Invoices and duplicate charges",
                "sre": "Production outages and latency spikes",
                "account": "SSO and user provisioning",
            },
            prior_logits=priors,
        )
    },
    prior_weight=0.2,
)

# 3. Construct calibrated split-conformal prediction sets (LAC / APS / RAPS)
calibrated = apply_temperature(sys1.answers["queue"].probabilities, temperature=1.35)
cset = conformal_prediction_set(calibrated, q_hat=0.18, config=ConformalConfig(method="raps"))
print(cset.prediction_set, cset.should_escalate)
```

## Agent framework adapters

- **Google ADK (`google-adk`)**: Pass `create_adk_before_tool_guard(client)` as `before_tool_callback` on an `LlmAgent` to gate tool execution on `tool_safety_gate.json.tmpl` with `<untrusted_data>` quarantine, hesitation gating, and optional conformal set checks.
- **LangGraph**: Register `create_langgraph_decision_node(client, template="support_triage", input_keys=["ticket"], route_question_id="team")` as a graph node feeding `add_conditional_edges`.

## Run unit tests

```bash
PYTHONPATH=sdk/python/src python3 -m unittest discover -s sdk/python/tests -v
```
