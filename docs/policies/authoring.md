---
title: "Policy Authoring and Stage 2 Cascades"
description: "Designing fast, reliable .json.tmpl decision policies (prefix caching, option design, samples) and escalating hesitant answers to Gemini with Stage 2 cascades in Studio, the HTTP API, MCP and the CLI."
---

# Policy Authoring and Stage 2 Cascades

For policy authors: how to design `.json.tmpl` policies that are fast and reliable, and how to escalate hesitant
answers to a larger model (Stage 2 cascade) on every surface. New to policies? Start with
[Your first decision policy](first-policy.md). Choosing a GPU backend (`vertex_first`, `vertex`, `cloudrun`) is
covered in [Gateway and routing](../deploy/gateway.md).

---

## 1. Stage 2 Gemini Cascade (`gemini-3.8-flash` Default) in Batch Eval, `/api/decide`, and MCP

When Stage 1 (`DiffusionGemma`) encounters an ambiguous input (where restricted-softmax Shannon entropy $H \ge 0.35\text{ nats}$, e.g. `ChaosNLI` human-disagreement items or complex multi-hop `ANLI-R3` contradictions), `dgem` can automatically escalate that item to a **Stage 2 Gemini Cascade**.

> [!IMPORTANT]
> **Supported Stage 2 Gemini Models**: Use **`gemini-3.8-flash`** (default) or **`gemini-3.5-flash-lite`** via standard Vertex AI `generateContent` routes.

| Parameter | Values / Default | Surface Support | Description |
| :--- | :--- | :--- | :--- |
| **`cascade_mode`** | `"off"` (default) \| `"entropy"` \| `"on_miss"` | Web Studio Batch Eval, `POST /api/decide`, MCP (`decide_policy`, `decide_custom_questions`) | **`"off"`**: Stage 1 `dgemma` only (one forward pass).<br/>**`"entropy"`**: Production uncertainty gate — early-exits confident decisions at Stage 1 ($H < \tau$) and escalates only high-entropy items ($H \ge \tau$) to Stage 2 Gemini.<br/>**`"on_miss"`**: Batch Eval ground-truth audit mode — escalates items where Stage 1 disagrees with the dataset's `expected` label. |
| **`cascade_threshold`** | `0.35` *(default, in nats)* | Web Studio Batch Eval, `POST /api/decide`, MCP, `dgem bench-calibration` | Shannon entropy gate $\tau$ in nats (`0.35` raw nats, or `0.16` normalized $\tilde{H} = H / \ln|\mathcal{V}_m|$). |
| **`cascade_model`** | `"gemini-3.8-flash"` *(default)* | Web Studio Batch Eval, `POST /api/decide`, MCP, `dgem serve`, `dgem bench-calibration` | Target Vertex AI Gemini model (`"gemini-3.8-flash"` or `"gemini-3.5-flash-lite"`). Configurable globally via `--cascade-model` (`DGEM_CASCADE_MODEL`) and `--cascade-models` (`DGEM_CASCADE_MODELS`). |
| **`stage2_prior`** | `"soft"` *(default)* \| `"full"` \| `"none"` | Web Studio Batch Eval ("Stage-1 hint to Gemini"), `POST /api/decide` (or header `X-DGem-Stage2-Prior`), MCP, `dgem bench-vision --cascade-prior`, env `DGEM_CASCADE_PRIOR` | How much of Stage 1's answer Gemini sees. **`soft`**: only that a fast first stage leaned toward an answer and may be wrong. **`full`**: answer, confidence, entropy and distribution as "the candidate" (the behaviour before v0.3.1); in our tests this anchored Gemini on low-threshold image cascades, which is why `soft` is the default. **`none`**: the question alone. On text decisions we saw no difference between `soft` and `full`. |

### Example: Calling `/api/decide` and MCP with `cascade_mode: "entropy"`

```bash
curl -sS "https://<your-dgem-gateway>/api/decide/calibration/chaos_nli" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -H "X-DGem-Backend: vertex_first" \
  -d '{
    "backend": "vertex_first",
    "cascade_mode": "entropy",
    "cascade_threshold": 0.35,
    "cascade_model": "gemini-3.8-flash",
    "variables": {
      "premise": "All four categories reached 50 to 75 percent of the cap.",
      "hypothesis": "Every category met the cap."
    }
  }' | jq .
```

```json
{
  "name": "decide_policy",
  "arguments": {
    "template": "calibration/chaos_nli",
    "backend": "vertex_first",
    "cascade_mode": "entropy",
    "cascade_threshold": 0.35,
    "cascade_model": "gemini-3.8-flash",
    "variables": {
      "premise": "All four categories reached 50 to 75 percent of the cap.",
      "hypothesis": "Every category met the cap."
    }
  }
}
```

---

## 2. Designing `.json.tmpl` Policy Schemas & Prefix-Cache Optimization

1. **Keep `"schema"` Static, Put Row Variables in `"state"`**:
   - `structured_server.py` serializes `"schema"` before `"state"` so `vLLM`'s automatic prefix cache can reuse the shared part of the prompt across rows in a batch evaluation. This applies to the `schema_first` layout; under the default `document_first` layout the input comes first, so little of the prompt is shared across rows ([prompt layout](prompt-layout.md)).
2. **Single-Pass Readout (`reads=1`)**:
   - Omit `depends_on` and `ask_if` unless you explicitly want a 2-stage conditional policy DAG (`reads=2`). All `level 0` slots (`boolean`, `choice [A–Z]`, `score`) are resolved simultaneously in a single forward pass.
3. **Evaluating a Policy on a Dataset**:
   - Run the policy on labelled rows and keep the receipt next to the template. For the full step-by-step Python batch runner (`run_dgem_dataset.py`), see **[Run a dataset](datasets.md)**; to check calibration, see [Calibrate your policy](../confidence/calibrate-your-policy.md).
