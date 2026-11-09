# Genkit Go v1.14.0 Calibrated RAI & Safety Guardrails (`genkit-rai-guardrails`)

This example demonstrates four **Responsible AI (RAI) & Safety** decision workflows pairing **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** with `dgem` `.json.tmpl` policies (`templates/rai/`):

1. **Input Safety Guardrail with Calibrated `(T, \tau)` Operating Points (`templates/rai/prompt_safety_guardrail.json.tmpl`)**:
   - Protects benign figurative, homonym, historical, and technical prompts (*"How do I kill a zombie child process in Linux?"* $\to$ `sensitive_benign_context`) from false-positive over-refusals while detecting direct and roleplay-wrapped harmful prompts.
   - Demonstrates `dgemgenkit.TemplateDecision.CalibratedProb` and `ExceedsThreshold(slot, "yes", T, tau)` across three illustrative dev-calibrated operating points (`strict`, `balanced`, `permissive`) from a **single** Stage-1 forward pass (`"seed_token": "fixed"`).
2. **Custom Multi-Rule Moderation with Post-Calibration Hesitation Gating (`templates/rai/custom_content_policy.json.tmpl`)**:
   - Evaluates organization-defined rules `R1`–`R5`, calls `dec.ApplyTemperature(T)`, and gates Stage-2 escalation on **post-calibration `Hesitation` ($H(p_T) / \ln K \ge 16\%$)** while preserving `RawHesitation` for telemetry.
   - Computing $H_{\text{norm}}$ *after* temperature scaling prevents overconfident raw logits from hiding borderline items inside the `< 16%` Clear band before escalating to `vertexai/gemini-3.8-flash` (or a human moderator hold).
3. **Single-Pass Red-Team Transcript Grading (`templates/rai/response_safety_grader.json.tmpl`)**:
   - Scores `(user_request, ai_response)` transcripts in one pass across `prompt_adversarial`, `response_harmful`, `response_refusal`, and 7-way `harm_category` (including `other`).
4. **Unbiased Context QA Abstention & C2PA Provenance Check (`templates/rai/unbiased_context_qa.json.tmpl`, `templates/rai/provenance_claim_check.json.tmpl`)**:
   - Enforces explicit `cannot_be_determined` abstention when context is underspecified rather than guessing from demographic/role stereotypes, and cross-checks C2PA Content Credentials validator reports against publisher captions (`publish_as_verified`, `attach_ai_disclosure_label`, `attach_unverified_origin_note`, `hold_for_editorial_review`).

> **Calibration note:** The `(T, \tau)` pairs in `main.go` are illustrative presets. Always fit policy-specific temperature $T$ and operating threshold $\tau$ on your own held-out labelled validation split (see [`docs/confidence/calibrate-your-policy.md`](../../docs/confidence/calibrate-your-policy.md)).

## Running Offline Unit Tests (No GPU or Cloud Credentials Required)

```bash
cd examples/genkit-rai-guardrails
go test -v ./...
```

## Running Live

```bash
cd examples/genkit-rai-guardrails

# Stage-1 dgem via gateway (hesitant items fall back to human_moderator_hold unless GOOGLE_CLOUD_PROJECT is set):
go run . -gateway-url "$DGEM_GATEWAY_URL"

# Stage-1 dgem + Stage-2 Vertex AI gemini-3.8-flash cascade:
GOOGLE_CLOUD_PROJECT=<PROJECT> go run . -gateway-url "$DGEM_GATEWAY_URL" -stage2-model vertexai/gemini-3.8-flash
```
