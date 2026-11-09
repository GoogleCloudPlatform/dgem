# Genkit Go v1.14.0 SecOps Alert DAG, RAG Quarantine & IAM Gate (`genkit-secops-triage`)

This example demonstrates three **SecOps** decision workflows pairing **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** with `dgem` `.json.tmpl` policies:

1. **2-Stage Conditional DAG Alert Response (`templates/secops_conditional_dag.json.tmpl`)**:
   - `dgemgenkit.DecideWithTemplate` schedules `depends_on` / `ask_if` questions into topological stages.
   - Routine backup telemetry (`active_breach == "no"`) finishes in **1 pass** (`Passes == 1`) and skips `containment_action` and `blast_radius`, avoiding an unnecessary second read.
   - Active compromise telemetry (`active_breach == "yes"`) runs **Stage 2** conditioned on `prior_answers` (`Passes == 2`) to select the containment playbook (`revoke_iam_and_snapshots`) and `blast_radius` (`account_wide_root`).
2. **RAG `Use`-vs-`Mention` Indirect Prompt-Injection Quarantine (`templates/secops_rag_quarantine.json.tmpl`)**:
   - Evaluates retrieved RAG passages across 3 decoupled choice slots (`"isolate": "all"`, `"seed_token": "fixed"`): `passage_relevance`, 4-tier `injection_posture` (`clean`, `benign_security_discussion`, `suspicious_framing`, `active_override_attempt`), and `quarantine_action`.
   - Security postmortems or runbooks that *quote* attack strings as examples are routed to `quote_as_inert_data` rather than false-positively quarantined, while active override payloads are routed to `quarantine_and_alert`.
3. **Control-Plane IAM / Terraform Diff Gating (`templates/secops_iam_change_audit.json.tmpl`)**:
   - Combines Genkit `v1.14.0`'s `&middleware.ToolApproval{}` with `genkit.DefineResumableTool` (`tool.Interrupt`):
     - **`auto_approve`** (least-privilege diff + Clear Hesitation `< 16%`): approved automatically.
     - **`block_change`** (wildcard `roles/owner` / `allUsers` or leaked credential): denied at the `ToolApproval` gate so the tool never executes.
     - **`require_secops_review`** (privileged cross-project production KMS binding or Hesitation `>= 16%`): raises a typed `tool.Interrupt` hold for human SecOps sign-off via `ai.WithResume(...)`.

## Running Offline Unit Tests (No GPU or Cloud Credentials Required)

```bash
cd examples/genkit-secops-triage
go test -v ./...
```

## Running Live

```bash
cd examples/genkit-secops-triage

# Against a dgem serve gateway:
go run . -gateway-url "$DGEM_GATEWAY_URL"

# Or against local Apple Silicon diffgemma (127.0.0.1:8080):
go run .
```
