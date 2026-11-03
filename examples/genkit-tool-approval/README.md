# Genkit Go v1.14.0 + `dgem` Policy Templates (`genkit-tool-approval`)

This example shows how to pair **[Genkit Go `v1.14.0`](https://github.com/genkit-ai/genkit/releases/tag/go%2Fv1.14.0)** with `dgem`'s executable `.json.tmpl` decision policies (`templates/`):

1. **Zero-Glue System One Wire Compatibility (`systemonex.SystemOne` $\leftrightarrow$ `POST /v1/systemone`)**:
   - Genkit Go `v1.14.0` includes `github.com/firebase/genkit/go/plugins/systemone/exp` (`systemonex`), a plugin for decision models that speak the System One protocol.
   - `dgem` (`pkg/decisionindex.NewSystemOneHTTPHandler`, `dgem serve`, and `dgem systemone serve`) serves `POST /v1/systemone` natively — including against local Apple Silicon Metal (`diffgemma`) and remote Cloud Run / Vertex AI endpoints.
2. **Executing Any `dgem` `.json.tmpl` Policy in Genkit (`../dgemgenkit/bridge.go`)**:
   - `dgemgenkit.DecideWithTemplate(ctx, g, modelName, tmplPath, vars)` renders a `.json.tmpl` file with `dgem/pkg/template`, maps its `boolean`/`noul`, `choice`, and `score` slots into a `systemonex.Schema`, executes them in **one forward pass** via `genkit.GenerateData[map[string]systemonex.Answer]`, and computes wall-time latency (`WallTimeMs`), per-slot Shannon entropy $H$, and **Hesitation %** ($H / \ln K$).
3. **Hesitation-Gated Tool Approval on Genkit `v1.14.0` Interrupts (`main.go`)**:
   - Uses Genkit `v1.14.0`'s typed interrupt claim loop (`middleware.ToolApprovalInterrupted` + `genkit.DefineResumableTool`'s `Interrupted`):
     ```go
     for _, part := range resp.Interrupts() {
     	if call, ok := middleware.ToolApprovalInterrupted(part); ok {
     		decision := middleware.ToolCallDecision{Approved: confirm(part.ToolRequest.Name, call.Input)}
     		if !decision.Approved {
     			decision.Reason = "the user declined"
     		}
     		parts = append(parts, call.Restart(decision))
     	}
     	if call, ok := transferMoney.Interrupted(part); ok {
     		parts = append(parts, call.Restart(Confirmation{Approved: true}))
     	}
     }
     ```
   - Inside `confirm(part.ToolRequest.Name, call.Input)`, `PolicyConfirmer.Confirm` evaluates `templates/calibration/agent_step_drift.json.tmpl` in one pass:
     - **Approve (`Approved: true`)** only when the tool call is classified as `benign` (or `failed_injection`) **and** hesitation is in the **Clear** band ($H / \ln K < 16\%$).
     - **Deny (`Approved: false, Reason: "the user declined"`)** when the call is classified as `hijacked` / `injection_point` or hesitation is high. Genkit's `ToolApproval` middleware converts the denial into a soft tool error (`tool.Fail`) so the agent sees the refusal reason on `ai.WithResume(parts...)` while sibling approved tools (`transferMoney`) execute normally.

---

## How to Run

### 1. Offline Unit Test (No GPU Required)

```bash
cd examples/genkit-tool-approval
go test -v ./...
```

### 2. Live on Local Apple Silicon Metal (`diffgemma`)

Make sure `diffgemma serve` is running on `127.0.0.1:8080` (from the repo root: `DIFFGEMMA_CTX=4096 ./scripts/serve.sh`), then run:

```bash
cd examples/genkit-tool-approval
go run .
```

Expected output:

```text
==> 1. Running templates/support_triage.json.tmpl via Genkit systemonex...
   • Latency: wall=2150ms (max_hesitation=68.5%)
   • sentiment  type=score  choice=            score=1.01 noul=0.000 conf=99.0% hesitation=5.2%
   • team       type=choice choice=billing     score=0.00 noul=0.000 conf=100.0% hesitation=0.2%
   • urgent     type=noul   choice=            score=0.00 noul=0.818 conf=0.0% hesitation=68.5%

==> 2. Running Genkit v1.14.0 ToolApproval + transferMoney interrupt loop with dgem policy confirm()...
   • Turn 1 finishReason="interrupted" interrupts=2
   • dgem policy audit on exportCustomerData: template=calibration/agent_step_drift slot=drift choice="hijacked" confidence=99.1% hesitation=4.1% wall=4401ms
   • Turn 2 finishReason="stop" response="Completed turn after tool results: [exportCustomerData={\"error\":\"the tool call was denied; do not retry it or work around the denial, continue without it or tell the user it was refused. Reason: the user declined\"} transferMoney={\"amount\":750,\"status\":\"completed\",\"toAccount\":\"acct-42\",\"transactionId\":\"tx-9481\"}]"
```

### 3. Stopping the Local Server

From the repository root:

```bash
make stop
# or:
./scripts/stop_server.sh
```
