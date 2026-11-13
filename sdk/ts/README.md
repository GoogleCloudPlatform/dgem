# `@dgem/client` — DiffusionGemma TypeScript SDK

Zero-runtime-dependency TypeScript client SDK (`Node.js >= 20`, native `fetch`), post-hoc calibration toolkit, and Genkit JS adapters for the `dgem` HTTP gateway (`POST /api/decide`) and serving endpoints (`POST /v1/systemone` and Vertex AI `:rawPredict`).

## Build and Test

```bash
npm --prefix sdk/ts install
npm --prefix sdk/ts test
```

## Quick Start (`dgem serve` Gateway)

```ts
import { DGemClient } from "@dgem/client";

const client = new DGemClient({
  baseUrl: "http://127.0.0.1:8090",
  gatewayApiKey: process.env.DGEM_GATEWAY_KEY,
});

const decision = await client.decide({
  template: "support_triage",
  variables: { ticket: "Charged twice on my April invoice" },
  cascadeMode: "entropy",
  cascadeThreshold: 0.16,
  cascadeThresholdMode: "normalized",
});

for (const [qid, answer] of Object.entries(decision.answers)) {
  console.log(qid, answer.value, answer.hesitationBand, answer.hesitation);
}
```

## Direct `/v1/systemone` & Vertex `:rawPredict`

```ts
import {
  DGemClient,
  conformalPredictionSet,
  normalizeCosinePriorLogits,
} from "@dgem/client";

const client = new DGemClient({
  baseUrl: process.env.DGEM_VERTEX_URL,
  bearerToken: process.env.DGEM_AUTH_TOKEN,
  serverKey: process.env.DGEM_SERVER_KEY,
});

const priorLogits = normalizeCosinePriorLogits(
  { billing: 0.84, sre: 0.41, account: 0.29 },
  0.07,
);

const result = await client.decideSystemOne({
  context: "Duplicate charge on VISA ending 4242",
  questions: {
    team: {
      type: "choice",
      instructions: "Select the owning support queue.",
      criteria: {
        billing: "Invoices, charges, and refunds",
        sre: "Service outages and latency",
        account: "Authentication and SSO",
      },
      priorLogits,
    },
  },
  priorWeight: 0.2,
});

const cset = conformalPredictionSet(result.answers.team.probabilities, 0.18, {
  method: "raps",
});
if (cset.shouldEscalate) {
  console.log("Escalate ambiguous candidate set:", cset.predictionSet);
}
```

## Genkit JS Tool Safety Guard

```ts
import { DGemClient, createGenkitToolGuard } from "@dgem/client";

const guard = createGenkitToolGuard(
  new DGemClient({ baseUrl: "http://127.0.0.1:8090" }),
  {
    template: "tool_safety_gate",
    hesitationCeiling: 0.16,
    failClosed: true,
  },
);

const verdict = await guard("delete_database", { target: "prod-orders" });
if (!verdict.allowed) {
  throw new Error(`${verdict.decision}: ${verdict.reason}`);
}
```

## Security and Cross-Language Parity Guarantees

- **Zero runtime dependencies**: Uses standard Web `fetch` and `node:crypto` only.
- **Per-service credential isolation**: Sends `X-DGem-Api-Key` exclusively to gateway endpoints and `X-DGem-Key` (first entry of a comma-separated rotation list) exclusively to serving endpoints.
- **Manual redirect origin guard**: Uses `redirect: "manual"` and permanently strips `Authorization`, `X-DGem-Key`, and `X-DGem-Api-Key` once any redirect leaves the initial `(protocol, hostname, port)` origin—including bounce-back redirects to the original host.
- **Byte-for-byte conformal parity**: `hmacUint64`, `hmacUniformFloat64`, `calibrateConformalQuantile`, and `conformalPredictionSet` (`lac`, `aps`, `raps`) match Go's `pkg/client/conformal.go` and Python's `dgem_sdk.calibration` identically.
