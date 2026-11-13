# `dgem` Client SDKs (`sdk/`)

Zero-runtime-dependency client libraries, post-hoc calibration primitives, and agent-framework safety guards for the `dgem` HTTP gateway (`POST /api/decide`) and serving endpoints (`POST /v1/systemone`, Vertex AI `:rawPredict`).

| SDK | Path | Runtime | Agent Adapters Included | Test Command |
| :--- | :--- | :--- | :--- | :--- |
| **Go** *(built-in)* | [`../pkg/client`](../pkg/client) | Go 1.24+ | [`../examples/dgemgenkit`](../examples/dgemgenkit) (Genkit Go) | `go test ./pkg/client/...` |
| **Python** (`dgem-sdk`) | [`./python`](./python) | Python 3.11+ (`stdlib` only) | Google ADK (`before_tool_callback`), LangGraph | `PYTHONPATH=sdk/python/src python3 -m unittest discover -s sdk/python/tests -v` |
| **TypeScript** (`@dgem/client`) | [`./ts`](./ts) | Node.js 20+ (`fetch` + `node:crypto`) | Genkit JS (`createGenkitToolGuard`, `createGenkitDecisionStep`) | `npm --prefix sdk/ts test` |

## Cross-Language Parity & Security Invariants

Every `dgem` client SDK implements the same wire, calibration, and security contracts:

1. **Per-Service Credential Isolation**: `X-DGem-Api-Key` is attached exclusively to `dgem serve` gateway calls (`POST /api/decide`, `POST /v1/systemone`), whereas `X-DGem-Key` (first entry of a comma-separated key-rotation list) is attached exclusively to serving containers (`/v1/systemone`, `:rawPredict`).
2. **Permanent Cross-Origin Redirect Stripping**: Once any HTTP redirect leaves the initial `(scheme, host, port)` origin, `Authorization`, `X-DGem-Key`, and `X-DGem-Api-Key` are stripped from that hop and every subsequent hop—including bounce-back redirects to the original host.
3. **Single-Route Vertex `:rawPredict` Support**: When `base_url` ends with `:rawPredict`, `health()` automatically dispatches `POST {"health": true}` and `decide_systemone()` / `decideSystemOne()` posts directly to the `:rawPredict` URL.
4. **Identical Calibration & Conformal Math**:
   - Post-hoc temperature scaling ($p_k^{1/T} / \sum_j p_j^{1/T}$, $T \in (0, 20]$) and cardinality-normalized hesitation ($H(p) / \ln K \in [0, 1]$).
   - Bi-encoder cosine-prior log-softmax normalization ($\tau = 0.07$ default) and log-linear prior fusion.
   - Stateless split-conformal prediction sets (`lac`, `aps`, `raps`) with byte-for-byte `HMAC-SHA256` tie-breaking parity across Go, Python, and TypeScript.
