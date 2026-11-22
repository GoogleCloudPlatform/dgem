# Google ADK, LangGraph, and Gemini CLI Guard (`python-adk-guard`)

Self-contained single-file Python standard-library recipe (`adk_guard.py`, zero third-party dependencies, Python 3.10+) showing how to connect **Google Agent Development Kit (ADK)**, **LangGraph**, and **Gemini CLI (`BeforeTool` hooks)** to **`dgem`** while keeping all calibration and Stage-2 cascade logic server-side on `dgem serve`.

## Design Principles

1. **MCP-first or async-safe `POST /api/decide` (zero duplicated client calibration):**
   - Attach `dgem mcp --remote https://<your-dgem-gateway>/mcp` directly via ADK `McpToolset` (`build_adk_mcp_toolset_command`), or call `POST /api/decide` (`GatewayDecideRequest` `"variables"`, template ID `"tool_safety_gate"`) on `dgem serve` (`evaluate_tool_call_async`) so per-policy temperature scaling and `cascade_mode: "entropy"` (`cascade_threshold_mode: "normalized"`, `0.16`) stay in one server-side implementation, while unescalated high-hesitation or failed-cascade responses fail safe to `"ask"`.
2. **Non-blocking `asyncio` execution (`Runner.run_async` & `ainvoke`):**
   - Dispatches `urllib.request` calls inside `asyncio.to_thread` so ADK `before_tool_callback` (`LlmAgent` and `BasePlugin`, reading `ToolContext.user_content`, `ToolContext.session.events` tool outputs, and `tool_confirmation` on resume) and LangGraph async tool wrappers never block the running event loop.
3. **Stage-0 `< 1 ms` deterministic static pre-filter:**
   - Fast-allows strictly read-only workspace inspection (`read_file`, `glob`, `pwd`, `git status` with read-only flags, excluding `.env` / `/proc/self/environ` / SSH / cloud credential paths and untrusted-output contexts) without network round-trips and fast-denies destructive or secret-exfiltration patterns (`rm -rf /`, `sudo`, `git push --force`, `@.env`, hook manifest edits) before calling [`templates/tool_safety_gate.json.tmpl`](../../templates/tool_safety_gate.json.tmpl).
4. **Endpoint readiness & credential hygiene:**
   - Probes true model readiness (`vllm_ready is True` and HTTP `200`, rejecting `503` during warmup) across every `dgem` topology: `POST` `{"ready": true}` directly to `:rawPredict`, `POST` `{"ready": true}` to `/invoke/predict` on Vertex dedicated `/invoke/*` routes, `GET /ready` on `/v1` serving containers, and `GET /health` on `dgem serve` gateways.
   - Sends `X-DGem-Key` (`DGEM_SERVER_KEY`) only to serving endpoints and `X-DGem-Api-Key` (`DGEM_GATEWAY_KEY`) only to matching `DGEM_GATEWAY_URL` origins, refusing cross-origin or `https` -> `http` downgrade redirects.
   - Groups single, multi-response, intermediate-thought, and chained `function_call` / `function_response` turns as indivisible units (`group_adk_turns_preserving_tool_pairs`, supporting both JSON dicts and `google.genai.types.Content`) before context pruning with [`templates/agent_context_pruning.json.tmpl`](../../templates/agent_context_pruning.json.tmpl).

## Quick Start

### 1. Run Offline Unit & Async Event-Loop Tests

```bash
python3 -m unittest -v examples/python-adk-guard/test_adk_guard.py
```

### 2. Google ADK (`LlmAgent` + `McpToolset`)

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters
from adk_guard import GuardConfig, build_adk_mcp_toolset_command, create_adk_before_tool_callback

cfg = GuardConfig(gateway_url="https://<your-dgem-gateway>", cascade_mode="entropy")
mcp_cmd = build_adk_mcp_toolset_command("https://<your-dgem-gateway>")

agent = LlmAgent(
    model="gemini-3.8-flash",
    name="ops_assistant",
    before_tool_callback=create_adk_before_tool_callback(cfg),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=mcp_cmd["command"],
                    args=mcp_cmd["args"],
                )
            )
        )
    ],
)
```

### 3. Gemini CLI (`BeforeTool` Extension Hook)

Install `adk_guard.py`, [`gemini-extension.example.json`](gemini-extension.example.json) as `~/.gemini/extensions/dgem-safety-guard/gemini-extension.json` (which allowlists `DGEM_GATEWAY_URL`, `DGEM_GATEWAY_KEY`, and `DGEM_AUTH_TOKEN` via `"settings"`), and [`hooks/hooks.example.json`](hooks/hooks.example.json) as `~/.gemini/extensions/dgem-safety-guard/hooks/hooks.json`. Malformed `stdin` payloads fail closed (`"deny"`), while transient gateway timeouts or uncertain classifications degrade safely to interactive confirmation (`"ask"` with `"systemMessage"`).
