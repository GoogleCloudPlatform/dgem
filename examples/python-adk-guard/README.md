# Google ADK, LangGraph, and Gemini CLI Guard (`python-adk-guard`)

Self-contained Python standard-library recipe (~260 lines, zero extra dependencies) showing how to connect **Google Agent Development Kit (ADK)**, **LangGraph**, and **Gemini CLI (`BeforeTool` hooks)** to **`dgem`** while keeping all calibration and Stage-2 cascade logic server-side on `dgem serve`.

## Design Principles

1. **MCP-first or async-safe `POST /api/decide` (zero duplicated client calibration):**
   - Attach `dgem mcp --remote https://<your-dgem-gateway>/mcp` directly via ADK `MCPToolset` (`build_adk_mcp_toolset_command`), or call `POST /api/decide` on `dgem serve` (`evaluate_tool_call_async`) so per-policy temperature scaling and `cascade_mode: "entropy"` (`cascade_threshold_mode: "normalized"`, `0.16`) stay in one server-side implementation.
2. **Non-blocking `asyncio` execution (`Runner.run_async` & `ainvoke`):**
   - Dispatches `urllib.request` calls inside `asyncio.to_thread` so ADK `before_tool_callback` and LangGraph async tool wrappers never block the running event loop.
3. **Stage-0 `< 1 ms` deterministic static pre-filter:**
   - Fast-allows read-only workspace inspection tools (`read_file`, `glob`, `git status`, `pytest`) without network round-trips and fast-denies destructive or secret-exfiltration patterns (`rm -rf /`, `git push --force`, `@.env`, hook manifest edits) before calling [`templates/tool_safety_gate.json.tmpl`](../../templates/tool_safety_gate.json.tmpl).
4. **Endpoint & credential hygiene:**
   - Recognizes Vertex AI dedicated `/invoke/*` paths (`is_serving_endpoint_url`), sending `X-DGem-Key` (`POST .../invoke/predict` with `{"health": true}` for readiness) to serving endpoints and `X-DGem-Api-Key` (`GET /health`) only to matching `DGEM_GATEWAY_URL` origins, stripping credentials permanently across cross-origin redirects.
   - Groups `function_call` and `function_response` turns as indivisible units (`group_adk_turns_preserving_tool_pairs`) before context pruning with [`templates/agent_context_pruning.json.tmpl`](../../templates/agent_context_pruning.json.tmpl).

## Quick Start

### 1. Run Offline Unit & Async Event-Loop Tests

```bash
python3 -m unittest -v examples/python-adk-guard/test_adk_guard.py
```

### 2. Google ADK (`LlmAgent` + `MCPToolset`)

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from adk_guard import GuardConfig, build_adk_mcp_toolset_command, create_adk_before_tool_callback

cfg = GuardConfig(gateway_url="https://<your-dgem-gateway>", cascade_mode="entropy")
mcp_cmd = build_adk_mcp_toolset_command("https://<your-dgem-gateway>")

agent = LlmAgent(
    model="gemini-3.8-flash",
    name="ops_assistant",
    before_tool_callback=create_adk_before_tool_callback(cfg),
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command=mcp_cmd["command"],
                args=mcp_cmd["args"],
            )
        )
    ],
)
```

### 3. Gemini CLI (`BeforeTool` Extension Hook)

Copy [`gemini-extension.example.json`](gemini-extension.example.json) into `~/.gemini/extensions/dgem-safety-guard/gemini-extension.json` and point `DGEM_GATEWAY_URL` at your running `dgem serve` gateway. Malformed `stdin` payloads fail closed (`"deny"`), while transient gateway timeouts degrade safely to interactive confirmation (`"ask"`).
