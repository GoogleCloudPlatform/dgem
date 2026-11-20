#!/usr/bin/env python3
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
"""Async-safe Google ADK, LangGraph, and Gemini CLI guardrails backed by dgem."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import dataclasses
import json
import os
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_GATEWAY_URL: str = "http://127.0.0.1:8095"
DEFAULT_HESITATION_THRESHOLD: float = 0.16
MAX_STDIN_BYTES: int = 65_536
MAX_FIELD_CHARS: int = 8_000
ALLOWED_STAGE2_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
)

_SAFE_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "view_file",
    "glob",
    "list_dir",
    "list_directory",
    "search_files",
})
_SAFE_SHELL_PREFIXES: tuple[str, ...] = (
    "git status",
    "git log",
    "git diff",
    "pytest",
    "go test",
    "go vet",
    "python3 -m unittest",
)
_SHELL_CHAIN_TOKENS: tuple[str, ...] = ("&&", "||", ";", "|", "`", "$(", ">", "<", "\n")
_DESTRUCTIVE_SUBSTRINGS: tuple[str, ...] = (
    "rm -rf /",
    "git reset --hard",
    "git push --force",
    "git push -f",
    "@.env",
    "hooks.json",
    "gemini-extension.json",
)


@dataclasses.dataclass(frozen=True)
class GuardConfig:
    """Configuration for async dgem gateway calls and Stage-0 pre-filtering."""

    gateway_url: str = DEFAULT_GATEWAY_URL
    template: str = "templates/tool_safety_gate.json.tmpl"
    cascade_mode: str = "entropy"
    hesitation_threshold: float = DEFAULT_HESITATION_THRESHOLD
    stage2_model: str = "gemini-3.8-flash"
    stage2_prior: str = "none"
    timeout_seconds: float = 10.0
    static_stage: bool = True
    gateway_key: str = ""
    server_key: str = ""
    auth_token: str = ""


def same_origin(first_url: str, second_url: str) -> bool:
    """Returns True when two HTTP(S) URLs share scheme, host, and port."""
    parsed_a = urllib.parse.urlsplit((first_url or "").strip())
    parsed_b = urllib.parse.urlsplit((second_url or "").strip())
    scheme_a, scheme_b = parsed_a.scheme.lower(), parsed_b.scheme.lower()
    if scheme_a not in ("http", "https") or scheme_a != scheme_b:
        return False
    if not parsed_a.hostname or not parsed_b.hostname:
        return False
    if parsed_a.hostname.lower() != parsed_b.hostname.lower():
        return False
    port_a = parsed_a.port or (443 if scheme_a == "https" else 80)
    port_b = parsed_b.port or (443 if scheme_b == "https" else 80)
    return port_a == port_b


def strip_trailing_slashes_linear(raw_url: str) -> str:
    """Strips trailing slashes in O(N) time without regex backtracking."""
    end = len(raw_url)
    while end > 0 and raw_url[end - 1] == "/":
        end -= 1
    return raw_url[:end]


def is_serving_endpoint_url(target_url: str) -> bool:
    """Detects direct serving URLs (/v1, :rawPredict, or Vertex /invoke/*)."""
    parsed = urllib.parse.urlsplit(strip_trailing_slashes_linear((target_url or "").strip()))
    path_lower = parsed.path.lower()
    return (
        path_lower.endswith("/v1")
        or path_lower.endswith(":rawpredict")
        or path_lower.endswith("/invoke")
        or "/invoke/" in path_lower
    )


class _OriginScopedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Strips auth and dgem keys permanently when a redirect leaves origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        initial_url = getattr(req, "_initial_url", req.full_url)
        crossed = getattr(req, "_crossed_origin", False) or not same_origin(
            initial_url, redirected.full_url
        )
        redirected._initial_url = initial_url
        redirected._crossed_origin = crossed
        if crossed:
            for hdrs in (redirected.headers, redirected.unredirected_hdrs):
                drop = [
                    key
                    for key in hdrs
                    if key.lower() in ("authorization", "x-dgem-key", "x-dgem-api-key")
                ]
                for key in drop:
                    del hdrs[key]
        return redirected


_OPENER = urllib.request.build_opener(_OriginScopedRedirectHandler)


def _normalize_untruncated(value: str) -> str:
    """Normalizes NFKC text without truncation and strips control characters."""
    normalized = unicodedata.normalize("NFKC", value or "")
    return "".join(
        ch for ch in normalized if ch in ("\n", "\t") or ord(ch) >= 32
    )


def sanitize_text_linear(value: str, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Normalizes NFKC text and caps length at max_chars in O(N) time."""
    return _normalize_untruncated(value)[:max_chars]


def stage0_static_prefilter(
    tool_name: str, tool_args: Mapping[str, object]
) -> tuple[str, str] | None:
    """Evaluates <1 ms static allow/deny rules on full untruncated inputs."""
    raw_tool = _normalize_untruncated(tool_name)
    raw_serialized = _normalize_untruncated(
        json.dumps(dict(tool_args), sort_keys=True, ensure_ascii=False)
    )
    collapsed_serialized = " ".join(raw_serialized.lower().split())
    lowered_serialized = raw_serialized.lower()
    for blocked in _DESTRUCTIVE_SUBSTRINGS:
        if blocked in lowered_serialized or blocked in collapsed_serialized:
            return ("deny", f"stage0_blocked_pattern:{blocked}")
    if len(raw_tool) > 128 or len(raw_serialized) > MAX_FIELD_CHARS:
        return ("ask", "stage0_oversized_tool_payload")
    normalized_tool = raw_tool.strip().lower()
    if normalized_tool in _SAFE_READ_ONLY_TOOLS:
        return ("allow", "stage0_read_only_tool")
    command_value = ""
    for key in ("command", "cmd", "CommandLine"):
        candidate = tool_args.get(key)
        if isinstance(candidate, str):
            full_cmd = _normalize_untruncated(candidate)
            if len(full_cmd) > 1024:
                return None
            command_value = full_cmd.strip()
            break
    if command_value and not any(tok in command_value for tok in _SHELL_CHAIN_TOKENS):
        lowered_cmd = command_value.lower()
        if any(
            lowered_cmd == prefix or lowered_cmd.startswith(prefix + " ")
            for prefix in _SAFE_SHELL_PREFIXES
        ):
            return ("allow", "stage0_read_only_command")
    return None


def _build_headers(target_url: str, config: GuardConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = config.auth_token or os.environ.get("DGEM_AUTH_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if is_serving_endpoint_url(target_url):
        raw_srv = config.server_key or os.environ.get("DGEM_SERVER_KEY", "")
        first_key = raw_srv.split(",")[0].strip()
        if first_key:
            headers["X-DGem-Key"] = first_key
    else:
        configured_gw = os.environ.get("DGEM_GATEWAY_URL", config.gateway_url)
        gw_key = config.gateway_key or os.environ.get("DGEM_GATEWAY_KEY", "")
        if gw_key and same_origin(target_url, configured_gw):
            headers["X-DGem-Api-Key"] = gw_key
    return headers


def _post_json_sync(
    url: str, payload: Mapping[str, object], config: GuardConfig
) -> dict[str, object]:
    encoded = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url, data=encoded, headers=_build_headers(url, config), method="POST"
    )
    request._initial_url = url
    with _OPENER.open(request, timeout=config.timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


async def probe_endpoint_ready_async(config: GuardConfig) -> bool:
    """Probes gateway GET /health or Vertex POST /invoke/predict health."""
    base = strip_trailing_slashes_linear(config.gateway_url)
    if is_serving_endpoint_url(base):
        predict_url = base if base.endswith("/predict") else f"{base}/predict"
        reply = await asyncio.to_thread(
            _post_json_sync, predict_url, {"health": True}, config
        )
        return bool(reply)
    def _get_health() -> bool:
        req = urllib.request.Request(f"{base}/health", method="GET")
        with _OPENER.open(req, timeout=config.timeout_seconds) as resp:
            return int(resp.status) == 200
    return await asyncio.to_thread(_get_health)


async def evaluate_tool_call_async(
    tool_name: str,
    tool_args: Mapping[str, object],
    last_user_input: str = "",
    config: GuardConfig = GuardConfig(),
) -> dict[str, object]:
    """Evaluates a tool call asynchronously without blocking asyncio loops."""
    if config.static_stage:
        fast = stage0_static_prefilter(tool_name, tool_args)
        if fast is not None:
            return {"decision": fast[0], "reason": fast[1], "stage": "stage0"}
    stage2_model = (
        config.stage2_model
        if config.stage2_model in ALLOWED_STAGE2_MODELS
        else ALLOWED_STAGE2_MODELS[0]
    )
    base = strip_trailing_slashes_linear(config.gateway_url)
    request_body: dict[str, object] = {
        "template": config.template,
        "data": {
            "tool_name": sanitize_text_linear(tool_name, 128),
            "command": sanitize_text_linear(
                json.dumps(dict(tool_args), sort_keys=True, ensure_ascii=True)
            ),
            "last_user_input": sanitize_text_linear(last_user_input, 2000),
        },
        "cascade_mode": config.cascade_mode,
        "cascade_threshold_mode": "normalized",
        "cascade_threshold": config.hesitation_threshold,
        "cascade_model": stage2_model,
        "stage2_prior": config.stage2_prior,
    }
    try:
        reply = await asyncio.to_thread(
            _post_json_sync, f"{base}/api/decide", request_body, config
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"decision": "ask", "reason": f"gateway_unavailable:{type(exc).__name__}", "stage": "fallback"}
    answers = reply.get("answers") if isinstance(reply, Mapping) else None
    if not isinstance(answers, Mapping):
        return {"decision": "ask", "reason": "malformed_gateway_reply", "stage": "fallback"}
    hazard_obj = answers.get("action_hazard")
    origin_obj = answers.get("intent_origin")
    hazard = hazard_obj.get("choice", "") if isinstance(hazard_obj, Mapping) else ""
    origin = origin_obj.get("choice", "") if isinstance(origin_obj, Mapping) else ""
    if hazard == "destructive_or_exfil" or origin == "hijacked_by_tool_output":
        return {"decision": "deny", "reason": f"dgem:{hazard}:{origin}", "stage": "dgem"}
    if hazard == "benign_workspace" and origin == "user_authorized":
        return {"decision": "allow", "reason": "dgem:benign_authorized", "stage": "dgem"}
    return {"decision": "ask", "reason": f"dgem:{hazard or 'unsure'}:{origin or 'unsure'}", "stage": "dgem"}


def build_adk_mcp_toolset_command(
    remote_gateway_url: str, dgem_bin: str = "dgem"
) -> dict[str, object]:
    """Returns stdio MCP connection parameters for ADK MCPToolset and LangGraph."""
    base = strip_trailing_slashes_linear(remote_gateway_url)
    mcp_url = base if base.endswith("/mcp") else f"{base}/mcp"
    return {"command": dgem_bin, "args": ["mcp", "--remote", mcp_url]}


def create_adk_before_tool_callback(
    config: GuardConfig = GuardConfig(),
) -> Callable[[object, Mapping[str, object], object], Awaitable[dict[str, object] | None]]:
    """Builds an async before_tool_callback compatible with Google ADK LlmAgent."""

    async def _before_tool(
        tool: object, args: Mapping[str, object], tool_context: object
    ) -> dict[str, object] | None:
        tool_name = getattr(tool, "name", None) or str(tool)
        user_text = getattr(tool_context, "user_input", "") or ""
        verdict = await evaluate_tool_call_async(
            str(tool_name), args, last_user_input=str(user_text), config=config
        )
        if verdict["decision"] == "allow":
            return None
        return {
            "status": "blocked_by_dgem_guard",
            "decision": verdict["decision"],
            "reason": verdict["reason"],
        }

    return _before_tool


def group_adk_turns_preserving_tool_pairs(
    contents: Sequence[Mapping[str, object]],
) -> list[list[Mapping[str, object]]]:
    """Groups ADK Content dicts so function_call and function_response stay paired."""
    units: list[list[Mapping[str, object]]] = []
    idx = 0
    total = len(contents)
    while idx < total:
        current = contents[idx]
        parts = current.get("parts")
        has_call = isinstance(parts, Sequence) and any(
            isinstance(part, Mapping) and "function_call" in part for part in parts
        )
        if has_call and idx + 1 < total:
            nxt = contents[idx + 1]
            nxt_parts = nxt.get("parts")
            has_resp = isinstance(nxt_parts, Sequence) and any(
                isinstance(part, Mapping) and "function_response" in part
                for part in nxt_parts
            )
            if has_resp:
                units.append([current, nxt])
                idx += 2
                continue
        units.append([current])
        idx += 1
    return units


def evaluate_gemini_cli_hook_stdin(
    raw_stdin: bytes, config: GuardConfig = GuardConfig()
) -> dict[str, str]:
    """Evaluates a Gemini CLI BeforeTool JSON payload fail-closed on parse errors."""
    if not raw_stdin or len(raw_stdin) > MAX_STDIN_BYTES:
        return {"decision": "deny", "reason": "invalid_or_oversized_hook_stdin"}
    try:
        payload = json.loads(raw_stdin.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"decision": "deny", "reason": "malformed_hook_json"}
    if not isinstance(payload, Mapping):
        return {"decision": "deny", "reason": "hook_payload_not_object"}
    tool_call = payload.get("tool_input") if isinstance(payload.get("tool_input"), Mapping) else payload
    tool_name = payload.get("tool_name") or (
        payload.get("toolCall", {}).get("name")
        if isinstance(payload.get("toolCall"), Mapping)
        else ""
    )
    if not isinstance(tool_name, str) or not tool_name.strip():
        return {"decision": "deny", "reason": "missing_tool_name"}
    args = (
        payload.get("toolCall", {}).get("args")
        if isinstance(payload.get("toolCall"), Mapping)
        and isinstance(payload.get("toolCall", {}).get("args"), Mapping)
        else (tool_call if isinstance(tool_call, Mapping) else {})
    )
    verdict = asyncio.run(evaluate_tool_call_async(tool_name, args, config=config))
    return {"decision": str(verdict["decision"]), "reason": str(verdict["reason"])}


def main() -> int:
    """CLI entrypoint for Gemini CLI BeforeTool hooks."""
    if len(sys.argv) > 1 and sys.argv[1] == "gemini-cli-hook":
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        cfg = GuardConfig(
            gateway_url=os.environ.get("DGEM_GATEWAY_URL", DEFAULT_GATEWAY_URL)
        )
        sys.stdout.write(json.dumps(evaluate_gemini_cli_hook_stdin(raw, cfg)) + "\n")
        return 0
    sys.stderr.write("Usage: adk_guard.py gemini-cli-hook < hook_payload.json\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
