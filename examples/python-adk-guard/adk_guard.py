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
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import math
import os
import pathlib
import sys
from typing import TypeVar
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_GATEWAY_URL: str = "http://127.0.0.1:8095"
DEFAULT_TEMPLATE_ID: str = "tool_safety_gate"
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
_SAFE_GIT_STATUS_ARGS: frozenset[str] = frozenset({
    "",
    "-s",
    "-sb",
    "-b",
    "--short",
    "--porcelain",
    "--branch",
})
_SHELL_CHAIN_TOKENS: tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    "`",
    "$(",
    ">",
    "<",
    "\n",
)
_DESTRUCTIVE_SUBSTRINGS: tuple[str, ...] = (
    "rm -rf /",
    "rm -fr /",
    "git reset --hard",
    "git clean -fd",
    "git push --force",
    "git push -f",
    "@.env",
    "hooks.json",
    "gemini-extension.json",
)
_SENSITIVE_PATH_MARKERS: tuple[str, ...] = (
    ".env",
    "/proc/self/environ",
    "/proc/1/environ",
    ".ssh/id_",
    ".ssh/config",
    ".netrc",
    ".git-credentials",
    ".kube/config",
    ".docker/config.json",
    "application_default_credentials.json",
    ".aws/credentials",
)
_VERTEX_SERVING_HOST_SUFFIXES: tuple[str, ...] = (
    ".prediction.vertexai.goog",
    ".prediction.p.vertexai.goog",
    ".aiplatform.googleapis.com",
)
_TurnT = TypeVar("_TurnT")


def _reject_nonfinite_json(constant_token: str) -> None:
    """Rejects non-standard JSON NaN and Infinity tokens."""
    raise ValueError(f"Non-standard JSON constant rejected: {constant_token}")


@dataclasses.dataclass(frozen=True)
class GuardConfig:
    """Configuration for async dgem gateway calls and Stage-0 pre-filtering."""

    gateway_url: str = DEFAULT_GATEWAY_URL
    template: str = DEFAULT_TEMPLATE_ID
    cascade_mode: str = "entropy"
    hesitation_threshold: float = DEFAULT_HESITATION_THRESHOLD
    stage2_model: str = "gemini-3.8-flash"
    stage2_prior: str = "none"
    timeout_seconds: float = 10.0
    static_stage: bool = True
    cwd: str = ""
    gateway_key: str = ""
    server_key: str = ""
    auth_token: str = ""


def sanitize_cascade_model(model_name: str) -> str:
    """Maps Stage-2 model names to GA Gemini identifiers.

    Args:
        model_name: Requested Stage-2 Gemini model identifier.

    Returns:
        Supported GA Gemini model name for dgem Stage-2 cascades.
    """
    normalized = (model_name or "").strip().lower()
    if "lite" in normalized:
        return "gemini-3.5-flash-lite"
    if normalized in ALLOWED_STAGE2_MODELS:
        return normalized
    return ALLOWED_STAGE2_MODELS[0]


def same_origin(first_url: str, second_url: str) -> bool:
    """Returns True when two HTTP(S) URLs share scheme, host, and port.

    Args:
        first_url: First candidate HTTP or HTTPS URL.
        second_url: Second candidate HTTP or HTTPS URL.

    Returns:
        True if both URLs parse cleanly and share scheme, hostname, and port.
    """
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
    """Strips trailing slashes in O(N) time without regex backtracking.

    Args:
        raw_url: Input URL string.

    Returns:
        The input string without trailing '/' characters.
    """
    end = len(raw_url)
    while end > 0 and raw_url[end - 1] == "/":
        end -= 1
    return raw_url[:end]


def is_serving_endpoint_url(target_url: str) -> bool:
    """Detects direct serving URLs (/v1, :rawPredict, or Vertex /invoke/*).

    Args:
        target_url: Target endpoint URL.

    Returns:
        True when target_url addresses a dgem serving container or Vertex
        prediction endpoint rather than a dgem serve gateway.
    """
    cleaned = strip_trailing_slashes_linear((target_url or "").strip())
    parsed = urllib.parse.urlsplit(cleaned)
    host_lower = (parsed.hostname or "").lower()
    path_lower = parsed.path.lower()
    if any(host_lower.endswith(suffix) for suffix in _VERTEX_SERVING_HOST_SUFFIXES):
        return True
    return (
        path_lower.endswith("/v1")
        or "/v1/" in path_lower
        or path_lower.endswith((":rawpredict", ":predict", ":streamrawpredict"))
        or path_lower.endswith(("/predict", "/rawpredict", "/ready", "/invoke"))
        or "/invoke/" in path_lower
    )


class _SameOriginOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuses cross-origin or scheme-downgrade redirects and keeps 307/308 bodies."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        initial_url = getattr(req, "_initial_url", req.full_url)
        if not same_origin(initial_url, resolved_url):
            raise urllib.error.HTTPError(
                resolved_url,
                code,
                "cross-origin or scheme-downgrade redirect refused",
                headers,
                fp,
            )
        if code in (307, 308) and req.data is not None:
            redirected = urllib.request.Request(
                resolved_url,
                data=req.data,
                headers=dict(req.headers),
                method=req.get_method(),
            )
            for key, val in req.unredirected_hdrs.items():
                redirected.add_unredirected_header(key, val)
            redirected._initial_url = initial_url
            return redirected
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected._initial_url = initial_url
        return redirected


_OPENER = urllib.request.build_opener(_SameOriginOnlyRedirectHandler)


def _normalize_untruncated(value: str) -> str:
    """Normalizes line endings + NFKC text without truncation and strips format/control chars."""
    line_normalized = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFKC", line_normalized)
    return "".join(
        ch
        for ch in normalized
        if ch in ("\n", "\t")
        or (
            ord(ch) >= 32
            and unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Zl", "Zp")
        )
    )


def sanitize_text_linear(value: str, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Normalizes NFKC text and caps length at max_chars in O(N) time.

    Args:
        value: Input string to normalize.
        max_chars: Maximum output character count.

    Returns:
        Normalized text truncated to max_chars.
    """
    return _normalize_untruncated(value)[:max_chars]


def _strip_untrusted_tags_linear(text: str) -> str:
    """Replaces any <untrusted_data...> or </untrusted_data...> tag in O(N) time."""
    lower = text.lower()
    needle = "untrusted_data"
    out: list[str] = []
    idx = 0
    total = len(text)
    while idx < total:
        lt_pos = lower.find("<", idx)
        if lt_pos == -1:
            out.append(text[idx:])
            break
        out.append(text[idx:lt_pos])
        cursor = lt_pos + 1
        while cursor < total and lower[cursor] in (" ", "\t", "\n", "/"):
            cursor += 1
        if lower.startswith(needle, cursor):
            gt_pos = lower.find(">", cursor + len(needle))
            if gt_pos != -1:
                out.append("[REDACTED_TAG]")
                idx = gt_pos + 1
                continue
            out.append("[REDACTED_TAG]")
            break
        out.append("<")
        idx = lt_pos + 1
    return "".join(out)


def _format_untrusted_output(untrusted_tool_output: str) -> str:
    """Wraps untrusted tool output in <untrusted_data> tags with tag neutralization."""
    cleaned = sanitize_text_linear(untrusted_tool_output, 2_000).strip()
    if not cleaned or cleaned == "none":
        return "none"
    neutralized = _strip_untrusted_tags_linear(cleaned)
    return f"<untrusted_data>\n{neutralized}\n</untrusted_data>"


def _has_sudo_token(collapsed_lower: str) -> bool:
    """Detects standalone sudo command or path tokens in O(N) time."""
    current: list[str] = []
    for ch in collapsed_lower:
        if ch.isalnum() or ch in ("_", "-"):
            current.append(ch)
        else:
            if current and "".join(current) == "sudo":
                return True
            current.clear()
    return bool(current and "".join(current) == "sudo")


def _collect_normalized_leaves(value: object, depth: int = 0) -> list[str]:
    """Collects normalized string leaves from nested tool args in O(N) time."""
    if depth > 12:
        return [ _normalize_untruncated(str(value)) ]
    if isinstance(value, str):
        return [_normalize_untruncated(value)]
    if isinstance(value, Mapping):
        leaves: list[str] = []
        for k, v in value.items():
            leaves.append(_normalize_untruncated(str(k)))
            leaves.extend(_collect_normalized_leaves(v, depth + 1))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        seq_leaves: list[str] = []
        for item in value:
            seq_leaves.extend(_collect_normalized_leaves(item, depth + 1))
        return seq_leaves
    return [_normalize_untruncated(str(value))]


def stage0_static_prefilter(
    tool_name: str,
    tool_args: Mapping[str, object],
    untrusted_tool_output: str = "",
) -> tuple[str, str] | None:
    """Evaluates <1 ms static allow/deny rules on full untruncated inputs.

    Args:
        tool_name: Name of the tool being invoked.
        tool_args: Mapping of tool arguments.
        untrusted_tool_output: Optional untrusted output from prior tool turns.

    Returns:
        A (decision, reason) tuple for clear static hits, or None to defer to
        dgem.
    """
    raw_tool = _normalize_untruncated(tool_name)
    raw_serialized = _normalize_untruncated(
        json.dumps(dict(tool_args), sort_keys=True, ensure_ascii=False, default=str)
    )
    joined_leaves = " ".join(_collect_normalized_leaves(tool_args))
    collapsed_leaves = " ".join(joined_leaves.lower().split())
    collapsed_serialized = " ".join(raw_serialized.lower().split())
    lowered_serialized = raw_serialized.lower()
    for blocked in _DESTRUCTIVE_SUBSTRINGS:
        if (
            blocked in lowered_serialized
            or blocked in collapsed_serialized
            or blocked in collapsed_leaves
        ):
            return ("deny", f"stage0_blocked_pattern:{blocked}")
    if _has_sudo_token(collapsed_serialized) or _has_sudo_token(collapsed_leaves):
        return ("deny", "stage0_blocked_pattern:sudo")
    if len(raw_tool) > 128 or len(raw_serialized) > MAX_FIELD_CHARS:
        return ("ask", "stage0_oversized_tool_payload")
    if any(
        marker in lowered_serialized or marker in collapsed_leaves
        for marker in _SENSITIVE_PATH_MARKERS
    ):
        return None
    if untrusted_tool_output and untrusted_tool_output.strip() not in ("", "none"):
        return None
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
        collapsed_cmd = " ".join(command_value.lower().split())
        if collapsed_cmd == "pwd":
            return ("allow", "stage0_read_only_command")
        if collapsed_cmd == "git status" or collapsed_cmd.startswith("git status "):
            suffix = collapsed_cmd[len("git status") :].strip()
            if suffix in _SAFE_GIT_STATUS_ARGS:
                return ("allow", "stage0_read_only_command")
    return None


def _build_headers(
    target_url: str,
    config: GuardConfig,
    *,
    include_content_type: bool = True,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    token = (config.auth_token or os.environ.get("DGEM_AUTH_TOKEN", "")).strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if is_serving_endpoint_url(target_url):
        raw_srv = config.server_key or os.environ.get("DGEM_SERVER_KEY", "")
        first_srv_key = raw_srv.split(",", 1)[0].strip()
        if first_srv_key:
            headers["X-DGem-Key"] = first_srv_key
    else:
        configured_gw = (
            os.environ.get("DGEM_GATEWAY_URL", "").strip() or config.gateway_url
        )
        raw_gw = config.gateway_key or os.environ.get("DGEM_GATEWAY_KEY", "")
        first_gw_key = raw_gw.split(",", 1)[0].strip()
        if first_gw_key and same_origin(target_url, configured_gw):
            headers["X-DGem-Api-Key"] = first_gw_key
    return headers


def _post_json_sync(
    url: str, payload: Mapping[str, object], config: GuardConfig
) -> tuple[int, dict[str, object]]:
    encoded = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=_build_headers(url, config, include_content_type=True),
        method="POST",
    )
    request._initial_url = url
    with _OPENER.open(request, timeout=config.timeout_seconds) as response:
        parsed = json.loads(
            response.read().decode("utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
        return int(response.status), (parsed if isinstance(parsed, dict) else {})


def _get_json_sync(url: str, config: GuardConfig) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        headers=_build_headers(url, config, include_content_type=False),
        method="GET",
    )
    request._initial_url = url
    with _OPENER.open(request, timeout=config.timeout_seconds) as response:
        parsed = json.loads(
            response.read().decode("utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
        return int(response.status), (parsed if isinstance(parsed, dict) else {})


def _resolve_readiness_probe(base_url: str) -> tuple[str, str, bool]:
    """Returns (http_method, probe_url, require_vllm_ready) for base_url."""
    base = strip_trailing_slashes_linear((base_url or "").strip())
    parsed = urllib.parse.urlsplit(base)
    host_lower = (parsed.hostname or "").lower()
    path = parsed.path
    path_lower = path.lower()

    if path_lower.endswith((":rawpredict", ":predict", ":streamrawpredict")):
        return ("POST", base, True)
    if path_lower.endswith(("/predict", "/rawpredict", "/invoke/ready", "/ready")):
        method = (
            "POST"
            if "/invoke/" in path_lower or path_lower.endswith(("/predict", "/rawpredict"))
            else "GET"
        )
        return (method, base, True)
    if path_lower.endswith("/invoke") or "/invoke/" in path_lower:
        invoke_idx = path_lower.find("/invoke")
        invoke_path = path[: invoke_idx + len("/invoke")]
        rebuilt = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, f"{invoke_path}/predict", "", "")
        )
        return ("POST", rebuilt, True)
    if any(host_lower.endswith(suffix) for suffix in _VERTEX_SERVING_HOST_SUFFIXES):
        clean_path = path[:-3] if path_lower.endswith("/v1") else path
        if host_lower.endswith(".prediction.vertexai.goog"):
            rebuilt = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, f"{clean_path}/invoke/predict", "", "")
            )
            return ("POST", rebuilt, True)
        rebuilt = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, f"{clean_path}:rawPredict", "", "")
        )
        return ("POST", rebuilt, True)
    if path_lower.endswith("/v1") or "/v1/" in path_lower:
        v1_idx = path_lower.find("/v1")
        service_prefix = path[:v1_idx]
        rebuilt = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, f"{service_prefix}/ready", "", "")
        )
        return ("GET", rebuilt, True)
    if path_lower.endswith("/api/decide"):
        base = base[: -len("/api/decide")]
    return ("GET", f"{base}/health", False)


def _probe_endpoint_ready_sync(
    config: GuardConfig, endpoint_url: str = ""
) -> bool:
    target = endpoint_url or config.gateway_url
    method, probe_url, require_vllm_ready = _resolve_readiness_probe(target)
    try:
        if method == "POST":
            status_code, reply = _post_json_sync(probe_url, {"ready": True}, config)
        else:
            status_code, reply = _get_json_sync(probe_url, config)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        RecursionError,
    ):
        return False
    if status_code != 200 or not isinstance(reply, Mapping):
        return False
    if require_vllm_ready or "vllm_ready" in reply:
        return reply.get("vllm_ready") is True
    return reply.get("status") == "ok"


async def probe_endpoint_ready_async(
    config: GuardConfig, endpoint_url: str = ""
) -> bool:
    """Probes gateway /health or serving /ready, /invoke/predict, or :rawPredict.

    Args:
        config: Guard configuration containing credentials and timeout.
        endpoint_url: Optional override URL to probe instead of config.gateway_url.

    Returns:
        True only when the endpoint responds HTTP 200 and reports readiness.
    """
    return await asyncio.to_thread(_probe_endpoint_ready_sync, config, endpoint_url)


def _slot_hesitation(
    slot_obj: Mapping[str, object],
    diagnostics: Mapping[str, object] | None,
    slot_id: str,
) -> float:
    """Computes normalized Shannon hesitation H / ln(K) in [0.0, 1.0]."""
    if isinstance(diagnostics, Mapping):
        questions = diagnostics.get("questions")
        if isinstance(questions, Mapping):
            slot_diag = questions.get(slot_id)
            if isinstance(slot_diag, Mapping):
                norm_ent = slot_diag.get("normalized_entropy")
                if isinstance(norm_ent, (int, float)) and not isinstance(norm_ent, bool):
                    if math.isfinite(float(norm_ent)):
                        return max(0.0, min(1.0, float(norm_ent)))
    probs_obj = slot_obj.get("probabilities")
    if not isinstance(probs_obj, Mapping) or not probs_obj:
        return 0.0
    positive: list[float] = []
    for raw_val in probs_obj.values():
        if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
            val = float(raw_val)
            if math.isfinite(val) and val > 0.0:
                positive.append(val)
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    option_count = max(len(probs_obj), 2)
    entropy_nats = -sum((p / total) * math.log(p / total) for p in positive)
    return max(0.0, min(1.0, entropy_nats / math.log(option_count)))


def _decide_via_gateway_sync(
    tool_name: str,
    tool_args: Mapping[str, object],
    last_user_input: str,
    untrusted_tool_output: str,
    config: GuardConfig,
) -> dict[str, object]:
    base = strip_trailing_slashes_linear(config.gateway_url)
    if base.lower().endswith("/api/decide"):
        base = base[: -len("/api/decide")]
    if is_serving_endpoint_url(base):
        return {
            "decision": "ask",
            "reason": "serving_url_requires_dgem_serve_gateway",
            "stage": "fallback",
        }
    serialized_command = _normalize_untruncated(
        json.dumps(dict(tool_args), sort_keys=True, ensure_ascii=False, default=str)
    )
    if len(serialized_command) > MAX_FIELD_CHARS:
        return {
            "decision": "ask",
            "reason": "oversized_tool_payload",
            "stage": "fallback",
        }
    resolved_cwd = config.cwd or os.getcwd()
    stage2_model = sanitize_cascade_model(config.stage2_model)
    request_body: dict[str, object] = {
        "template": config.template,
        "variables": {
            "tool_name": sanitize_text_linear(tool_name, 128),
            "cwd": sanitize_text_linear(resolved_cwd, 512),
            "command": serialized_command,
            "last_user_input": sanitize_text_linear(last_user_input, 2_000),
            "untrusted_tool_output": _format_untrusted_output(untrusted_tool_output),
        },
        "cascade_mode": config.cascade_mode,
        "cascade_threshold_mode": "normalized",
        "cascade_threshold": config.hesitation_threshold,
        "cascade_model": stage2_model,
        "stage2_prior": config.stage2_prior,
    }
    try:
        _, reply = _post_json_sync(f"{base}/api/decide", request_body, config)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        RecursionError,
    ) as exc:
        return {
            "decision": "ask",
            "reason": f"gateway_unavailable:{type(exc).__name__}",
            "stage": "fallback",
        }
    if not isinstance(reply, Mapping):
        return {
            "decision": "ask",
            "reason": "malformed_gateway_reply",
            "stage": "fallback",
        }
    cascade_summary = reply.get("cascade")
    cascade_triggered = False
    if isinstance(cascade_summary, Mapping):
        if cascade_summary.get("error"):
            return {
                "decision": "ask",
                "reason": "stage2_cascade_error",
                "stage": "fallback",
            }
        cascade_triggered = bool(cascade_summary.get("triggered"))
    answers = reply.get("answers")
    if not isinstance(answers, Mapping):
        return {
            "decision": "ask",
            "reason": "malformed_gateway_reply",
            "stage": "fallback",
        }
    diagnostics = reply.get("diagnostics")
    diag_map = diagnostics if isinstance(diagnostics, Mapping) else None
    hazard_obj = answers.get("action_hazard")
    origin_obj = answers.get("intent_origin")
    hazard = hazard_obj.get("choice", "") if isinstance(hazard_obj, Mapping) else ""
    origin = origin_obj.get("choice", "") if isinstance(origin_obj, Mapping) else ""
    if hazard == "destructive_or_exfil" or origin == "hijacked_by_tool_output":
        return {"decision": "deny", "reason": f"dgem:{hazard}:{origin}", "stage": "dgem"}
    if hazard == "benign_workspace" and origin == "user_authorized":
        if not cascade_triggered:
            hazard_h = (
                _slot_hesitation(hazard_obj, diag_map, "action_hazard")
                if isinstance(hazard_obj, Mapping)
                else 0.0
            )
            origin_h = (
                _slot_hesitation(origin_obj, diag_map, "intent_origin")
                if isinstance(origin_obj, Mapping)
                else 0.0
            )
            hesitation = max(hazard_h, origin_h)
            if hesitation >= config.hesitation_threshold:
                return {
                    "decision": "ask",
                    "reason": f"dgem:high_hesitation:{hesitation:.3f}",
                    "stage": "dgem",
                }
        return {"decision": "allow", "reason": "dgem:benign_authorized", "stage": "dgem"}
    return {
        "decision": "ask",
        "reason": f"dgem:{hazard or 'unsure'}:{origin or 'unsure'}",
        "stage": "dgem",
    }


async def evaluate_tool_call_async(
    tool_name: str,
    tool_args: Mapping[str, object],
    last_user_input: str = "",
    config: GuardConfig = GuardConfig(),
    untrusted_tool_output: str = "",
) -> dict[str, object]:
    """Evaluates a tool call asynchronously without blocking asyncio loops.

    Args:
        tool_name: Name of the candidate tool.
        tool_args: Tool call arguments mapping.
        last_user_input: Explicit user message for the current invocation.
        config: Guard configuration.
        untrusted_tool_output: Optional untrusted output from prior tool calls.

    Returns:
        A dict containing 'decision' ('allow' | 'ask' | 'deny'), 'reason', and
        'stage'.
    """
    if config.static_stage:
        fast_verdict = stage0_static_prefilter(
            tool_name, tool_args, untrusted_tool_output=untrusted_tool_output
        )
        if fast_verdict is not None:
            return {
                "decision": fast_verdict[0],
                "reason": fast_verdict[1],
                "stage": "stage0",
            }
    return await asyncio.to_thread(
        _decide_via_gateway_sync,
        tool_name,
        tool_args,
        last_user_input,
        untrusted_tool_output,
        config,
    )


def build_adk_mcp_toolset_command(
    remote_gateway_url: str, dgem_bin: str = "dgem"
) -> dict[str, object]:
    """Returns stdio MCP connection parameters for ADK McpToolset and LangGraph.

    Args:
        remote_gateway_url: Base URL of a running dgem serve gateway.
        dgem_bin: Path or binary name of the dgem CLI.

    Returns:
        Dictionary with 'command' and 'args' for StdioServerParameters.
    """
    base = strip_trailing_slashes_linear(remote_gateway_url)
    mcp_url = base if base.endswith("/mcp") else f"{base}/mcp"
    return {"command": dgem_bin, "args": ["mcp", "--remote", mcp_url]}


def _extract_adk_user_text(tool_context: object) -> str:
    """Extracts user text from Google ADK ToolContext.user_content or user_input."""
    user_content = getattr(tool_context, "user_content", None)
    if isinstance(user_content, str) and user_content.strip():
        return user_content
    parts = (
        user_content.get("parts")
        if isinstance(user_content, Mapping)
        else getattr(user_content, "parts", None)
    )
    if isinstance(parts, Sequence) and not isinstance(parts, (str, bytes)):
        chunks: list[str] = []
        for part in parts:
            text_val = (
                part.get("text")
                if isinstance(part, Mapping)
                else getattr(part, "text", None)
            )
            if isinstance(text_val, str) and text_val:
                chunks.append(text_val)
        if chunks:
            return "\n".join(chunks)
    fallback = getattr(tool_context, "user_input", "")
    return fallback if isinstance(fallback, str) else ""


def _extract_adk_untrusted_tool_output(tool_context: object) -> str:
    """Extracts recent function_response payloads from ADK session.events."""
    explicit = getattr(tool_context, "untrusted_tool_output", "")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    session = getattr(tool_context, "session", None)
    events = getattr(session, "events", None)
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return ""
    snippets: list[str] = []
    for event in events[-8:]:
        content = (
            event.get("content")
            if isinstance(event, Mapping)
            else getattr(event, "content", None)
        )
        parts = (
            content.get("parts")
            if isinstance(content, Mapping)
            else getattr(content, "parts", None)
        )
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
            continue
        for part in parts:
            func_resp = (
                part.get("function_response")
                if isinstance(part, Mapping)
                else getattr(part, "function_response", None)
            )
            if func_resp is not None:
                resp_payload = (
                    func_resp.get("response", func_resp)
                    if isinstance(func_resp, Mapping)
                    else getattr(func_resp, "response", func_resp)
                )
                snippets.append(
                    json.dumps(resp_payload, ensure_ascii=False, default=str)[:600]
                )
    return "\n".join(snippets[-4:])


def create_adk_before_tool_callback(
    config: GuardConfig = GuardConfig(),
) -> Callable[..., Awaitable[dict[str, object] | None]]:
    """Builds an async before_tool_callback compatible with ADK LlmAgent & BasePlugin.

    Args:
        config: Guard configuration.

    Returns:
        Async callback accepting ADK LlmAgent or bound BasePlugin arguments.
    """

    async def _before_tool(
        *positional: object, **keyword: object
    ) -> dict[str, object] | None:
        if "tool" in keyword:
            tool = keyword["tool"]
        elif len(positional) == 3:
            tool = positional[0]
        elif len(positional) >= 2 and hasattr(positional[0], "before_tool_callback"):
            tool = positional[1]
        elif positional:
            tool = positional[0]
        else:
            tool = None

        raw_args = (
            keyword["args"]
            if keyword.get("args") is not None
            else keyword.get("tool_args")
        )
        if raw_args is None:
            for candidate in positional[1:]:
                if isinstance(candidate, Mapping):
                    raw_args = candidate
                    break
        resolved_args: Mapping[str, object] = (
            raw_args if isinstance(raw_args, Mapping) else {}
        )

        if "tool_context" in keyword:
            tool_context = keyword["tool_context"]
        elif len(positional) >= 3:
            tool_context = positional[-1]
        else:
            tool_context = None

        confirmation = getattr(tool_context, "tool_confirmation", None)
        if confirmation is not None:
            if getattr(confirmation, "confirmed", False) is True:
                return None
            return {
                "status": "blocked_by_dgem_guard",
                "decision": "deny",
                "reason": "user_rejected_confirmation",
            }

        tool_name = getattr(tool, "name", None) or str(tool or "")
        user_text = _extract_adk_user_text(tool_context)
        untrusted_out = _extract_adk_untrusted_tool_output(tool_context)
        verdict = await evaluate_tool_call_async(
            str(tool_name),
            resolved_args,
            last_user_input=user_text,
            config=config,
            untrusted_tool_output=untrusted_out,
        )
        if verdict["decision"] == "allow":
            return None
        return {
            "status": "blocked_by_dgem_guard",
            "decision": verdict["decision"],
            "reason": verdict["reason"],
        }

    return _before_tool


def _extract_turn_tool_tokens(turn: object) -> tuple[list[str], list[str]]:
    """Extracts (call_keys, response_keys) for a dict or ADK Content turn."""
    parts = (
        turn.get("parts")
        if isinstance(turn, Mapping)
        else getattr(turn, "parts", None)
    )
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return ([], [])
    calls: list[str] = []
    responses: list[str] = []
    for idx, part in enumerate(parts):
        fc = (
            part.get("function_call")
            if isinstance(part, Mapping)
            else getattr(part, "function_call", None)
        )
        if fc is not None:
            cid = (
                fc.get("id") or fc.get("name")
                if isinstance(fc, Mapping)
                else (getattr(fc, "id", None) or getattr(fc, "name", None))
            )
            calls.append(str(cid or f"call_{idx}"))
        fr = (
            part.get("function_response")
            if isinstance(part, Mapping)
            else getattr(part, "function_response", None)
        )
        if fr is not None:
            rid = (
                fr.get("id") or fr.get("name")
                if isinstance(fr, Mapping)
                else (getattr(fr, "id", None) or getattr(fr, "name", None))
            )
            responses.append(str(rid or f"resp_{idx}"))
    return (calls, responses)


def group_adk_turns_preserving_tool_pairs(
    contents: Sequence[_TurnT],
) -> list[list[_TurnT]]:
    """Groups ADK Content turns so function_call and function_response stay paired.

    Merges spans across intermediate model turns and attaches leading orphaned
    function_response turns to their neighboring group so pruning never leaves
    an unmatched function_call or function_response turn.

    Args:
        contents: Ordered sequence of ADK conversation turns.

    Returns:
        List of indivisible turn groups safe for context-pruning selection.
    """
    total = len(contents)
    if total == 0:
        return []
    parent = list(range(total))

    def find_root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def merge_span(start_idx: int, end_idx: int) -> None:
        base_root = find_root(start_idx)
        for pos in range(start_idx + 1, end_idx + 1):
            pos_root = find_root(pos)
            if pos_root != base_root:
                lo, hi = (base_root, pos_root) if base_root < pos_root else (pos_root, base_root)
                parent[hi] = lo
                base_root = lo

    open_call_turns: dict[str, int] = {}
    last_call_idx: int | None = None
    for idx, turn in enumerate(contents):
        call_keys, resp_keys = _extract_turn_tool_tokens(turn)
        if resp_keys:
            matched = False
            for rkey in resp_keys:
                if rkey in open_call_turns:
                    merge_span(open_call_turns[rkey], idx)
                    matched = True
            if not matched and last_call_idx is not None:
                merge_span(last_call_idx, idx)
            elif not matched and idx + 1 < total:
                merge_span(idx, idx + 1)
        for ckey in call_keys:
            open_call_turns[ckey] = idx
            last_call_idx = idx

    buckets: dict[int, list[_TurnT]] = {}
    for idx, turn in enumerate(contents):
        root = find_root(idx)
        buckets.setdefault(root, []).append(turn)
    return [buckets[root] for root in sorted(buckets)]


def _read_transcript_context(
    transcript_path: str,
) -> tuple[str, str]:
    """Reads (last_user_input, untrusted_tool_output) from a Gemini CLI transcript."""
    if not transcript_path:
        return ("", "")
    try:
        resolved = pathlib.Path(os.path.expanduser(transcript_path))
        if not resolved.is_file() or resolved.stat().st_size > 2_097_152:
            return ("", "")
        raw_text = resolved.read_text(encoding="utf-8", errors="replace").strip()
        if not raw_text:
            return ("", "")
        user_snippets: list[str] = []
        tool_snippets: list[str] = []
        if raw_text.startswith("{") and "\n" not in raw_text[:200]:
            doc = json.loads(raw_text)
            messages = doc.get("messages", []) if isinstance(doc, Mapping) else []
            if isinstance(messages, Sequence):
                for msg in messages[-12:]:
                    if not isinstance(msg, Mapping):
                        continue
                    role = str(msg.get("type") or msg.get("role") or "").lower()
                    content = json.dumps(msg.get("content", ""), ensure_ascii=False, default=str)
                    if role in ("user", "human"):
                        user_snippets.append(content[:800])
                    elif "tool" in role or "function" in role:
                        tool_snippets.append(content[:600])
        else:
            for line in raw_text.splitlines()[-12:]:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, Mapping):
                    continue
                step_type = str(obj.get("type") or obj.get("role") or "").upper()
                content = str(obj.get("content") or "").strip()
                if step_type in ("USER_INPUT", "USER") and content:
                    user_snippets.append(content[:800])
                elif step_type in ("TOOL_RESPONSE", "TOOL_RESULT", "GENERIC") and content:
                    tool_snippets.append(content[:600])
        return (
            user_snippets[-1] if user_snippets else "",
            "\n".join(tool_snippets[-4:]),
        )
    except (OSError, ValueError, TypeError, RecursionError):
        return ("", "")


def evaluate_gemini_cli_hook_stdin(
    raw_stdin: bytes, config: GuardConfig = GuardConfig()
) -> dict[str, str]:
    """Evaluates a Gemini CLI BeforeTool JSON payload fail-closed on parse errors.

    Args:
        raw_stdin: Raw UTF-8 JSON bytes read from stdin.
        config: Guard configuration.

    Returns:
        Hook response dictionary with 'decision', 'reason', and 'systemMessage'.
    """
    if not raw_stdin or len(raw_stdin) > MAX_STDIN_BYTES:
        return {
            "decision": "deny",
            "reason": "invalid_or_oversized_hook_stdin",
            "systemMessage": "invalid_or_oversized_hook_stdin",
        }
    try:
        payload = json.loads(
            raw_stdin.decode("utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        return {
            "decision": "deny",
            "reason": "malformed_hook_json",
            "systemMessage": "malformed_hook_json",
        }
    if not isinstance(payload, Mapping):
        return {
            "decision": "deny",
            "reason": "hook_payload_not_object",
            "systemMessage": "hook_payload_not_object",
        }
    tool_call_obj = payload.get("toolCall")
    raw_tool_name = payload.get("tool_name") or (
        tool_call_obj.get("name") if isinstance(tool_call_obj, Mapping) else ""
    )
    if not isinstance(raw_tool_name, str):
        return {
            "decision": "deny",
            "reason": "missing_tool_name",
            "systemMessage": "missing_tool_name",
        }
    cleaned_tool_name = _normalize_untruncated(raw_tool_name).strip()
    if not cleaned_tool_name:
        return {
            "decision": "deny",
            "reason": "missing_tool_name",
            "systemMessage": "missing_tool_name",
        }

    if isinstance(payload.get("tool_input"), Mapping):
        args: Mapping[str, object] = payload["tool_input"]  # type: ignore[assignment]
    elif isinstance(tool_call_obj, Mapping) and isinstance(
        tool_call_obj.get("args"), Mapping
    ):
        args = tool_call_obj["args"]  # type: ignore[assignment]
    else:
        args = {}

    raw_cwd = payload.get("cwd")
    effective_cfg = (
        dataclasses.replace(config, cwd=raw_cwd)
        if isinstance(raw_cwd, str) and raw_cwd.strip()
        else config
    )
    raw_user_input = payload.get("lastUserInput") or payload.get("user_input") or ""
    user_input = raw_user_input if isinstance(raw_user_input, str) else ""
    raw_untrusted = (
        payload.get("untrustedToolOutput")
        or payload.get("untrusted_tool_output")
        or ""
    )
    untrusted_output = raw_untrusted if isinstance(raw_untrusted, str) else ""
    raw_transcript = payload.get("transcript_path") or payload.get("transcriptPath") or ""
    if (not user_input or not untrusted_output) and isinstance(raw_transcript, str):
        tx_user, tx_untrusted = _read_transcript_context(raw_transcript)
        if not user_input:
            user_input = tx_user
        if not untrusted_output:
            untrusted_output = tx_untrusted

    if effective_cfg.static_stage:
        fast_verdict = stage0_static_prefilter(
            cleaned_tool_name, args, untrusted_tool_output=untrusted_output
        )
        if fast_verdict is not None:
            out_fast = {"decision": fast_verdict[0], "reason": fast_verdict[1]}
            if fast_verdict[0] != "allow":
                out_fast["systemMessage"] = fast_verdict[1]
            return out_fast
    verdict = _decide_via_gateway_sync(
        cleaned_tool_name,
        args,
        user_input,
        untrusted_output,
        effective_cfg,
    )
    decision_str = str(verdict["decision"])
    reason_str = str(verdict["reason"])
    out = {"decision": decision_str, "reason": reason_str}
    if decision_str != "allow":
        out["systemMessage"] = reason_str
    return out


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
