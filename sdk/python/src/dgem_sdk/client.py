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

"""Credential-isolated HTTP client for the dgem gateway and serving endpoints."""

from collections.abc import Callable, Mapping, Sequence
import http.client
import inspect
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from dgem_sdk.types import DGemHTTPError
from dgem_sdk.types import DGemTransportError
from dgem_sdk.types import DGemValidationError
from dgem_sdk.types import GatewayDecideResponse
from dgem_sdk.types import StructuredDecisionResponse
from dgem_sdk.types import SystemOneQuestion

TransportResult = tuple[int, bytes, Mapping[str, str]]
TransportCallable = Callable[
    [str, str, bytes | None, Mapping[str, str], float],
    TransportResult,
]

_CREDENTIAL_HEADER_NAMES = frozenset(
    {"authorization", "x-dgem-key", "x-dgem-api-key"}
)


def parse_url_origin(raw_url: str) -> tuple[str, str, int]:
  """Parse and validate an HTTP(S) URL into `(scheme, hostname, port)`.

  Args:
    raw_url: Target HTTP or HTTPS URL string.

  Returns:
    Normalized `(scheme, hostname, port)` tuple.

  Raises:
    DGemValidationError: If `raw_url` uses a non-HTTP(S) scheme or lacks a host.
  """
  if "\r" in raw_url or "\n" in raw_url:
    raise DGemValidationError("URL must not contain CR or LF characters")
  try:
    parsed = urllib.parse.urlparse(raw_url.strip())
    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
  except ValueError as error:
    raise DGemValidationError(f"invalid URL {raw_url!r}: {error}") from error

  if scheme not in ("http", "https") or not hostname:
    raise DGemValidationError(
        f"URL must use http:// or https:// with a hostname (got {raw_url!r})"
    )
  resolved_port = port if port is not None else (443 if scheme == "https" else 80)
  return (scheme, hostname, resolved_port)


def _validate_header_value(header_name: str, header_value: str) -> str:
  """Reject CR/LF header injection sequences in outgoing header values."""
  if "\r" in header_value or "\n" in header_value:
    raise DGemValidationError(
        f"header {header_name!r} must not contain CR or LF characters"
    )
  return header_value


class _OriginGuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
  """Strip credentials permanently once a redirect leaves the first origin."""

  def redirect_request(
      self,
      req: urllib.request.Request,
      fp: object,
      code: int,
      msg: str,
      headers: http.client.HTTPMessage,
      newurl: str,
  ) -> urllib.request.Request | None:
    resolved_url = urllib.parse.urljoin(req.full_url, newurl.replace(" ", "%20"))
    next_origin = parse_url_origin(resolved_url)
    method = req.get_method()
    if code in (307, 308) and method not in ("GET", "HEAD"):
      copied_headers = {
          key: val
          for key, val in req.header_items()
          if key.lower() not in ("host", "content-length")
      }
      redirected: urllib.request.Request | None = urllib.request.Request(
          resolved_url,
          data=req.data,
          headers=copied_headers,
          origin_req_host=req.origin_req_host,
          unverifiable=True,
          method=method,
      )
    else:
      redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
      if redirected is None:
        return None
      next_origin = parse_url_origin(redirected.full_url)

    first_origin = getattr(
        req, "_dgem_first_origin", parse_url_origin(req.full_url)
    )
    crossed_origin = bool(
        getattr(req, "_dgem_crossed_origin", False)
        or (first_origin != next_origin)
    )
    setattr(redirected, "_dgem_first_origin", first_origin)
    setattr(redirected, "_dgem_crossed_origin", crossed_origin)
    if crossed_origin:
      for header_dict in (redirected.headers, redirected.unredirected_hdrs):
        removable = [
            key
            for key in header_dict
            if key.lower() in _CREDENTIAL_HEADER_NAMES
        ]
        for key in removable:
          del header_dict[key]
    return redirected


def default_urllib_transport(
    method: str,
    url: str,
    body: bytes | None,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> TransportResult:
  """Execute an HTTP request using `urllib` with cross-origin key stripping.

  Args:
    method: HTTP verb (`"GET"` or `"POST"`).
    url: Validated HTTP(S) target URL.
    body: Optional UTF-8 encoded request body.
    headers: Outgoing request headers.
    timeout_seconds: Socket timeout in seconds.

  Returns:
    Tuple of `(status_code, response_bytes, response_headers)`.

  Raises:
    DGemHTTPError: If the endpoint returns a non-2xx HTTP response.
    DGemTransportError: If a network or timeout error occurs.
  """
  request = urllib.request.Request(
      url,
      data=body,
      headers=dict(headers),
      method=method,
  )
  opener = urllib.request.build_opener(_OriginGuardedRedirectHandler())
  try:
    with opener.open(request, timeout=timeout_seconds) as http_response:
      response_bytes = http_response.read()
      response_headers = {
          str(k): str(v) for k, v in http_response.headers.items()
      }
      return int(http_response.status), response_bytes, response_headers
  except urllib.error.HTTPError as http_err:
    error_bytes = http_err.read() if http_err.fp is not None else b""
    error_text = error_bytes.decode("utf-8", errors="replace")
    raise DGemHTTPError(
        int(http_err.code),
        http_err.reason or "request failed",
        error_text,
    ) from http_err
  except (
      urllib.error.URLError,
      http.client.HTTPException,
      TimeoutError,
      OSError,
  ) as transport_err:
    raise DGemTransportError(
        f"failed calling {method} {url}: {transport_err}"
    ) from transport_err


class DGemClient:
  """Synchronous client for the dgem HTTP gateway and serving endpoints."""

  def __init__(
      self,
      base_url: str | None = None,
      *,
      bearer_token: str | None = None,
      token_provider: Callable[[str], str | None] | None = None,
      gateway_api_key: str | None = None,
      server_key: str | None = None,
      default_backend: str = "",
      timeout_seconds: float = 60.0,
      transport: TransportCallable | None = None,
  ) -> None:
    """Initialize a credential-isolated `DGemClient`.

    Args:
      base_url: Gateway base URL (e.g. `http://localhost:8090`) or serving URL
        (`http://localhost:8080/v1` or Vertex `:rawPredict`). Reads
        `DGEM_GATEWAY_URL` then `DGEM_URL` when omitted.
      bearer_token: Static `Authorization: Bearer` token. Reads `DGEM_AUTH_TOKEN`
        when omitted.
      token_provider: Optional synchronous `(url) -> token` callable.
      gateway_api_key: `X-DGem-Api-Key` sent exclusively to `base_url` when
        operating against a gateway (`DGEM_GATEWAY_KEY` fallback).
      server_key: `X-DGem-Key` sent exclusively to a serving image; takes the
        first non-empty entry when comma-separated (`DGEM_SERVER_KEY` fallback).
      default_backend: Optional `X-DGem-Backend` routing header (`vertex_first`,
        `vertex`, `cloudrun`, `local`).
      timeout_seconds: Per-request timeout in seconds (`> 0`).
      transport: Optional custom synchronous transport for testing.

    Raises:
      DGemValidationError: If URLs, timeouts, or coroutine callbacks are invalid.
    """
    if transport is not None and (
        not callable(transport) or inspect.iscoroutinefunction(transport)
    ):
      raise DGemValidationError(
          "transport must be a synchronous callable, not a coroutine function"
      )
    if token_provider is not None and (
        not callable(token_provider)
        or inspect.iscoroutinefunction(token_provider)
    ):
      raise DGemValidationError(
          "token_provider must be a synchronous callable, not a coroutine"
      )
    if timeout_seconds <= 0.0:
      raise DGemValidationError(
          f"timeout_seconds must be > 0 (got {timeout_seconds})"
      )

    resolved_url = (
        base_url
        or os.environ.get("DGEM_GATEWAY_URL")
        or os.environ.get("DGEM_URL")
        or "http://127.0.0.1:8090"
    ).strip()
    self._base_origin = parse_url_origin(resolved_url)
    self._base_url = resolved_url.rstrip("/")

    env_token = os.environ.get("DGEM_AUTH_TOKEN", "").strip()
    self._bearer_token = (
        bearer_token.strip() if bearer_token is not None else env_token
    )
    self._token_provider = token_provider

    raw_gateway_key = (
        gateway_api_key
        if gateway_api_key is not None
        else os.environ.get("DGEM_GATEWAY_KEY", "")
    ).strip()
    self._gateway_api_key = raw_gateway_key

    raw_server_key = (
        server_key
        if server_key is not None
        else os.environ.get("DGEM_SERVER_KEY", "")
    )
    first_server_key = raw_server_key.split(",")[0].strip()
    self._server_key = first_server_key

    self._default_backend = default_backend.strip()
    self._timeout_seconds = float(timeout_seconds)
    self._transport: TransportCallable = transport or default_urllib_transport

  @property
  def base_url(self) -> str:
    """Return the normalized base URL configured on this client."""
    return self._base_url

  def _is_rawpredict(self) -> bool:
    """Return True if `base_url` is a single-route Vertex `:rawPredict` URL."""
    return self._base_url.lower().endswith(":rawpredict")

  def _is_serving_v1_base(self) -> bool:
    """Return True if `base_url` ends with `/v1` or `:rawPredict`."""
    lowered = self._base_url.lower()
    return lowered.endswith("/v1") or lowered.endswith(":rawpredict")

  def _build_headers(
      self,
      target_url: str,
      *,
      target_role: str,
      backend_override: str = "",
  ) -> dict[str, str]:
    """Construct outgoing headers enforcing per-service credential isolation."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    target_origin = parse_url_origin(target_url)

    token = self._bearer_token
    if not token and self._token_provider is not None:
      minted = self._token_provider(target_url)
      if inspect.iscoroutine(minted):
        minted.close()
        raise DGemValidationError("token_provider returned a coroutine")
      if minted:
        token = str(minted).strip()
    if token and target_origin == self._base_origin:
      headers["Authorization"] = _validate_header_value(
          "Authorization", f"Bearer {token}"
      )

    if (
        target_role == "gateway"
        and self._gateway_api_key
        and target_origin == self._base_origin
    ):
      headers["X-DGem-Api-Key"] = _validate_header_value(
          "X-DGem-Api-Key", self._gateway_api_key
      )
    elif (
        target_role == "serving"
        and self._server_key
        and target_origin == self._base_origin
    ):
      headers["X-DGem-Key"] = _validate_header_value(
          "X-DGem-Key", self._server_key
      )

    selected_backend = backend_override.strip() or self._default_backend
    if selected_backend and target_role == "gateway":
      headers["X-DGem-Backend"] = _validate_header_value(
          "X-DGem-Backend", selected_backend
      )
    return headers

  def _decode_json_object(self, raw_bytes: bytes) -> dict[str, object]:
    """Parse UTF-8 JSON bytes and require a top-level JSON object."""
    try:
      decoded = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
      raise DGemTransportError(f"invalid JSON response: {error}") from error
    if not isinstance(decoded, dict):
      raise DGemTransportError("expected JSON object in response")
    return decoded

  def health(self) -> dict[str, object]:
    """Query liveness and readiness metadata from the gateway or serving node.

    On Vertex `:rawPredict` endpoints, `GET /health` is dispatched as
    `POST {"health": true}` per the single-route serving contract.

    Returns:
      Parsed health JSON dictionary.
    """
    if self._is_rawpredict():
      headers = self._build_headers(self._base_url, target_role="serving")
      payload = json.dumps({"health": True}).encode("utf-8")
      _, response_bytes, _ = self._transport(
          "POST", self._base_url, payload, headers, self._timeout_seconds
      )
      return self._decode_json_object(response_bytes)

    base_without_v1 = (
        self._base_url[:-3]
        if self._base_url.lower().endswith("/v1")
        else self._base_url
    )
    role = "serving" if self._is_serving_v1_base() else "gateway"
    target_url = f"{base_without_v1}/health"
    headers = self._build_headers(target_url, target_role=role)
    _, response_bytes, _ = self._transport(
        "GET", target_url, None, headers, self._timeout_seconds
    )
    return self._decode_json_object(response_bytes)

  def decide(
      self,
      template: str = "",
      *,
      variables: Mapping[str, object] | None = None,
      custom_template: str = "",
      images: Sequence[str] | None = None,
      backend: str = "",
      cascade_mode: str = "",
      cascade_threshold: float | None = None,
      cascade_threshold_mode: str = "",
      cascade_model: str = "",
      layout: str = "",
      temperature: float | Mapping[str, float] | None = None,
  ) -> GatewayDecideResponse:
    """Evaluate a `.json.tmpl` decision policy via `POST /api/decide`.

    Args:
      template: Named policy template (e.g. `"support_triage"`).
      variables: Template input variable mapping.
      custom_template: Inline `.json.tmpl` text when `template` is omitted.
      images: Optional sequence of `data:` URIs or image URLs.
      backend: Optional backend override (`"vertex_first"`, `"vertex"`, `"cloudrun"`).
      cascade_mode: Optional Stage 2 mode (`"off"`, `"entropy"`, `"on_miss"`).
      cascade_threshold: Optional Stage 2 entropy or hesitation cutoff.
      cascade_threshold_mode: `"nats"` or `"normalized"`.
      cascade_model: GA Gemini cascade model (`"gemini-3.8-flash"`).
      layout: Optional prompt layout (`"document_first"` or `"schema_first"`).
      temperature: Optional post-hoc temperature (`float` or `{type: float}`).

    Returns:
      Typed `GatewayDecideResponse` with parsed `QuestionAnswer` entries.

    Raises:
      DGemValidationError: If neither `template` nor `custom_template` is given.
    """
    if not template.strip() and not custom_template.strip():
      raise DGemValidationError(
          "either template or custom_template must be non-empty"
      )
    if self._is_serving_v1_base():
      raise DGemValidationError(
          "decide() requires a dgem gateway URL; use decide_systemone() on /v1"
      )

    request_body: dict[str, object] = {}
    if template.strip():
      request_body["template"] = template.strip()
    if custom_template.strip():
      request_body["custom_template"] = custom_template
    if variables is not None:
      request_body["variables"] = dict(variables)
    if images:
      request_body["images"] = list(images)
    if backend.strip():
      request_body["backend"] = backend.strip()
    if cascade_mode.strip():
      request_body["cascade_mode"] = cascade_mode.strip()
    if cascade_threshold is not None:
      request_body["cascade_threshold"] = float(cascade_threshold)
    if cascade_threshold_mode.strip():
      request_body["cascade_threshold_mode"] = cascade_threshold_mode.strip()
    if cascade_model.strip():
      request_body["cascade_model"] = cascade_model.strip()
    if layout.strip():
      request_body["layout"] = layout.strip()
    if temperature is not None:
      request_body["temperature"] = (
          dict(temperature)
          if isinstance(temperature, Mapping)
          else float(temperature)
      )

    target_url = f"{self._base_url}/api/decide"
    headers = self._build_headers(
        target_url,
        target_role="gateway",
        backend_override=backend,
    )
    encoded_body = json.dumps(request_body).encode("utf-8")
    _, response_bytes, response_headers = self._transport(
        "POST", target_url, encoded_body, headers, self._timeout_seconds
    )
    parsed = self._decode_json_object(response_bytes)
    backend_used = ""
    for header_key, header_val in response_headers.items():
      if header_key.lower() == "x-dgem-backend-used":
        backend_used = header_val
        break
    return GatewayDecideResponse.from_dict(
        parsed, backend_used_header=backend_used
    )

  def decide_systemone(
      self,
      context_payload: str | Mapping[str, object],
      questions: Mapping[str, SystemOneQuestion | Mapping[str, object]],
      *,
      images: Sequence[str] | None = None,
      instructions: str = "",
      samples: int = 1,
      layout: str = "",
      prior_weight: float | None = None,
      shortlist_topk: int | None = None,
      backend: str = "",
  ) -> StructuredDecisionResponse:
    """Execute a structured decision request against `POST /v1/systemone`.

    Works transparently against a `dgem serve` gateway, a `/v1` serving image,
    or a single-route Vertex `:rawPredict` endpoint.

    Args:
      context_payload: Input text or JSON state mapping to evaluate.
      questions: Mapping of `question_id` to `SystemOneQuestion` or raw dict.
      images: Optional sequence of `data:` image URIs.
      instructions: Optional global system rubric.
      samples: Number of parallel canvas reads (`>= 1`, default `1`).
      layout: Optional `"document_first"` or `"schema_first"`.
      prior_weight: Optional log-linear prior fusion weight (`>= 0.0`).
      shortlist_topk: Optional top-K shortlist size for wide choice sets.
      backend: Optional `X-DGem-Backend` header when calling a gateway.

    Returns:
      Parsed `StructuredDecisionResponse`.

    Raises:
      DGemValidationError: If `questions` is empty or `samples < 1`.
    """
    if not questions:
      raise DGemValidationError("questions must be non-empty")
    if samples < 1:
      raise DGemValidationError(f"samples must be >= 1 (got {samples})")

    serialized_questions: dict[str, object] = {}
    for question_id, spec in questions.items():
      if isinstance(spec, SystemOneQuestion):
        serialized_questions[str(question_id)] = spec.to_dict()
      elif isinstance(spec, Mapping):
        serialized_questions[str(question_id)] = dict(spec)
      else:
        raise DGemValidationError(
            f"question {question_id!r} must be SystemOneQuestion or Mapping"
        )

    payload: dict[str, object] = {
        "context": (
            dict(context_payload)
            if isinstance(context_payload, Mapping)
            else str(context_payload)
        ),
        "questions": serialized_questions,
        "samples": int(samples),
    }
    if instructions.strip():
      payload["instructions"] = instructions.strip()
    if images:
      payload["images"] = list(images)
    if layout.strip():
      payload["layout"] = layout.strip()
    if prior_weight is not None:
      payload["prior_weight"] = float(prior_weight)
    if shortlist_topk is not None:
      payload["shortlist_topk"] = int(shortlist_topk)

    if self._is_rawpredict():
      target_url = self._base_url
      role = "serving"
    elif self._base_url.lower().endswith("/v1"):
      target_url = f"{self._base_url}/systemone"
      role = "serving"
    else:
      target_url = f"{self._base_url}/v1/systemone"
      role = "gateway"

    headers = self._build_headers(
        target_url,
        target_role=role,
        backend_override=backend,
    )
    encoded_body = json.dumps(payload).encode("utf-8")
    _, response_bytes, _ = self._transport(
        "POST", target_url, encoded_body, headers, self._timeout_seconds
    )
    return StructuredDecisionResponse.from_dict(
        self._decode_json_object(response_bytes)
    )
