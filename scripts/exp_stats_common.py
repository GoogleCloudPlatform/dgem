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

"""Reusable stdlib-only statistical, conformal, and split-lock primitives."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import http.client
import json
import math
import pathlib
import random
import re
import struct
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_CONFORMAL_SEED_KEY: str = "dgem-conformal-v1"
CONFORMAL_SCORE_TOLERANCE: float = 1e-12
MAX_RESPONSE_BYTES: int = 16 * 1024 * 1024
ALLOWED_GA_GEMINI_MODELS: frozenset[str] = frozenset({
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
})
LOCALHOST_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
SENSITIVE_AUTH_HEADERS: frozenset[str] = frozenset({
    "authorization",
    "x-dgem-key",
    "x-dgem-api-key",
    "x-goog-api-key",
})

# Bounded-quantifier O(N) pack-multiplier patterns (zero polynomial backtracking).
_RE_COMPOUND_UNIT_PREFIX = re.compile(
    r"\b(\d{1,4})[ \t]{0,3}[x×][ \t]{0,3}\d{1,5}(?:\.\d{1,3})?[ \t]{0,2}"
    r"(?:ml|l|ltr|oz|fl\.?[ \t]{0,2}oz|g|gm|kg|lb|lbs|mg|cl|ct|count|pk|"
    r"pack|packs|pcs|caps|capsules|tabs|tablets|rolls|sheets|wipes|bars|"
    r"cans|bottles|pouches|pods|sachets)\b",
    re.IGNORECASE,
)
_RE_COMPOUND_UNIT_SUFFIX = re.compile(
    r"\b\d{1,5}(?:\.\d{1,3})?[ \t]{0,2}"
    r"(?:ml|l|ltr|oz|fl\.?[ \t]{0,2}oz|g|gm|kg|lb|lbs|mg|cl)"
    r"[ \t]{0,3}[x×][ \t]{0,3}(\d{1,4})\b",
    re.IGNORECASE,
)
_RE_PACK_OF_COUNT = re.compile(
    r"\b(?:pack|case|box|bundle|carton)[ \t]{1,3}of[ \t]{1,3}(\d{1,4})\b",
    re.IGNORECASE,
)
_RE_HYPHENATED_PACK = re.compile(
    r"\b(\d{1,4})(?:-[ \t]{0,2}|[ \t]{0,2})"
    r"(?:pack|packs|pk|count|ct|cans|bottles|bars|pods|pouches|sachets)\b",
    re.IGNORECASE,
)
_RE_BARE_COMPOUND = re.compile(
    r"\b(\d{1,2})[ \t]{0,2}[x×][ \t]{0,2}(\d{2,4})\b",
    re.IGNORECASE,
)
_RE_TWIN_PACK = re.compile(
    r"\b(?:twin|double|duo)(?:-[ \t]{0,2}|[ \t]{0,2})pack\b",
    re.IGNORECASE,
)
_RE_TRIPLE_PACK = re.compile(
    r"\btriple(?:-[ \t]{0,2}|[ \t]{0,2})pack\b",
    re.IGNORECASE,
)
_RE_QUAD_PACK = re.compile(
    r"\bquad(?:-[ \t]{0,2}|[ \t]{0,2})pack\b",
    re.IGNORECASE,
)


class SplitLeakageError(ValueError):
  """Raised when split items, hashes, or cluster IDs violate disjointness."""


class StatisticalValidationError(ValueError):
  """Raised when inputs to statistical or conformal helpers are invalid."""


@dataclass(frozen=True)
class BenchmarkSplitItem:
  """Canonical deduplicated item tracked in a 30/20/50 split lockfile."""

  item_id: str
  cluster_id: str
  content_text: str
  content_sha256: str


@dataclass(frozen=True)
class ConformalParameters:
  """Stateless split-conformal configuration matching Go ConformalConfig."""

  method: str = "raps"
  alpha: float = 0.10
  k_reg: int = 1
  lambda_reg: float = 0.01
  randomized: bool = False
  seed_key: str = DEFAULT_CONFORMAL_SEED_KEY
  item_key: str = ""
  abstain_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConformalFitSummary:
  """Calibrated split-conformal quantile summary fitted on D_cal."""

  method: str
  alpha: float
  num_calibration: int
  order_index: int
  q_hat: float
  empirical_coverage: float
  sorted_scores: tuple[float, ...]


@dataclass(frozen=True)
class ConformalPredictionResult:
  """Prediction set and escalation flag for one probability distribution."""

  prediction_set: tuple[str, ...]
  scores: dict[str, float]
  q_hat: float
  top_option: str
  top_probability: float
  set_size: int
  singleton: bool
  contains_abstain: bool
  should_escalate: bool


@dataclass(frozen=True)
class McNemarTestResult:
  """Exact two-sided binomial Paired McNemar contingency and test statistics."""

  n_total: int
  n_11: int
  n_10: int
  n_01: int
  n_00: int
  baseline_accuracy: float
  treatment_accuracy: float
  delta_accuracy: float
  chi2_continuity: float
  p_value_exact: float


@dataclass(frozen=True)
class TangoNonInferiorityResult:
  """Tango (1998) asymptotic score z-test for paired binary non-inferiority."""

  n_total: int
  n_10: int
  n_01: int
  margin: float
  delta_accuracy: float
  z_score: float
  p_value_one_sided: float
  non_inferior_at_005: bool


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
  """Paired cluster-bootstrap point estimate and percentile interval."""

  estimate: float
  ci_lower: float
  ci_upper: float
  num_clusters: int
  num_draws: int


def validate_local_file_path(
    raw_path: str | pathlib.Path,
    must_exist: bool = False,
    allowed_root: pathlib.Path | None = None,
) -> pathlib.Path:
  """Validate and canonicalize a local filesystem path against traversal.

  Args:
    raw_path: Input filesystem path string or `Path` object.
    must_exist: Whether to require an existing regular file at `raw_path`.
    allowed_root: Optional directory boundary that must contain the target path.

  Returns:
    Canonicalized `pathlib.Path` instance.

  Raises:
    StatisticalValidationError: If the path contains NUL bytes, escapes
      `allowed_root`, or fails the `must_exist` check.
  """
  raw_text = str(raw_path).strip()
  if not raw_text or "\x00" in raw_text:
    raise StatisticalValidationError("file path must be non-empty without NUL")
  resolved_path = pathlib.Path(raw_text).expanduser().resolve()
  if allowed_root is not None:
    canonical_root = allowed_root.expanduser().resolve()
    if resolved_path != canonical_root and not resolved_path.is_relative_to(
        canonical_root
    ):
      raise StatisticalValidationError(
          f"path {resolved_path} escapes allowed root {canonical_root}"
      )
  if must_exist and not resolved_path.is_file():
    raise StatisticalValidationError(f"expected existing file: {resolved_path}")
  return resolved_path


def validate_endpoint_url(raw_url: str) -> str:
  """Enforce HTTPS on remote target URLs while allowing HTTP only on loopback.

  Args:
    raw_url: Candidate HTTP or HTTPS endpoint URL.

  Returns:
    Normalized URL string without trailing whitespace.

  Raises:
    StatisticalValidationError: If scheme, hostname, or embedded credentials
      violate security constraints.
  """
  cleaned_url = raw_url.strip()
  if not cleaned_url or "\x00" in cleaned_url:
    raise StatisticalValidationError("endpoint URL must be non-empty")
  parsed_url = urllib.parse.urlparse(cleaned_url)
  if parsed_url.username or parsed_url.password:
    raise StatisticalValidationError(
        "endpoint URL must not contain embedded user credentials"
    )
  scheme = parsed_url.scheme.lower()
  hostname = (parsed_url.hostname or "").lower()
  if not hostname:
    raise StatisticalValidationError(f"endpoint URL missing host: {raw_url!r}")
  if scheme == "https":
    return cleaned_url
  if scheme == "http" and hostname in LOCALHOST_HOSTS:
    return cleaned_url
  raise StatisticalValidationError(
      f"remote endpoint URL must use https:// (got {raw_url!r})"
  )


def validate_ga_gemini_model(model_name: str) -> str:
  """Validate Stage-2 model names against the GA Gemini model allowlist.

  Args:
    model_name: Requested Stage-2 Gemini model identifier.

  Returns:
    Canonical GA Gemini model identifier.

  Raises:
    StatisticalValidationError: If `model_name` is not in the GA allowlist.
  """
  normalized_name = model_name.strip()
  if normalized_name not in ALLOWED_GA_GEMINI_MODELS:
    allowed_csv = ", ".join(sorted(ALLOWED_GA_GEMINI_MODELS))
    raise StatisticalValidationError(
        f"model {model_name!r} not in GA allowlist ({allowed_csv})"
    )
  return normalized_name


class _CredentialStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
  """Validates redirect targets and strips auth headers across origins."""

  def redirect_request(
      self,
      req: urllib.request.Request,
      fp: object,
      code: int,
      msg: str,
      headers: object,
      newurl: str,
  ) -> urllib.request.Request | None:
    validated_target = validate_endpoint_url(newurl)
    source_parsed = urllib.parse.urlparse(req.full_url)
    target_parsed = urllib.parse.urlparse(validated_target)
    source_origin = (source_parsed.scheme.lower(), source_parsed.netloc.lower())
    target_origin = (target_parsed.scheme.lower(), target_parsed.netloc.lower())
    cross_origin = source_origin != target_origin

    if code in (307, 308):
      merged_headers: dict[str, str] = {}
      for header_key, header_val in req.header_items():
        if cross_origin and header_key.lower() in SENSITIVE_AUTH_HEADERS:
          continue
        merged_headers[header_key] = header_val
      return urllib.request.Request(
          validated_target,
          data=req.data,
          headers=merged_headers,
          origin_req_host=req.origin_req_host,
          unverifiable=True,
          method=req.get_method(),
      )

    redirected = super().redirect_request(
        req, fp, code, msg, headers, validated_target  # type: ignore[arg-type]
    )
    if redirected is None:
      return None
    if cross_origin:
      for header_key in list(redirected.headers.keys()):
        if header_key.lower() in SENSITIVE_AUTH_HEADERS:
          del redirected.headers[header_key]
      for header_key in list(redirected.unredirected_hdrs.keys()):
        if header_key.lower() in SENSITIVE_AUTH_HEADERS:
          del redirected.unredirected_hdrs[header_key]
    return redirected


def post_json_secure(
    endpoint_url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str] | None = None,
    timeout_sec: float = 30.0,
) -> dict[str, object]:
  """Send a JSON POST request with HTTPS validation and safe redirect stripping.

  Args:
    endpoint_url: Validated HTTPS (or localhost HTTP) endpoint URL.
    payload: JSON-serializable dictionary payload.
    headers: Optional HTTP request headers.
    timeout_sec: Socket timeout in seconds (`> 0`).

  Returns:
    Decoded JSON object dictionary from the endpoint response.

  Raises:
    StatisticalValidationError: If URL/timeout validation fails or response is
      not a valid JSON object within `MAX_RESPONSE_BYTES`.
  """
  safe_url = validate_endpoint_url(endpoint_url)
  if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
    raise StatisticalValidationError("timeout_sec must be finite and > 0")
  encoded_body = json.dumps(dict(payload)).encode("utf-8")
  request_headers: dict[str, str] = {
      "Content-Type": "application/json",
      "Accept": "application/json",
  }
  if headers is not None:
    for header_name, header_value in headers.items():
      if "\r" in header_name or "\n" in header_name:
        raise StatisticalValidationError("invalid CR/LF in header name")
      if "\r" in header_value or "\n" in header_value:
        raise StatisticalValidationError("invalid CR/LF in header value")
      request_headers[header_name] = header_value

  http_request = urllib.request.Request(
      safe_url,
      data=encoded_body,
      headers=request_headers,
      method="POST",
  )
  opener = urllib.request.build_opener(_CredentialStrippingRedirectHandler())
  try:
    with opener.open(http_request, timeout=timeout_sec) as http_response:
      raw_bytes = http_response.read(MAX_RESPONSE_BYTES + 1)
  except (
      urllib.error.URLError,
      TimeoutError,
      OSError,
      http.client.HTTPException,
  ) as transport_err:
    raise StatisticalValidationError(
        f"HTTP request failed for {safe_url}: {transport_err}"
    ) from transport_err

  if len(raw_bytes) > MAX_RESPONSE_BYTES:
    raise StatisticalValidationError("HTTP response exceeded maximum byte cap")
  try:
    decoded_payload = json.loads(raw_bytes.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as decode_err:
    raise StatisticalValidationError(
        f"invalid JSON response from {safe_url}"
    ) from decode_err
  if not isinstance(decoded_payload, dict):
    raise StatisticalValidationError("expected JSON object from endpoint")
  return decoded_payload


def compute_content_sha256(content_text: str, media_digest: str = "") -> str:
  """Compute a canonical SHA-256 content digest over normalized text and media.

  Args:
    content_text: Prompt or item payload text.
    media_digest: Optional hex SHA-256 digest of associated image bytes.

  Returns:
    Lowercase 64-character SHA-256 hex digest.

  Raises:
    StatisticalValidationError: If normalized `content_text` and `media_digest`
      are both empty.
  """
  normalized_text = " ".join(content_text.strip().lower().split())
  normalized_media = media_digest.strip().lower()
  if not normalized_text and not normalized_media:
    raise StatisticalValidationError(
        "cannot hash empty content_text and empty media_digest"
    )
  hasher = hashlib.sha256()
  hasher.update(normalized_text.encode("utf-8"))
  hasher.update(b"\x00")
  hasher.update(normalized_media.encode("utf-8"))
  return hasher.hexdigest()


def compute_file_sha256(file_path: str | pathlib.Path) -> str:
  """Compute the streaming SHA-256 digest of a local file.

  Args:
    file_path: Path to an existing local file.

  Returns:
    Lowercase 64-character SHA-256 hex digest.

  Raises:
    StatisticalValidationError: If `file_path` is invalid or missing.
  """
  resolved = validate_local_file_path(file_path, must_exist=True)
  hasher = hashlib.sha256()
  with resolved.open("rb") as handle:
    while True:
      chunk = handle.read(65536)
      if not chunk:
        break
      hasher.update(chunk)
  return hasher.hexdigest()


def hmac_uint64(seed_key: str, item_key: str, tag: str) -> int:
  """Compute big-endian uint64 HMAC-SHA256 matching Go `hmacUint64`.

  Args:
    seed_key: Secret/public deterministic seed key (defaults when empty).
    item_key: Per-item or per-cluster scoping key.
    tag: Domain-separation suffix tag.

  Returns:
    Unsigned 64-bit integer in `[0, 2**64 - 1]`.

  Raises:
    TypeError: If arguments are not strings.
  """
  effective_key = seed_key if seed_key else DEFAULT_CONFORMAL_SEED_KEY
  mac = hmac.new(effective_key.encode("utf-8"), digestmod=hashlib.sha256)
  mac.update(item_key.encode("utf-8"))
  mac.update(b"\x00")
  mac.update(tag.encode("utf-8"))
  return int(struct.unpack(">Q", mac.digest()[:8])[0])


def hmac_uniform_float64(seed_key: str, item_key: str, tag: str) -> float:
  """Draw a deterministic Uniform[0, 1) float64 matching Go `hmacUniformFloat64`.

  Args:
    seed_key: Deterministic HMAC key string.
    item_key: Per-item identifier string.
    tag: Domain-separation tag string.

  Returns:
    IEEE-754 float64 in `[0.0, 1.0)`.

  Raises:
    TypeError: If arguments are not strings.
  """
  top_53_bits = hmac_uint64(seed_key, item_key, tag) >> 11
  return float(top_53_bits) * (1.0 / float(1 << 53))


def deduplicate_and_partition_clusters(
    items: Sequence[BenchmarkSplitItem],
    seed_tag: str = "dgem-exp-v1",
    cal_ratio: float = 0.30,
    val_ratio: float = 0.20,
) -> dict[str, list[BenchmarkSplitItem]]:
  """Deduplicate by SHA-256 and partition clusters into 30/20/50 disjoint folds.

  Args:
    items: Raw candidate `BenchmarkSplitItem` sequence.
    seed_tag: Deterministic HMAC seed tag controlling cluster ordering.
    cal_ratio: Target fraction of items assigned to `cal` (default `0.30`).
    val_ratio: Target fraction of items assigned to `val` (default `0.20`).

  Returns:
    Dictionary with keys `"cal"`, `"val"`, and `"test"` mapping to disjoint
    lists of `BenchmarkSplitItem`.

  Raises:
    SplitLeakageError: If duplicate `item_id` values appear or fewer than 3
      unique clusters remain after content-hash deduplication.
  """
  if not (0.0 < cal_ratio < 1.0 and 0.0 < val_ratio < 1.0):
    raise SplitLeakageError("cal_ratio and val_ratio must lie in (0, 1)")
  if cal_ratio + val_ratio >= 1.0:
    raise SplitLeakageError("cal_ratio + val_ratio must be < 1.0")

  seen_item_ids: set[str] = set()
  seen_hashes: set[str] = set()
  deduped_items: list[BenchmarkSplitItem] = []

  def _item_order_key(entry: BenchmarkSplitItem) -> str:
    return entry.item_id

  for candidate in sorted(items, key=_item_order_key):
    if not candidate.item_id.strip() or not candidate.cluster_id.strip():
      raise SplitLeakageError("item_id and cluster_id must be non-empty")
    if candidate.item_id in seen_item_ids:
      raise SplitLeakageError(f"duplicate item_id {candidate.item_id!r}")
    seen_item_ids.add(candidate.item_id)
    if candidate.content_sha256 in seen_hashes:
      continue
    seen_hashes.add(candidate.content_sha256)
    deduped_items.append(candidate)

  clusters_map: dict[str, list[BenchmarkSplitItem]] = {}
  for entry in deduped_items:
    clusters_map.setdefault(entry.cluster_id, []).append(entry)

  if len(clusters_map) < 3:
    raise SplitLeakageError(
        f"need at least 3 disjoint clusters, got {len(clusters_map)}"
    )

  def _cluster_sort_key(cluster_name: str) -> tuple[int, str]:
    return (hmac_uint64(seed_tag, cluster_name, "cluster-split"), cluster_name)

  ordered_clusters = sorted(clusters_map.keys(), key=_cluster_sort_key)
  total_count = len(deduped_items)
  target_counts = {
      "cal": int(round(total_count * cal_ratio)),
      "val": int(round(total_count * val_ratio)),
  }
  target_counts["test"] = (
      total_count - target_counts["cal"] - target_counts["val"]
  )

  fold_buckets: dict[str, list[BenchmarkSplitItem]] = {
      "cal": [],
      "val": [],
      "test": [],
  }
  fold_names = ("cal", "val", "test")

  # Seed at least one cluster per fold first so every fold is non-empty, then
  # assign remaining clusters to the fold with the largest remaining deficit.
  for idx, cluster_name in enumerate(ordered_clusters):
    cluster_rows = clusters_map[cluster_name]
    if idx < 3:
      chosen_fold = fold_names[idx]
    else:
      chosen_fold = "test"
      best_deficit = -math.inf
      for fold_name in fold_names:
        remaining = target_counts[fold_name] - len(fold_buckets[fold_name])
        if remaining > best_deficit:
          best_deficit = float(remaining)
          chosen_fold = fold_name
    fold_buckets[chosen_fold].extend(cluster_rows)

  for fold_name in fold_names:
    fold_buckets[fold_name].sort(key=_item_order_key)
  return fold_buckets


def _compute_track_split_digest(
    folds: Mapping[str, Sequence[str]],
    item_hashes: Mapping[str, str],
) -> str:
  """Compute deterministic SHA-256 digest over fold assignments and hashes."""
  canonical_blob = json.dumps(
      {
          "folds": {
              fold_name: list(folds[fold_name])
              for fold_name in ("cal", "val", "test")
          },
          "item_hashes": {
              key: item_hashes[key] for key in sorted(item_hashes.keys())
          },
      },
      sort_keys=True,
      separators=(",", ":"),
  )
  return hashlib.sha256(canonical_blob.encode("utf-8")).hexdigest()


def build_track_lock_entry(
    clustering_unit: str,
    fold_items: Mapping[str, Sequence[BenchmarkSplitItem]],
) -> dict[str, object]:
  """Construct a serializable split-lock dictionary for one experiment track.

  Args:
    clustering_unit: Name of the correlation cluster field (e.g. `domain_id`).
    fold_items: Mapping of `"cal"`, `"val"`, `"test"` to `BenchmarkSplitItem`s.

  Returns:
    JSON-serializable dictionary recording fold IDs, cluster IDs, and hashes.

  Raises:
    SplitLeakageError: If `fold_items` contains any cross-fold item, hash, or
      cluster overlap.
  """
  folds_ids: dict[str, list[str]] = {}
  cluster_assignments: dict[str, list[str]] = {}
  item_hashes: dict[str, str] = {}
  item_clusters: dict[str, str] = {}

  seen_ids: set[str] = set()
  seen_hashes_by_fold: dict[str, str] = {}
  seen_clusters_by_fold: dict[str, str] = {}

  for fold_name in ("cal", "val", "test"):
    entries = list(fold_items.get(fold_name, ()))
    if not entries:
      raise SplitLeakageError(f"fold {fold_name!r} must be non-empty")
    fold_id_list: list[str] = []
    fold_cluster_set: set[str] = set()
    for entry in entries:
      if entry.item_id in seen_ids:
        raise SplitLeakageError(
            f"item_id {entry.item_id!r} appears in multiple folds"
        )
      seen_ids.add(entry.item_id)
      prior_hash_fold = seen_hashes_by_fold.get(entry.content_sha256)
      if prior_hash_fold is not None and prior_hash_fold != fold_name:
        raise SplitLeakageError(
            f"content_sha256 {entry.content_sha256} crosses {prior_hash_fold} "
            f"and {fold_name}"
        )
      seen_hashes_by_fold[entry.content_sha256] = fold_name

      prior_cluster_fold = seen_clusters_by_fold.get(entry.cluster_id)
      if prior_cluster_fold is not None and prior_cluster_fold != fold_name:
        raise SplitLeakageError(
            f"cluster_id {entry.cluster_id!r} crosses {prior_cluster_fold} "
            f"and {fold_name}"
        )
      seen_clusters_by_fold[entry.cluster_id] = fold_name

      fold_id_list.append(entry.item_id)
      fold_cluster_set.add(entry.cluster_id)
      item_hashes[entry.item_id] = entry.content_sha256
      item_clusters[entry.item_id] = entry.cluster_id

    folds_ids[fold_name] = sorted(fold_id_list)
    cluster_assignments[fold_name] = sorted(fold_cluster_set)

  split_digest = _compute_track_split_digest(folds_ids, item_hashes)
  return {
      "clustering_unit": clustering_unit,
      "counts": {
          fold_name: len(folds_ids[fold_name])
          for fold_name in ("cal", "val", "test")
      },
      "cluster_counts": {
          fold_name: len(cluster_assignments[fold_name])
          for fold_name in ("cal", "val", "test")
      },
      "split_digest_sha256": split_digest,
      "folds": folds_ids,
      "cluster_assignments": cluster_assignments,
      "item_hashes": item_hashes,
      "item_clusters": item_clusters,
  }


def verify_split_lockfile(
    lockfile_payload: Mapping[str, object],
    track_name: str,
    items: Sequence[BenchmarkSplitItem],
) -> dict[str, list[BenchmarkSplitItem]]:
  """Verify zero item, content-hash, and cluster leakage against a lockfile.

  Args:
    lockfile_payload: Parsed JSON lockfile dictionary.
    track_name: Track key inside `lockfile_payload["tracks"]`.
    items: Live `BenchmarkSplitItem` sequence to validate and partition.

  Returns:
    Dictionary mapping `"cal"`, `"val"`, `"test"` to verified item lists.

  Raises:
    SplitLeakageError: If any item ID, SHA-256 content hash, cluster ID, or
      track digest diverges from the lockfile or crosses folds.
  """
  if lockfile_payload.get("schema_version") != "dgem.exp.splits.v1":
    raise SplitLeakageError("unsupported or missing lockfile schema_version")
  tracks_obj = lockfile_payload.get("tracks")
  if not isinstance(tracks_obj, Mapping) or track_name not in tracks_obj:
    raise SplitLeakageError(f"track {track_name!r} missing from lockfile")
  track_entry = tracks_obj[track_name]
  if not isinstance(track_entry, Mapping):
    raise SplitLeakageError(f"invalid track entry for {track_name!r}")

  folds_obj = track_entry.get("folds")
  hashes_obj = track_entry.get("item_hashes")
  clusters_obj = track_entry.get("cluster_assignments")
  expected_digest = track_entry.get("split_digest_sha256")
  if (
      not isinstance(folds_obj, Mapping)
      or not isinstance(hashes_obj, Mapping)
      or not isinstance(clusters_obj, Mapping)
      or not isinstance(expected_digest, str)
  ):
    raise SplitLeakageError(f"malformed lockfile structure for {track_name!r}")

  locked_folds: dict[str, list[str]] = {}
  locked_hashes: dict[str, str] = {}
  for key, val in hashes_obj.items():
    if not isinstance(key, str) or not isinstance(val, str):
      raise SplitLeakageError("lockfile item_hashes must map str to str")
    locked_hashes[key] = val

  seen_locked_ids: set[str] = set()
  seen_locked_hashes: dict[str, str] = {}
  seen_locked_clusters: dict[str, str] = {}

  for fold_name in ("cal", "val", "test"):
    raw_fold_ids = folds_obj.get(fold_name)
    raw_fold_clusters = clusters_obj.get(fold_name)
    if not isinstance(raw_fold_ids, list) or not raw_fold_ids:
      raise SplitLeakageError(f"lockfile fold {fold_name!r} empty or invalid")
    if not isinstance(raw_fold_clusters, list) or not raw_fold_clusters:
      raise SplitLeakageError(
          f"lockfile cluster_assignments[{fold_name!r}] empty or invalid"
      )
    clean_ids: list[str] = []
    for raw_id in raw_fold_ids:
      if not isinstance(raw_id, str):
        raise SplitLeakageError("fold item IDs must be strings")
      if raw_id in seen_locked_ids:
        raise SplitLeakageError(
            f"lockfile item_id {raw_id!r} appears in multiple folds"
        )
      if raw_id not in locked_hashes:
        raise SplitLeakageError(
            f"lockfile item_id {raw_id!r} missing from item_hashes"
        )
      digest_val = locked_hashes[raw_id]
      prior_fold = seen_locked_hashes.get(digest_val)
      if prior_fold is not None and prior_fold != fold_name:
        raise SplitLeakageError(
            f"lockfile SHA-256 {digest_val} crosses {prior_fold} and "
            f"{fold_name}"
        )
      seen_locked_hashes[digest_val] = fold_name
      seen_locked_ids.add(raw_id)
      clean_ids.append(raw_id)

    for raw_cluster in raw_fold_clusters:
      if not isinstance(raw_cluster, str):
        raise SplitLeakageError("cluster IDs must be strings")
      prior_cluster_fold = seen_locked_clusters.get(raw_cluster)
      if prior_cluster_fold is not None and prior_cluster_fold != fold_name:
        raise SplitLeakageError(
            f"lockfile cluster_id {raw_cluster!r} crosses "
            f"{prior_cluster_fold} and {fold_name}"
        )
      seen_locked_clusters[raw_cluster] = fold_name

    locked_folds[fold_name] = clean_ids

  computed_digest = _compute_track_split_digest(locked_folds, locked_hashes)
  if computed_digest != expected_digest:
    raise SplitLeakageError(
        f"lockfile split_digest_sha256 mismatch for {track_name!r}: "
        f"got {computed_digest}, expected {expected_digest}"
    )

  items_by_id: dict[str, BenchmarkSplitItem] = {}
  for live_item in items:
    if live_item.item_id in items_by_id:
      raise SplitLeakageError(f"duplicate live item_id {live_item.item_id!r}")
    items_by_id[live_item.item_id] = live_item

  if set(items_by_id.keys()) != seen_locked_ids:
    missing_count = len(seen_locked_ids - set(items_by_id.keys()))
    extra_count = len(set(items_by_id.keys()) - seen_locked_ids)
    raise SplitLeakageError(
        f"live corpus mismatch against lockfile {track_name!r} "
        f"(missing={missing_count}, extra={extra_count})"
    )

  live_hash_fold: dict[str, str] = {}
  live_cluster_fold: dict[str, str] = {}
  partitioned: dict[str, list[BenchmarkSplitItem]] = {
      "cal": [],
      "val": [],
      "test": [],
  }

  for fold_name in ("cal", "val", "test"):
    allowed_clusters = set(clusters_obj[fold_name])  # type: ignore[arg-type]
    for item_id in locked_folds[fold_name]:
      live_item = items_by_id[item_id]
      if live_item.content_sha256 != locked_hashes[item_id]:
        raise SplitLeakageError(
            f"SHA-256 content drift for {item_id!r}: "
            f"live={live_item.content_sha256} != lock={locked_hashes[item_id]}"
        )
      if live_item.cluster_id not in allowed_clusters:
        raise SplitLeakageError(
            f"live cluster_id {live_item.cluster_id!r} for {item_id!r} "
            f"not permitted in fold {fold_name!r}"
        )
      prior_live_hash = live_hash_fold.get(live_item.content_sha256)
      if prior_live_hash is not None and prior_live_hash != fold_name:
        raise SplitLeakageError(
            f"live content_sha256 {live_item.content_sha256} crosses folds"
        )
      live_hash_fold[live_item.content_sha256] = fold_name

      prior_live_cluster = live_cluster_fold.get(live_item.cluster_id)
      if prior_live_cluster is not None and prior_live_cluster != fold_name:
        raise SplitLeakageError(
            f"live cluster_id {live_item.cluster_id!r} crosses folds"
        )
      live_cluster_fold[live_item.cluster_id] = fold_name
      partitioned[fold_name].append(live_item)

  return partitioned


def write_split_lockfile(
    lockfile_path: str | pathlib.Path,
    datasets_metadata: Sequence[Mapping[str, str]],
    tracks_payload: Mapping[str, Mapping[str, object]],
) -> str:
  """Write the canonical SHA-256 split lockfile JSON and return its file digest.

  Args:
    lockfile_path: Destination `.json` lockfile path.
    datasets_metadata: Source dataset provenance and license entries.
    tracks_payload: Mapping from track ID to `build_track_lock_entry` dicts.

  Returns:
    SHA-256 hex digest of the written lockfile.

  Raises:
    StatisticalValidationError: If `lockfile_path` is invalid.
  """
  resolved_target = validate_local_file_path(lockfile_path, must_exist=False)
  resolved_target.parent.mkdir(parents=True, exist_ok=True)
  document = {
      "schema_version": "dgem.exp.splits.v1",
      "license": "Apache-2.0",
      "split_protocol": "30_cal_20_val_50_test_cluster_disjoint",
      "datasets": [dict(entry) for entry in datasets_metadata],
      "tracks": {key: dict(val) for key, val in tracks_payload.items()},
  }
  serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
  resolved_target.write_text(serialized, encoding="utf-8")
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _ordered_normalized_probabilities(
    probabilities: Mapping[str, float],
) -> list[tuple[str, float]]:
  """Validate, normalize, and sort (name, prob) descending by prob then key."""
  if not probabilities:
    raise StatisticalValidationError("probabilities mapping must be non-empty")
  sorted_keys = sorted(str(key) for key in probabilities.keys())
  raw_pairs: list[tuple[str, float]] = []
  total_mass = 0.0
  for key in sorted_keys:
    raw_value = probabilities[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
      raise StatisticalValidationError(
          f"probability for {key!r} must be numeric"
      )
    prob_val = float(raw_value)
    if not math.isfinite(prob_val) or prob_val < 0.0:
      raise StatisticalValidationError(
          f"probability for {key!r} must be finite and >= 0 (got {prob_val})"
      )
    raw_pairs.append((key, prob_val))
    total_mass += prob_val
  if not math.isfinite(total_mass) or total_mass <= 0.0:
    raise StatisticalValidationError("total probability mass must be > 0")

  normalized_pairs = [(key, val / total_mass) for key, val in raw_pairs]

  def _prob_desc_key(pair: tuple[str, float]) -> tuple[float, str]:
    return (-pair[1], pair[0])

  normalized_pairs.sort(key=_prob_desc_key)
  return normalized_pairs


def normalize_probabilities(
    probabilities: Mapping[str, float],
) -> dict[str, float]:
  """Normalize an option probability dictionary to sum to 1.0.

  Args:
    probabilities: Mapping from option key to non-negative probability weight.

  Returns:
    Normalized dictionary ordered by descending probability then key.

  Raises:
    StatisticalValidationError: If `probabilities` is empty or non-positive.
  """
  return dict(_ordered_normalized_probabilities(probabilities))


def shannon_entropy(probabilities: Mapping[str, float]) -> float:
  """Compute Shannon entropy H(p) in nats for a discrete distribution.

  Args:
    probabilities: Mapping from option key to non-negative probability weight.

  Returns:
    Non-negative entropy in nats.

  Raises:
    StatisticalValidationError: If `probabilities` is invalid.
  """
  ordered = _ordered_normalized_probabilities(probabilities)
  entropy_nats = 0.0
  for _, prob_val in ordered:
    if prob_val > 0.0:
      entropy_nats -= prob_val * math.log(prob_val)
  return max(0.0, entropy_nats)


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
  """Compute cardinality-normalized hesitation H(p) / ln(K) in [0.0, 1.0].

  Args:
    probabilities: Mapping from option key to non-negative probability weight.

  Returns:
    Normalized entropy in `[0.0, 1.0]` (`0.0` when `K <= 1`).

  Raises:
    StatisticalValidationError: If `probabilities` is invalid.
  """
  ordered = _ordered_normalized_probabilities(probabilities)
  if len(ordered) <= 1:
    return 0.0
  entropy_nats = 0.0
  for _, prob_val in ordered:
    if prob_val > 0.0:
      entropy_nats -= prob_val * math.log(prob_val)
  return max(0.0, min(1.0, entropy_nats / math.log(len(ordered))))


def normalized_ordinal_variance(ordered_probs: Sequence[float]) -> float:
  """Compute Popoviciu-normalized ordinal variance Var(S) / ((L-1)^2 / 4).

  Args:
    ordered_probs: Sequence of probabilities ordered from lowest to highest bin.

  Returns:
    Scale-invariant ordinal variance in `[0.0, 1.0]` (`0.0` if invalid/`L < 2`).

  Raises:
    TypeError: If `ordered_probs` is not a sequence of numeric values.
  """
  num_buckets = len(ordered_probs)
  if num_buckets < 2:
    return 0.0
  total_mass = 0.0
  clean_values: list[float] = []
  for entry in ordered_probs:
    if isinstance(entry, bool) or not isinstance(entry, (int, float)):
      return 0.0
    numeric_val = float(entry)
    if not math.isfinite(numeric_val) or numeric_val < 0.0:
      return 0.0
    clean_values.append(numeric_val)
    total_mass += numeric_val
  if not math.isfinite(total_mass) or total_mass <= 0.0:
    return 0.0

  mean_index = 0.0
  for bucket_idx, numeric_val in enumerate(clean_values):
    mean_index += float(bucket_idx) * (numeric_val / total_mass)

  raw_variance = 0.0
  for bucket_idx, numeric_val in enumerate(clean_values):
    delta = float(bucket_idx) - mean_index
    raw_variance += (numeric_val / total_mass) * delta * delta

  popoviciu_bound = float((num_buckets - 1) * (num_buckets - 1)) / 4.0
  if popoviciu_bound <= 0.0:
    return 0.0
  return max(0.0, min(1.0, raw_variance / popoviciu_bound))


def parse_pack_multiplier(title: str) -> int:
  """Extract integer multipack count ahead of bare unit volumes (`O(N)` time).

  Args:
    title: Product title string (for example `"Sparkling Water 2 x 500ml"`).

  Returns:
    Positive integer pack multiplier (`1` when single-unit or unspecified).

  Raises:
    TypeError: If `title` is not a string.
  """
  trimmed = title.strip()
  if not trimmed:
    return 1

  for compiled_regex in (
      _RE_COMPOUND_UNIT_PREFIX,
      _RE_COMPOUND_UNIT_SUFFIX,
      _RE_PACK_OF_COUNT,
      _RE_HYPHENATED_PACK,
  ):
    regex_match = compiled_regex.search(trimmed)
    if regex_match is not None:
      parsed_count = int(regex_match.group(1))
      if 1 <= parsed_count <= 500:
        return parsed_count

  if _RE_TWIN_PACK.search(trimmed) is not None:
    return 2
  if _RE_TRIPLE_PACK.search(trimmed) is not None:
    return 3
  if _RE_QUAD_PACK.search(trimmed) is not None:
    return 4

  bare_match = _RE_BARE_COMPOUND.search(trimmed)
  if bare_match is not None:
    pack_count = int(bare_match.group(1))
    unit_size = int(bare_match.group(2))
    if 1 <= pack_count <= 96 and 10 <= unit_size <= 5000:
      return pack_count

  return 1


def apply_temperature(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
  """Scale discrete probabilities by post-hoc temperature p_k^(1 / T).

  Args:
    probabilities: Option probability dictionary.
    temperature: Positive scalar temperature in `(0.0, 20.0]`.

  Returns:
    Normalized temperature-scaled probability dictionary.

  Raises:
    StatisticalValidationError: If `temperature` or `probabilities` is invalid.
  """
  if (
      isinstance(temperature, bool)
      or not isinstance(temperature, (int, float))
      or not math.isfinite(float(temperature))
      or float(temperature) <= 0.0
      or float(temperature) > 20.0
  ):
    raise StatisticalValidationError(
        f"temperature must lie in (0, 20.0] (got {temperature!r})"
    )
  temp_scalar = float(temperature)
  ordered = _ordered_normalized_probabilities(probabilities)
  if abs(temp_scalar - 1.0) <= 1e-12:
    return dict(ordered)

  scaled_logits: list[tuple[str, float]] = []
  max_logit = -math.inf
  for option_name, prob_val in ordered:
    logit_val = -30.0 if prob_val <= 1e-13 else math.log(prob_val) / temp_scalar
    scaled_logits.append((option_name, logit_val))
    if logit_val > max_logit:
      max_logit = logit_val

  exp_pairs = [
      (option_name, math.exp(logit_val - max_logit))
      for option_name, logit_val in scaled_logits
  ]
  partition_sum = sum(weight for _, weight in exp_pairs)
  return {
      option_name: weight / partition_sum for option_name, weight in exp_pairs
  }


def resolve_gold_probability_key(
    probabilities: Mapping[str, float],
    expected_label: str,
) -> str | None:
  """Resolve a gold label against probability dictionary keys and aliases.

  Args:
    probabilities: Option probability mapping.
    expected_label: Ground-truth label string.

  Returns:
    Matching key in `probabilities`, or `None` if unresolved.

  Raises:
    TypeError: If `expected_label` is not a string.
  """
  if not probabilities:
    return None
  if expected_label in probabilities:
    return expected_label
  trimmed = expected_label.strip()
  if trimmed in probabilities:
    return trimmed
  lowered = trimmed.lower()
  for candidate_key in probabilities:
    if candidate_key.strip().lower() == lowered:
      return candidate_key
  if lowered in ("true", "yes", "1"):
    for alias in ("yes", "true", "1"):
      for candidate_key in probabilities:
        if candidate_key.strip().lower() == alias:
          return candidate_key
  elif lowered in ("false", "no", "0"):
    for alias in ("no", "false", "0"):
      for candidate_key in probabilities:
        if candidate_key.strip().lower() == alias:
          return candidate_key
  try:
    expected_float = float(trimmed)
  except ValueError:
    return None
  for candidate_key in probabilities:
    try:
      if abs(float(candidate_key.strip()) - expected_float) <= 1e-9:
        return candidate_key
    except ValueError:
      continue
  return None


def fit_temperature_nll(
    probabilities_list: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
    min_temp: float = 0.05,
    max_temp: float = 10.0,
) -> float:
  """Fit post-hoc temperature T minimizing multiclass cross-entropy NLL.

  Exploits strict convexity of negative log-likelihood in inverse temperature
  `beta = 1 / T` via exact golden-section search.

  Args:
    probabilities_list: Sequence of option probability mappings.
    gold_labels: Ground-truth option labels aligned with `probabilities_list`.
    min_temp: Lower bound on fitted temperature (`> 0`).
    max_temp: Upper bound on fitted temperature (`> min_temp`).

  Returns:
    Optimal scalar temperature `T*` in `[min_temp, max_temp]`.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or missing gold
      keys.
  """
  if not probabilities_list:
    raise StatisticalValidationError("probabilities_list must be non-empty")
  if len(probabilities_list) != len(gold_labels):
    raise StatisticalValidationError(
        "probabilities_list and gold_labels length mismatch"
    )
  if min_temp <= 0.0 or max_temp <= min_temp or max_temp > 20.0:
    raise StatisticalValidationError("invalid [min_temp, max_temp] bounds")

  prepared_logits: list[tuple[list[float], int]] = []
  for row_idx, probs_map in enumerate(probabilities_list):
    ordered = _ordered_normalized_probabilities(probs_map)
    resolved_gold = resolve_gold_probability_key(
        probs_map, gold_labels[row_idx]
    )
    if resolved_gold is None:
      raise StatisticalValidationError(
          f"gold label {gold_labels[row_idx]!r} missing at row {row_idx}"
      )
    row_logits: list[float] = []
    gold_pos = -1
    for opt_idx, (opt_name, prob_val) in enumerate(ordered):
      row_logits.append(-30.0 if prob_val <= 1e-13 else math.log(prob_val))
      if opt_name == resolved_gold:
        gold_pos = opt_idx
    prepared_logits.append((row_logits, gold_pos))

  def _mean_nll_at_inv_temp(inv_temp: float) -> float:
    total_nll = 0.0
    for row_logits, gold_pos in prepared_logits:
      scaled = [val * inv_temp for val in row_logits]
      max_val = max(scaled)
      log_sum_exp = max_val + math.log(
          sum(math.exp(val - max_val) for val in scaled)
      )
      total_nll += log_sum_exp - scaled[gold_pos]
    return total_nll / float(len(prepared_logits))

  beta_low = 1.0 / max_temp
  beta_high = 1.0 / min_temp
  golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0

  left_probe = beta_high - golden_ratio * (beta_high - beta_low)
  right_probe = beta_low + golden_ratio * (beta_high - beta_low)
  left_loss = _mean_nll_at_inv_temp(left_probe)
  right_loss = _mean_nll_at_inv_temp(right_probe)

  for _ in range(60):
    if abs(beta_high - beta_low) <= 1e-8:
      break
    if left_loss < right_loss:
      beta_high = right_probe
      right_probe = left_probe
      right_loss = left_loss
      left_probe = beta_high - golden_ratio * (beta_high - beta_low)
      left_loss = _mean_nll_at_inv_temp(left_probe)
    else:
      beta_low = left_probe
      left_probe = right_probe
      left_loss = right_loss
      right_probe = beta_low + golden_ratio * (beta_high - beta_low)
      right_loss = _mean_nll_at_inv_temp(right_probe)

  optimal_beta = 0.5 * (beta_low + beta_high)
  return max(min_temp, min(max_temp, 1.0 / optimal_beta))


def fit_oof_temperature_nll(
    probabilities_list: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
    cluster_ids: Sequence[str],
    num_folds: int = 5,
    seed_tag: str = "dgem-oof-v1",
) -> tuple[float, list[dict[str, float]], tuple[float, ...]]:
  """Fit 5-fold cluster-disjoint out-of-fold temperatures on D_cal.

  Produces OOF-calibrated D_cal probability maps prior to LAC/RAPS quantile
  fitting so temperature estimation cannot overfit conformal scores.

  Args:
    probabilities_list: Calibration probability dictionaries on `D_cal`.
    gold_labels: Ground-truth labels aligned with `probabilities_list`.
    cluster_ids: Cluster identifiers aligned with `probabilities_list`.
    num_folds: Number of cluster-disjoint cross-validation folds (`>= 2`).
    seed_tag: Deterministic HMAC seed tag for cluster-to-fold assignment.

  Returns:
    Tuple `(global_temperature, oof_scaled_probabilities, fold_temperatures)`.

  Raises:
    StatisticalValidationError: If lengths mismatch or fewer than `num_folds`
      unique clusters are supplied.
  """
  num_rows = len(probabilities_list)
  if num_rows == 0:
    raise StatisticalValidationError("probabilities_list must be non-empty")
  if len(gold_labels) != num_rows or len(cluster_ids) != num_rows:
    raise StatisticalValidationError(
        "probabilities_list, gold_labels, and cluster_ids length mismatch"
    )
  if num_folds < 2:
    raise StatisticalValidationError("num_folds must be >= 2")

  unique_clusters = sorted(set(cluster_ids))
  if len(unique_clusters) < num_folds:
    raise StatisticalValidationError(
        f"need >= {num_folds} unique clusters for OOF scaling, got "
        f"{len(unique_clusters)}"
    )

  def _oof_cluster_key(cluster_name: str) -> tuple[int, str]:
    return (hmac_uint64(seed_tag, cluster_name, "oof-fold"), cluster_name)

  ordered_clusters = sorted(unique_clusters, key=_oof_cluster_key)
  cluster_to_fold = {
      cluster_name: idx % num_folds
      for idx, cluster_name in enumerate(ordered_clusters)
  }

  oof_scaled: list[dict[str, float] | None] = [None] * num_rows
  fold_temperatures: list[float] = []

  for fold_idx in range(num_folds):
    train_probs = [
        probabilities_list[row_idx]
        for row_idx in range(num_rows)
        if cluster_to_fold[cluster_ids[row_idx]] != fold_idx
    ]
    train_labels = [
        gold_labels[row_idx]
        for row_idx in range(num_rows)
        if cluster_to_fold[cluster_ids[row_idx]] != fold_idx
    ]
    fold_temp = fit_temperature_nll(train_probs, train_labels)
    fold_temperatures.append(fold_temp)
    for row_idx in range(num_rows):
      if cluster_to_fold[cluster_ids[row_idx]] == fold_idx:
        oof_scaled[row_idx] = apply_temperature(
            probabilities_list[row_idx], fold_temp
        )

  global_temperature = fit_temperature_nll(probabilities_list, gold_labels)
  finalized_oof = [
      entry if entry is not None else dict(probabilities_list[idx])
      for idx, entry in enumerate(oof_scaled)
  ]
  return global_temperature, finalized_oof, tuple(fold_temperatures)


def _validate_conformal_params(
    params: ConformalParameters,
) -> tuple[str, int, float]:
  """Validate method, k_reg, and lambda_reg matching Go ConformalConfig."""
  raw_method = params.method.strip().lower()
  if raw_method in ("", "raps"):
    method = "raps"
  elif raw_method in ("lac", "aps"):
    method = raw_method
  else:
    raise StatisticalValidationError(
        f"unknown conformal method: {params.method!r}"
    )
  if params.k_reg < 0:
    raise StatisticalValidationError(f"k_reg must be >= 0 (got {params.k_reg})")
  if not math.isfinite(params.lambda_reg) or params.lambda_reg < 0.0:
    raise StatisticalValidationError(
        f"lambda_reg must be finite and >= 0 (got {params.lambda_reg})"
    )
  k_reg = params.k_reg
  lambda_reg = params.lambda_reg
  if raw_method == "" and k_reg == 0 and lambda_reg == 0.0:
    k_reg = 1
    lambda_reg = 0.01
  return method, k_reg, lambda_reg


def _partition_tie_groups(
    ordered: Sequence[tuple[str, float]],
) -> list[list[tuple[str, float]]]:
  """Group consecutive options whose probabilities differ by <= 1e-12."""
  groups: list[list[tuple[str, float]]] = []
  cursor = 0
  total = len(ordered)
  while cursor < total:
    anchor_prob = ordered[cursor][1]
    end_cursor = cursor + 1
    while (
        end_cursor < total
        and abs(ordered[end_cursor][1] - anchor_prob)
        <= CONFORMAL_SCORE_TOLERANCE
    ):
      end_cursor += 1
    groups.append(list(ordered[cursor:end_cursor]))
    cursor = end_cursor
  return groups


def compute_nonconformity_scores(
    probabilities: Mapping[str, float],
    params: ConformalParameters,
    item_key_override: str | None = None,
) -> dict[str, float]:
  """Compute LAC, APS, or RAPS nonconformity scores matching Go `conformal.go`.

  Args:
    probabilities: Option probability dictionary.
    params: `ConformalParameters` configuration.
    item_key_override: Optional item key overriding `params.item_key`.

  Returns:
    Dictionary mapping each option name to its nonconformity score.

  Raises:
    StatisticalValidationError: If `probabilities` or `params` is invalid.
  """
  method, k_reg, lambda_reg = _validate_conformal_params(params)
  ordered = _ordered_normalized_probabilities(probabilities)
  effective_item_key = (
      item_key_override if item_key_override is not None else params.item_key
  )

  if not params.randomized:
    if method == "lac":
      return {opt_name: 1.0 - prob_val for opt_name, prob_val in ordered}
    deterministic_scores: dict[str, float] = {}
    cum_strictly_greater = 0.0
    strict_count = 0
    for tie_group in _partition_tie_groups(ordered):
      group_start_rank = strict_count + 1
      rank_penalty = 0.0
      if method == "raps" and group_start_rank > k_reg:
        rank_penalty = lambda_reg * float(group_start_rank - k_reg)
      group_score = cum_strictly_greater + rank_penalty
      group_mass = 0.0
      for opt_name, prob_val in tie_group:
        deterministic_scores[opt_name] = group_score
        group_mass += prob_val
      cum_strictly_greater += group_mass
      strict_count += len(tie_group)
    return deterministic_scores

  uniform_draw = hmac_uniform_float64(params.seed_key, effective_item_key, "u")
  flattened: list[tuple[str, float]] = []
  for tie_group in _partition_tie_groups(ordered):
    if len(tie_group) <= 1:
      flattened.extend(tie_group)
      continue

    def _tie_break_key(entry: tuple[str, float]) -> tuple[int, str]:
      return (
          hmac_uint64(params.seed_key, effective_item_key, f"tie:{entry[0]}"),
          entry[0],
      )

    flattened.extend(sorted(tie_group, key=_tie_break_key))

  randomized_scores: dict[str, float] = {}
  cum_before = 0.0
  for rank_zero, (opt_name, prob_val) in enumerate(flattened):
    rank_one = rank_zero + 1
    if method == "lac":
      randomized_scores[opt_name] = 1.0 - prob_val
    elif method == "aps":
      randomized_scores[opt_name] = cum_before + uniform_draw * prob_val
    else:
      rank_penalty = (
          lambda_reg * float(rank_one - k_reg) if rank_one > k_reg else 0.0
      )
      randomized_scores[opt_name] = (
          cum_before + uniform_draw * prob_val + rank_penalty
      )
    cum_before += prob_val
  return randomized_scores


def calibrate_conformal_quantile(
    calibration_probabilities: Sequence[Mapping[str, float]],
    calibration_labels: Sequence[str],
    params: ConformalParameters,
    item_keys: Sequence[str] | None = None,
) -> ConformalFitSummary:
  """Fit finite-sample split-conformal quantile q_hat on calibration items.

  Args:
    calibration_probabilities: Sequence of per-item probability dictionaries.
    calibration_labels: Sequence of gold labels aligned with probabilities.
    params: `ConformalParameters` configuration including target `alpha`.
    item_keys: Optional per-item HMAC keys when `params.randomized` is True.

  Returns:
    `ConformalFitSummary` with `q_hat` at `ceil((n+1)*(1-alpha) - 1e-12)`.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or invalid.
  """
  if not calibration_probabilities:
    raise StatisticalValidationError(
        "calibration_probabilities must be non-empty"
    )
  num_cal = len(calibration_probabilities)
  if len(calibration_labels) != num_cal:
    raise StatisticalValidationError(
        "calibration_probabilities and calibration_labels length mismatch"
    )
  if item_keys is not None and len(item_keys) != num_cal:
    raise StatisticalValidationError("item_keys length mismatch")
  if not math.isfinite(params.alpha) or not (0.0 < params.alpha < 1.0):
    raise StatisticalValidationError(
        f"alpha must lie strictly in (0, 1) (got {params.alpha})"
    )

  method, _, _ = _validate_conformal_params(params)
  gold_scores: list[float] = []
  for row_idx in range(num_cal):
    probs_map = calibration_probabilities[row_idx]
    resolved_gold = resolve_gold_probability_key(
        probs_map, calibration_labels[row_idx]
    )
    if resolved_gold is None:
      raise StatisticalValidationError(
          f"gold label {calibration_labels[row_idx]!r} missing at index "
          f"{row_idx}"
      )
    key_str = (
        item_keys[row_idx]
        if item_keys is not None and item_keys[row_idx]
        else f"cal:{row_idx}"
    )
    score_map = compute_nonconformity_scores(
        probs_map, params, item_key_override=key_str
    )
    gold_scores.append(score_map[resolved_gold])

  sorted_scores = sorted(gold_scores)
  order_index = max(
      1,
      int(
          math.ceil(
              float(num_cal + 1) * (1.0 - params.alpha)
              - CONFORMAL_SCORE_TOLERANCE
          )
      ),
  )
  if order_index > num_cal:
    order_index = num_cal + 1
    q_hat = math.inf
  else:
    q_hat = sorted_scores[order_index - 1]

  covered_count = sum(
      1
      for score_val in gold_scores
      if math.isinf(q_hat) or score_val <= q_hat + CONFORMAL_SCORE_TOLERANCE
  )
  return ConformalFitSummary(
      method=method,
      alpha=params.alpha,
      num_calibration=num_cal,
      order_index=order_index,
      q_hat=q_hat,
      empirical_coverage=float(covered_count) / float(num_cal),
      sorted_scores=tuple(sorted_scores),
  )


def conformal_prediction_set(
    probabilities: Mapping[str, float],
    q_hat: float,
    params: ConformalParameters,
) -> ConformalPredictionResult:
  """Construct the calibrated prediction set C(x) = {y : s(x, y) <= q_hat}.

  Args:
    probabilities: Option probability dictionary.
    q_hat: Calibrated nonconformity quantile (`>= 0.0` or `+inf`).
    params: `ConformalParameters` configuration.

  Returns:
    `ConformalPredictionResult` ordered by descending probability.

  Raises:
    StatisticalValidationError: If `q_hat`, `probabilities`, or `params` is
      invalid.
  """
  if math.isnan(q_hat) or q_hat == -math.inf or q_hat < 0.0:
    raise StatisticalValidationError(
        f"q_hat must be >= 0 and non-NaN (got {q_hat})"
    )
  ordered = _ordered_normalized_probabilities(probabilities)
  score_map = compute_nonconformity_scores(probabilities, params)
  abstain_lookup = {
      label.strip().lower() for label in params.abstain_labels if label.strip()
  }

  members: list[str] = []
  contains_abstain = False
  for opt_name, _ in ordered:
    if (
        math.isinf(q_hat)
        or score_map[opt_name] <= q_hat + CONFORMAL_SCORE_TOLERANCE
    ):
      members.append(opt_name)
      if opt_name.strip().lower() in abstain_lookup:
        contains_abstain = True

  top_option, top_probability = ordered[0]
  is_singleton = len(members) == 1
  return ConformalPredictionResult(
      prediction_set=tuple(members),
      scores=score_map,
      q_hat=q_hat,
      top_option=top_option,
      top_probability=top_probability,
      set_size=len(members),
      singleton=is_singleton,
      contains_abstain=contains_abstain,
      should_escalate=(not is_singleton) or contains_abstain,
  )


def solve_matched_escalation_threshold(
    val_scores: Sequence[float],
    target_escalation_rate: float,
) -> float:
  """Select the D_val score cutoff escalating target_escalation_rate of items.

  Args:
    val_scores: Uncertainty scores on `D_val` where larger values escalate.
    target_escalation_rate: Target escalation fraction strictly in `(0.0, 1.0)`.

  Returns:
    Scalar threshold `tau` such that `score > tau` triggers Stage-2 escalation.

  Raises:
    StatisticalValidationError: If `val_scores` is empty/non-finite or
      `target_escalation_rate` lies outside `(0.0, 1.0)`.
  """
  if not val_scores:
    raise StatisticalValidationError("val_scores must be non-empty")
  if (
      not math.isfinite(target_escalation_rate)
      or not 0.0 < target_escalation_rate < 1.0
  ):
    raise StatisticalValidationError(
        f"target_escalation_rate must be in (0, 1), got "
        f"{target_escalation_rate}"
    )
  cleaned_scores: list[float] = []
  for raw_score in val_scores:
    numeric_score = float(raw_score)
    if not math.isfinite(numeric_score):
      raise StatisticalValidationError("val_scores must contain finite values")
    cleaned_scores.append(numeric_score)
  ordered_scores = sorted(cleaned_scores)
  keep_fraction = 1.0 - target_escalation_rate
  cutoff_index = min(
      len(ordered_scores) - 1,
      max(0, math.ceil(keep_fraction * len(ordered_scores)) - 1),
  )
  return ordered_scores[cutoff_index]


def expected_calibration_error_equal_mass(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    num_bins: int = 15,
) -> float:
  """Compute equal-mass (quantile-binned) Expected Calibration Error (ECE).

  Args:
    confidences: Predicted top-1 confidence values in `[0.0, 1.0]`.
    correctness: Boolean correctness indicators aligned with `confidences`.
    num_bins: Number of equal-mass quantile bins (`default = 15`).

  Returns:
    Weighted mean absolute calibration gap across `num_bins` quantile bins.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or `num_bins`
      is `< 1`.
  """
  total_items = len(confidences)
  if total_items == 0:
    raise StatisticalValidationError("confidences must be non-empty")
  if len(correctness) != total_items:
    raise StatisticalValidationError(
        "confidences and correctness length mismatch"
    )
  if num_bins < 1:
    raise StatisticalValidationError("num_bins must be >= 1")

  paired_rows = [
      (float(confidences[idx]), 1.0 if correctness[idx] else 0.0)
      for idx in range(total_items)
  ]
  paired_rows.sort()

  effective_bins = min(num_bins, total_items)
  ece_sum = 0.0
  for bin_idx in range(effective_bins):
    start_pos = int(round(bin_idx * total_items / effective_bins))
    end_pos = int(round((bin_idx + 1) * total_items / effective_bins))
    bin_slice = paired_rows[start_pos:end_pos]
    if not bin_slice:
      continue
    mean_conf = sum(row[0] for row in bin_slice) / float(len(bin_slice))
    mean_acc = sum(row[1] for row in bin_slice) / float(len(bin_slice))
    ece_sum += (float(len(bin_slice)) / float(total_items)) * abs(
        mean_conf - mean_acc
    )
  return ece_sum


def multiclass_brier_score(
    probabilities_list: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
) -> float:
  """Compute proper multiclass Brier score mean_i sum_k (p_{i,k} - y_{i,k})^2.

  Args:
    probabilities_list: Sequence of normalized or raw probability dictionaries.
    gold_labels: Ground-truth option labels aligned with `probabilities_list`.

  Returns:
    Mean multiclass Brier score in `[0.0, 2.0]`.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or missing gold
      keys.
  """
  num_items = len(probabilities_list)
  if num_items == 0:
    raise StatisticalValidationError("probabilities_list must be non-empty")
  if len(gold_labels) != num_items:
    raise StatisticalValidationError(
        "probabilities_list and gold_labels length mismatch"
    )

  total_brier = 0.0
  for idx in range(num_items):
    probs_map = probabilities_list[idx]
    ordered = _ordered_normalized_probabilities(probs_map)
    resolved_gold = resolve_gold_probability_key(probs_map, gold_labels[idx])
    if resolved_gold is None:
      raise StatisticalValidationError(
          f"gold label {gold_labels[idx]!r} missing at index {idx}"
      )
    item_brier = 0.0
    for opt_name, prob_val in ordered:
      indicator = 1.0 if opt_name == resolved_gold else 0.0
      diff = prob_val - indicator
      item_brier += diff * diff
    total_brier += item_brier
  return total_brier / float(num_items)


def paired_mcnemar_exact(
    baseline_correct: Sequence[bool],
    treatment_correct: Sequence[bool],
) -> McNemarTestResult:
  """Run the exact two-sided binomial Paired McNemar test on discordant pairs.

  Args:
    baseline_correct: Boolean correctness outcomes for the baseline arm.
    treatment_correct: Boolean correctness outcomes for the treatment arm.

  Returns:
    `McNemarTestResult` with contingency counts, continuity-corrected chi2, and
    exact two-sided binomial p-value.

  Raises:
    StatisticalValidationError: If inputs are empty or differ in length.
  """
  num_total = len(baseline_correct)
  if num_total == 0:
    raise StatisticalValidationError("baseline_correct must be non-empty")
  if len(treatment_correct) != num_total:
    raise StatisticalValidationError(
        "baseline_correct and treatment_correct length mismatch"
    )

  n_11 = 0
  n_10 = 0
  n_01 = 0
  n_00 = 0
  for base_ok, treat_ok in zip(baseline_correct, treatment_correct):
    if base_ok and treat_ok:
      n_11 += 1
    elif base_ok and not treat_ok:
      n_10 += 1
    elif not base_ok and treat_ok:
      n_01 += 1
    else:
      n_00 += 1

  baseline_acc = float(n_11 + n_10) / float(num_total)
  treatment_acc = float(n_11 + n_01) / float(num_total)
  delta_acc = treatment_acc - baseline_acc

  discordant = n_10 + n_01
  if discordant == 0:
    return McNemarTestResult(
        n_total=num_total,
        n_11=n_11,
        n_10=n_10,
        n_01=n_01,
        n_00=n_00,
        baseline_accuracy=baseline_acc,
        treatment_accuracy=treatment_acc,
        delta_accuracy=delta_acc,
        chi2_continuity=0.0,
        p_value_exact=1.0,
    )

  corrected_diff = max(0.0, float(abs(n_01 - n_10)) - 1.0)
  chi2_cc = (corrected_diff * corrected_diff) / float(discordant)

  tail_limit = min(n_10, n_01)
  log_half_power = float(discordant) * math.log(2.0)
  log_fact_discordant = math.lgamma(float(discordant + 1))
  lower_tail_prob = 0.0
  for count_j in range(tail_limit + 1):
    log_choose = (
        log_fact_discordant
        - math.lgamma(float(count_j + 1))
        - math.lgamma(float(discordant - count_j + 1))
    )
    lower_tail_prob += math.exp(log_choose - log_half_power)

  p_exact = min(1.0, max(0.0, 2.0 * lower_tail_prob))
  return McNemarTestResult(
      n_total=num_total,
      n_11=n_11,
      n_10=n_10,
      n_01=n_01,
      n_00=n_00,
      baseline_accuracy=baseline_acc,
      treatment_accuracy=treatment_acc,
      delta_accuracy=delta_acc,
      chi2_continuity=chi2_cc,
      p_value_exact=p_exact,
  )


def tango_non_inferiority_test(
    baseline_correct: Sequence[bool],
    treatment_correct: Sequence[bool],
    margin: float = 0.010,
) -> TangoNonInferiorityResult:
  """Compute Tango's (1998) restricted-MLE score z-test for non-inferiority.

  Tests `H_0: p_treatment - p_baseline <= -margin` against
  `H_1: p_treatment - p_baseline > -margin` on paired binary outcomes.

  Args:
    baseline_correct: Boolean correctness outcomes for the reference arm.
    treatment_correct: Boolean correctness outcomes for the candidate arm.
    margin: Positive non-inferiority margin `delta_0` in `(0.0, 1.0)`.

  Returns:
    `TangoNonInferiorityResult` containing score z-statistic and one-sided p.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or `margin`
      lies outside `(0.0, 1.0)`.
  """
  if not math.isfinite(margin) or not 0.0 < margin < 1.0:
    raise StatisticalValidationError(
        f"non-inferiority margin must lie in (0, 1), got {margin}"
    )
  mcnemar_summary = paired_mcnemar_exact(baseline_correct, treatment_correct)
  total_n = float(mcnemar_summary.n_total)
  n_10 = float(mcnemar_summary.n_10)
  n_01 = float(mcnemar_summary.n_01)

  # Restricted MLE for pi_01 under H_0: pi_10 - pi_01 = margin (Tango 1998).
  quad_a = 2.0 * total_n
  quad_b = (2.0 * total_n - n_10 + n_01) * margin - n_10 - n_01
  quad_c = -n_01 * margin * (1.0 - margin)
  discriminant = max(0.0, quad_b * quad_b - 4.0 * quad_a * quad_c)
  pi_01_hat = max(
      0.0,
      min(
          0.5 * (1.0 - margin),
          (-quad_b + math.sqrt(discriminant)) / (2.0 * quad_a),
      ),
  )

  null_variance = total_n * (2.0 * pi_01_hat + margin - margin * margin)
  if null_variance <= 0.0:
    z_stat = 0.0
  else:
    z_stat = ((n_01 - n_10) + total_n * margin) / math.sqrt(null_variance)

  p_one_sided = 0.5 * math.erfc(z_stat / math.sqrt(2.0))
  return TangoNonInferiorityResult(
      n_total=mcnemar_summary.n_total,
      n_10=mcnemar_summary.n_10,
      n_01=mcnemar_summary.n_01,
      margin=margin,
      delta_accuracy=mcnemar_summary.delta_accuracy,
      z_score=z_stat,
      p_value_one_sided=p_one_sided,
      non_inferior_at_005=(p_one_sided < 0.05),
  )


def paired_cluster_bootstrap_ci(
    cluster_ids: Sequence[str],
    baseline_values: Sequence[float],
    treatment_values: Sequence[float],
    num_draws: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20261111,
) -> BootstrapConfidenceInterval:
  """Compute a B-draw paired cluster-bootstrap CI for mean(treatment - base).

  Args:
    cluster_ids: Cluster identifier per observation.
    baseline_values: Baseline scalar metric per observation.
    treatment_values: Treatment scalar metric per observation.
    num_draws: Number of cluster-level bootstrap resamples (`default = 2000`).
    confidence_level: Confidence level in `(0.0, 1.0)` (`default = 0.95`).
    seed: Deterministic PRNG seed.

  Returns:
    `BootstrapConfidenceInterval` for `mean(treatment) - mean(baseline)`.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or invalid.
  """
  if len(baseline_values) != len(treatment_values):
    raise StatisticalValidationError(
        "baseline_values and treatment_values length mismatch"
    )
  diffs = [
      float(treat_val) - float(base_val)
      for base_val, treat_val in zip(baseline_values, treatment_values)
  ]
  return cluster_bootstrap_mean_ci(
      cluster_ids=cluster_ids,
      values=diffs,
      num_draws=num_draws,
      confidence_level=confidence_level,
      seed=seed,
  )


def cluster_bootstrap_mean_ci(
    cluster_ids: Sequence[str],
    values: Sequence[float],
    num_draws: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20261111,
) -> BootstrapConfidenceInterval:
  """Compute a B-draw cluster-bootstrap percentile CI for a sample mean.

  Pre-aggregates per-cluster `(sum, count)` tuples so `2,000` resamples run in
  `O(B * num_clusters)` time while preserving within-cluster correlation.

  Args:
    cluster_ids: Cluster identifier per observation.
    values: Numeric observation values.
    num_draws: Number of cluster resamples (`>= 100`).
    confidence_level: Interval coverage level in `(0.0, 1.0)`.
    seed: Deterministic PRNG seed.

  Returns:
    `BootstrapConfidenceInterval` with point estimate and percentile bounds.

  Raises:
    StatisticalValidationError: If inputs are empty, mismatched, or invalid.
  """
  total_obs = len(values)
  if total_obs == 0:
    raise StatisticalValidationError("values must be non-empty")
  if len(cluster_ids) != total_obs:
    raise StatisticalValidationError("cluster_ids and values length mismatch")
  if num_draws < 100:
    raise StatisticalValidationError("num_draws must be >= 100")
  if not 0.0 < confidence_level < 1.0:
    raise StatisticalValidationError("confidence_level must lie in (0, 1)")

  cluster_totals: dict[str, list[float]] = {}
  overall_sum = 0.0
  for cluster_name, raw_val in zip(cluster_ids, values):
    numeric_val = float(raw_val)
    overall_sum += numeric_val
    bucket = cluster_totals.get(cluster_name)
    if bucket is None:
      cluster_totals[cluster_name] = [numeric_val, 1.0]
    else:
      bucket[0] += numeric_val
      bucket[1] += 1.0

  ordered_names = sorted(cluster_totals.keys())
  cluster_stats = [
      (cluster_totals[name][0], cluster_totals[name][1])
      for name in ordered_names
  ]
  num_clusters = len(cluster_stats)
  point_estimate = overall_sum / float(total_obs)

  if num_clusters < 2:
    return BootstrapConfidenceInterval(
        estimate=point_estimate,
        ci_lower=point_estimate,
        ci_upper=point_estimate,
        num_clusters=num_clusters,
        num_draws=num_draws,
    )

  rng = random.Random(seed)
  replicate_means: list[float] = []
  for _ in range(num_draws):
    sampled_sum = 0.0
    sampled_count = 0.0
    for _ in range(num_clusters):
      cluster_sum, cluster_count = cluster_stats[rng.randrange(num_clusters)]
      sampled_sum += cluster_sum
      sampled_count += cluster_count
    replicate_means.append(sampled_sum / sampled_count)

  replicate_means.sort()
  alpha_tail = 0.5 * (1.0 - confidence_level)
  lower_idx = max(0, min(num_draws - 1, int(math.floor(alpha_tail * num_draws))))
  upper_idx = max(
      0, min(num_draws - 1, int(math.ceil((1.0 - alpha_tail) * num_draws)) - 1)
  )
  return BootstrapConfidenceInterval(
      estimate=point_estimate,
      ci_lower=replicate_means[lower_idx],
      ci_upper=replicate_means[upper_idx],
      num_clusters=num_clusters,
      num_draws=num_draws,
  )


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
  """Apply Holm-Bonferroni step-down family-wise error rate (FWER) adjustment.

  Args:
    p_values: Sequence of raw hypothesis-test p-values in `[0.0, 1.0]`.

  Returns:
    Monotone FWER-adjusted p-values aligned with `p_values`.

  Raises:
    StatisticalValidationError: If any p-value is non-finite or outside `[0, 1]`.
  """
  num_tests = len(p_values)
  if num_tests == 0:
    return []
  indexed_p: list[tuple[float, int]] = []
  for idx, raw_p in enumerate(p_values):
    numeric_p = float(raw_p)
    if not math.isfinite(numeric_p) or not 0.0 <= numeric_p <= 1.0:
      raise StatisticalValidationError(
          f"p-value at index {idx} must lie in [0, 1], got {raw_p}"
      )
    indexed_p.append((numeric_p, idx))

  indexed_p.sort()
  adjusted = [0.0] * num_tests
  running_max = 0.0
  for step_idx, (raw_p, orig_idx) in enumerate(indexed_p):
    multiplier = float(num_tests - step_idx)
    candidate_p = min(1.0, multiplier * raw_p)
    if candidate_p > running_max:
      running_max = candidate_p
    adjusted[orig_idx] = running_max
  return adjusted
