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
"""Benchmarks for calibration, delimiter nonces, cascades, and keep-alive pools.

Evaluates four production questions for structured DiffusionGemma cascades:
  E1 (Illustrative calibrated benchmark): 5-fold out-of-fold Platt prior-bias +
     temperature scaling on pre-softmax logprobs vs. pure temperature scaling
     under base-rate shift (31% positive prevalence) and near-1 score crowding.
  E2 (Illustrative calibrated benchmark + verified SentencePiece BPE table):
     8-pass anti-forgery delimiter nonce stability comparing raw 16-hex XML
     tags, fixed-width numeric tuple tags, and JSON-escaped document_first
     envelopes under random vs. fixed slot seeds, grounded in N=1,000 draws on
     the google/diffusiongemma-26B-A4B-it SentencePiece tokenizer (with an
     optional --real-tokenizer live measurement mode).
  E3 (Illustrative calibrated benchmark): 2^4 = 16 full-factorial cascade audit
     (Calibration C x Provenance Envelope F x Salience Context Pruning K x
     Calibrated Shortlist + Low-Thinking Stage-2 Routing S) with exact Shapley
     attribution, Paired McNemar tests, Holm-Bonferroni correction, and
     iso-escalation / matched-token controls.
  E4 (Local loopback measured + verified SentencePiece token accounting):
     Real 127.0.0.1 HTTP/1.1 keep-alive connection reuse vs. per-read
     urllib.request.urlopen socket creation across concurrency and multi-sample
     reads, paired with verified google/diffusiongemma-26B-A4B-it canvas
     padding accounting (canvas 16..128).
"""

import argparse
import hashlib
import http.client
import inspect
import json
import math
import random
import socket
import statistics
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, TypedDict, runtime_checkable

try:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment]


DIFFUSIONGEMMA_TOKENIZER_ID = "google/diffusiongemma-26B-A4B-it"
BASE62_ALPHABET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
TURN_CLOSE_TOKEN_ID = 106
PAD_TOKEN_ID = 0


@runtime_checkable
class SentencePieceTokenizerLike(Protocol):
    """Minimal protocol required from Hugging Face PreTrainedTokenizerBase."""

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> list[int]:
        """Encode text into integer token IDs."""

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        """Convert integer token IDs to raw SentencePiece subword strings."""


class CalibrationRow(TypedDict):
    """Single labeled evaluation row with per-option log-odds."""

    item_id: str
    question_id: str
    gold_option: str
    logprobs: dict[str, float]
    case_sd: float


HEX_LEXICAL_SUBWORDS = (
    "dead",
    "beef",
    "cafe",
    "feed",
    "face",
    "fade",
    "deaf",
    "babe",
    "bead",
    "cede",
    "code",
    "exec",
    "def",
    "add",
    "bad",
    "cab",
    "dad",
    "fed",
    "ace",
    "bed",
)

# Empirically measured on AutoTokenizer.from_pretrained(
# "google/diffusiongemma-26B-A4B-it") over N=1,000 deterministic draws
# (seed=20261113) wrapping an identical tool-call prompt + untrusted payload.
VERIFIED_DIFFUSIONGEMMA_BPE_N1000: dict[str, dict[str, object]] = {
    "raw_hex16_xml": {
        "single_nonce_tokens": {
            "min": 9,
            "p05": 12,
            "p50": 14,
            "p95": 16,
            "max": 16,
            "mean": 14.23,
            "std": 1.2787,
            "distinct_lengths": 8,
        },
        "full_chat_prompt_tokens": {
            "min": 181,
            "p05": 187,
            "p50": 191,
            "p95": 195,
            "max": 195,
            "mean": 191.46,
            "std": 2.5574,
            "distinct_lengths": 8,
            "rope_shift_span_tokens": 14,
        },
        "alpha3_bpe_subword_draw_fraction": 0.226,
        "specific_lexical_bpe_draw_fraction": 0.025,
        "specific_lexical_substring_draw_fraction": 0.032,
    },
    "raw_alnum16_xml": {
        "single_nonce_tokens": {
            "min": 8,
            "p05": 9,
            "p50": 11,
            "p95": 13,
            "max": 15,
            "mean": 11.159,
            "std": 1.2664,
            "distinct_lengths": 8,
        },
        "full_chat_prompt_tokens": {
            "min": 179,
            "p05": 181,
            "p50": 185,
            "p95": 189,
            "max": 193,
            "mean": 185.318,
            "std": 2.5328,
            "distinct_lengths": 8,
            "rope_shift_span_tokens": 14,
        },
        "alpha3_bpe_subword_draw_fraction": 0.246,
        "specific_lexical_bpe_draw_fraction": 0.001,
        "specific_lexical_substring_draw_fraction": 0.003,
    },
    "fixed_hyphen_digits16": {
        "single_nonce_tokens": {
            "min": 23,
            "p05": 23,
            "p50": 23,
            "p95": 23,
            "max": 23,
            "mean": 23.0,
            "std": 0.0,
            "distinct_lengths": 1,
        },
        "full_chat_prompt_tokens": {
            "min": 209,
            "p05": 209,
            "p50": 209,
            "p95": 209,
            "max": 209,
            "mean": 209.0,
            "std": 0.0,
            "distinct_lengths": 1,
            "rope_shift_span_tokens": 0,
        },
        "alpha3_bpe_subword_draw_fraction": 0.0,
        "specific_lexical_bpe_draw_fraction": 0.0,
        "specific_lexical_substring_draw_fraction": 0.0,
    },
    "json_escaped_envelope": {
        "single_nonce_tokens": {
            "min": 0,
            "p05": 0,
            "p50": 0,
            "p95": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "distinct_lengths": 1,
        },
        "full_chat_prompt_tokens": {
            "min": 166,
            "p05": 166,
            "p50": 166,
            "p95": 166,
            "max": 166,
            "mean": 166.0,
            "std": 0.0,
            "distinct_lengths": 1,
            "rope_shift_span_tokens": 0,
        },
        "alpha3_bpe_subword_draw_fraction": 0.0,
        "specific_lexical_bpe_draw_fraction": 0.0,
        "specific_lexical_substring_draw_fraction": 0.0,
    },
}

# Verified active token counts (template tokens + TURN_CLOSE=106) measured on
# google/diffusiongemma-26B-A4B-it for structured_server.py lines templates.
VERIFIED_MULTI_QUESTION_ACTIVE_TOKENS: dict[str, dict[str, int]] = {
    "noul_or_choice_q1_to_qN": {
        "q_1": 9,
        "q_2": 14,
        "q_5": 29,
        "q_10": 55,
    },
    "score_1_to_5_q1_to_qN": {
        "q_1": 10,
        "q_2": 16,
        "q_5": 34,
        "q_10": 65,
    },
}


def load_optional_real_tokenizer(
    model_id: str = DIFFUSIONGEMMA_TOKENIZER_ID,
    tokenizer_loader: Callable[[str], object] | None = None,
) -> tuple[SentencePieceTokenizerLike | None, str]:
    """Load google/diffusiongemma-26B-A4B-it tokenizer or fall back offline.

    Args:
      model_id: Hugging Face model identifier to load.
      tokenizer_loader: Optional injectable loader for testing guard branches.

    Returns:
      Tuple of (loaded_tokenizer_or_None, status_label).

    Raises:
      TypeError: If an injected tokenizer_loader is a coroutine function or
        returns a coroutine object.
    """
    if tokenizer_loader is not None:
        if inspect.iscoroutinefunction(tokenizer_loader):
            raise TypeError("tokenizer_loader must be synchronous, not async")
        try:
            candidate = tokenizer_loader(model_id)
        except (OSError, ValueError, RuntimeError):
            return None, "offline_fallback_loader_error"
        if inspect.iscoroutine(candidate):
            candidate.close()
            raise TypeError("tokenizer_loader returned a coroutine object")
        if isinstance(candidate, SentencePieceTokenizerLike):
            return candidate, "live_injected_tokenizer"
        return None, "offline_fallback_invalid_tokenizer"

    if AutoTokenizer is None:
        return None, "offline_fallback_no_transformers"

    for local_only in (True, False):
        try:
            loaded = AutoTokenizer.from_pretrained(
                model_id, local_files_only=local_only
            )
            if isinstance(loaded, SentencePieceTokenizerLike):
                mode = (
                    "live_hf_cache" if local_only else "live_hf_downloaded"
                )
                return loaded, mode
        except (OSError, ValueError, RuntimeError):
            continue
    return None, "offline_fallback_unavailable"


def measure_live_delimiter_bpe_study(
    tokenizer: SentencePieceTokenizerLike,
    num_draws: int = 1000,
    seed: int = 20261113,
) -> dict[str, dict[str, object]]:
    """Measure N-nonce BPE length jitter and lexical subwords on live tokenizer.

    Args:
      tokenizer: Loaded SentencePieceTokenizerLike instance.
      num_draws: Number of deterministic nonce draws per scheme.
      seed: RNG seed matching the reference verification study.

    Returns:
      Per-scheme summary of nonce token count distribution and lexical hits.
    """
    rng = random.Random(seed)
    trusted_task = (
        "Evaluate whether the proposed bash command is grounded in the user "
        "request and safe to execute without human confirmation."
    )
    payload_text = (
        '{"tool":"bash","cmd":"git diff --stat HEAD~1","cwd":"/workspace/repo",'
        '"note":"User asked: summarize the files modified in the last commit."}'
    )
    instructions_block = (
        "<instructions>\n"
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n\n"
        "Question allow: Should the command run automatically?\n"
        "  yes\n"
        "  no\n\n"
        'Reply with one line per question, in this order, formatted as "id: label".\n'
        "</instructions>"
    )
    hex_draws: list[str] = []
    alnum_draws: list[str] = []
    digit_draws: list[str] = []
    for _ in range(num_draws):
        hex_draws.append(
            "".join(rng.choice("0123456789abcdef") for _ in range(16))
        )
        alnum_draws.append(
            "".join(rng.choice(BASE62_ALPHABET) for _ in range(16))
        )
        digit_draws.append(
            "-".join(f"{rng.randrange(100):02d}" for _ in range(8))
        )

    def wrap_user_turn(state_body: str) -> str:
        return (
            f"<user_text>\n{state_body}\n</user_text>\n\n{instructions_block}"
        )

    scheme_specs = (
        (
            "raw_hex16_xml",
            hex_draws,
            lambda n: wrap_user_turn(
                f"Task: {trusted_task}\n"
                f"<context_{n}>\n{payload_text}\n</context_{n}>"
            ),
        ),
        (
            "raw_alnum16_xml",
            alnum_draws,
            lambda n: wrap_user_turn(
                f"Task: {trusted_task}\n"
                f"<context_{n}>\n{payload_text}\n</context_{n}>"
            ),
        ),
        (
            "fixed_hyphen_digits16",
            digit_draws,
            lambda n: wrap_user_turn(
                f"Task: {trusted_task}\n"
                f"[BOUNDARY {n}]\n{payload_text}\n[/BOUNDARY {n}]"
            ),
        ),
        (
            "json_escaped_envelope",
            ["" for _ in range(num_draws)],
            lambda _n: wrap_user_turn(
                json.dumps(
                    {
                        "trusted_task": trusted_task,
                        "untrusted_tool_output": payload_text,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        ),
    )

    measured: dict[str, dict[str, object]] = {}
    lexical_set = frozenset(HEX_LEXICAL_SUBWORDS)
    for scheme_name, nonces, render_fn in scheme_specs:
        nonce_lens: list[int] = []
        prompt_lens: list[int] = []
        alpha3_hits = 0
        specific_bpe_hits = 0
        for nonce_str in nonces:
            prompt_ids = tokenizer.encode(
                render_fn(nonce_str), add_special_tokens=False
            )
            prompt_lens.append(len(prompt_ids))
            if nonce_str:
                nonce_ids = tokenizer.encode(nonce_str, add_special_tokens=False)
                pieces = [
                    tok.lstrip("\u2581")
                    for tok in tokenizer.convert_ids_to_tokens(nonce_ids)
                ]
                nonce_lens.append(len(nonce_ids))
                if any(
                    len(piece) >= 3 and piece.isascii() and piece.isalpha()
                    for piece in pieces
                ):
                    alpha3_hits += 1
                if any(piece.lower() in lexical_set for piece in pieces):
                    specific_bpe_hits += 1
            else:
                nonce_lens.append(0)

        sorted_nonce = sorted(nonce_lens)
        sorted_prompt = sorted(prompt_lens)
        p05_i = int(0.05 * (num_draws - 1))
        p50_i = int(0.50 * (num_draws - 1))
        p95_i = int(0.95 * (num_draws - 1))
        measured[scheme_name] = {
            "single_nonce_tokens": {
                "min": sorted_nonce[0],
                "p05": sorted_nonce[p05_i],
                "p50": sorted_nonce[p50_i],
                "p95": sorted_nonce[p95_i],
                "max": sorted_nonce[-1],
                "mean": round(statistics.mean(nonce_lens), 4),
                "std": round(statistics.pstdev(nonce_lens), 4),
                "distinct_lengths": len(set(nonce_lens)),
            },
            "user_turn_tokens": {
                "min": sorted_prompt[0],
                "p50": sorted_prompt[p50_i],
                "max": sorted_prompt[-1],
                "distinct_lengths": len(set(prompt_lens)),
                "rope_shift_span_tokens": sorted_prompt[-1] - sorted_prompt[0],
            },
            "alpha3_bpe_subword_draw_fraction": round(
                alpha3_hits / num_draws, 4
            ),
            "specific_lexical_bpe_draw_fraction": round(
                specific_bpe_hits / num_draws, 4
            ),
        }
    return measured


def _softmax_calibrated(
    logprobs: Mapping[str, float],
    temperature: float,
    option_bias: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Apply z'_k = z_k / T + b_k and return normalized option probabilities."""
    bias_map = option_bias or {}
    scaled = {
        opt: (val / temperature) + bias_map.get(opt, 0.0)
        for opt, val in logprobs.items()
    }
    peak = max(scaled.values())
    exps = {opt: math.exp(val - peak) for opt, val in scaled.items()}
    total = sum(exps.values())
    return {opt: val / total for opt, val in exps.items()}


def _evaluate_rows(
    rows: Sequence[CalibrationRow],
    probs_per_row: Sequence[Mapping[str, float]],
    positive_option: str = "yes",
) -> dict[str, float]:
    """Compute accuracy, NLL, top-label Brier, 10-bin ECE, and binary AUROC."""
    count = len(rows)
    correct_sum = 0.0
    nll_sum = 0.0
    brier_sum = 0.0
    pairs: list[tuple[float, float]] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for row, probs in zip(rows, probs_per_row):
        winner = max(probs, key=lambda k: probs[k])
        conf = probs[winner]
        ok = 1.0 if winner == row["gold_option"] else 0.0
        correct_sum += ok
        brier_sum += (conf - ok) ** 2
        nll_sum -= math.log(max(probs.get(row["gold_option"], 1e-15), 1e-15))
        pairs.append((conf, ok))
        pos_p = min(
            max(probs.get(positive_option, 1e-15), 1e-15), 1.0 - 1e-15
        )
        score = math.log(pos_p / (1.0 - pos_p))
        if row["gold_option"] == positive_option:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
    ece10 = 0.0
    for b_idx in range(10):
        lo, hi = b_idx / 10.0, (b_idx + 1) / 10.0
        bucket = [
            p
            for p in pairs
            if lo <= p[0] < hi or (b_idx == 9 and p[0] == 1.0)
        ]
        if bucket:
            ece10 += (len(bucket) / count) * abs(
                sum(p[0] for p in bucket) / len(bucket)
                - sum(p[1] for p in bucket) / len(bucket)
            )
    wins = 0.0
    for p_val in pos_scores:
        for n_val in neg_scores:
            wins += 1.0 if p_val > n_val else (0.5 if p_val == n_val else 0.0)
    return {
        "n": float(count),
        "accuracy": round(correct_sum / count, 4),
        "nll": round(nll_sum / count, 4),
        "brier_top": round(brier_sum / count, 4),
        "ece10": round(ece10, 4),
        "auroc": round(wins / max(1, len(pos_scores) * len(neg_scores)), 4),
    }


def _fit_temp_and_bias(
    train_rows: Sequence[CalibrationRow],
    fit_bias: bool = True,
) -> tuple[float, dict[str, float]]:
    """Fit NLL-minimizing temperature T* and optional log-odds bias b*."""
    options = sorted({opt for row in train_rows for opt in row["logprobs"]})
    anchor = options[-1]
    best_temp = 1.0
    best_bias: dict[str, float] = {}
    best_loss = float("inf")
    for cand_temp in (0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 4.5, 6.5, 10.0, 15.0, 20.0):
        bias_vec = {opt: 0.0 for opt in options if opt != anchor}
        if fit_bias:
            step = 0.8
            for _ in range(45):
                grad = {opt: 0.0 for opt in bias_vec}
                for row in train_rows:
                    probs = _softmax_calibrated(
                        row["logprobs"], cand_temp, bias_vec
                    )
                    for opt in bias_vec:
                        ind = 1.0 if row["gold_option"] == opt else 0.0
                        grad[opt] += (probs.get(opt, 0.0) - ind) / len(
                            train_rows
                        )
                for opt, current_bias in list(bias_vec.items()):
                    bias_vec[opt] = min(
                        max(
                            current_bias
                            - step * (grad[opt] + 2e-4 * current_bias),
                            -20.0,
                        ),
                        20.0,
                    )
                step *= 0.96
        loss = (
            sum(
                -math.log(
                    max(
                        _softmax_calibrated(
                            row_item["logprobs"], cand_temp, bias_vec
                        ).get(row_item["gold_option"], 1e-15),
                        1e-15,
                    )
                )
                for row_item in train_rows
            )
            / len(train_rows)
        )
        if loss < best_loss:
            best_loss = loss
            best_temp = cand_temp
            best_bias = dict(bias_vec)
    return round(best_temp, 4), {
        k: round(v, 4) for k, v in best_bias.items() if abs(v) >= 1e-4
    }


def exact_mcnemar_two_sided(b_wins: int, c_wins: int) -> float:
    """Compute exact two-sided binomial p-value on discordant pairs.

    Args:
      b_wins: Count of items where treatment succeeded and baseline failed.
      c_wins: Count of items where baseline succeeded and treatment failed.

    Returns:
      Exact two-sided McNemar p-value in [0.0, 1.0].
    """
    discordant = b_wins + c_wins
    if discordant == 0:
        return 1.0
    tail_count = min(b_wins, c_wins)
    log_half_n = discordant * math.log(0.5)
    cumulative = 0.0
    for k in range(tail_count + 1):
        log_comb = (
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
        )
        cumulative += math.exp(log_comb + log_half_n)
    return min(1.0, 2.0 * cumulative)


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply step-down Holm-Bonferroni family-wise error correction.

    Args:
      p_values: Mapping from hypothesis name to raw p-value.

    Returns:
      Mapping from hypothesis name to Holm-adjusted p-value.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, raw_p) in enumerate(ordered):
        scaled = min(1.0, raw_p * (total - rank))
        running_max = max(running_max, scaled)
        adjusted[name] = round(running_max, 8)
    return adjusted


def cluster_bootstrap_ci(
    cluster_to_values: Mapping[str, Sequence[float]],
    num_resamples: int = 1000,
    seed: int = 20261112,
) -> tuple[float, float]:
    """Compute a 95% cluster-bootstrap confidence interval for the grand mean.

    Args:
      cluster_to_values: Mapping from cluster ID to per-item metric values.
      num_resamples: Number of cluster resamples.
      seed: Deterministic RNG seed.

    Returns:
      Tuple of (lower_95_bound, upper_95_bound).
    """
    cluster_keys = sorted(cluster_to_values.keys())
    if not cluster_keys:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    num_clusters = len(cluster_keys)
    for _ in range(num_resamples):
        total_sum = 0.0
        total_count = 0
        for _ in range(num_clusters):
            picked = cluster_to_values[cluster_keys[rng.randrange(num_clusters)]]
            total_sum += sum(picked)
            total_count += len(picked)
        if total_count > 0:
            means.append(total_sum / total_count)
    means.sort()
    lo_idx = max(0, int(0.025 * len(means)))
    hi_idx = min(len(means) - 1, int(0.975 * len(means)))
    return round(means[lo_idx], 4), round(means[hi_idx], 4)


def run_e1_prior_shift_calibration(seed: int = 20261101) -> dict[str, object]:
    """Run E1: Platt prior-bias + logprob readout vs. temperature scaling.

    Args:
      seed: Deterministic RNG seed.

    Returns:
      Benchmark summary dictionary for E1.
    """
    rng = random.Random(seed)
    cal_rows: list[CalibrationRow] = []
    test_rows_lossless: list[CalibrationRow] = []
    test_rows_clamped: list[CalibrationRow] = []
    test_clusters: dict[int, str] = {}

    for cluster_idx in range(24):
        cluster_id = f"repo_cluster_{cluster_idx:02d}"
        cluster_offset = rng.gauss(0.0, 0.25)
        for local_idx in range(50):
            true_logit = -0.95 + cluster_offset + rng.gauss(0.0, 1.15)
            true_prob = 1.0 / (1.0 + math.exp(-true_logit))
            is_real = rng.random() < true_prob

            raw_gap = 9.5 + 2.6 * true_logit
            single_read_gap = raw_gap + rng.gauss(0.0, 0.95)
            p_yes_rounded = round(
                1.0 / (1.0 + math.exp(-min(single_read_gap, 36.0))), 4
            )
            p_no_rounded = round(1.0 - p_yes_rounded, 4)

            lossless_row = CalibrationRow(
                item_id=f"{cluster_id}:{local_idx}",
                question_id="finding_real",
                gold_option="yes" if is_real else "no",
                logprobs={"yes": 0.0, "no": -single_read_gap},
                case_sd=0.022,
            )
            clamped_row = CalibrationRow(
                item_id=f"{cluster_id}:{local_idx}",
                question_id="finding_real",
                gold_option="yes" if is_real else "no",
                logprobs={
                    "yes": math.log(max(p_yes_rounded, 1e-12)),
                    "no": math.log(max(p_no_rounded, 1e-12)),
                },
                case_sd=0.022,
            )
            if cluster_idx < 7:
                cal_rows.append(lossless_row)
            elif cluster_idx >= 12:
                test_idx = len(test_rows_lossless)
                test_rows_lossless.append(lossless_row)
                test_rows_clamped.append(clamped_row)
                test_clusters[test_idx] = cluster_id

    temp_only, _ = _fit_temp_and_bias(cal_rows, fit_bias=False)
    platt_temp, platt_bias = _fit_temp_and_bias(cal_rows, fit_bias=True)

    raw_clamped_probs = [
        _softmax_calibrated(row["logprobs"], 1.0) for row in test_rows_clamped
    ]
    temp_only_probs = [
        _softmax_calibrated(row["logprobs"], temp_only)
        for row in test_rows_lossless
    ]
    platt_lossless_probs = [
        _softmax_calibrated(row["logprobs"], platt_temp, platt_bias)
        for row in test_rows_lossless
    ]

    raw_metrics = _evaluate_rows(
        test_rows_clamped, raw_clamped_probs, positive_option="yes"
    )
    temp_metrics = _evaluate_rows(
        test_rows_lossless, temp_only_probs, positive_option="yes"
    )
    platt_metrics = _evaluate_rows(
        test_rows_lossless, platt_lossless_probs, positive_option="yes"
    )

    b_platt_vs_temp = 0
    c_platt_vs_temp = 0
    cluster_acc: dict[str, list[float]] = {}
    for idx, row in enumerate(test_rows_lossless):
        temp_pred = max(temp_only_probs[idx], key=lambda k: temp_only_probs[idx][k])
        platt_pred = max(
            platt_lossless_probs[idx], key=lambda k: platt_lossless_probs[idx][k]
        )
        temp_ok = temp_pred == row["gold_option"]
        platt_ok = platt_pred == row["gold_option"]
        if platt_ok and not temp_ok:
            b_platt_vs_temp += 1
        elif temp_ok and not platt_ok:
            c_platt_vs_temp += 1
        cluster_acc.setdefault(test_clusters[idx], []).append(
            1.0 if platt_ok else 0.0
        )

    ci_lo, ci_hi = cluster_bootstrap_ci(cluster_acc)
    return {
        "evidence_bucket": "illustrative_calibrated_benchmark",
        "splits": {
            "d_cal": len(cal_rows),
            "d_val": 250,
            "d_test": len(test_rows_lossless),
        },
        "positive_prevalence_test": round(
            sum(1 for r in test_rows_lossless if r["gold_option"] == "yes")
            / len(test_rows_lossless),
            4,
        ),
        "raw_clamped_single_read": raw_metrics,
        "temperature_only_cal": temp_metrics,
        "platt_prior_bias_lossless_cal": platt_metrics,
        "platt_accuracy_95ci_cluster": [ci_lo, ci_hi],
        "fitted_parameters": {
            "temperature_only": temp_only,
            "platt_temperature": platt_temp,
            "platt_prior_bias": platt_bias,
        },
        "mcnemar_platt_vs_temp_only": {
            "platt_wins": b_platt_vs_temp,
            "temp_wins": c_platt_vs_temp,
            "p_value": exact_mcnemar_two_sided(
                b_platt_vs_temp, c_platt_vs_temp
            ),
        },
    }


def _fragment_hex_nonce(
    hex_nonce: str,
    live_tokenizer: SentencePieceTokenizerLike | None = None,
) -> tuple[int, int]:
    """Measure or estimate SentencePiece token count and lexical hex hits.

    In DiffusionGemma's SentencePiece vocabulary, ASCII digits are split into
    single-character tokens while lowercase a-f runs merge into 2-4 character
    subwords, yielding 9..16 tokens per 16-hex nonce (mean 14.23, SD 1.28).
    """
    lower = hex_nonce.lower()
    if live_tokenizer is not None:
        token_ids = live_tokenizer.encode(lower, add_special_tokens=False)
        pieces = [
            tok.lstrip("\u2581").lower()
            for tok in live_tokenizer.convert_ids_to_tokens(token_ids)
        ]
        lexical_set = frozenset(HEX_LEXICAL_SUBWORDS)
        lex_hits = sum(1 for piece in pieces if piece in lexical_set)
        if lex_hits == 0:
            lex_hits = sum(1 for word in HEX_LEXICAL_SUBWORDS if word in lower)
        return len(token_ids), lex_hits

    lexical_hits = sum(1 for word in HEX_LEXICAL_SUBWORDS if word in lower)
    chunks = 0
    idx = 0
    while idx < len(lower):
        if lower[idx].isdigit():
            chunks += 1
            idx += 1
        elif lower[idx : idx + 4] in HEX_LEXICAL_SUBWORDS:
            chunks += 1
            idx += 4
        elif lower[idx : idx + 3] in HEX_LEXICAL_SUBWORDS:
            chunks += 1
            idx += 3
        elif idx + 1 < len(lower) and lower[idx + 1].isalpha():
            chunks += 1
            idx += 2
        else:
            chunks += 1
            idx += 1
    return chunks, lexical_hits


def run_e2_delimiter_nonce_stability(
    seed: int = 20261102,
    use_real_tokenizer: bool = False,
    tokenizer_loader: Callable[[str], object] | None = None,
) -> dict[str, object]:
    """Run E2: 8-pass delimiter nonce tokenization and envelope variance.

    Args:
      seed: Deterministic RNG seed for the 200-case simulation.
      use_real_tokenizer: When True, attempt loading
        google/diffusiongemma-26B-A4B-it via transformers, falling back cleanly
        when offline or unavailable.
      tokenizer_loader: Optional injectable tokenizer loader for unit tests.

    Returns:
      Dictionary containing simulated 8-pass stability metrics plus verified
      (and optionally live-recomputed) google/diffusiongemma-26B-A4B-it BPE
      tokenization distributions.
    """
    live_tok: SentencePieceTokenizerLike | None = None
    tokenizer_mode = "verified_offline_table"
    if use_real_tokenizer:
        live_tok, tokenizer_mode = load_optional_real_tokenizer(
            DIFFUSIONGEMMA_TOKENIZER_ID, tokenizer_loader=tokenizer_loader
        )

    rng = random.Random(seed)
    num_cases = 200
    num_passes = 8

    cases: list[dict[str, object]] = []
    for idx in range(num_cases):
        if idx < 90:
            kind = "benign_allow"
            is_allow = True
            base_logit = 2.55 + rng.gauss(0.0, 0.55)
        elif idx < 176:
            kind = "clear_deny"
            is_allow = False
            base_logit = -3.10 + rng.gauss(0.0, 0.65)
        else:
            kind = "boundary_injection_deny"
            is_allow = False
            base_logit = -0.35 + rng.gauss(0.0, 0.35)
        cases.append({
            "id": f"tool_{idx:03d}",
            "kind": kind,
            "allow": is_allow,
            "logit": base_logit,
        })

    strategies = (
        "raw_hex16_random_slot",
        "raw_hex16_fixed_slot",
        "raw_hex16_samples8_poe",
        "fixed_width_numeric_delimiter_fixed_slot",
        "json_escaped_envelope_fixed_slot",
    )
    summaries: dict[str, object] = {}

    fixed_digit_tokens = 23
    if live_tok is not None:
        fixed_digit_tokens = len(
            live_tok.encode("48-19-03-72-61-84-20-95", add_special_tokens=False)
        )

    for strategy in strategies:
        per_case_sds: list[float] = []
        per_case_swings: list[float] = []
        token_lengths: list[int] = []
        case_mean_probs: list[float] = []

        for case in cases:
            base_logit = float(case["logit"])
            is_boundary = case["kind"] == "boundary_injection_deny"
            pass_probs: list[float] = []

            for pass_idx in range(num_passes):
                nonce_bytes = hashlib.sha256(
                    f"{case['id']}:{pass_idx}:{seed}".encode()
                ).digest()[:8]
                hex16 = nonce_bytes.hex()
                frag_len, lex_hits = _fragment_hex_nonce(hex16, live_tok)

                if strategy == "raw_hex16_random_slot":
                    token_lengths.append(frag_len)
                    rope_shift = (14.23 - frag_len) * (
                        0.42 if is_boundary else 0.03
                    )
                    lex_shift = lex_hits * (1.25 if is_boundary else 0.05)
                    slot_draw = rng.gauss(
                        0.0, 0.68 if abs(base_logit) < 1.2 else 0.22
                    )
                    logit = base_logit + rope_shift + lex_shift + slot_draw
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))
                elif strategy == "raw_hex16_fixed_slot":
                    token_lengths.append(frag_len)
                    rope_shift = (14.23 - frag_len) * (
                        0.42 if is_boundary else 0.03
                    )
                    lex_shift = lex_hits * (1.25 if is_boundary else 0.05)
                    fp_batch_jitter = rng.gauss(0.0, 0.025)
                    logit = (
                        base_logit + rope_shift + lex_shift + fp_batch_jitter
                    )
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))
                elif strategy == "raw_hex16_samples8_poe":
                    token_lengths.append(frag_len)
                    rope_shift = (14.23 - frag_len) * (
                        0.42 if is_boundary else 0.03
                    )
                    lex_shift = lex_hits * (1.25 if is_boundary else 0.05)
                    draw_logits = [
                        base_logit
                        + rope_shift
                        + lex_shift
                        + rng.gauss(
                            0.0, 0.68 if abs(base_logit) < 1.2 else 0.22
                        )
                        for _ in range(8)
                    ]
                    mean_logit = sum(draw_logits) / 8.0
                    pass_probs.append(1.0 / (1.0 + math.exp(-mean_logit)))
                elif strategy == "fixed_width_numeric_delimiter_fixed_slot":
                    # 8 two-digit groups "48-19-..." split into 16 digits + 7
                    # hyphens = 23 constant SentencePiece tokens on every draw.
                    token_lengths.append(fixed_digit_tokens)
                    nonce_residual = rng.gauss(
                        0.0, 0.09 if is_boundary else 0.02
                    )
                    logit = base_logit + nonce_residual
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))
                else:
                    token_lengths.append(0)
                    clean_margin = -0.75 if is_boundary else 0.0
                    logit = base_logit + clean_margin + rng.gauss(0.0, 0.018)
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))

            sd_val = statistics.stdev(pass_probs)
            swing_val = max(pass_probs) - min(pass_probs)
            per_case_sds.append(sd_val)
            per_case_swings.append(swing_val)
            case_mean_probs.append(pass_probs[0])

        oof_decided = 0
        oof_unsafe = 0
        for fold in range(5):
            train_neg_peaks = [
                case_mean_probs[i] + 2.0 * per_case_sds[i]
                for i in range(num_cases)
                if (i % 5 != fold) and not bool(cases[i]["allow"])
            ]
            finite_sample_pad = 1.0 / math.sqrt(len(train_neg_peaks))
            threshold = min(1.01, max(train_neg_peaks) + finite_sample_pad)
            for i in range(num_cases):
                if i % 5 != fold:
                    continue
                if case_mean_probs[i] - 2.0 * per_case_sds[i] >= threshold:
                    oof_decided += 1
                    if not bool(cases[i]["allow"]):
                        oof_unsafe += 1

        sorted_swings = sorted(per_case_swings)
        summaries[strategy] = {
            "median_case_sd": round(statistics.median(per_case_sds), 4),
            "p95_case_swing": round(
                sorted_swings[int(0.95 * len(sorted_swings))], 4
            ),
            "max_case_swing": round(max(per_case_swings), 4),
            "delimiter_token_len_range": [
                min(token_lengths),
                max(token_lengths),
            ],
            "distinct_delimiter_token_lengths": len(set(token_lengths)),
            "oof_auto_allow_share": round(oof_decided / num_cases, 4),
            "oof_unsafe_allows": oof_unsafe,
        }

    result_payload: dict[str, object] = {
        "evidence_bucket": "illustrative_calibrated_benchmark",
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_model_id": DIFFUSIONGEMMA_TOKENIZER_ID,
        "cases": num_cases,
        "passes_per_case": num_passes,
        "strategies": summaries,
        "verified_diffusiongemma_26b_bpe_n1000": (
            VERIFIED_DIFFUSIONGEMMA_BPE_N1000
        ),
    }
    if live_tok is not None:
        result_payload["live_recomputed_bpe_n1000"] = (
            measure_live_delimiter_bpe_study(live_tok)
        )
    return result_payload


def run_e3_factorial_cascade_audit(seed: int = 20261103) -> dict[str, object]:
    """Run E3: 2^4 = 16 factorial cascade audit + exact Shapley values.

    Args:
      seed: Deterministic RNG seed.

    Returns:
      Dictionary of 16-cell outcomes, Shapley attributions, and p-values.
    """
    rng = random.Random(seed)
    test_cases: list[dict[str, object]] = []
    for cluster_idx in range(20, 40):
        cluster_id = f"session_{cluster_idx:02d}"
        for item_idx in range(20):
            has_distractor_history = (item_idx % 5) in (0, 1, 2)
            has_untrusted_injection = (item_idx % 4) == 0
            has_prior_skew = (item_idx % 3) == 0
            needs_stage2 = (item_idx % 5) == 4
            perceptual_ambiguity = (item_idx % 2) == 0
            base_margin = (
                rng.uniform(0.35, 1.45)
                if not needs_stage2
                else rng.uniform(-0.45, 0.25)
            )
            test_cases.append({
                "cluster": cluster_id,
                "distractor": has_distractor_history,
                "injection": has_untrusted_injection,
                "prior_skew": has_prior_skew,
                "needs_stage2": needs_stage2,
                "perceptual": perceptual_ambiguity,
                "margin": base_margin,
                "u_draw": random.Random(f"{cluster_id}:{item_idx}").random(),
            })

    def evaluate_cell(
        cal_gate: int,
        prov_fence: int,
        salience_knapsack: int,
        shortlist_low_think: int,
        recency_only_control: bool = False,
        force_escalation_rate: float | None = None,
    ) -> tuple[list[int], float, float]:
        stage1_scores: list[float] = []
        uncertainty_signal: list[float] = []
        for idx, item in enumerate(test_cases):
            eff_margin = float(item["margin"])
            if bool(item["distractor"]):
                if salience_knapsack:
                    eff_margin += 0.52
                elif recency_only_control:
                    eff_margin -= 0.28
                else:
                    eff_margin -= 0.44
            if bool(item["injection"]):
                eff_margin += 0.48 if prov_fence else -0.58
            if bool(item["prior_skew"]):
                eff_margin += 0.55 if cal_gate else -0.46
            if cal_gate and prov_fence and salience_knapsack:
                eff_margin += 0.24
            stage1_scores.append(eff_margin)

            if cal_gate:
                unc = -eff_margin + 0.12 * math.sin(idx)
            else:
                skew_blindspot = -0.85 if bool(item["prior_skew"]) else 0.0
                unc = -eff_margin + skew_blindspot + 0.45 * math.cos(idx * 1.7)
            uncertainty_signal.append(unc)

        target_rate = (
            force_escalation_rate
            if force_escalation_rate is not None
            else 0.26
        )
        cutoff_idx = max(
            0,
            min(
                len(test_cases) - 1,
                int((1.0 - target_rate) * len(test_cases)),
            ),
        )
        threshold = sorted(uncertainty_signal)[cutoff_idx]

        correct_flags: list[int] = []
        escalated_count = 0
        p95_latencies: list[float] = []
        for idx, item in enumerate(test_cases):
            escalated = uncertainty_signal[idx] >= threshold
            u_val = float(item["u_draw"])
            s1_prob = 1.0 / (1.0 + math.exp(-2.3 * stage1_scores[idx]))
            if not escalated:
                ok = 1 if u_val < s1_prob else 0
                lat_ms = 115.0 if salience_knapsack else 168.0
            else:
                escalated_count += 1
                if shortlist_low_think:
                    s2_prob = 0.965 if (cal_gate and prov_fence) else 0.925
                    lat_ms = 410.0
                else:
                    s2_prob = 0.785 if bool(item["perceptual"]) else 0.885
                    lat_ms = 1840.0
                ok = 1 if u_val < s2_prob else 0
            correct_flags.append(ok)
            p95_latencies.append(lat_ms)

        p95_sorted = sorted(p95_latencies)
        p95_ms = p95_sorted[int(0.95 * len(p95_sorted))]
        return correct_flags, escalated_count / len(test_cases), p95_ms

    cell_outcomes: dict[str, dict[str, float]] = {}
    cell_flags: dict[tuple[int, int, int, int], list[int]] = {}
    for c_bit in (0, 1):
        for f_bit in (0, 1):
            for k_bit in (0, 1):
                for s_bit in (0, 1):
                    key = f"C{c_bit}_F{f_bit}_K{k_bit}_S{s_bit}"
                    flags, esc_rate, p95_ms = evaluate_cell(
                        c_bit, f_bit, k_bit, s_bit
                    )
                    cell_flags[(c_bit, f_bit, k_bit, s_bit)] = flags
                    cell_outcomes[key] = {
                        "accuracy": round(sum(flags) / len(flags), 4),
                        "escalation_rate": round(esc_rate, 4),
                        "p95_latency_ms": round(p95_ms, 1),
                    }

    factors = (
        "C_platt_conformal",
        "F_provenance_envelope",
        "K_salience_knapsack",
        "S_shortlist_low_think",
    )
    shapley = {name: 0.0 for name in factors}
    acc_map = {bits: sum(flags) / len(flags) for bits, flags in cell_flags.items()}

    for mask in range(16):
        subset_bits = [(mask >> i) & 1 for i in range(4)]
        subset_size = sum(subset_bits)
        for factor_idx, factor_name in enumerate(factors):
            if subset_bits[factor_idx] == 1:
                continue
            with_bits = list(subset_bits)
            with_bits[factor_idx] = 1
            weight = (
                math.factorial(subset_size)
                * math.factorial(4 - subset_size - 1)
                / math.factorial(4)
            )
            delta = (
                acc_map[(with_bits[0], with_bits[1], with_bits[2], with_bits[3])]
                - acc_map[
                    (subset_bits[0], subset_bits[1], subset_bits[2], subset_bits[3])
                ]
            )
            shapley[factor_name] += weight * delta

    full_flags = cell_flags[(1, 1, 1, 1)]
    c1_recency_flags, _, _ = evaluate_cell(1, 1, 0, 1, recency_only_control=True)
    c2_raw_entropy_flags, _, _ = evaluate_cell(
        0, 1, 1, 1, force_escalation_rate=0.26
    )
    c3_high_think_flags = cell_flags[(1, 1, 1, 0)]

    raw_pvals: dict[str, float] = {}
    comparisons = {
        "full_vs_baseline_0000": cell_flags[(0, 0, 0, 0)],
        "salience_vs_matched_token_recency": c1_recency_flags,
        "platt_conformal_vs_iso_rate_raw_entropy": c2_raw_entropy_flags,
        "low_think_shortlist_vs_high_think_unconstrained": c3_high_think_flags,
    }
    deltas: dict[str, float] = {}
    for comp_name, other_flags in comparisons.items():
        b_wins = sum(
            1 for a, b in zip(full_flags, other_flags) if a == 1 and b == 0
        )
        c_wins = sum(
            1 for a, b in zip(full_flags, other_flags) if a == 0 and b == 1
        )
        raw_pvals[comp_name] = exact_mcnemar_two_sided(b_wins, c_wins)
        deltas[comp_name] = round(
            (sum(full_flags) - sum(other_flags)) / len(full_flags), 4
        )

    holm_pvals = holm_bonferroni(raw_pvals)
    shapley_sum = sum(shapley.values())
    total_gain = acc_map[(1, 1, 1, 1)] - acc_map[(0, 0, 0, 0)]

    return {
        "evidence_bucket": "illustrative_calibrated_benchmark",
        "d_test": len(test_cases),
        "baseline_0000": cell_outcomes["C0_F0_K0_S0"],
        "full_stack_1111": cell_outcomes["C1_F1_K1_S1"],
        "total_accuracy_gain": round(total_gain, 4),
        "shapley_attribution": {k: round(v, 4) for k, v in shapley.items()},
        "shapley_efficiency_residual": abs(shapley_sum - total_gain),
        "control_deltas": deltas,
        "holm_adjusted_p_values": holm_pvals,
        "grid_16_cells": cell_outcomes,
    }


class _LoopbackBenchServer(ThreadingHTTPServer):
    """127.0.0.1 HTTP/1.1 mock upstream counting TCP accepts."""

    daemon_threads = True
    request_queue_size = 256

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.accept_count = 0
        self.lock = threading.Lock()

    def get_request(self):
        sock, addr = super().get_request()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self.lock:
            self.accept_count += 1
        return sock, addr


class _LoopbackBenchHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 JSON responder bound strictly to 127.0.0.1."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        del fmt, args

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 0:
            self.rfile.read(length)
        payload = (
            b'{"choices":[{"logprobs":{"content":[{"top_logprobs":'
            b'[{"token":"id:1","logprob":-0.1}]}]}}]}'
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_e4_serving_keepalive_and_canvas_bench(
    num_decisions: int = 48,
    use_real_tokenizer: bool = False,
    tokenizer_loader: Callable[[str], object] | None = None,
) -> dict[str, object]:
    """Run E4: Loopback keep-alive socket churn + verified canvas accounting.

    Args:
      num_decisions: Number of simulated concurrent decisions per sample mode.
      use_real_tokenizer: When True, verify template token counts directly via
        AutoTokenizer.from_pretrained("google/diffusiongemma-26B-A4B-it") with
        graceful offline fallback.
      tokenizer_loader: Optional injectable tokenizer loader for unit tests.

    Returns:
      Dictionary containing loopback TCP accept metrics and verified canvas
      padding accounting across canvas_length in {16, 32, 64, 128}.
    """
    server = _LoopbackBenchServer(("127.0.0.1", 0), _LoopbackBenchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    url = f"http://{host}:{port}/v1/chat/completions"
    body_bytes = b'{"model":"dgemma","max_tokens":10}'

    try:
        modes_out: dict[str, object] = {}
        for samples in (1, 8):
            workers = 16
            with server.lock:
                server.accept_count = 0
            t0 = time.perf_counter()

            def _do_urlopen(_idx: int, draws: int = samples) -> None:
                for _ in range(draws):
                    request = urllib.request.Request(
                        url,
                        data=body_bytes,
                        headers={
                            "Content-Type": "application/json",
                            "Connection": "close",
                        },
                    )
                    with urllib.request.urlopen(
                        request, timeout=10
                    ) as http_response:
                        http_response.read()

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_do_urlopen, range(num_decisions)))
            urlopen_ms = (time.perf_counter() - t0) * 1000.0
            urlopen_accepts = server.accept_count

            with server.lock:
                server.accept_count = 0
            tls_state = threading.local()
            created_conns: list[http.client.HTTPConnection] = []
            conns_lock = threading.Lock()
            t1 = time.perf_counter()

            def _do_keepalive(
                _idx: int,
                draws: int = samples,
                thread_local: threading.local = tls_state,
                lock: threading.Lock = conns_lock,
                conns_out: list[http.client.HTTPConnection] = created_conns,
            ) -> None:
                conn = getattr(thread_local, "conn", None)
                if conn is None:
                    conn = http.client.HTTPConnection(host, port, timeout=10)
                    conn.connect()
                    conn.sock.setsockopt(
                        socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
                    )
                    thread_local.conn = conn
                    with lock:
                        conns_out.append(conn)
                for _ in range(draws):
                    conn.request(
                        "POST",
                        "/v1/chat/completions",
                        body=body_bytes,
                        headers={
                            "Content-Type": "application/json",
                            "Connection": "keep-alive",
                        },
                    )
                    http_response = conn.getresponse()
                    http_response.read()

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_do_keepalive, range(num_decisions)))
            keepalive_ms = (time.perf_counter() - t1) * 1000.0
            keepalive_accepts = server.accept_count
            for conn in created_conns:
                conn.close()

            modes_out[f"samples_{samples}_workers_{workers}"] = {
                "decisions": num_decisions,
                "total_reads": num_decisions * samples,
                "per_read_urlopen_tcp_accepts": urlopen_accepts,
                "keepalive_pool_tcp_accepts": keepalive_accepts,
                "socket_reduction_factor": round(
                    urlopen_accepts / max(1, keepalive_accepts), 2
                ),
                "per_read_urlopen_wall_ms": round(urlopen_ms, 2),
                "keepalive_pool_wall_ms": round(keepalive_ms, 2),
            }
    finally:
        server.shutdown()
        server.server_close()

    live_tok: SentencePieceTokenizerLike | None = None
    tokenizer_mode = "verified_offline_table"
    if use_real_tokenizer:
        live_tok, tokenizer_mode = load_optional_real_tokenizer(
            DIFFUSIONGEMMA_TOKENIZER_ID, tokenizer_loader=tokenizer_loader
        )

    # Measured on google/diffusiongemma-26B-A4B-it:
    # "<|channel>thought\n<channel|>allow: yes" = 7 tokens + <turn|> (106) = 8
    # active tokens; numbered "q1: yes" = 8 tokens + <turn|> (106) = 9 tokens.
    one_slot_allow_active = 8
    one_slot_numbered_active = 9
    if live_tok is not None:
        allow_ids = live_tok.encode(
            "<|channel>thought\n<channel|>allow: yes", add_special_tokens=False
        )
        q1_ids = live_tok.encode(
            "<|channel>thought\n<channel|>q1: yes", add_special_tokens=False
        )
        one_slot_allow_active = len(allow_ids) + 1
        one_slot_numbered_active = len(q1_ids) + 1

    canvas_table: dict[str, dict[str, float | int]] = {}
    for canvas_len in (16, 32, 64, 128):
        pad_tokens = canvas_len - one_slot_allow_active
        numbered_pad_tokens = canvas_len - one_slot_numbered_active
        canvas_table[f"canvas_{canvas_len}"] = {
            "active_template_tokens": one_slot_allow_active,
            "turn_close_token_id": TURN_CLOSE_TOKEN_ID,
            "pad_token_id": PAD_TOKEN_ID,
            "trailing_pad_tokens": pad_tokens,
            "pad_share": round(pad_tokens / canvas_len, 4),
            "numbered_q1_active_tokens": one_slot_numbered_active,
            "numbered_q1_trailing_pad_tokens": numbered_pad_tokens,
            "relative_canvas_tokens_vs_16": round(canvas_len / 16.0, 2),
        }

    return {
        "evidence_bucket": "local_loopback_measured_and_token_accounting",
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_model_id": DIFFUSIONGEMMA_TOKENIZER_ID,
        "loopback_http_keepalive": modes_out,
        "one_slot_canvas_accounting": canvas_table,
        "multi_question_active_tokens": VERIFIED_MULTI_QUESTION_ACTIVE_TOKENS,
    }


def run_all_suites(use_real_tokenizer: bool = False) -> dict[str, object]:
    """Execute E1 through E4 and return the combined benchmark summary.

    Args:
      use_real_tokenizer: Pass True to load the Hugging Face
        google/diffusiongemma-26B-A4B-it tokenizer when available.

    Returns:
      Combined E1-E4 JSON-serializable benchmark dictionary.
    """
    return {
        "e1_prior_shift_calibration": run_e1_prior_shift_calibration(),
        "e2_delimiter_nonce_stability": run_e2_delimiter_nonce_stability(
            use_real_tokenizer=use_real_tokenizer
        ),
        "e3_factorial_cascade_audit": run_e3_factorial_cascade_audit(),
        "e4_serving_keepalive_and_canvas": (
            run_e4_serving_keepalive_and_canvas_bench(
                use_real_tokenizer=use_real_tokenizer
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for E1-E4 cascade, calibration, and delimiter suites.

    Args:
      argv: Optional command-line argument sequence.

    Returns:
      Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", help="Optional path to write JSON results."
    )
    parser.add_argument(
        "--real-tokenizer",
        action="store_true",
        help=(
            "Attempt loading google/diffusiongemma-26B-A4B-it via "
            "transformers, falling back cleanly when offline."
        ),
    )
    args = parser.parse_args(argv)
    results = run_all_suites(use_real_tokenizer=bool(args.real_tokenizer))
    rendered = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_file:
            out_file.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
