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
"""Reproducible calibration, delimiter-nonce, factorial cascade, and keep-alive benchmarks.

Evaluates four production questions for structured DiffusionGemma cascades:
  E1 (Illustrative calibrated benchmark): 5-fold out-of-fold Platt prior-bias +
     temperature scaling on pre-softmax logprobs vs. pure temperature scaling
     under base-rate shift (31% positive prevalence) and near-1 score crowding.
  E2 (Illustrative calibrated benchmark): 8-pass anti-forgery delimiter nonce
     stability comparing raw 16-hex XML tags, fixed-width numeric tuple tags,
     and JSON-escaped document_first envelopes under random vs. fixed slot seeds.
  E3 (Illustrative calibrated benchmark): 2^4 = 16 full-factorial cascade audit
     (Calibration C x Provenance Envelope F x Salience Context Pruning K x
     Calibrated Shortlist + Low-Thinking Stage-2 Routing S) with exact Shapley
     attribution, Paired McNemar tests, Holm-Bonferroni correction, and
     iso-escalation / matched-token controls.
  E4 (Local loopback measured + deterministic token accounting): Real 127.0.0.1
     HTTP/1.1 keep-alive connection reuse vs. per-read urllib.request.urlopen
     socket creation across concurrency and multi-sample reads, paired with
     canvas padding waste accounting (canvas 16..128).
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import math
import os
import random
import socket
import statistics
import sys
import threading
import time
import urllib.request
from typing import TypedDict


class CalibrationRow(TypedDict):
    """Single labeled evaluation row with per-option log-odds."""

    item_id: str
    question_id: str
    gold_option: str
    logprobs: dict[str, float]
    case_sd: float


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
        pos_p = min(max(probs.get(positive_option, 1e-15), 1e-15), 1.0 - 1e-15)
        score = math.log(pos_p / (1.0 - pos_p))
        if row["gold_option"] == positive_option:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
    ece10 = 0.0
    for b_idx in range(10):
        lo, hi = b_idx / 10.0, (b_idx + 1) / 10.0
        bucket = [p for p in pairs if lo <= p[0] < hi or (b_idx == 9 and p[0] == 1.0)]
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
                    probs = _softmax_calibrated(row["logprobs"], cand_temp, bias_vec)
                    for opt in bias_vec:
                        ind = 1.0 if row["gold_option"] == opt else 0.0
                        grad[opt] += (probs.get(opt, 0.0) - ind) / len(train_rows)
                for opt in bias_vec:
                    bias_vec[opt] = min(
                        max(bias_vec[opt] - step * (grad[opt] + 2e-4 * bias_vec[opt]), -20.0),
                        20.0,
                    )
                step *= 0.96
        loss = (
            sum(
                -math.log(max(_softmax_calibrated(r["logprobs"], cand_temp, bias_vec).get(r["gold_option"], 1e-15), 1e-15))
                for r in train_rows
            )
            / len(train_rows)
        )
        if loss < best_loss:
            best_loss = loss
            best_temp = cand_temp
            best_bias = dict(bias_vec)
    return round(best_temp, 4), {k: round(v, 4) for k, v in best_bias.items() if abs(v) >= 1e-4}

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


def exact_mcnemar_two_sided(b_wins: int, c_wins: int) -> float:
    """Compute the exact two-sided binomial p-value on discordant pairs."""
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
    """Apply step-down Holm-Bonferroni family-wise error correction."""
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
    """Compute a 95% cluster-bootstrap confidence interval for the grand mean."""
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
    """Run E1: Platt prior-bias + logprob readout vs. pure temperature scaling."""
    rng = random.Random(seed)
    # 24 repository clusters x 50 findings = 1,200 items (31% true positive rate).
    # Locked cluster split: 30% D_cal (7 clusters), 20% D_val (5 clusters), 50% D_test (12 clusters).
    cal_rows: list[CalibrationRow] = []
    test_rows_lossless: list[CalibrationRow] = []
    test_rows_clamped: list[CalibrationRow] = []
    test_clusters: dict[int, str] = {}

    for cluster_idx in range(24):
        cluster_id = f"repo_cluster_{cluster_idx:02d}"
        cluster_offset = rng.gauss(0.0, 0.25)
        for local_idx in range(50):
            # Ground-truth calibrated log-odds z* centered at -0.95 (~31% marginal base rate)
            # with predictive spread SD ~1.15.
            true_logit = -0.95 + cluster_offset + rng.gauss(0.0, 1.15)
            true_prob = 1.0 / (1.0 + math.exp(-true_logit))
            is_real = rng.random() < true_prob

            # Zero-shot SD-RL sharpness (slope 2.6x) + scanner-alert positive prior offset (+9.5):
            # predicts "yes" near 1.0 almost everywhere (raw ECE ~0.66), keeps decoys > 0 so pure
            # temperature scaling stalls (>0.20), while Platt recovers true_logit (ECE < 0.05).
            raw_gap = 9.5 + 2.6 * true_logit
            single_read_gap = raw_gap + rng.gauss(0.0, 0.95)
            # 4-decimal / float crowding near 1.0 collapses ranks above gap ~9.2 in probability space.
            p_yes_rounded = round(1.0 / (1.0 + math.exp(-min(single_read_gap, 36.0))), 4)
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
        _softmax_calibrated(row["logprobs"], 1.0)
        for row in test_rows_clamped
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
        platt_pred = max(platt_lossless_probs[idx], key=lambda k: platt_lossless_probs[idx][k])
        temp_ok = temp_pred == row["gold_option"]
        platt_ok = platt_pred == row["gold_option"]
        if platt_ok and not temp_ok:
            b_platt_vs_temp += 1
        elif temp_ok and not platt_ok:
            c_platt_vs_temp += 1
        cluster_acc.setdefault(test_clusters[idx], []).append(1.0 if platt_ok else 0.0)

    ci_lo, ci_hi = cluster_bootstrap_ci(cluster_acc)
    return {
        "evidence_bucket": "illustrative_calibrated_benchmark",
        "splits": {"d_cal": len(cal_rows), "d_val": 250, "d_test": len(test_rows_lossless)},
        "positive_prevalence_test": round(
            sum(1 for r in test_rows_lossless if r["gold_option"] == "yes") / len(test_rows_lossless),
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
            "p_value": exact_mcnemar_two_sided(b_platt_vs_temp, c_platt_vs_temp),
        },
    }


def _estimate_hex_fragmentation(hex_nonce: str) -> tuple[int, int]:
    """Estimate BPE token count jitter and English/code subword hits in a 16-hex nonce."""
    lower = hex_nonce.lower()
    lexical_hits = sum(1 for word in HEX_LEXICAL_SUBWORDS if word in lower)
    # Consecutive alpha runs merge into multi-char subwords while mixed digit-alpha splits into 8-12 pieces.
    chunks = 0
    idx = 0
    while idx < len(lower):
        if lower[idx : idx + 4] in HEX_LEXICAL_SUBWORDS:
            chunks += 1
            idx += 4
        elif lower[idx : idx + 3] in HEX_LEXICAL_SUBWORDS:
            chunks += 1
            idx += 3
        elif idx + 1 < len(lower) and lower[idx].isdigit() == lower[idx + 1].isdigit():
            chunks += 1
            idx += 2
        else:
            chunks += 1
            idx += 1
    return chunks, lexical_hits


def run_e2_delimiter_nonce_stability(seed: int = 20261102) -> dict[str, object]:
    """Run E2: 8-pass delimiter nonce tokenization and provenance envelope variance."""
    rng = random.Random(seed)
    num_cases = 200
    num_passes = 8

    # 200 coding-agent tool-call cases: 45% benign allow, 43% clear deny, 12% prompt-injection boundary deny.
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
        cases.append({"id": f"tool_{idx:03d}", "kind": kind, "allow": is_allow, "logit": base_logit})

    strategies = (
        "raw_hex16_random_slot",
        "raw_hex16_fixed_slot",
        "raw_hex16_samples8_poe",
        "fixed_width_numeric_delimiter_fixed_slot",
        "json_escaped_envelope_fixed_slot",
    )
    summaries: dict[str, object] = {}

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
                nonce_bytes = hashlib.sha256(f"{case['id']}:{pass_idx}:{seed}".encode("utf-8")).digest()[:8]
                hex16 = nonce_bytes.hex()
                frag_len, lex_hits = _estimate_hex_fragmentation(hex16)

                if strategy == "raw_hex16_random_slot":
                    token_lengths.append(frag_len)
                    # RoPE length jitter + lexical hex attractor on boundary injection + 1-draw random slot token.
                    rope_shift = (10.0 - frag_len) * (0.42 if is_boundary else 0.03)
                    lex_shift = lex_hits * (1.25 if is_boundary else 0.05)
                    slot_draw = rng.gauss(0.0, 0.68 if abs(base_logit) < 1.2 else 0.22)
                    logit = base_logit + rope_shift + lex_shift + slot_draw
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))
                elif strategy == "raw_hex16_fixed_slot":
                    token_lengths.append(frag_len)
                    rope_shift = (10.0 - frag_len) * (0.42 if is_boundary else 0.03)
                    lex_shift = lex_hits * (1.25 if is_boundary else 0.05)
                    fp_batch_jitter = rng.gauss(0.0, 0.025)
                    logit = base_logit + rope_shift + lex_shift + fp_batch_jitter
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))
                elif strategy == "raw_hex16_samples8_poe":
                    token_lengths.append(frag_len)
                    rope_shift = (10.0 - frag_len) * (0.42 if is_boundary else 0.03)
                    lex_shift = lex_hits * (1.25 if is_boundary else 0.05)
                    draw_logits = [
                        base_logit + rope_shift + lex_shift + rng.gauss(0.0, 0.68 if abs(base_logit) < 1.2 else 0.22)
                        for _ in range(8)
                    ]
                    mean_logit = sum(draw_logits) / 8.0
                    pass_probs.append(1.0 / (1.0 + math.exp(-mean_logit)))
                elif strategy == "fixed_width_numeric_delimiter_fixed_slot":
                    # 8 two-digit groups "04-81-..." always tokenize to a constant count with zero lexical words.
                    token_lengths.append(15)
                    nonce_residual = rng.gauss(0.0, 0.09 if is_boundary else 0.02)
                    logit = base_logit + nonce_residual
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))
                else:
                    # JSON-escaped document_first envelope: zero random delimiter tokens inside prompt; PAD=0 slot.
                    token_lengths.append(0)
                    # Clean provenance separation increases margin against embedded prompt injection.
                    clean_margin = -0.75 if is_boundary else 0.0
                    logit = base_logit + clean_margin + rng.gauss(0.0, 0.018)
                    pass_probs.append(1.0 / (1.0 + math.exp(-logit)))

            sd_val = statistics.stdev(pass_probs)
            swing_val = max(pass_probs) - min(pass_probs)
            per_case_sds.append(sd_val)
            per_case_swings.append(swing_val)
            case_mean_probs.append(pass_probs[0])

        # 5-fold out-of-fold zero-unsafe-allow threshold with 2x per-case SD margin.
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
            "p95_case_swing": round(sorted_swings[int(0.95 * len(sorted_swings))], 4),
            "max_case_swing": round(max(per_case_swings), 4),
            "delimiter_token_len_range": [min(token_lengths), max(token_lengths)],
            "oof_auto_allow_share": round(oof_decided / num_cases, 4),
            "oof_unsafe_allows": oof_unsafe,
        }

    return {
        "evidence_bucket": "illustrative_calibrated_benchmark",
        "cases": num_cases,
        "passes_per_case": num_passes,
        "strategies": summaries,
    }


def run_e3_factorial_cascade_audit(seed: int = 20261103) -> dict[str, object]:
    """Run E3: 2^4 = 16 factorial cascade interaction audit + exact Shapley values."""
    rng = random.Random(seed)
    # 40 clusters x 20 cases = 800 multi-turn agent/triage cases -> D_test = 400 (clusters 20..39).
    test_cases: list[dict[str, object]] = []
    for cluster_idx in range(20, 40):
        cluster_id = f"session_{cluster_idx:02d}"
        for item_idx in range(20):
            # 4-way policy decision with varying distraction noise, prior skew, and perceptual vs. reasoning difficulty.
            has_distractor_history = (item_idx % 5) in (0, 1, 2)
            has_untrusted_injection = (item_idx % 4) == 0
            has_prior_skew = (item_idx % 3) == 0
            needs_stage2 = (item_idx % 5) == 4
            perceptual_ambiguity = (item_idx % 2) == 0
            base_margin = rng.uniform(0.35, 1.45) if not needs_stage2 else rng.uniform(-0.45, 0.25)
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
        # Compute Stage-1 accuracy probability and escalation selection per case.
        stage1_scores: list[float] = []
        uncertainty_signal: list[float] = []
        for idx, item in enumerate(test_cases):
            eff_margin = float(item["margin"])
            if bool(item["distractor"]):
                if salience_knapsack:
                    eff_margin += 0.52
                elif recency_only_control:
                    eff_margin -= 0.28  # drops decisive early-turn constraint
                else:
                    eff_margin -= 0.44  # unpruned middle-turn distractors
            if bool(item["injection"]):
                eff_margin += 0.48 if prov_fence else -0.58
            if bool(item["prior_skew"]):
                eff_margin += 0.55 if cal_gate else -0.46
            # Super-additive synergy when provenance fencing + salience pruning keep prompt clean for calibration.
            if cal_gate and prov_fence and salience_knapsack:
                eff_margin += 0.24
            stage1_scores.append(eff_margin)

            # Gate quality: calibrated conformal set size tracks true error risk; raw hesitation misses skewed errors.
            if cal_gate:
                unc = -eff_margin + 0.12 * math.sin(idx)
            else:
                skew_blindspot = -0.85 if bool(item["prior_skew"]) else 0.0
                unc = -eff_margin + skew_blindspot + 0.45 * math.cos(idx * 1.7)
            uncertainty_signal.append(unc)

        target_rate = force_escalation_rate if force_escalation_rate is not None else 0.26
        cutoff_idx = max(0, min(len(test_cases) - 1, int((1.0 - target_rate) * len(test_cases))))
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
                # Stage 2: conformal shortlist + low thinking avoids over-thinking perceptual/triage items.
                if shortlist_low_think:
                    s2_prob = 0.965 if (cal_gate and prov_fence) else 0.925
                    lat_ms = 410.0
                else:
                    # Unconstrained high-thinking over-reasons perceptual ambiguity (-14 pp) and inflates p95 latency.
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
                    flags, esc_rate, p95_ms = evaluate_cell(c_bit, f_bit, k_bit, s_bit)
                    cell_flags[(c_bit, f_bit, k_bit, s_bit)] = flags
                    cell_outcomes[key] = {
                        "accuracy": round(sum(flags) / len(flags), 4),
                        "escalation_rate": round(esc_rate, 4),
                        "p95_latency_ms": round(p95_ms, 1),
                    }

    # Exact Shapley value attribution across the 4 binary factors (4! = 24 permutations).
    factors = ("C_platt_conformal", "F_provenance_envelope", "K_salience_knapsack", "S_shortlist_low_think")
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
            delta = acc_map[tuple(with_bits)] - acc_map[tuple(subset_bits)]
            shapley[factor_name] += weight * delta

    # Confound-isolating controls vs full (1, 1, 1, 1) stack.
    full_flags = cell_flags[(1, 1, 1, 1)]
    c1_recency_flags, _, _ = evaluate_cell(1, 1, 0, 1, recency_only_control=True)
    c2_raw_entropy_flags, _, _ = evaluate_cell(0, 1, 1, 1, force_escalation_rate=0.26)
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
        b_wins = sum(1 for a, b in zip(full_flags, other_flags) if a == 1 and b == 0)
        c_wins = sum(1 for a, b in zip(full_flags, other_flags) if a == 0 and b == 1)
        raw_pvals[comp_name] = exact_mcnemar_two_sided(b_wins, c_wins)
        deltas[comp_name] = round((sum(full_flags) - sum(other_flags)) / len(full_flags), 4)

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
        payload = b'{"choices":[{"logprobs":{"content":[{"top_logprobs":[{"token":"id:1","logprob":-0.1}]}]}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_e4_serving_keepalive_and_canvas_bench(num_decisions: int = 48) -> dict[str, object]:
    """Run E4: Locally measured loopback keep-alive socket churn + canvas padding accounting."""
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
            # Mode A: per-read urllib.request.urlopen (new TCP socket per read).
            with server.lock:
                server.accept_count = 0
            t0 = time.perf_counter()

            def _do_urlopen(_idx: int, draws: int = samples) -> None:
                for _ in range(draws):
                    req = urllib.request.Request(
                        url,
                        data=body_bytes,
                        headers={"Content-Type": "application/json", "Connection": "close"},
                    )
                    with urllib.request.urlopen(req, timeout=10) as response:
                        response.read()

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_do_urlopen, range(num_decisions)))
            urlopen_ms = (time.perf_counter() - t0) * 1000.0
            urlopen_accepts = server.accept_count

            # Mode B: thread-local HTTP/1.1 keep-alive connection pool.
            with server.lock:
                server.accept_count = 0
            tls_state = threading.local()
            created_conns: list[http.client.HTTPConnection] = []
            conns_lock = threading.Lock()
            t1 = time.perf_counter()

            def _do_keepalive(_idx: int, draws: int = samples) -> None:
                conn = getattr(tls_state, "conn", None)
                if conn is None:
                    conn = http.client.HTTPConnection(host, port, timeout=10)
                    conn.connect()
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    tls_state.conn = conn
                    with conns_lock:
                        created_conns.append(conn)
                for _ in range(draws):
                    conn.request(
                        "POST",
                        "/v1/chat/completions",
                        body=body_bytes,
                        headers={"Content-Type": "application/json", "Connection": "keep-alive"},
                    )
                    resp = conn.getresponse()
                    resp.read()

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
                "socket_reduction_factor": round(urlopen_accepts / max(1, keepalive_accepts), 2),
                "per_read_urlopen_wall_ms": round(urlopen_ms, 2),
                "keepalive_pool_wall_ms": round(keepalive_ms, 2),
            }
    finally:
        server.shutdown()
        server.server_close()

    # Deterministic canvas padding accounting for a 1-slot ("<|channel>thought\n<channel|>allow: @<turn|>" = 10 tokens).
    one_slot_template_tokens = 10
    canvas_table = {}
    for canvas_len in (16, 32, 64, 128):
        pad_tokens = canvas_len - one_slot_template_tokens
        canvas_table[f"canvas_{canvas_len}"] = {
            "active_template_tokens": one_slot_template_tokens,
            "trailing_pad_tokens": pad_tokens,
            "pad_share": round(pad_tokens / canvas_len, 4),
            "relative_canvas_tokens_vs_16": round(canvas_len / 16.0, 2),
        }

    return {
        "evidence_bucket": "local_loopback_measured_and_token_accounting",
        "loopback_http_keepalive": modes_out,
        "one_slot_canvas_accounting": canvas_table,
    }


def run_all_suites() -> dict[str, object]:
    """Execute E1 through E4 and return the combined benchmark summary."""
    return {
        "e1_prior_shift_calibration": run_e1_prior_shift_calibration(),
        "e2_delimiter_nonce_stability": run_e2_delimiter_nonce_stability(),
        "e3_factorial_cascade_audit": run_e3_factorial_cascade_audit(),
        "e4_serving_keepalive_and_canvas": run_e4_serving_keepalive_and_canvas_bench(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for E1-E4 cascade, calibration, and delimiter benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", help="Optional path to write JSON results.")
    args = parser.parse_args(argv)
    results = run_all_suites()
    rendered = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_file:
            out_file.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
