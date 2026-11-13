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

"""Experiment D3: Popoviciu ordinal variance gating & entity deduplication."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import exp_stats_common as stats_lib  # noqa: E402

TRACK_D3A: str = "d3a_ordinal"
TRACK_D3B: str = "d3b_entity_dedup"
DEFAULT_LOCKFILE: pathlib.Path = (
    REPO_ROOT / "benchmarks" / "matrix" / "locks" / "dgem_exp_splits_v1.json"
)
TOTAL_DOCUMENTS: int = 80
BUNDLES_PER_DOCUMENT: int = 30
SLOTS_PER_BUNDLE: int = 8
RUBRIC_LEVELS: int = 5

TOTAL_ENTITY_COMPONENTS: int = 100
PAIRS_PER_COMPONENT: int = 30
TRIANGLES_PER_COMPONENT: int = 6


@dataclass(frozen=True)
class OrdinalSlotRecord:
  """One 5-point ordered verification slot inside an 8-slot bundle."""

  slot_index: int
  regime: str
  probs: tuple[float, float, float, float, float]
  gold_level: int


@dataclass(frozen=True)
class D3aChecklistBundle:
  """8-slot ordinal checklist verification bundle clustered by `document_id`."""

  item_id: str
  document_id: str
  slots: tuple[OrdinalSlotRecord, ...]
  split_item: stats_lib.BenchmarkSplitItem


@dataclass(frozen=True)
class D3bEntityPair:
  """Product title deduplication pair clustered by `entity_component_id`."""

  item_id: str
  entity_component_id: str
  node_a: str
  node_b: str
  title_a: str
  title_b: str
  is_duplicate: bool
  pair_category: str
  triangle_group: int
  split_item: stats_lib.BenchmarkSplitItem


def _norm_z(seed_key: str, item_key: str, tag: str) -> float:
  """Generate deterministic standard normal variate via Box-Muller."""
  u_one = max(
      1e-12, stats_lib.hmac_uniform_float64(seed_key, item_key, f"{tag}:1")
  )
  u_two = stats_lib.hmac_uniform_float64(seed_key, item_key, f"{tag}:2")
  return math.sqrt(-2.0 * math.log(u_one)) * math.cos(2.0 * math.pi * u_two)


def build_d3a_corpus() -> list[D3aChecklistBundle]:
  """Build 2,400 8-slot checklist bundles (`80` documents x `30` bundles).

  Each slot falls into one of three realistic ordinal regimes:
    - `clear` (~74% of slots): concentrated unimodal probability mass.
    - `benign_adjacent` (~22% of slots): ~50/50 split between two adjacent
      levels (e.g. `[0, 0.48, 0.50, 0.02, 0]`). In an 8-slot bundle,
      `1 - (1 - 0.22)^8 = 86.3%` of bundles have >= 1 adjacent split, causing
      the ~86.7% bundle escalation blow-up under `max_j H_norm(p_j) > 0.25`.
    - `bimodal_polarized` (~4% of slots): polarized opposition across extremes
      (e.g. `[0.49, 0.01, 0.0, 0.01, 0.49]`), which shares almost identical
      `H_norm` (~0.45) with `benign_adjacent` but has Popoviciu variance ~0.98!

  Returns:
    Deterministic list of `2,400` `D3aChecklistBundle` items.

  Raises:
    stats_lib.SplitLeakageError: If content hashing fails.
  """
  bundles: list[D3aChecklistBundle] = []
  for doc_idx in range(TOTAL_DOCUMENTS):
    document_id = f"doc_{doc_idx:03d}"
    for local_idx in range(BUNDLES_PER_DOCUMENT):
      item_id = f"d3a-{doc_idx:03d}-{local_idx:02d}"
      slot_records: list[OrdinalSlotRecord] = []

      for slot_j in range(SLOTS_PER_BUNDLE):
        slot_tag = f"s{slot_j}"
        draw_u = stats_lib.hmac_uniform_float64(
            "dgem-d3a-regime", item_id, slot_tag
        )
        jitter = 0.03 * _norm_z("dgem-d3a-jit", item_id, slot_tag)

        if draw_u < 0.042:
          # Bimodal polarized opposition across extremes 0 and 4.
          left_mass = max(0.44, min(0.56, 0.50 + jitter))
          right_mass = 0.99 - left_mass
          probs = (left_mass, 0.004, 0.002, 0.004, right_mass)
          gold_coin = stats_lib.hmac_uniform_float64(
              "dgem-d3a-gold", item_id, f"bi:{slot_j}"
          )
          gold_level = 0 if gold_coin < left_mass / 0.99 else 4
          regime = "bimodal_polarized"
        elif draw_u < 0.223:
          # Benign adjacent rounding between neighboring levels (k, k+1).
          anchor = int(
              stats_lib.hmac_uint64("dgem-d3a-adj", item_id, slot_tag) % 4
          )
          raw_vec = [0.004, 0.004, 0.004, 0.004, 0.004]
          lower_share = max(0.44, min(0.54, 0.49 + jitter))
          raw_vec[anchor] = lower_share
          raw_vec[anchor + 1] = 0.988 - lower_share
          norm_sum = sum(raw_vec)
          probs = (
              raw_vec[0] / norm_sum,
              raw_vec[1] / norm_sum,
              raw_vec[2] / norm_sum,
              raw_vec[3] / norm_sum,
              raw_vec[4] / norm_sum,
          )
          gold_coin = stats_lib.hmac_uniform_float64(
              "dgem-d3a-gold", item_id, f"adj:{slot_j}"
          )
          gold_level = anchor if gold_coin < lower_share / 0.988 else anchor + 1
          regime = "benign_adjacent"
        else:
          # Clear unimodal rubric slot (H_norm < 0.21 < 0.25).
          center = int(
              stats_lib.hmac_uint64("dgem-d3a-clr", item_id, slot_tag) % 5
          )
          peak = max(
              0.92,
              min(0.98, 0.95 + 0.015 * _norm_z("dgem-d3a-pk", item_id, slot_tag)),
          )
          rem = 1.0 - peak
          raw_vec = [0.0, 0.0, 0.0, 0.0, 0.0]
          raw_vec[center] = peak
          if center > 0 and center < 4:
            raw_vec[center - 1] = 0.5 * rem
            raw_vec[center + 1] = 0.5 * rem
          elif center == 0:
            raw_vec[1] = 0.85 * rem
            raw_vec[2] = 0.15 * rem
          else:
            raw_vec[3] = 0.85 * rem
            raw_vec[2] = 0.15 * rem
          probs = (
              raw_vec[0],
              raw_vec[1],
              raw_vec[2],
              raw_vec[3],
              raw_vec[4],
          )
          err_coin = stats_lib.hmac_uniform_float64(
              "dgem-d3a-gold", item_id, f"clr:{slot_j}"
          )
          if err_coin < 0.99:
            gold_level = center
          else:
            gold_level = max(0, min(4, center + (1 if center < 4 else -1)))
          regime = "clear"

        slot_records.append(
            OrdinalSlotRecord(
                slot_index=slot_j,
                regime=regime,
                probs=probs,
                gold_level=gold_level,
            )
        )

      content_text = (
          f"[{document_id}] Checklist bundle {item_id}: "
          + "|".join(
              f"s{s.slot_index}:{s.regime}:{s.gold_level}:{s.probs[0]:.4f}"
              for s in slot_records
          )
      )
      split_item = stats_lib.BenchmarkSplitItem(
          item_id=item_id,
          cluster_id=document_id,
          content_text=content_text,
          content_sha256=stats_lib.compute_content_sha256(content_text),
      )
      bundles.append(
          D3aChecklistBundle(
              item_id=item_id,
              document_id=document_id,
              slots=tuple(slot_records),
              split_item=split_item,
          )
      )
  return bundles


def build_d3b_corpus() -> list[D3bEntityPair]:
  """Build 3,000 product deduplication pairs + 600 embedded 3-cycles.

  Within each of the `100` connected `entity_component_id` clusters (`30` pairs
  per component):
    - First `18` pairs form `6` explicit 3-cycles (`(A, B), (B, C), (A, C)`),
      totaling `600` 3-cycles (`180 D_cal / 120 D_val / 300 D_test`).
    - Remaining `12` pairs cover single-vs-multipack traps (`"500ml"` vs
      `"2 x 500ml"`, `"6-pack"`), SKU capacity/color variants, and reordered
      exact duplicates modeled on `WDC Products` and `Abt-Buy`.

  Returns:
    Deterministic list of `3,000` `D3bEntityPair` records.

  Raises:
    stats_lib.SplitLeakageError: If content hashing fails.
  """
  product_bases = (
      ("Alpine Sparkling Mineral Water", "500ml", "Beverages"),
      ("RoastMaster Whole Bean Espresso", "250g", "Coffee"),
      ("Nordic Oat Barista Drink", "1L", "DairyAlt"),
      ("Citrus Burst Isotonic Electrolyte", "330ml", "SportsDrink"),
      ("PureCare Sensitive Toothpaste", "75ml", "OralCare"),
      ("UltraClean Dishwasher Gel Pods", "400g", "HomeCare"),
      ("ProCharge Braided USB-C Cable", "2m", "Electronics"),
      ("Veloce Carbon Road Tire", "700x28c", "Cycling"),
      ("Summit Alkaline AA Cell", "1500mAh", "Batteries"),
      ("ClearView IPS Monitor 1920x1080", "27in", "Displays"),
  )
  multipack_templates = (
      "2 x {unit}",
      "6x{unit}",
      "{unit} x 4",
      "6-pack {unit}",
      "{unit} (Pack of 12)",
      "Twin-Pack {unit}",
  )

  pairs: list[D3bEntityPair] = []
  for comp_idx in range(TOTAL_ENTITY_COMPONENTS):
    comp_id = f"ent_comp_{comp_idx:03d}"
    base_name, unit_str, category_tag = product_bases[
        comp_idx % len(product_bases)
    ]
    sku_code = f"SKU{1000 + comp_idx * 13}"

    # 6 triangles per component = 18 pairs (indices 0..17).
    for tri_idx in range(TRIANGLES_PER_COMPONENT):
      tri_global = comp_idx * TRIANGLES_PER_COMPONENT + tri_idx
      node_a = f"{comp_id}_t{tri_idx}_A"
      node_b = f"{comp_id}_t{tri_idx}_B"
      node_c = f"{comp_id}_t{tri_idx}_C"

      if tri_idx % 2 == 0:
        # All 3 nodes are true duplicates (bridge B connects A and C, while
        # A vs C has low surface lexical overlap causing raw triangle rift!).
        title_a = f"{base_name} {unit_str} [{sku_code}-T{tri_idx}]"
        title_b = f"{sku_code}-T{tri_idx} {base_name} {category_tag} {unit_str}"
        title_c = (
            f"{category_tag} {unit_str} Authentic {sku_code}-T{tri_idx} Edition"
        )
        gold_flags = (True, True, True)
        cat_labels = ("tri_true_all", "tri_true_all", "tri_true_all")
      else:
        # A and B are single-unit duplicates; C is a multipack confuser!
        mp_fmt = multipack_templates[
            (comp_idx + tri_idx) % len(multipack_templates)
        ].format(unit=unit_str)
        title_a = f"{base_name} {unit_str} Single [{sku_code}-T{tri_idx}]"
        title_b = f"{base_name} {unit_str} [{sku_code}-T{tri_idx}]"
        title_c = f"{base_name} {mp_fmt} [{sku_code}-T{tri_idx}]"
        gold_flags = (True, False, False)
        cat_labels = (
            "tri_dup_edge",
            "tri_multipack_trap",
            "tri_multipack_trap",
        )

      tri_edges = (
          (node_a, node_b, title_a, title_b, gold_flags[0], cat_labels[0]),
          (node_b, node_c, title_b, title_c, gold_flags[1], cat_labels[1]),
          (node_a, node_c, title_a, title_c, gold_flags[2], cat_labels[2]),
      )
      for edge_pos, (na, nb, ta, tb, is_dup, cat_str) in enumerate(tri_edges):
        local_idx = tri_idx * 3 + edge_pos
        item_id = f"d3b-{comp_idx:03d}-{local_idx:02d}"
        content_text = f"[{comp_id}] pair={item_id} A={ta} || B={tb}"
        split_item = stats_lib.BenchmarkSplitItem(
            item_id=item_id,
            cluster_id=comp_id,
            content_text=content_text,
            content_sha256=stats_lib.compute_content_sha256(content_text),
        )
        pairs.append(
            D3bEntityPair(
                item_id=item_id,
                entity_component_id=comp_id,
                node_a=na,
                node_b=nb,
                title_a=ta,
                title_b=tb,
                is_duplicate=is_dup,
                pair_category=cat_str,
                triangle_group=tri_global,
                split_item=split_item,
            )
        )

    # Remaining 12 standalone pairs per component (indices 18..29).
    for extra_idx in range(12):
      local_idx = 18 + extra_idx
      item_id = f"d3b-{comp_idx:03d}-{local_idx:02d}"
      node_a = f"{comp_id}_p{extra_idx}_A"
      node_b = f"{comp_id}_p{extra_idx}_B"
      variant_tag = f"{sku_code}-P{extra_idx}"

      if extra_idx < 4:
        # True reordered duplicate (single or matching multipack).
        if extra_idx == 3:
          title_a = f"{base_name} 6 x {unit_str} ({variant_tag})"
          title_b = f"{variant_tag} {base_name} {unit_str} 6-pack"
        else:
          title_a = f"{base_name} {unit_str} Official {variant_tag}"
          title_b = f"{variant_tag} - {unit_str} {base_name} ({category_tag})"
        is_dup = True
        cat_str = "standalone_duplicate"
      elif extra_idx < 8:
        # Multipack vs single trap!
        mp_str = multipack_templates[
            (comp_idx + extra_idx) % len(multipack_templates)
        ].format(unit=unit_str)
        title_a = f"{base_name} {unit_str} {variant_tag}"
        title_b = f"{base_name} {mp_str} {variant_tag}"
        is_dup = False
        cat_str = "multipack_trap"
      else:
        # Distinct model/accessory/flavor non-duplicate.
        flavor_b = ("Zero Sugar", "Replacement Lid", "Pro Max", "Mini Edition")[
            extra_idx - 8
        ]
        title_a = f"{base_name} {unit_str} Original {variant_tag}"
        title_b = f"{base_name} {unit_str} {flavor_b} {variant_tag}X"
        is_dup = False
        cat_str = "variant_non_duplicate"

      content_text = f"[{comp_id}] pair={item_id} A={title_a} || B={title_b}"
      split_item = stats_lib.BenchmarkSplitItem(
          item_id=item_id,
          cluster_id=comp_id,
          content_text=content_text,
          content_sha256=stats_lib.compute_content_sha256(content_text),
      )
      pairs.append(
          D3bEntityPair(
              item_id=item_id,
              entity_component_id=comp_id,
              node_a=node_a,
              node_b=node_b,
              title_a=title_a,
              title_b=title_b,
              is_duplicate=is_dup,
              pair_category=cat_str,
              triangle_group=-1,
              split_item=split_item,
          )
      )
  return pairs


def _evaluate_d3a(
    cal_bundles: Sequence[D3aChecklistBundle],
    val_bundles: Sequence[D3aChecklistBundle],
    test_bundles: Sequence[D3aChecklistBundle],
) -> dict[str, object]:
  """Run Experiment D3a comparing bundle max-H_norm vs Popoviciu + Sidak/Bonf."""
  # Family-wise error rate control across M = 8 slots per bundle at alpha = 0.10.
  alpha_bundle = 0.10
  alpha_sidak = 1.0 - math.pow(1.0 - alpha_bundle, 1.0 / float(SLOTS_PER_BUNDLE))
  alpha_bonferroni = alpha_bundle / float(SLOTS_PER_BUNDLE)

  # Fit LAC nonconformity 1 - p_gold on D_cal slots (720 * 8 = 5,760 slots)
  # under Sidak and Bonferroni family-wise error control across M = 8 slots.
  cal_lac_scores: list[float] = []
  for bundle in cal_bundles:
    for slot in bundle.slots:
      cal_lac_scores.append(1.0 - float(slot.probs[slot.gold_level]))
  cal_lac_scores.sort()
  n_cal_slots = len(cal_lac_scores)

  idx_sidak = min(
      n_cal_slots - 1,
      max(0, int(math.ceil((n_cal_slots + 1) * (1.0 - alpha_sidak))) - 1),
  )
  idx_bonf = min(
      n_cal_slots - 1,
      max(0, int(math.ceil((n_cal_slots + 1) * (1.0 - alpha_bonferroni))) - 1),
  )
  q_lac_sidak = cal_lac_scores[idx_sidak]
  q_lac_bonf = cal_lac_scores[idx_bonf]
  min_prob_sidak = max(0.02, 1.0 - q_lac_sidak)
  min_prob_bonf = max(0.02, 1.0 - q_lac_bonf)

  # Select Popoviciu normalized variance cutoff tau_sigma2 strictly on D_val
  # maximizing F1 for detecting slots with error > 1 rubric step (polarized).
  best_tau_var = 0.20
  best_f1 = -1.0
  for candidate_tau in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
    tp = 0
    fp = 0
    fn = 0
    for bundle in val_bundles:
      for slot in bundle.slots:
        pred_lvl = max(range(5), key=lambda k: slot.probs[k])
        severe_risk = slot.regime == "bimodal_polarized" or (
            abs(pred_lvl - slot.gold_level) >= 2
        )
        flagged = stats_lib.normalized_ordinal_variance(slot.probs) > candidate_tau
        if flagged and severe_risk:
          tp += 1
        elif flagged and not severe_risk:
          fp += 1
        elif not flagged and severe_risk:
          fn += 1
    denom = 2 * tp + fp + fn
    f1_val = (2.0 * tp / float(denom)) if denom > 0 else 0.0
    if f1_val > best_f1:
      best_f1 = f1_val
      best_tau_var = candidate_tau

  # Solve a slot-matched H_norm threshold on D_val that escalates the exact
  # same fraction of slots as Popoviciu+Sidak using the identical span rule.
  val_pop_flags = 0
  val_hnorm_scores: list[float] = []
  for bundle in val_bundles:
    for slot in bundle.slots:
      prob_map = {str(k): slot.probs[k] for k in range(5)}
      jitter = 1e-12 * stats_lib.hmac_uniform_float64(
          "d3a-j", bundle.item_id, f"s{slot.slot_index}"
      )
      val_hnorm_scores.append(stats_lib.normalized_entropy(prob_map) + jitter)
      ord_var = stats_lib.normalized_ordinal_variance(slot.probs)
      c_levels = [k for k in range(5) if slot.probs[k] >= min_prob_sidak]
      span_wide = (max(c_levels) - min(c_levels) > 1) if c_levels else True
      if ord_var > best_tau_var or span_wide:
        val_pop_flags += 1

  matched_slot_rate = max(
      0.01, min(0.99, val_pop_flags / float(len(val_hnorm_scores)))
  )
  matched_hnorm_tau = stats_lib.solve_matched_escalation_threshold(
      val_hnorm_scores, matched_slot_rate
  )

  # Evaluate 4 policies on locked D_test (1,200 bundles = 9,600 slots):
  # 1. bundle_max_hnorm_025 (PR #35 baseline: escalates all 8 slots if max H_norm > 0.25)
  # 2. slot_matched_hnorm (per-slot H_norm at matched slot escalation rate)
  # 3. popoviciu_sidak_gate (per-slot Popoviciu variance + Sidak interval span)
  # 4. popoviciu_bonferroni_gate (per-slot Popoviciu variance + Bonferroni span)
  policies = (
      "bundle_max_hnorm_025",
      "slot_matched_hnorm",
      "popoviciu_sidak_gate",
      "popoviciu_bonferroni_gate",
  )

  bundle_escalated: dict[str, list[float]] = {p: [] for p in policies}
  slot_escalated_counts: dict[str, int] = {p: 0 for p in policies}
  bimodal_total = 0
  adjacent_total = 0
  bimodal_caught: dict[str, int] = {p: 0 for p in policies}
  adjacent_false_esc: dict[str, int] = {p: 0 for p in policies}
  slot_abs_errors: dict[str, list[float]] = {p: [] for p in policies}
  slot_within_one: dict[str, list[bool]] = {p: [] for p in policies}
  slot_clusters: list[str] = []

  for bundle in test_bundles:
    slot_hnorms = [
        stats_lib.normalized_entropy({str(k): s.probs[k] for k in range(5)})
        + 1e-12
        * stats_lib.hmac_uniform_float64(
            "d3a-j", bundle.item_id, f"s{s.slot_index}"
        )
        for s in bundle.slots
    ]
    bundle_max_h = max(slot_hnorms)

    any_esc_flags = {p: False for p in policies}
    for slot_idx, slot in enumerate(bundle.slots):
      slot_clusters.append(bundle.document_id)
      if slot.regime == "bimodal_polarized":
        bimodal_total += 1
      elif slot.regime == "benign_adjacent":
        adjacent_total += 1

      ord_var = stats_lib.normalized_ordinal_variance(slot.probs)
      support_sidak = [k for k in range(5) if slot.probs[k] >= min_prob_sidak]
      span_sidak = (
          (max(support_sidak) - min(support_sidak)) > 1
          if support_sidak
          else True
      )
      support_bonf = [k for k in range(5) if slot.probs[k] >= min_prob_bonf]
      span_bonf = (
          (max(support_bonf) - min(support_bonf)) > 1 if support_bonf else True
      )

      decisions = {
          "bundle_max_hnorm_025": bundle_max_h > 0.25,
          "slot_matched_hnorm": slot_hnorms[slot_idx] > matched_hnorm_tau,
          "popoviciu_sidak_gate": (ord_var > best_tau_var) or span_sidak,
          "popoviciu_bonferroni_gate": (ord_var > best_tau_var) or span_bonf,
      }

      s1_pred = max(range(5), key=lambda k: slot.probs[k])
      # Stage-2 rubric arbiter resolves escalated slot with 96% exact accuracy.
      s2_ok = (
          stats_lib.hmac_uniform_float64(
              "d3a-s2", bundle.item_id, f"s{slot_idx}"
          )
          < 0.96
      )
      s2_pred = slot.gold_level if s2_ok else s1_pred

      for pol_name, esc_flag in decisions.items():
        if esc_flag:
          any_esc_flags[pol_name] = True
          slot_escalated_counts[pol_name] += 1
          if slot.regime == "bimodal_polarized":
            bimodal_caught[pol_name] += 1
          elif slot.regime == "benign_adjacent":
            adjacent_false_esc[pol_name] += 1
        final_pred = s2_pred if esc_flag else s1_pred
        abs_err = float(abs(final_pred - slot.gold_level))
        slot_abs_errors[pol_name].append(abs_err)
        slot_within_one[pol_name].append(abs_err <= 1.0)

    for pol_name in policies:
      bundle_escalated[pol_name].append(1.0 if any_esc_flags[pol_name] else 0.0)

  total_test_slots = len(test_bundles) * SLOTS_PER_BUNDLE
  bundle_clusters = [b.document_id for b in test_bundles]
  policy_metrics: dict[str, dict[str, object]] = {}

  for pol_name in policies:
    b_esc_ci = stats_lib.cluster_bootstrap_mean_ci(
        bundle_clusters,
        bundle_escalated[pol_name],
        num_draws=2000,
        seed=20263101,
    )
    mae_ci = stats_lib.cluster_bootstrap_mean_ci(
        slot_clusters,
        slot_abs_errors[pol_name],
        num_draws=2000,
        seed=20263102,
    )
    policy_metrics[pol_name] = {
        "n_bundles": len(test_bundles),
        "n_slots": total_test_slots,
        "bundle_escalation_rate": round(b_esc_ci.estimate, 4),
        "bundle_escalation_ci95": [
            round(b_esc_ci.ci_lower, 4),
            round(b_esc_ci.ci_upper, 4),
        ],
        "slot_escalation_rate": round(
            slot_escalated_counts[pol_name] / float(total_test_slots), 4
        ),
        "bimodal_polarized_recall": round(
            bimodal_caught[pol_name] / float(max(1, bimodal_total)), 4
        ),
        "benign_adjacent_false_escalation_rate": round(
            adjacent_false_esc[pol_name] / float(max(1, adjacent_total)), 4
        ),
        "slot_mae": round(mae_ci.estimate, 4),
        "slot_mae_ci95": [round(mae_ci.ci_lower, 4), round(mae_ci.ci_upper, 4)],
        "within_1_step_accuracy": round(
            sum(1 for ok in slot_within_one[pol_name] if ok)
            / float(total_test_slots),
            4,
        ),
    }

  # Paired McNemar on within-1-step accuracy: Popoviciu+Sidak vs Slot-H_norm
  mcnemar_sidak_vs_slot_h = stats_lib.paired_mcnemar_exact(
      slot_within_one["slot_matched_hnorm"],
      slot_within_one["popoviciu_sidak_gate"],
  )
  bundle_esc_diff_ci = stats_lib.paired_cluster_bootstrap_ci(
      bundle_clusters,
      bundle_escalated["bundle_max_hnorm_025"],
      bundle_escalated["popoviciu_sidak_gate"],
      num_draws=2000,
      seed=20263109,
  )

  return {
      "n_cal_bundles": len(cal_bundles),
      "n_val_bundles": len(val_bundles),
      "n_test_bundles": len(test_bundles),
      "slots_per_bundle": SLOTS_PER_BUNDLE,
      "fitted_tau_popoviciu_var": best_tau_var,
      "fitted_matched_slot_hnorm_tau": round(matched_hnorm_tau, 4),
      "sidak_slot_alpha": round(alpha_sidak, 6),
      "bonferroni_slot_alpha": round(alpha_bonferroni, 6),
      "policies": policy_metrics,
      "comparisons": {
          "popoviciu_sidak_vs_bundle_max_hnorm_escalation_delta": {
              "delta_bundle_escalation": round(bundle_esc_diff_ci.estimate, 4),
              "ci95": [
                  round(bundle_esc_diff_ci.ci_lower, 4),
                  round(bundle_esc_diff_ci.ci_upper, 4),
              ],
          },
          "popoviciu_sidak_vs_slot_matched_hnorm_within1_mcnemar": {
              "delta_accuracy": round(
                  mcnemar_sidak_vs_slot_h.delta_accuracy, 4
              ),
              "n_10": mcnemar_sidak_vs_slot_h.n_10,
              "n_01": mcnemar_sidak_vs_slot_h.n_01,
              "p_value_exact": mcnemar_sidak_vs_slot_h.p_value_exact,
          },
      },
  }


def _token_jaccard(title_a: str, title_b: str) -> float:
  """Compute lowercase alphanumeric token Jaccard similarity."""
  tokens_a = {
      tok.strip("[]()-.,")
      for tok in title_a.lower().split()
      if tok.strip("[]()-.,")
  }
  tokens_b = {
      tok.strip("[]()-.,")
      for tok in title_b.lower().split()
      if tok.strip("[]()-.,")
  }
  union_set = tokens_a | tokens_b
  if not union_set:
    return 0.0
  return float(len(tokens_a & tokens_b)) / float(len(union_set))


def _simulate_entity_equiv_prob(pair: D3bEntityPair) -> float:
  """Simulate raw Stage-1 equivalence probability prior to pack/triangle guards."""
  noise = 0.09 * _norm_z("dgem-d3b-prob", pair.item_id, "equiv")
  if pair.pair_category == "tri_true_all":
    # On edge (A, C) (the 3rd edge in each triangle), surface lexical divergence
    # often depresses raw Stage-1 probability, creating a 3-cycle contradiction!
    if pair.item_id.endswith(("02", "05", "08", "11", "14", "17")):
      base_p = 0.24 + 0.12 * stats_lib.hmac_uniform_float64(
          "dgem-d3b-tri", pair.item_id, "weak"
      )
    else:
      base_p = 0.88
  elif pair.pair_category in ("tri_multipack_trap", "multipack_trap"):
    # Unguarded Stage-1 is frequently fooled by high brand/SKU/unit overlap!
    # In odd triangles, one of (B,C) or (A,C) gets high prob while the other is
    # uncertain, creating multipack false merges AND 3-cycle contradictions.
    if pair.item_id.endswith(("04", "10", "16")) or (
        pair.pair_category == "multipack_trap"
        and stats_lib.hmac_uniform_float64("dgem-d3b-mp", pair.item_id, "fool")
        < 0.68
    ):
      base_p = 0.81
    else:
      base_p = 0.22
  elif pair.is_duplicate:
    base_p = 0.86
  else:
    base_p = 0.14
  return max(0.01, min(0.99, base_p + noise))


def _compute_f1_summary(
    gold_flags: Sequence[bool],
    pred_flags: Sequence[bool],
) -> dict[str, float]:
  """Compute precision, recall, F1, and accuracy on binary pair predictions."""
  tp = sum(1 for g, p in zip(gold_flags, pred_flags) if g and p)
  fp = sum(1 for g, p in zip(gold_flags, pred_flags) if (not g) and p)
  fn = sum(1 for g, p in zip(gold_flags, pred_flags) if g and (not p))
  correct = sum(1 for g, p in zip(gold_flags, pred_flags) if g == p)
  precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
  f1_score = (
      2.0 * precision * recall / (precision + recall)
      if (precision + recall) > 0.0
      else 0.0
  )
  return {
      "precision": round(precision, 4),
      "recall": round(recall, 4),
      "f1": round(f1_score, 4),
      "accuracy": round(correct / float(len(gold_flags)), 4),
  }


def _count_triangle_contradictions(
    pairs_seq: Sequence[D3bEntityPair],
    pred_flags: Sequence[bool],
) -> tuple[int, int]:
  """Count hard transitive contradictions `(2 True, 1 False)` across 3-cycles."""
  by_tri: dict[int, list[bool]] = {}
  for pair_obj, pred_val in zip(pairs_seq, pred_flags):
    if pair_obj.triangle_group >= 0:
      by_tri.setdefault(pair_obj.triangle_group, []).append(pred_val)
  total_tris = 0
  contradictions = 0
  for edge_bools in by_tri.values():
    if len(edge_bools) != 3:
      continue
    total_tris += 1
    if sum(1 for flag in edge_bools if flag) == 2:
      contradictions += 1
  return contradictions, total_tris


def _evaluate_d3b(
    val_pairs: Sequence[D3bEntityPair],
    test_pairs: Sequence[D3bEntityPair],
) -> dict[str, object]:
  """Run Experiment D3b across lexical, unprotected, pack-guard, and arbiter."""
  # Fit Jaccard threshold and Stage-1 probability threshold strictly on D_val.
  best_jaccard_tau = 0.50
  best_jaccard_f1 = -1.0
  val_jaccards = [_token_jaccard(p.title_a, p.title_b) for p in val_pairs]
  val_golds = [p.is_duplicate for p in val_pairs]
  for cand_t in (0.35, 0.45, 0.55, 0.65, 0.75):
    preds = [j >= cand_t for j in val_jaccards]
    f1_val = _compute_f1_summary(val_golds, preds)["f1"]
    if f1_val > best_jaccard_f1:
      best_jaccard_f1 = f1_val
      best_jaccard_tau = cand_t

  val_probs = [_simulate_entity_equiv_prob(p) for p in val_pairs]
  best_prob_tau = 0.50
  best_prob_f1 = -1.0
  for cand_t in (0.40, 0.45, 0.50, 0.55, 0.60):
    preds = [prob >= cand_t for prob in val_probs]
    f1_val = _compute_f1_summary(val_golds, preds)["f1"]
    if f1_val > best_prob_f1:
      best_prob_f1 = f1_val
      best_prob_tau = cand_t

  # Evaluate on locked D_test (1,500 pairs + 300 embedded 3-cycles).
  test_golds = [p.is_duplicate for p in test_pairs]
  test_jaccards = [_token_jaccard(p.title_a, p.title_b) for p in test_pairs]
  test_raw_probs = [_simulate_entity_equiv_prob(p) for p in test_pairs]

  # Arm 1: Lexical Jaccard baseline.
  preds_lexical = [val >= best_jaccard_tau for val in test_jaccards]

  # Arm 2: Stage-1 unprotected (without ParsePackMultiplier guard).
  preds_unprotected = [prob >= best_prob_tau for prob in test_raw_probs]

  # Arm 3: Stage-1 + ParsePackMultiplier guard.
  guarded_probs: list[float] = []
  preds_pack_guard: list[bool] = []
  for idx, pair_obj in enumerate(test_pairs):
    mult_a = stats_lib.parse_pack_multiplier(pair_obj.title_a)
    mult_b = stats_lib.parse_pack_multiplier(pair_obj.title_b)
    if mult_a != mult_b:
      guarded_probs.append(0.02)
      preds_pack_guard.append(False)
    else:
      guarded_probs.append(test_raw_probs[idx])
      preds_pack_guard.append(test_raw_probs[idx] >= best_prob_tau)

  # Arm 4: Stage-1 + ParsePackMultiplier + Probabilistic Triangle Arbiter.
  # Detect 3-cycles with continuous violation mass m_max >= 0.30 (matching
  # ScoreProbabilisticTriangles in arbiters.go) and escalate weak/inconsistent
  # edges to Stage-2 joint 3-node consistency arbiter.
  preds_arbiter = list(preds_pack_guard)
  tri_indices: dict[int, list[int]] = {}
  for idx, pair_obj in enumerate(test_pairs):
    if pair_obj.triangle_group >= 0:
      tri_indices.setdefault(pair_obj.triangle_group, []).append(idx)

  arbitrated_triangles = 0
  for tri_id, idx_list in tri_indices.items():
    if len(idx_list) != 3:
      continue
    p_ab, p_bc, p_ac = (
        guarded_probs[idx_list[0]],
        guarded_probs[idx_list[1]],
        guarded_probs[idx_list[2]],
    )
    violation_masses = (
        p_bc * p_ac * (1.0 - p_ab),
        p_ab * p_ac * (1.0 - p_bc),
        p_ab * p_bc * (1.0 - p_ac),
    )
    has_hard_contradiction = (
        sum(1 for i in idx_list if preds_arbiter[i]) == 2
    )
    if max(violation_masses) >= 0.30 or has_hard_contradiction:
      arbitrated_triangles += 1
      # Joint 3-node Stage-2 arbiter sees all three titles simultaneously and
      # enforces transitive equivalence (97% joint resolution accuracy).
      arbiter_ok = (
          stats_lib.hmac_uniform_float64(
              "dgem-d3b-arbiter", f"tri:{tri_id}", "resolve"
          )
          < 0.97
      )
      if arbiter_ok:
        for edge_idx in idx_list:
          preds_arbiter[edge_idx] = test_pairs[edge_idx].is_duplicate

  arms_preds = {
      "lexical_jaccard_baseline": preds_lexical,
      "stage1_unprotected": preds_unprotected,
      "stage1_pack_guard": preds_pack_guard,
      "stage1_pack_guard_plus_triangle_arbiter": preds_arbiter,
  }

  multipack_indices = [
      idx
      for idx, p in enumerate(test_pairs)
      if "multipack_trap" in p.pair_category
  ]
  test_clusters = [p.entity_component_id for p in test_pairs]

  arms_summary: dict[str, dict[str, object]] = {}
  correct_flags_by_arm: dict[str, list[bool]] = {}
  for arm_name, pred_list in arms_preds.items():
    metrics = _compute_f1_summary(test_golds, pred_list)
    correct_vec = [g == p for g, p in zip(test_golds, pred_list)]
    correct_flags_by_arm[arm_name] = correct_vec
    acc_ci = stats_lib.cluster_bootstrap_mean_ci(
        test_clusters,
        [1.0 if ok else 0.0 for ok in correct_vec],
        num_draws=2000,
        seed=20263201,
    )
    mp_false_merges = sum(1 for i in multipack_indices if pred_list[i])
    tri_contra, total_tris = _count_triangle_contradictions(
        test_pairs, pred_list
    )
    arms_summary[arm_name] = {
        **metrics,
        "accuracy_ci95": [round(acc_ci.ci_lower, 4), round(acc_ci.ci_upper, 4)],
        "multipack_trap_n": len(multipack_indices),
        "multipack_false_merge_rate": round(
            mp_false_merges / float(max(1, len(multipack_indices))), 4
        ),
        "test_triangles": total_tris,
        "triangle_contradictions": tri_contra,
        "triangle_contradiction_rate": round(
            tri_contra / float(max(1, total_tris)), 4
        ),
    }

  # Pairwise McNemar + Holm-Bonferroni across planned comparisons.
  planned_pairs = (
      ("stage1_unprotected", "stage1_pack_guard"),
      ("stage1_pack_guard", "stage1_pack_guard_plus_triangle_arbiter"),
      ("lexical_jaccard_baseline", "stage1_pack_guard_plus_triangle_arbiter"),
      ("stage1_unprotected", "stage1_pack_guard_plus_triangle_arbiter"),
  )
  raw_ps: list[float] = []
  comparisons: dict[str, dict[str, object]] = {}
  comp_keys: list[str] = []
  for base_arm, treat_arm in planned_pairs:
    mcnemar = stats_lib.paired_mcnemar_exact(
        correct_flags_by_arm[base_arm],
        correct_flags_by_arm[treat_arm],
    )
    diff_ci = stats_lib.paired_cluster_bootstrap_ci(
        test_clusters,
        [1.0 if ok else 0.0 for ok in correct_flags_by_arm[base_arm]],
        [1.0 if ok else 0.0 for ok in correct_flags_by_arm[treat_arm]],
        num_draws=2000,
        seed=20263250,
    )
    key = f"{treat_arm}_vs_{base_arm}"
    comp_keys.append(key)
    raw_ps.append(mcnemar.p_value_exact)
    comparisons[key] = {
        "delta_accuracy": round(mcnemar.delta_accuracy, 4),
        "delta_accuracy_ci95": [
            round(diff_ci.ci_lower, 4),
            round(diff_ci.ci_upper, 4),
        ],
        "n_10_regressions": mcnemar.n_10,
        "n_01_improvements": mcnemar.n_01,
        "mcnemar_chi2_cc": round(mcnemar.chi2_continuity, 4),
        "p_value_exact": mcnemar.p_value_exact,
    }

  holm_ps = stats_lib.holm_bonferroni_adjust(raw_ps)
  for key, adj_p in zip(comp_keys, holm_ps):
    comparisons[key]["p_value_holm"] = adj_p
    comparisons[key]["significant_fwer_005"] = adj_p < 0.05

  return {
      "n_test_pairs": len(test_pairs),
      "arbitrated_test_triangles": arbitrated_triangles,
      "fitted_jaccard_tau": best_jaccard_tau,
      "fitted_stage1_prob_tau": best_prob_tau,
      "arms": arms_summary,
      "comparisons": comparisons,
  }


def run_d3_suite(lockfile_path: pathlib.Path) -> dict[str, object]:
  """Execute Experiment D3a (ordinal bundles) and D3b (entity deduplication).

  Args:
    lockfile_path: Path to `dgem_exp_splits_v1.json`.

  Returns:
    Structured JSON summary of D3a and D3b results.

  Raises:
    stats_lib.SplitLeakageError: If split lockfile verification fails.
  """
  lock_payload = json.loads(
      stats_lib.validate_local_file_path(lockfile_path, must_exist=True).read_text(
          encoding="utf-8"
      )
  )

  d3a_corpus = build_d3a_corpus()
  d3a_by_id = {b.item_id: b for b in d3a_corpus}
  d3a_folds = stats_lib.verify_split_lockfile(
      lock_payload, TRACK_D3A, [b.split_item for b in d3a_corpus]
  )
  d3a_results = _evaluate_d3a(
      cal_bundles=[d3a_by_id[r.item_id] for r in d3a_folds["cal"]],
      val_bundles=[d3a_by_id[r.item_id] for r in d3a_folds["val"]],
      test_bundles=[d3a_by_id[r.item_id] for r in d3a_folds["test"]],
  )

  d3b_corpus = build_d3b_corpus()
  d3b_by_id = {p.item_id: p for p in d3b_corpus}
  d3b_folds = stats_lib.verify_split_lockfile(
      lock_payload, TRACK_D3B, [p.split_item for p in d3b_corpus]
  )
  d3b_results = _evaluate_d3b(
      val_pairs=[d3b_by_id[r.item_id] for r in d3b_folds["val"]],
      test_pairs=[d3b_by_id[r.item_id] for r in d3b_folds["test"]],
  )

  return {
      "experiment_id": "D3",
      "title": (
          "Popoviciu Ordinal Variance Gating & Entity Deduplication Arbiters"
      ),
      "execution_mode": "calibrated_parametric_fixture",
      "claim_bucket": "Modeled / Calibrated Offline Benchmark Fixture",
      "d3a_ordinal_checklist": d3a_results,
      "d3b_entity_deduplication": d3b_results,
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for Experiment D3."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lockfile", type=pathlib.Path, default=DEFAULT_LOCKFILE)
  parser.add_argument("--verify-splits-only", action="store_true")
  parser.add_argument("--output-json", type=pathlib.Path, default=None)
  parsed = parser.parse_args(argv)

  if parsed.verify_splits_only:
    lock_payload = json.loads(
        stats_lib.validate_local_file_path(
            parsed.lockfile, must_exist=True
        ).read_text(encoding="utf-8")
    )
    d3a_corpus = build_d3a_corpus()
    d3b_corpus = build_d3b_corpus()
    folds_a = stats_lib.verify_split_lockfile(
        lock_payload, TRACK_D3A, [b.split_item for b in d3a_corpus]
    )
    folds_b = stats_lib.verify_split_lockfile(
        lock_payload, TRACK_D3B, [p.split_item for p in d3b_corpus]
    )
    print(
        json.dumps({
            "tracks": {
                TRACK_D3A: {k: len(v) for k, v in folds_a.items()},
                TRACK_D3B: {k: len(v) for k, v in folds_b.items()},
            },
            "verified": True,
        })
    )
    return 0

  summary = run_d3_suite(lockfile_path=parsed.lockfile)
  serialized = json.dumps(summary, indent=2)
  if parsed.output_json is not None:
    out_path = stats_lib.validate_local_file_path(
        parsed.output_json, must_exist=False
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized + "\n", encoding="utf-8")
  print(serialized)
  return 0


if __name__ == "__main__":
  sys.exit(main())
