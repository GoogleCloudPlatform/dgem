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

"""Unit tests for scripts/policy_calibration.py."""

import io
import json
import math
import pathlib
import random
import tempfile
import unittest
from unittest import mock

from scripts import policy_calibration


class PolicyCalibrationTest(unittest.TestCase):
  """Verifies fold parity, OOF temperature, conformal sets, and matched gates."""

  def test_fold_index_matches_go_bench_calibration(self) -> None:
    """Asserts SHA-1 fold assignment matches cmd/temperature_fit_test.go."""
    expected_folds = {
        "easy-intent-00": 2,
        "hard-opus-a-long_policy-01": 4,
        "x": 0,
    }
    for item_id, expected_fold in expected_folds.items():
      self.assertEqual(
          policy_calibration.compute_fold_index(item_id, 5),
          expected_fold,
      )

  def test_raps_and_aps_scores_and_abstain_escalation(self) -> None:
    """Verifies RAPS rank regularization and 5-fold out-of-sample coverage."""
    six_options = {
        "o1": 0.30,
        "o2": 0.25,
        "o3": 0.20,
        "o4": 0.12,
        "o5": 0.08,
        "o6": 0.05,
    }
    raps_scores = policy_calibration.compute_option_nonconformity_scores(
        six_options,
        "raps",
    )
    self.assertAlmostEqual(raps_scores["o5"], 0.91, places=9)
    self.assertAlmostEqual(raps_scores["o6"], 1.00, places=9)

    generator = random.Random(42)
    rows: list[str] = []
    for index in range(120):
      draw_val = generator.random()
      if draw_val >= 0.92:
        gold_label = "other"
      elif draw_val >= 0.72:
        gold_label = "technical"
      else:
        gold_label = "billing"
      row_payload = {
          "id": f"ticket-{index:03d}",
          "slots": {
              "team": {
                  "predicted": "billing",
                  "expected": gold_label,
                  "correct": gold_label == "billing",
                  "confidence": 0.72,
                  "hesitation": 0.48,
                  "probabilities": {
                      "billing": 0.72,
                      "technical": 0.20,
                      "other": 0.08,
                  },
              }
          },
      }
      rows.append(json.dumps(row_payload))

    with tempfile.TemporaryDirectory() as temp_dir:
      receipt_path = pathlib.Path(temp_dir) / "receipt.jsonl"
      receipt_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

      stdout_buffer = io.StringIO()
      with mock.patch("sys.stdout", stdout_buffer):
        policy_calibration.main([
            str(receipt_path),
            "--conformal",
            "raps",
            "--conformal-alpha",
            "0.10",
            "--json",
        ])
      parsed_output = json.loads(stdout_buffer.getvalue())
      conformal_stats = parsed_output["team"]["conformal"]
      self.assertEqual(conformal_stats["method"], "raps")
      self.assertEqual(conformal_stats["evaluated_n"], 120)
      self.assertGreaterEqual(conformal_stats["empirical_coverage"], 0.84)
      self.assertNotIn("oof_temperature", conformal_stats)

      oos_rows = [
          json.dumps({
              "id": f"clinc-{idx:02d}",
              "slots": {
                  "intent": {
                      "predicted": "oos",
                      "expected": "oos",
                      "correct": True,
                      "confidence": 0.97,
                      "hesitation": 0.08,
                      "probabilities": {"oos": 0.97, "balance": 0.03},
                  }
              },
          })
          for idx in range(20)
      ]
      oos_receipt = pathlib.Path(temp_dir) / "clinc_oos.jsonl"
      oos_receipt.write_text("\n".join(oos_rows) + "\n", encoding="utf-8")

      default_buffer = io.StringIO()
      with mock.patch("sys.stdout", default_buffer):
        policy_calibration.main([
            str(oos_receipt),
            "--conformal",
            "lac",
            "--json",
        ])
      default_oos = json.loads(default_buffer.getvalue())["intent"]["conformal"]
      self.assertEqual(default_oos["singleton_share"], 1.0)
      self.assertEqual(default_oos["escalated_share"], 0.0)

      opt_in_buffer = io.StringIO()
      with mock.patch("sys.stdout", opt_in_buffer):
        policy_calibration.main([
            str(oos_receipt),
            "--conformal",
            "lac",
            "--conformal-abstain-labels",
            "oos,abstain",
            "--json",
        ])
      opt_in_oos = json.loads(opt_in_buffer.getvalue())["intent"]["conformal"]
      self.assertEqual(opt_in_oos["singleton_share"], 1.0)
      self.assertEqual(opt_in_oos["escalated_share"], 1.0)

  def test_oof_temperature_scaling_reduces_high_k_set_inflation(self) -> None:
    """Verifies 5-fold OOF NLL temperature scaling sharpens diluted wide-K sets."""
    generator = random.Random(2026)
    num_options = 32
    option_names = [f"opt_{idx:02d}" for idx in range(num_options)]
    rows: list[str] = []
    for row_idx in range(160):
      true_probs: dict[str, float] = {}
      lead_opt = option_names[row_idx % num_options]
      runner_opt = option_names[(row_idx + 1) % num_options]
      true_probs[lead_opt] = 0.92
      true_probs[runner_opt] = 0.05
      tail_each = 0.03 / float(num_options - 2)
      for opt_name in option_names:
        if opt_name not in true_probs:
          true_probs[opt_name] = tail_each

      draw_u = generator.random()
      if draw_u < 0.92:
        gold_label = lead_opt
      elif draw_u < 0.97:
        gold_label = runner_opt
      else:
        gold_label = option_names[(row_idx + 2) % num_options]

      # Simulate option-count dilution (underconfident raw softmax at T = 2.4).
      diluted_probs = policy_calibration.apply_temperature(true_probs, 2.4)
      top_conf = max(diluted_probs.values())
      hesitation = policy_calibration.compute_normalized_entropy(diluted_probs)
      rows.append(
          json.dumps({
              "id": f"wide-{row_idx:03d}",
              "slots": {
                  "intent": {
                      "predicted": lead_opt,
                      "expected": gold_label,
                      "correct": gold_label == lead_opt,
                      "confidence": top_conf,
                      "hesitation": hesitation,
                      "probabilities": diluted_probs,
                  }
              },
          })
      )

    with tempfile.TemporaryDirectory() as temp_dir:
      receipt_path = pathlib.Path(temp_dir) / "wide_k.jsonl"
      receipt_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

      raw_buf = io.StringIO()
      with mock.patch("sys.stdout", raw_buf):
        policy_calibration.main([
            str(receipt_path),
            "--conformal",
            "raps",
            "--conformal-alpha",
            "0.10",
            "--json",
        ])
      raw_raps = json.loads(raw_buf.getvalue())["intent"]["conformal"]

      oof_buf = io.StringIO()
      with mock.patch("sys.stdout", oof_buf):
        policy_calibration.main([
            str(receipt_path),
            "--conformal",
            "raps",
            "--conformal-alpha",
            "0.10",
            "--oof-temperature",
            "--json",
        ])
      oof_raps = json.loads(oof_buf.getvalue())["intent"]["conformal"]

      self.assertTrue(oof_raps["oof_temperature"])
      self.assertLess(oof_raps["mean_temperature"], 0.65)
      self.assertGreaterEqual(oof_raps["empirical_coverage"], 0.85)
      self.assertLessEqual(oof_raps["mean_set_size"], raw_raps["mean_set_size"])

      # Verify tabular console rendering with --oof-temperature and matched rates.
      text_buf = io.StringIO()
      with mock.patch("sys.stdout", text_buf):
        policy_calibration.main([
            str(receipt_path),
            "--conformal",
            "raps",
            "--oof-temperature",
            "--matched-escalation-rates",
            "0.20,0.30",
        ])
      rendered_text = text_buf.getvalue()
      self.assertIn("T_oof=", rendered_text)
      self.assertIn("matched escalation (target=30%):", rendered_text)
      self.assertIn("oof_temp_raps:", rendered_text)

  def test_matched_escalation_threshold_solver_and_policy_comparison(
      self,
  ) -> None:
    """Verifies matched D_val escalation outperforms Raw-H_norm on focal ties."""
    num_options = 24
    option_names = [f"c_{idx:02d}" for idx in range(num_options)]
    rows: list[str] = []

    for row_idx in range(200):
      lead_opt = option_names[0]
      rival_opt = option_names[1]
      prob_map: dict[str, float] = {}
      if row_idx % 4 == 0:
        # Focal 2-way tie on K = 24: low H_norm (~0.22), 50% top-1 error!
        prob_map[lead_opt] = 0.51
        prob_map[rival_opt] = 0.489
        tail_val = 0.001 / float(num_options - 2)
        for opt_name in option_names[2:]:
          prob_map[opt_name] = tail_val
        gold_label = lead_opt if (row_idx // 4) % 2 == 0 else rival_opt
      else:
        # Diffuse background tail: higher H_norm (~0.36), 98% top-1 accuracy!
        prob_map[lead_opt] = 0.72
        tail_val = 0.28 / float(num_options - 1)
        for opt_name in option_names[1:]:
          prob_map[opt_name] = tail_val
        gold_label = lead_opt if row_idx % 50 != 1 else rival_opt

      hesitation = policy_calibration.compute_normalized_entropy(prob_map)
      rows.append(
          json.dumps({
              "id": f"item-{row_idx:03d}",
              "slots": {
                  "route": {
                      "predicted": lead_opt,
                      "expected": gold_label,
                      "correct": gold_label == lead_opt,
                      "confidence": prob_map[lead_opt],
                      "hesitation": hesitation,
                      "probabilities": prob_map,
                  }
              },
          })
      )

    with tempfile.TemporaryDirectory() as temp_dir:
      receipt_path = pathlib.Path(temp_dir) / "matched.jsonl"
      receipt_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

      json_buf = io.StringIO()
      with mock.patch("sys.stdout", json_buf):
        policy_calibration.main([
            str(receipt_path),
            "--matched-escalation-rates",
            "0.30",
            "--json",
        ])
      sweep_list = json.loads(json_buf.getvalue())["route"][
          "matched_escalation"
      ]
      self.assertEqual(len(sweep_list), 1)
      policies = sweep_list[0]["policies"]
      raw_h = policies["raw_h_norm"]
      oof_raps = policies["oof_temp_raps"]

      self.assertLess(oof_raps["kept_error_rate"], raw_h["kept_error_rate"])
      self.assertLess(
          oof_raps["mean_stage2_candidates"],
          raw_h["mean_stage2_candidates"],
      )

  def test_validation_guards_fail_closed(self) -> None:
    """Verifies strict input validation on temperatures, quantiles, and rates."""
    for invalid_temp in (0.0, -1.0, 25.0, math.nan, math.inf, True):
      with self.assertRaises(ValueError):
        policy_calibration.apply_temperature({"a": 0.7, "b": 0.3}, invalid_temp)

    for invalid_rate in ("0.0", "1.0", "-0.2", "1.5", "abc"):
      with self.assertRaises(ValueError):
        policy_calibration.parse_escalation_rates(invalid_rate)

    with self.assertRaises(ValueError):
      policy_calibration.solve_matched_escalation_threshold([], 0.25)
    with self.assertRaises(ValueError):
      policy_calibration.solve_matched_escalation_threshold([0.1, math.nan], 0.2)
    with self.assertRaises(ValueError):
      policy_calibration.calibrate_conformal_quantile([0.1, 0.2], True)


if __name__ == "__main__":
  unittest.main()
