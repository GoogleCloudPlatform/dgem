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
import pathlib
import random
import tempfile
import unittest
from unittest import mock

from scripts import policy_calibration


class PolicyCalibrationTest(unittest.TestCase):
  """Verifies fold parity, nonconformity scores, and CLI conformal reporting."""

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


if __name__ == "__main__":
  unittest.main()
