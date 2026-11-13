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
"""Unit tests for out-of-fold Platt prior-bias and temperature calibration."""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import platt_prior_calibration  # noqa: E402


class PlattPriorCalibrationTest(unittest.TestCase):
    """Tests prior-shifted overconfidence recovery and safety-margin gating."""

    def test_platt_beats_temperature_only_under_prior_shift_and_near_one_crowding(self):
        # Simulate an SD-RL overconfident binary triage question with 30% true positives:
        # True positives have delta_logit ~ +9.0, decoys have delta_logit ~ +3.2 (still > 0!).
        # Pure temperature scaling cannot move any decoy below 0.50 (ECE >= 0.20),
        # while Platt prior bias shifts the intercept and drops OOF ECE below 0.05.
        rows = []
        for idx in range(200):
            is_real = (idx % 10) < 3
            raw_gap = (8.5 + 0.6 * math.sin(idx)) if is_real else (3.1 + 0.7 * math.cos(idx))
            p_yes = 1.0 / (1.0 + math.exp(-raw_gap))
            rows.append({
                "id": f"case_{idx}",
                "slots": {
                    "finding_real": {
                        "expected": is_real,
                        "predicted": "yes",
                        "confidence": p_yes,
                        "probabilities": {"yes": p_yes, "no": 1.0 - p_yes},
                        "stderr": 0.015,
                    }
                },
                "diagnostics": {
                    "questions": {
                        "finding_real": {
                            "logprobs": {"yes": 0.0, "no": -raw_gap},
                        }
                    }
                },
            })

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            for item in rows:
                tmp.write(json.dumps(item) + "\n")
            receipt_path = tmp.name
        self.addCleanup(lambda: os.path.exists(receipt_path) and os.remove(receipt_path))

        loaded = platt_prior_calibration.load_calibration_rows(receipt_path)
        self.assertEqual(len(loaded), 200)
        summary = platt_prior_calibration.cross_validate_question(
            loaded, num_folds=5, safety_option="yes", safety_sigma_margin=2.0
        )
        raw_ece = summary["raw"]["ece10"]
        temp_ece = summary["oof_temperature_only"]["ece10"]
        platt_ece = summary["oof_platt_prior_bias"]["ece10"]

        self.assertGreater(raw_ece, 0.60)
        # Pure temperature scaling stays bounded away from 0 because all decoys have gap > 0.
        self.assertGreater(temp_ece, 0.20)
        self.assertLess(platt_ece, 0.05)
        self.assertGreater(summary["oof_platt_prior_bias"]["accuracy"], 0.95)
        self.assertEqual(summary["safety_gate"]["oof_unsafe_allows"], 0)
        self.assertGreater(summary["safety_gate"]["oof_auto_decided_share"], 0.20)


if __name__ == "__main__":
    unittest.main()
