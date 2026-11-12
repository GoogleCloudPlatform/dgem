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
"""Unit tests for E1-E4 calibration, delimiter, factorial cascade, and keep-alive suites."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_cascade_calibration_delimiters as bench  # noqa: E402


class CascadeCalibrationDelimitersBenchTest(unittest.TestCase):
    """Verifies statistical invariants and guard branches across E1-E4."""

    def test_e1_platt_recovers_prior_shift_over_temperature_only(self):
        e1 = bench.run_e1_prior_shift_calibration()
        self.assertGreater(e1["raw_clamped_single_read"]["ece10"], 0.60)
        self.assertGreater(e1["temperature_only_cal"]["ece10"], 0.20)
        self.assertLess(e1["platt_prior_bias_lossless_cal"]["ece10"], 0.05)
        self.assertGreater(
            e1["platt_prior_bias_lossless_cal"]["auroc"],
            e1["raw_clamped_single_read"]["auroc"],
        )
        self.assertLess(e1["mcnemar_platt_vs_temp_only"]["p_value"], 1e-10)

    def test_e2_provenance_envelopes_collapse_nonce_swing_with_zero_unsafe_allows(self):
        e2 = bench.run_e2_delimiter_nonce_stability()
        strat = e2["strategies"]
        self.assertGreater(strat["raw_hex16_fixed_slot"]["max_case_swing"], 0.45)
        self.assertLess(
            strat["fixed_width_numeric_delimiter_fixed_slot"]["max_case_swing"],
            0.15,
        )
        self.assertLess(
            strat["json_escaped_envelope_fixed_slot"]["max_case_swing"],
            0.05,
        )
        for name, metrics in strat.items():
            self.assertEqual(metrics["oof_unsafe_allows"], 0, f"{name} had unsafe allows")
        self.assertGreater(
            strat["json_escaped_envelope_fixed_slot"]["oof_auto_allow_share"],
            strat["raw_hex16_random_slot"]["oof_auto_allow_share"],
        )

    def test_e3_factorial_shapley_sums_exactly_and_beats_iso_controls(self):
        e3 = bench.run_e3_factorial_cascade_audit()
        self.assertEqual(len(e3["grid_16_cells"]), 16)
        self.assertLess(e3["shapley_efficiency_residual"], 1e-12)
        self.assertGreater(e3["total_accuracy_gain"], 0.20)
        for comp_name, p_adj in e3["holm_adjusted_p_values"].items():
            self.assertLess(p_adj, 0.01, f"{comp_name} Holm p={p_adj} not < 0.01")

    def test_e4_keepalive_reduces_loopback_sockets_and_canvas_padding(self):
        e4 = bench.run_e4_serving_keepalive_and_canvas_bench(num_decisions=24)
        s8 = e4["loopback_http_keepalive"]["samples_8_workers_16"]
        self.assertEqual(s8["per_read_urlopen_tcp_accepts"], 192)
        self.assertLessEqual(s8["keepalive_pool_tcp_accepts"], 16)
        self.assertGreaterEqual(s8["socket_reduction_factor"], 12.0)
        self.assertEqual(
            e4["one_slot_canvas_accounting"]["canvas_128"]["trailing_pad_tokens"],
            118,
        )


if __name__ == "__main__":
    unittest.main()
