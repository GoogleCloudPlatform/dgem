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
"""Unit tests for E1-E4 calibration, delimiter, cascade, and keep-alive suites."""

import unittest
from collections.abc import Sequence

try:
    from scripts import exp_cascade_calibration_delimiters as bench
except ImportError:
    import exp_cascade_calibration_delimiters as bench  # type: ignore[no-redef]


class _FakeSentencePieceTokenizer:
    """Deterministic fake SentencePiece tokenizer for guard-branch testing."""

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> list[int]:
        del add_special_tokens
        if text == "<|channel>thought\n<channel|>allow: yes":
            return [100, 45518, 107, 101, 21950, 235292, 11262]
        if text == "<|channel>thought\n<channel|>q1: yes":
            return [100, 45518, 107, 101, 235280, 235274, 235292, 11262]
        if text == "48-19-03-72-61-84-20-95":
            return list(range(23))
        return [ord(ch) for ch in text]

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [chr(val) if 32 <= val < 127 else "x" for val in ids]


class CascadeCalibrationDelimitersBenchTest(unittest.TestCase):
    """Verifies statistical invariants and guard branches across E1-E4."""

    def test_e1_platt_recovers_prior_shift_over_temperature_only(self) -> None:
        e1_result = bench.run_e1_prior_shift_calibration()
        self.assertGreater(e1_result["raw_clamped_single_read"]["ece10"], 0.60)
        self.assertGreater(e1_result["temperature_only_cal"]["ece10"], 0.20)
        self.assertLess(
            e1_result["platt_prior_bias_lossless_cal"]["ece10"], 0.05
        )
        self.assertGreater(
            e1_result["platt_prior_bias_lossless_cal"]["auroc"],
            e1_result["raw_clamped_single_read"]["auroc"],
        )
        self.assertLess(
            e1_result["mcnemar_platt_vs_temp_only"]["p_value"], 1e-10
        )

    def test_e2_provenance_envelopes_and_verified_bpe_stats(self) -> None:
        e2_result = bench.run_e2_delimiter_nonce_stability()
        strategies = e2_result["strategies"]
        self.assertGreater(
            strategies["raw_hex16_fixed_slot"]["max_case_swing"], 0.45
        )
        self.assertLess(
            strategies["fixed_width_numeric_delimiter_fixed_slot"][
                "max_case_swing"
            ],
            0.15,
        )
        self.assertLess(
            strategies["json_escaped_envelope_fixed_slot"]["max_case_swing"],
            0.05,
        )
        self.assertEqual(
            strategies["fixed_width_numeric_delimiter_fixed_slot"][
                "delimiter_token_len_range"
            ],
            [23, 23],
        )
        self.assertEqual(
            strategies["json_escaped_envelope_fixed_slot"][
                "delimiter_token_len_range"
            ],
            [0, 0],
        )
        for name, metrics in strategies.items():
            self.assertEqual(
                metrics["oof_unsafe_allows"], 0, f"{name} had unsafe allows"
            )
        self.assertGreater(
            strategies["json_escaped_envelope_fixed_slot"][
                "oof_auto_allow_share"
            ],
            strategies["raw_hex16_random_slot"]["oof_auto_allow_share"],
        )

        verified_bpe = e2_result["verified_diffusiongemma_26b_bpe_n1000"]
        self.assertEqual(
            verified_bpe["raw_hex16_xml"]["single_nonce_tokens"][
                "distinct_lengths"
            ],
            8,
        )
        self.assertEqual(
            verified_bpe["fixed_hyphen_digits16"]["single_nonce_tokens"][
                "distinct_lengths"
            ],
            1,
        )
        self.assertEqual(
            verified_bpe["json_escaped_envelope"]["single_nonce_tokens"][
                "distinct_lengths"
            ],
            1,
        )

    def test_e3_factorial_shapley_sums_exactly_and_beats_controls(self) -> None:
        e3_result = bench.run_e3_factorial_cascade_audit()
        self.assertEqual(len(e3_result["grid_16_cells"]), 16)
        self.assertLess(e3_result["shapley_efficiency_residual"], 1e-12)
        self.assertGreater(e3_result["total_accuracy_gain"], 0.20)
        for comp_name, p_adj in e3_result["holm_adjusted_p_values"].items():
            self.assertLess(
                p_adj, 0.01, f"{comp_name} Holm p={p_adj} not < 0.01"
            )

    def test_e4_keepalive_reduces_loopback_sockets_and_canvas_padding(
        self,
    ) -> None:
        e4_result = bench.run_e4_serving_keepalive_and_canvas_bench(
            num_decisions=24
        )
        samples_eight = e4_result["loopback_http_keepalive"][
            "samples_8_workers_16"
        ]
        self.assertEqual(samples_eight["per_read_urlopen_tcp_accepts"], 192)
        self.assertLessEqual(samples_eight["keepalive_pool_tcp_accepts"], 16)
        self.assertGreaterEqual(samples_eight["socket_reduction_factor"], 12.0)

        canvas_accounting = e4_result["one_slot_canvas_accounting"]
        self.assertEqual(
            canvas_accounting["canvas_16"]["active_template_tokens"], 8
        )
        self.assertEqual(
            canvas_accounting["canvas_16"]["trailing_pad_tokens"], 8
        )
        self.assertEqual(
            canvas_accounting["canvas_128"]["trailing_pad_tokens"], 120
        )
        self.assertAlmostEqual(
            canvas_accounting["canvas_128"]["pad_share"], 0.9375, places=4
        )
        self.assertEqual(
            canvas_accounting["canvas_128"]["numbered_q1_trailing_pad_tokens"],
            119,
        )

    def test_real_tokenizer_flag_offline_fallback_and_guard_branches(
        self,
    ) -> None:
        def failing_loader(_model_id: str) -> object:
            raise OSError("Simulated offline environment")

        fallback_tok, fallback_mode = bench.load_optional_real_tokenizer(
            tokenizer_loader=failing_loader
        )
        self.assertIsNone(fallback_tok)
        self.assertEqual(fallback_mode, "offline_fallback_loader_error")

        async def async_bad_loader(_model_id: str) -> object:
            return _FakeSentencePieceTokenizer()

        with self.assertRaises(TypeError):
            bench.load_optional_real_tokenizer(tokenizer_loader=async_bad_loader)

        e2_live = bench.run_e2_delimiter_nonce_stability(
            use_real_tokenizer=True,
            tokenizer_loader=lambda _model_id: _FakeSentencePieceTokenizer(),
        )
        self.assertEqual(e2_live["tokenizer_mode"], "live_injected_tokenizer")
        self.assertIn("live_recomputed_bpe_n1000", e2_live)

        e4_fallback = bench.run_e4_serving_keepalive_and_canvas_bench(
            num_decisions=8,
            use_real_tokenizer=True,
            tokenizer_loader=failing_loader,
        )
        self.assertEqual(
            e4_fallback["tokenizer_mode"], "offline_fallback_loader_error"
        )
        self.assertEqual(
            e4_fallback["one_slot_canvas_accounting"]["canvas_128"][
                "trailing_pad_tokens"
            ],
            120,
        )


if __name__ == "__main__":
    unittest.main()
