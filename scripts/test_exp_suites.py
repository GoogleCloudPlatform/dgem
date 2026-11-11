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

"""Unit, mutation, security, and localhost HTTP tests for experiment suites."""

import ast
import http.server
import json
import math
import pathlib
import re
import sys
import tempfile
import threading
import time
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import exp_conformal_matched_escalation as d1_mod  # noqa: E402
import exp_ordinal_and_entity_dedup as d3_mod  # noqa: E402
import exp_som_shortlist_eval as d2_mod  # noqa: E402
import exp_stats_common as stats_lib  # noqa: E402

BANNED_SHORT_NAMES: frozenset[str] = frozenset({
    "res",
    "resp",
    "env",
    "tid",
    "r1",
    "r2",
    "probe_ct",
    "inc_1",
    "rb_1",
})


class SplitLockAndLeakageTests(unittest.TestCase):
  """Tests 30/20/50 cluster-disjoint partitioning and leakage rejection."""

  def _make_sample_items(self) -> list[stats_lib.BenchmarkSplitItem]:
    items: list[stats_lib.BenchmarkSplitItem] = []
    for cluster_idx in range(10):
      cluster_id = f"cluster_{cluster_idx:02d}"
      for row_idx in range(10):
        item_id = f"item-{cluster_idx:02d}-{row_idx:02d}"
        text = f"Sample payload for {cluster_id} row {row_idx}"
        items.append(
            stats_lib.BenchmarkSplitItem(
                item_id=item_id,
                cluster_id=cluster_id,
                content_text=text,
                content_sha256=stats_lib.compute_content_sha256(text),
            )
        )
    return items

  def test_cluster_disjoint_30_20_50_and_roundtrip_lock(self) -> None:
    items = self._make_sample_items()
    folds = stats_lib.deduplicate_and_partition_clusters(items, "test-seed")
    self.assertEqual(len(folds["cal"]), 30)
    self.assertEqual(len(folds["val"]), 20)
    self.assertEqual(len(folds["test"]), 50)

    cal_clusters = {x.cluster_id for x in folds["cal"]}
    val_clusters = {x.cluster_id for x in folds["val"]}
    test_clusters = {x.cluster_id for x in folds["test"]}
    self.assertFalse(cal_clusters & val_clusters)
    self.assertFalse(cal_clusters & test_clusters)
    self.assertFalse(val_clusters & test_clusters)

    with tempfile.TemporaryDirectory() as tmp_str:
      lock_path = pathlib.Path(tmp_str) / "splits.lock.json"
      track_entry = stats_lib.build_track_lock_entry("cluster_id", folds)
      stats_lib.write_split_lockfile(
          lock_path,
          [{"path": "synthetic", "license": "Apache-2.0"}],
          {"sample_track": track_entry},
      )
      loaded = json.loads(lock_path.read_text(encoding="utf-8"))
      verified = stats_lib.verify_split_lockfile(loaded, "sample_track", items)
      self.assertEqual(len(verified["test"]), 50)

  def test_mutation_rejects_cross_fold_sha256_leakage(self) -> None:
    items = self._make_sample_items()
    folds = stats_lib.deduplicate_and_partition_clusters(items, "test-seed")
    track_entry = stats_lib.build_track_lock_entry("cluster_id", folds)
    lock_doc = {
        "schema_version": "dgem.exp.splits.v1",
        "tracks": {"sample_track": track_entry},
    }
    # Mutate one test item's hash in lockfile to match a cal item's hash!
    cal_id = folds["cal"][0].item_id
    test_id = folds["test"][0].item_id
    item_hashes = dict(track_entry["item_hashes"])  # type: ignore[arg-type]
    item_hashes[test_id] = item_hashes[cal_id]
    mutated_track = dict(track_entry)
    mutated_track["item_hashes"] = item_hashes
    mutated_doc = {
        "schema_version": "dgem.exp.splits.v1",
        "tracks": {"sample_track": mutated_track},
    }
    with self.assertRaises(stats_lib.SplitLeakageError):
      stats_lib.verify_split_lockfile(mutated_doc, "sample_track", items)

    # Also verify live content drift is rejected against untouched lock_doc.
    drifted_items = list(items)
    drifted_items[0] = stats_lib.BenchmarkSplitItem(
        item_id=drifted_items[0].item_id,
        cluster_id=drifted_items[0].cluster_id,
        content_text="tampered prompt text",
        content_sha256=stats_lib.compute_content_sha256("tampered prompt text"),
    )
    with self.assertRaises(stats_lib.SplitLeakageError):
      stats_lib.verify_split_lockfile(lock_doc, "sample_track", drifted_items)

  def test_mutation_rejects_cross_fold_cluster_leakage(self) -> None:
    items = self._make_sample_items()
    folds = stats_lib.deduplicate_and_partition_clusters(items, "test-seed")
    # Force one cal item into test fold while keeping its cal cluster_id!
    leaked_folds = {
        "cal": list(folds["cal"][1:]),
        "val": list(folds["val"]),
        "test": list(folds["test"]) + [folds["cal"][0]],
    }
    with self.assertRaises(stats_lib.SplitLeakageError):
      stats_lib.build_track_lock_entry("cluster_id", leaked_folds)


class StatisticalMathTests(unittest.TestCase):
  """Tests OOF scaling, matched thresholds, McNemar, Tango, ECE, and Holm."""

  def test_oof_temperature_scaling_recovers_sharpened_logits(self) -> None:
    probs_list: list[dict[str, float]] = []
    labels: list[str] = []
    clusters: list[str] = []
    for idx in range(150):
      clusters.append(f"c_{idx % 10}")
      # Overconfident 92% confidence on something only ~68% accurate.
      probs_list.append({"a": 0.92, "b": 0.08})
      labels.append("a" if (idx % 100) < 68 else "b")

    global_t, oof_probs, fold_ts = stats_lib.fit_oof_temperature_nll(
        probs_list, labels, clusters, num_folds=5
    )
    self.assertGreater(global_t, 1.5)
    self.assertEqual(len(fold_ts), 5)
    raw_brier = stats_lib.multiclass_brier_score(probs_list, labels)
    oof_brier = stats_lib.multiclass_brier_score(oof_probs, labels)
    self.assertLess(oof_brier, raw_brier)

  def test_matched_escalation_solver_exact_quantiles(self) -> None:
    scores = [float(i) / 600.0 for i in range(600)]
    for target_rate in (0.10, 0.20, 0.30, 0.40):
      cutoff = stats_lib.solve_matched_escalation_threshold(scores, target_rate)
      empirical = sum(1 for s in scores if s > cutoff) / float(len(scores))
      self.assertAlmostEqual(empirical, target_rate, places=9)

    with self.assertRaises(stats_lib.StatisticalValidationError):
      stats_lib.solve_matched_escalation_threshold([], 0.20)
    with self.assertRaises(stats_lib.StatisticalValidationError):
      stats_lib.solve_matched_escalation_threshold(scores, 1.0)

  def test_paired_mcnemar_exact_closed_form(self) -> None:
    # 50 concordant true, 2 regressions (n_10=2), 10 improvements (n_01=10), 10 false.
    baseline = [True] * 50 + [True] * 2 + [False] * 10 + [False] * 10
    treatment = [True] * 50 + [False] * 2 + [True] * 10 + [False] * 10
    mcnemar = stats_lib.paired_mcnemar_exact(baseline, treatment)
    self.assertEqual(mcnemar.n_10, 2)
    self.assertEqual(mcnemar.n_01, 10)
    # Exact 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12 = 158 / 4096.
    self.assertAlmostEqual(mcnemar.p_value_exact, 158.0 / 4096.0, places=12)

    # Zero discordant pairs -> p = 1.0, chi2 = 0.0.
    zero_disc = stats_lib.paired_mcnemar_exact([True, False], [True, False])
    self.assertEqual(zero_disc.p_value_exact, 1.0)
    self.assertEqual(zero_disc.chi2_continuity, 0.0)

  def test_tango_non_inferiority_score_test(self) -> None:
    # Identical 1,200 outcomes with margin = 0.010 -> strongly non-inferior!
    base = [True] * 1080 + [False] * 120
    same = list(base)
    tango_same = stats_lib.tango_non_inferiority_test(base, same, margin=0.010)
    self.assertGreater(tango_same.z_score, 3.0)
    self.assertLess(tango_same.p_value_one_sided, 0.002)
    self.assertTrue(tango_same.non_inferior_at_005)

    # Inferior by 2.5% (30 regressions, 0 improvements on 1,200) -> fails NI!
    worse = [True] * 1050 + [False] * 150
    tango_worse = stats_lib.tango_non_inferiority_test(
        base, worse, margin=0.010
    )
    self.assertFalse(tango_worse.non_inferior_at_005)
    self.assertGreater(tango_worse.p_value_one_sided, 0.90)

  def test_holm_bonferroni_step_down_monotonicity(self) -> None:
    raw = [0.04, 0.005, 0.018, 0.25]
    adjusted = stats_lib.holm_bonferroni_adjust(raw)
    # Sorted: 0.005*4=0.02, 0.018*3=0.054, 0.04*2=0.08, 0.25*1=0.25.
    expected = [0.08, 0.02, 0.054, 0.25]
    for got_p, exp_p in zip(adjusted, expected):
      self.assertAlmostEqual(got_p, exp_p, places=12)

  def test_popoviciu_ordinal_variance_and_redos_free_pack_parser(self) -> None:
    self.assertAlmostEqual(
        stats_lib.normalized_ordinal_variance([0.5, 0.0, 0.0, 0.0, 0.5]),
        1.0,
        places=12,
    )
    self.assertAlmostEqual(
        stats_lib.normalized_ordinal_variance([0.0, 0.5, 0.5, 0.0, 0.0]),
        0.0625,
        places=12,
    )
    self.assertEqual(
        stats_lib.parse_pack_multiplier("Sparkling Water 2 x 500ml"), 2
    )
    self.assertEqual(stats_lib.parse_pack_multiplier("Sparkling Water 500ml"), 1)
    self.assertEqual(stats_lib.parse_pack_multiplier("Oat Milk 6-pack 1L"), 6)
    self.assertEqual(stats_lib.parse_pack_multiplier("Monitor 1920x1080"), 1)

    # 50,000-character pathological input completes in O(N) (< 100 ms).
    adversarial_title = "12   " * 10000 + "x   500"
    start = time.perf_counter()
    parsed_mult = stats_lib.parse_pack_multiplier(adversarial_title)
    elapsed_sec = time.perf_counter() - start
    self.assertGreaterEqual(parsed_mult, 1)
    self.assertLess(elapsed_sec, 0.20)


class SecurityAndLocalhostHTTPTests(unittest.TestCase):
  """Tests URL/path/model guards and localhost-only mock HTTP endpoints."""

  def test_url_path_and_model_validators(self) -> None:
    with self.assertRaises(stats_lib.StatisticalValidationError):
      stats_lib.validate_endpoint_url("http://example.com/v1/systemone")
    with self.assertRaises(stats_lib.StatisticalValidationError):
      stats_lib.validate_endpoint_url("https://user:pass@example.com/v1")
    self.assertEqual(
        stats_lib.validate_endpoint_url("https://example.com/v1/systemone"),
        "https://example.com/v1/systemone",
    )
    self.assertEqual(
        stats_lib.validate_endpoint_url("http://127.0.0.1:8080/v1/systemone"),
        "http://127.0.0.1:8080/v1/systemone",
    )

    with self.assertRaises(stats_lib.StatisticalValidationError):
      stats_lib.validate_ga_gemini_model("gemini-1.5-pro")
    self.assertEqual(
        stats_lib.validate_ga_gemini_model("gemini-3.5-flash-lite"),
        "gemini-3.5-flash-lite",
    )

    with tempfile.TemporaryDirectory() as sandbox_dir:
      root = pathlib.Path(sandbox_dir)
      with self.assertRaises(stats_lib.StatisticalValidationError):
        stats_lib.validate_local_file_path(
            root / ".." / "escape.json", allowed_root=root
        )

  def test_localhost_mock_server_and_cross_origin_header_stripping(
      self,
  ) -> None:
    captured_target_headers: dict[str, str] = {}

    class TargetHandler(http.server.BaseHTTPRequestHandler):

      def do_POST(self) -> None:  # noqa: N802
        for hdr_k, hdr_v in self.headers.items():
          captured_target_headers[hdr_k.lower()] = hdr_v
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = json.dumps({
            "answers": {
                "intent": {
                    "probabilities": {"opt_a": 0.75, "opt_b": 0.25},
                }
            }
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def log_message(self, format_str: str, *args: object) -> None:
        del format_str, args

    target_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), TargetHandler
    )
    target_port = target_server.server_address[1]
    target_thread = threading.Thread(
        target=target_server.serve_forever, daemon=True
    )
    target_thread.start()

    class RedirectHandler(http.server.BaseHTTPRequestHandler):

      def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self.send_response(307)
        self.send_header(
            "Location", f"http://127.0.0.1:{target_port}/v1/systemone"
        )
        self.end_headers()

      def log_message(self, format_str: str, *args: object) -> None:
        del format_str, args

    redirect_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), RedirectHandler
    )
    redirect_port = redirect_server.server_address[1]
    redirect_thread = threading.Thread(
        target=redirect_server.serve_forever, daemon=True
    )
    redirect_thread.start()

    try:
      reply = stats_lib.post_json_secure(
          f"http://127.0.0.1:{redirect_port}/v1/systemone",
          {"state": "test"},
          headers={"X-DGem-Key": "ephemeral-test-token"},
          timeout_sec=5.0,
      )
      self.assertIn("answers", reply)
      # Different port = cross-origin -> X-DGem-Key MUST be stripped!
      self.assertNotIn("x-dgem-key", captured_target_headers)
    finally:
      redirect_server.shutdown()
      redirect_server.server_close()
      target_server.shutdown()
      target_server.server_close()
      redirect_thread.join(timeout=2.0)
      target_thread.join(timeout=2.0)


class CodeHygieneAndSurfaceIsolationTests(unittest.TestCase):
  """Audits AST style rules, headers, and 4-surface locate isolation."""

  def test_four_surface_locate_isolation(self) -> None:
    status = d2_mod.verify_four_surface_locate_isolation()
    self.assertTrue(all(status.values()))
    self.assertEqual(len(d1_mod.CANONICAL_CATALOG_151), 151)
    self.assertEqual(len(d3_mod.build_d3a_corpus()), 2400)

  def test_ast_pystyle_and_apache_headers(self) -> None:
    target_scripts = (
        SCRIPT_DIR / "exp_stats_common.py",
        SCRIPT_DIR / "exp_conformal_matched_escalation.py",
        SCRIPT_DIR / "exp_som_shortlist_eval.py",
        SCRIPT_DIR / "exp_ordinal_and_entity_dedup.py",
        SCRIPT_DIR / "exp_sdk_conformal_parity.py",
        SCRIPT_DIR / "test_exp_suites.py",
    )
    multiline_from_import = re.compile(r"^from\s+\S+\s+import\s+\(", re.M)
    for py_path in target_scripts:
      source = py_path.read_text(encoding="utf-8")
      self.assertIn("Copyright 2026 Google LLC", source, msg=str(py_path))
      self.assertIsNone(
          multiline_from_import.search(source),
          msg=f"parenthesized multi-line import in {py_path}",
      )
      tree = ast.parse(source, filename=str(py_path))
      for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
          for child in ast.walk(node):
            if child is not node and isinstance(
                child, (ast.Import, ast.ImportFrom)
            ):
              self.fail(f"nested import inside {node.name} in {py_path}")
        if isinstance(node, ast.Lambda):
          if (
              node.end_lineno is not None
              and node.lineno != node.end_lineno
          ):
            self.fail(f"multi-line lambda at {py_path}:{node.lineno}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
          if node.id in BANNED_SHORT_NAMES:
            self.fail(
                f"banned short variable name {node.id!r} at "
                f"{py_path}:{node.lineno}"
            )


if __name__ == "__main__":
  unittest.main()
