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

"""Offline tests for the regression matrix (no network, no GPU).

  cd scripts && python3 -m unittest matrix.test_matrix -v
"""
import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matrix import cases as C  # noqa: E402
from matrix import datasets as D  # noqa: E402
from matrix import metrics as M  # noqa: E402
from matrix import net  # noqa: E402
from matrix import report as R  # noqa: E402
from matrix import runners  # noqa: E402
from matrix.targets import Target  # noqa: E402

MATRIX = os.path.join(D.REPO, "benchmarks", "matrix", "matrix_example.json")
HAVE_JEVBENCH = os.path.exists(C.JEVBENCH)  # fetched by `dgem bench-jev --sync`, not in the repository


def _rows(n, acc, seed, flip=0.0):
    """Synthetic systemone rows: ~acc correct, `flip` share of answers changed vs seed 0."""
    rng, base = random.Random(seed), random.Random(0)
    out = []
    for i in range(n):
        right = base.random() < acc
        if rng.random() < flip:
            right = not right
        p = 0.9 if right else 0.6
        exp, act = "a", ("a" if right else "b")
        out.append({"id": f"x-{i}", "status": 200, "expected": exp, "actual": act, "accurate": right,
                    "confidence": p, "probabilities": {act: p, ("b" if act == "a" else "a"): 1 - p},
                    "wall_ms": 100.0, "suite": "jev_systemone"})
    return out


def _w(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _run(tmp, cand_acc, cand_flip):
    man = {"run_id": "test", "git_commit": "0" * 40, "receipts": [],
           "matrix": {"matrix_version": "example-1", "tier": "T1", "baseline": "base",
                      "targets": [{"name": "base", "kind": "vertex"}, {"name": "new", "kind": "vertex"}]}}
    for t, acc, flip in (("base", 0.85, 0.05), ("new", cand_acc, cand_flip)):
        for r in (1, 2, 3):
            p = f"jev_systemone__{t}__r{r}.json"
            rows = _rows(231, acc, seed=r + (10 if t == "new" else 0), flip=flip)
            _w(os.path.join(tmp, p), {"kind": "systemone", "suite": "jev_systemone", "cases": rows})
            man["receipts"].append({"path": p, "suite": "jev_systemone", "config": t, "run": r, "perm": None})
        p = f"health__{t}__r1.json"
        _w(os.path.join(tmp, p), {"kind": "health", "ok": True, "status": 200, "body": {}})
        man["receipts"].append({"path": p, "suite": "health", "config": t, "run": 1, "perm": None})
    _w(os.path.join(tmp, "manifest.json"), man)
    with open(MATRIX) as f:
        return R.build(tmp, json.load(f))


class TestServerKey(unittest.TestCase):
    """X-DGem-Key (the serving image's service key) is sent on every target request when DGEM_SERVER_KEY is set: the
    first entry of a rotation list, and no header at all when it is unset."""

    def test_header_on_target_requests(self):
        sent = []
        orig = net.request
        net.request = lambda url, body=None, **kw: sent.append((url, kw.get("headers"))) or (200, {}, 1.0, {})
        old = os.environ.get("DGEM_SERVER_KEY")
        try:
            os.environ["DGEM_SERVER_KEY"] = "svc-key,old-key"
            Target("prod", "https://serving.example/v1").systemone({"questions": {}})
            Target("prod", "https://serving.example").chat({"questions": []}, {})
            Target("prod", "https://serving.example").request("/v1/systemone", {"questions": {}})
            os.environ["DGEM_SERVER_KEY"] = ""
            Target("prod", "https://serving.example").systemone({"questions": {}})
        finally:
            net.request = orig
            if old is None:
                os.environ.pop("DGEM_SERVER_KEY", None)
            else:
                os.environ["DGEM_SERVER_KEY"] = old
        self.assertEqual(sent[0][1], {"X-DGem-Key": "svc-key"})
        self.assertEqual(sent[1][1], {"X-DGem-Key": "svc-key"})
        self.assertEqual(sent[2][1], {"X-DGem-Key": "svc-key"})
        self.assertEqual(sent[3][1], {})

    def test_keepalive_runner_sends_key(self):
        class Conn:
            def __init__(self):
                self.headers = None

            def request(self, method, path, body=None, headers=None):
                self.headers = headers

            def getresponse(self):
                return mock.Mock(status=401, read=lambda: b'{"error": {"message": "missing or wrong X-DGem-Key"}}')

        conn = Conn()
        ka = runners._KeepAlive("https://serving.example")
        ka._conn = lambda: conn
        with mock.patch.dict(os.environ, {"DGEM_SERVER_KEY": " new-key , old-key "}), \
                mock.patch.object(net, "token_for", return_value=None):
            ka.call({"questions": {}})
        self.assertEqual(conn.headers.get("X-DGem-Key"), "new-key")

    def test_is_vertex_checks_parsed_hostname(self):
        import serving_speed as ss
        for fn in (net.is_vertex, ss.is_vertex):
            self.assertTrue(fn("https://1234.us-central1-999.prediction.vertexai.goog/v1/projects/p/locations/r/endpoints/1234/invoke"))
            self.assertTrue(fn("https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/1234:rawPredict"))
            self.assertTrue(fn("https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/1234:rawPredict"))
            self.assertTrue(fn("https://us-central1.aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/1234:rawPredict"))
            self.assertFalse(fn("https://dgemma-service.a.run.app/v1"))
            self.assertFalse(fn("https://evil.example/https://aiplatform.googleapis.com"))
            self.assertFalse(fn("https://evil.example?target=prediction.vertexai.goog"))
            self.assertFalse(fn("https://aiplatform.googleapis.com.evil.example/v1"))
            self.assertFalse(fn("https://evilaiplatform.googleapis.com/v1"))
            self.assertFalse(fn("https://nested.us-central1-aiplatform.googleapis.com/v1"))


class TestMetrics(unittest.TestCase):
    def test_ece_perfect_and_off(self):
        self.assertAlmostEqual(M.ece([1.0, 1.0], [1, 1]), 0.0)
        self.assertAlmostEqual(M.ece([0.9] * 10, [1] * 5 + [0] * 5), 0.4)

    def test_auroc(self):
        self.assertEqual(M.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(M.auroc([0.5, 0.5], [1, 0]), 0.5)
        self.assertIsNone(M.auroc([0.5], [1]))

    def test_mcnemar(self):
        a = [{"id": i, "accurate": True} for i in range(20)]
        b = [{"id": i, "accurate": i >= 15} for i in range(20)]
        r = M.mcnemar([(a, b)])
        self.assertEqual((r["a_only"], r["b_only"]), (15, 0))
        self.assertLess(r["p"], 0.001)
        self.assertEqual(M.mcnemar([(a, a)])["p"], 1.0)

    def test_heldout_temperature_softens_overconfidence(self):
        rows = [{"id": f"r{i}", "expected": "a", "probabilities": ({"a": 0.99, "b": 0.01} if i % 3 else {"a": 0.01, "b": 0.99})}
                for i in range(90)]
        h = M.heldout_temperature(rows)
        self.assertGreater(min(h["T"]), 1.5)
        self.assertLess(h["ece_heldout"], h["ece_raw"])

    def test_normalize_native_receipts(self):
        jev = {"cases": [{"id": "1", "question_type": "noul", "expected": "yes", "actual": "true", "accurate": True,
                          "confidence": 0.9, "top_probabilities": {"true": 0.9, "false": 0.1}}]}
        r = M.rows(jev)[0]
        self.assertEqual((r["expected"], r["actual"]), ("yes", "yes"))
        self.assertAlmostEqual(r["brier"], 0.02)
        intents = {"results": [{"id": "b", "expected_intent": "x", "actual_intent": "x", "intent_accurate": True,
                                "confidence": 0.5, "top_probabilities": {"x": 0.5, "y": 0.5}}]}
        self.assertTrue(M.rows(intents)[0]["accurate"])


class TestMetricsV2(unittest.TestCase):
    def test_macro_f1(self):
        rs = [{"expected": "a", "actual": "a"}, {"expected": "a", "actual": "b"}, {"expected": "b", "actual": "b"},
              {"expected": "c", "actual": "b"}]
        # a: tp1 fn1 -> 2/3; b: tp1 fp2 -> 0.5; c: fn1 -> 0
        self.assertAlmostEqual(M.macro_f1(rs), (2 / 3 + 0.5 + 0) / 3)
        self.assertEqual(M.macro_f1([{"expected": "x", "actual": "x"}]), 1.0)
        q = [{"qid": "q1", "expected": "0", "actual": "0"}, {"qid": "q2", "expected": "0", "actual": "1"}]
        self.assertAlmostEqual(M.macro_f1(q), 0.5)  # labels namespaced per question

    def test_attempted_counts_refusals(self):
        sys1 = {"kind": "systemone", "cases": [{"status": 200}, {"status": 422}, {"status": "n/a"}, {"status": 200}]}
        self.assertEqual(M.attempted(sys1), (2, 4))
        native = {"cases": [{"id": 1}, {"id": 2, "error": "boom"}]}
        self.assertEqual(M.attempted(native), (1, 2))

    def test_reliability_bins(self):
        rs = [{"confidence": 0.95, "accurate": True}, {"confidence": 0.92, "accurate": False}, {"confidence": 0.15, "accurate": False}]
        b = M.reliability(rs)
        self.assertEqual(len(b), 10)
        self.assertEqual(b[9]["count"], 2)
        self.assertAlmostEqual(b[9]["acc"], 0.5)
        self.assertAlmostEqual(sum(x["share"] for x in b), 1.0)


def _check_schema(obj, schema, path="$", root=None):
    """Minimal JSON-schema check (type, required, enum, const, items, additionalProperties, $ref) without jsonschema."""
    root = root or schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return _check_schema(obj, node, path, root)
    errs = []
    types = schema.get("type")
    if types:
        types = [types] if isinstance(types, str) else types
        py = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "null": type(None),
              "boolean": bool}
        if not any(isinstance(obj, py[t]) and not (t in ("integer", "number") and isinstance(obj, bool)) for t in types):
            return [f"{path}: {type(obj).__name__} not in {types}"]
    if "const" in schema and obj != schema["const"]:
        errs.append(f"{path}: {obj!r} != {schema['const']!r}")
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing {k}")
        for k, v in obj.items():
            if k in schema.get("properties", {}):
                errs += _check_schema(v, schema["properties"][k], f"{path}.{k}", root)
            elif isinstance(schema.get("additionalProperties"), dict):
                errs += _check_schema(v, schema["additionalProperties"], f"{path}.{k}", root)
    if isinstance(obj, list) and "items" in schema:
        for i, v in enumerate(obj):
            errs += _check_schema(v, schema["items"], f"{path}[{i}]", root)
    return errs


class TestSummaryContract(unittest.TestCase):
    SCHEMA = os.path.join(D.REPO, "benchmarks", "matrix", "summary.schema.json")

    def _schema(self):
        with open(self.SCHEMA) as f:
            return json.load(f)

    def test_built_summary_matches_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            _md, s = _run(tmp, 0.85, 0.05)
        self.assertEqual(s["schema"], R.SUMMARY_SCHEMA)
        self.assertEqual(_check_schema(s, self._schema()), [])
        blk = s["suites"]["jev_systemone"]["new"]
        self.assertEqual(len(blk["runs"]), 3)
        self.assertEqual(blk["runs"][0]["attempted"], 231)
        self.assertIsNotNone(blk["macro_f1"])

    def test_summary_targets_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            _md, s = _run(tmp, 0.85, 0.05)
        self.assertTrue(set(s) <= set(self._schema()["properties"]), set(s) - set(self._schema()["properties"]))
        for t in s["targets"]:
            self.assertEqual(set(t), {"name", "kind", "health"})


class TestCaseExact(unittest.TestCase):
    def test_case_exact(self):
        rs = [{"case": "a", "qid": "q1", "accurate": True}, {"case": "a", "qid": "q2", "accurate": True},
              {"case": "b", "qid": "q1", "accurate": True}, {"case": "b", "qid": "q2", "accurate": False}]
        self.assertEqual(M.case_exact(rs), 0.5)
        self.assertIsNone(M.case_exact([{"id": "x", "accurate": True}]))
        self.assertIsNone(M.case_exact([{"case": "a", "qid": "q", "accurate": True}]))


class TestDecisionIndexTrack(unittest.TestCase):
    def _report_with_probes(self, tmp, cases_):
        man = {"run_id": "t", "git_commit": "0" * 40, "receipts": [],
               "matrix": {"matrix_version": "example-1", "tier": "T1", "baseline": None, "targets": [{"name": "p", "kind": "vertex"}]}}
        _w(os.path.join(tmp, "di_probes__p__r1.json"), {"kind": "adapter_probes", "cases": cases_})
        man["receipts"].append({"path": "di_probes__p__r1.json", "suite": "di_probes", "config": "p", "run": 1, "perm": None})
        _w(os.path.join(tmp, "di_kit_compat__p__r1.json"), {"kind": "kit_compat", "skipped": True, "reason": "x"})
        man["receipts"].append({"path": "di_kit_compat__p__r1.json", "suite": "di_kit_compat", "config": "p", "run": 1, "perm": None})
        _w(os.path.join(tmp, "manifest.json"), man)
        with open(MATRIX) as f:
            return R.build(tmp, json.load(f))

    def test_probe_failure_fails_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _md, s = self._report_with_probes(tmp, [
                {"name": "wide_41", "status": 500, "ok": False, "notes": ["HTTP 500"]},
                {"name": "wide_27", "status": 200, "ok": True, "notes": [], "keys": 27, "prob_sum": 1.0, "top_p": 0.9}])
        g = {x["gate"]: x for x in s["gates"]}
        self.assertEqual(g["di_probes"]["verdict"], "FAIL")
        self.assertIn("wide_41", g["di_probes"]["detail"])
        self.assertEqual(g["di_kit_compat"]["verdict"], "INFO")
        self.assertEqual(s["overall"]["p"], "FAIL")

    def test_info_only_probe_does_not_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _md, s = self._report_with_probes(tmp, [
                {"name": "wide_27", "status": 200, "ok": True, "notes": []},
                {"name": "wide_catchall_151", "status": 200, "ok": False, "notes": ["intent: out_of_scope != refund_request"],
                 "info_only": True}])
        g = {x["gate"]: x for x in s["gates"]}
        self.assertEqual(g["di_probes"]["verdict"], "PASS")
        self.assertEqual(g["di_probe_wide_catchall_151"]["verdict"], "INFO")

    def test_new_probes_present(self):
        names = {p["name"] for p in C.di_probes()}
        self.assertTrue({"criteria_objects", "noul_criteria", "wide_catchall_151", "long_9k"} <= names)

    def test_probes_cover_awkward_bracket_sizes(self):
        names = {p["name"] for p in C.di_probes()}
        for k in (41, 61, 101):  # K % 20 == 1 once produced a one-option bracket
            self.assertIn(f"wide_{k}", names)
        refusal = next(p for p in C.di_probes() if p["name"] == "context_refusal")
        self.assertTrue(refusal["expect"]["marker_if_422"])


class TestCases(unittest.TestCase):
    @unittest.skipUnless(HAVE_JEVBENCH, "JevBench not synced (./bin/dgem bench-jev --sync)")
    def test_jevbench_shapes(self):
        cs = C.jevbench()
        self.assertEqual(len(cs), 231)
        types = {c["qs"][0]["type"] for c in cs}
        self.assertEqual(types, {"noul", "choice", "score"})
        score = next(c for c in cs if c["qs"][0]["type"] == "score")
        self.assertIsInstance(C.body(score)["questions"]["decision"]["criteria"], list)

    def test_distribution_and_score(self):
        q = {"type": "noul", "labels": ["yes", "no"], "expected": "no", "values": None}
        d = C.distribution(q, {"noul": 0.2})
        self.assertEqual(C.score(q, d)["actual"], "no")
        q = {"type": "score", "labels": ["0", "1", "2"], "expected": "1", "values": [0, 1, 2]}
        d = C.distribution(q, {"probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}})
        s = C.score(q, d)
        self.assertTrue(s["accurate"])
        self.assertAlmostEqual(s["abs_err_ev"], 0.0)

    def test_jevbench_missing_message(self):
        if HAVE_JEVBENCH:
            self.skipTest("JevBench is synced")
        with self.assertRaises(RuntimeError) as cm:
            C.jevbench()
        self.assertIn("bench-jev --sync", str(cm.exception))

    def test_latency_payloads_without_jevbench(self):
        P = runners.latency_payloads()
        self.assertTrue({"q1", "q5", "q10", "q5_s4"} <= set(P))
        self.assertEqual("long_q1" in P, HAVE_JEVBENCH)

    def test_permute_keeps_labels(self):
        c = {"id": "c", "suite": "s", "subset": "", "tier": None, "state": {},
             "qs": [{"qid": "decision", "type": "choice", "instructions": "", "criteria": {"a": "A", "b": "B", "c": "C"},
                     "labels": ["a", "b", "c"], "expected": "a", "values": None}]}
        p = C.permute(c, "reverse")
        self.assertEqual(set(p["qs"][0]["criteria"]), set(c["qs"][0]["criteria"]))
        self.assertEqual(list(p["qs"][0]["criteria"]), list(c["qs"][0]["criteria"])[::-1])


class TestLockAndRedaction(unittest.TestCase):
    def test_lock_pins_every_file(self):
        lock = D.lock()
        for repo, ent in lock.items():
            self.assertEqual(len(ent["revision"]), 40, repo)
            for f, meta in ent["files"].items():
                self.assertEqual(len(meta["sha256"]), 64, f)
        self.assertEqual(len(D.massive_langs()), 51)
        self.assertEqual(len(D.xnli_langs()), 15)
        for lg in C.SPOT_LANGS:
            self.assertIn(f"validation/{lg}.json.gz", lock[D.MASSIVE]["files"])

    def test_dump_handles_nested_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "c.json")
            runners._dump(p, {"cases": {"a": {"probabilities": {"q": {"x": 0.5}}}}})
            runners._dump(p, {"kind": "systemone", "cases": [{"probabilities": {"x": 0.999999, "y": 1e-7}}]})
            with open(p) as f:
                self.assertEqual(json.load(f)["cases"][0]["probabilities"], {"x": 1.0})

    def test_redact(self):
        t = Target("prod", "https://1234.us-central1-999.prediction.vertexai.goog/v1/projects/p/locations/r/endpoints/1234/invoke")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"target_url": t.base + "/v1/chat/completions", "k": "secret-token"}))
        runners.redact_file(f.name, t, "secret-token")
        txt = open(f.name).read()
        self.assertNotIn("prediction.vertexai.goog", txt)
        self.assertNotIn("secret-token", txt)
        self.assertEqual(t.kind, "vertex")
        self.assertEqual(Target("x", "http://h:8080/v1").cli_url, "http://h:8080/v1")


class TestVisionRows(unittest.TestCase):
    def test_bench_vision_receipt_rows(self):
        rec = {"receipt_kind": "bench-vision", "items": [
            {"id": "a#grid_cell", "expected": "top_left", "actual": "top_left", "accurate": True, "confidence": 0.9,
             "top_probabilities": {"top_left": 0.9, "middle_center": 0.1}, "wall_time_ms": 300},
            {"id": "a#present", "expected": "yes", "actual": "no", "accurate": False, "confidence": 0.6, "wall_time_ms": 300}]}
        rs = M.rows(rec)
        self.assertEqual([r["id"] for r in rs], ["a#grid_cell", "a#present"])
        self.assertEqual(M.summary(rs)["accuracy"], 0.5)


class TestReport(unittest.TestCase):
    def test_same_model_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            md, s = _run(tmp, 0.85, 0.05)
        self.assertEqual(s["overall"]["new"], "PASS", md)

    def test_regression_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            md, s = _run(tmp, 0.65, 0.05)
        self.assertEqual(s["overall"]["new"], "FAIL", md)

    def test_dev_suites_registered(self):
        mx = json.load(open(MATRIX))
        for sid in ("di_catchall", "rag_dev"):
            self.assertIn(sid, mx["tiers"]["T1"])
            self.assertEqual(mx["suites"][sid]["via"], "adapter")
            self.assertIn(mx["suites"][sid]["cases"], C.SUITES)
        lock = D.lock()
        self.assertIn(D.CLINC, lock)
        self.assertIn(D.RAGTRUTH, lock)

    def test_refusal_reasons(self):
        self.assertEqual(M.refusal_reason(400, "This model's maximum context length is 4096 tokens"), "context")
        self.assertEqual(M.refusal_reason(422, "unsupported"), "capacity")
        self.assertEqual(M.refusal_reason(400, "schema: at most 26 alternatives"), "capacity")
        self.assertEqual(M.refusal_reason("n/a", None), "na")
        self.assertEqual(M.refusal_reason(None, "timed out"), "error")
        r = {"kind": "systemone", "cases": [{"status": 200}, {"status": 502, "error": "maximum context length"},
                                            {"status": "n/a"}]}
        self.assertEqual(M.refusals(r), {"context": 1, "na": 1})
        self.assertEqual(M.refusals({"cases": [{"error": "HTTP 502 maximum context length"}, {}]}), {"context": 1})

    def test_target_options(self):
        from matrix.targets import Target
        t = Target("doc", "http://h:8080/v1#layout=document_first")
        self.assertEqual((t.base, t.options), ("http://h:8080", {"layout": "document_first"}))
        self.assertEqual(Target("p", "http://h:8080").options, {})
        self.assertEqual(t.redacted()["options"], {"layout": "document_first"})
        with self.assertRaises(SystemExit):
            Target("bad", "http://h#layout=sideways")

    def test_matrix_definition(self):
        mx = json.load(open(MATRIX))
        for tier, suites in mx["tiers"].items():
            for s in suites:
                self.assertIn(s, mx["suites"], f"{tier}: {s}")
        self.assertTrue(set(mx["tiers"]["T0"]) <= set(mx["tiers"]["T1"]) <= set(mx["tiers"]["T2"]))
        frozen = {s for s, v in mx["suites"].items() if v.get("frozen")}
        self.assertFalse(frozen & set(mx["tiers"]["T1"]), "frozen sets belong to T2 only")
        for s, v in mx["suites"].items():
            if v["kind"] == "systemone":
                self.assertIn(v["cases"], C.SUITES)
            for f in v.get("requires", []):
                self.assertFalse(os.path.isabs(f), s)
        self.assertEqual(set(mx["tiers"]), {"T0", "T1", "T2"})
        self.assertNotIn("reference", mx)
        self.assertEqual(mx["version"], "example-1")
        for k in ("agreement_margin_pp", "mcnemar_p", "multilingual_min_accuracy", "latency_p50_ratio_review",
                  "latency_p50_ratio_fail", "throughput_ratio_min", "order_net_flip_delta_review"):
            self.assertIn(k, mx["thresholds"])
        self.assertTrue(R.accuracy_suites(mx))

    def test_single_target_without_reference_is_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = {"run_id": "solo", "git_commit": "0" * 40, "receipts": [],
                   "matrix": {"matrix_version": "example-1", "tier": "T1", "baseline": None,
                              "targets": [{"name": "gpu", "kind": "self-hosted"}]}}
            _w(os.path.join(tmp, "jev_systemone__gpu__r1.json"),
               {"kind": "systemone", "suite": "jev_systemone", "cases": _rows(50, 0.8, 1)})
            man["receipts"].append({"path": "jev_systemone__gpu__r1.json", "suite": "jev_systemone", "config": "gpu",
                                    "run": 1, "perm": None})
            _w(os.path.join(tmp, "manifest.json"), man)
            with open(MATRIX) as f:
                md, s = R.build(tmp, json.load(f))
        g = {(x["gate"], x["target"]): x for x in s["gates"]}
        self.assertEqual(g[("jev_systemone", "gpu")]["verdict"], "INFO")
        self.assertEqual(s["overall"]["gpu"], "PASS")
        self.assertIn("no baseline and no reference ranges", md)


class TestPyarrowPreflight(unittest.TestCase):
    """A run that includes a parquet suite checks for pyarrow before it sends anything, instead of stopping at the
    first parquet suite after the suites before it have already sent requests. The tests patch `have_pyarrow`, so they
    give the same result with and without pyarrow installed."""

    @classmethod
    def setUpClass(cls):
        import bench_matrix  # scripts/ is on sys.path (above)
        cls.bm = bench_matrix
        cls.mx = bench_matrix.load_matrix(MATRIX)

    def _args(self, tier="T1", **kw):
        a = dict(matrix=None, tier=tier, confirm=False, plan=False, only=None, skip=None, baseline=None,
                 target=["gpu=http://127.0.0.1:9"], label=None, out_dir=None, resume=False, dgem=None, workers=4,
                 latency_n=100, fail_fast=False, log_json=False, note=None, upload=None)
        a.update(kw)
        return argparse.Namespace(**a)

    def _hint(self, tier, installed, skip=()):
        suites = [s for s in self.mx["tiers"][tier] if s not in skip]
        with mock.patch.object(D, "have_pyarrow", return_value=installed):
            return self.bm.parquet_hint(self.mx, suites)

    @staticmethod
    def _named(hint):
        return sorted(hint.split("parquet datasets: ", 1)[1].split("\n", 1)[0].split(", "))

    def _plan(self, installed, **kw):
        out = io.StringIO()
        with mock.patch.object(D, "have_pyarrow", return_value=installed), contextlib.redirect_stdout(out):
            rc = self.bm.cmd_run(self._args(plan=True, **kw))
        return rc, out.getvalue()

    def test_parquet_suites_are_registered(self):
        self.assertLessEqual(C.PARQUET_SUITES, set(C.SUITES))
        for name, build in C.SUITES.items():
            if "parquet" in inspect.getsource(build):
                self.assertIn(name, C.PARQUET_SUITES, f"{name} reads parquet: add it to cases.PARQUET_SUITES")

    def test_have_pyarrow_reports_a_failed_import(self):
        with mock.patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):  # None: import raises
            self.assertFalse(D.have_pyarrow())

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is not installed")
    def test_have_pyarrow_when_installed(self):
        self.assertTrue(D.have_pyarrow())

    def test_hint_names_the_suites_that_need_pyarrow(self):
        hint = self._hint("T1", installed=False)
        self.assertTrue(hint.startswith("pyarrow is not installed"), hint)
        self.assertEqual(self._named(hint), ["di_catchall", "rag_dev"])
        self.assertIn("pip install -r scripts/requirements-matrix.txt", hint)
        self.assertIn("--skip", hint)
        self.assertEqual(self._named(self._hint("T2", installed=False)),
                         ["di_catchall", "order", "rag_dev", "typed", "xnli"])

    def test_no_hint_when_nothing_needs_it(self):
        self.assertEqual(self._hint("T0", installed=False), "")  # T0 reads no parquet
        self.assertEqual(self._hint("T1", installed=False, skip=("di_catchall", "rag_dev")), "")
        for tier in ("T0", "T1", "T2"):
            self.assertEqual(self._hint(tier, installed=True), "")

    def test_run_stops_before_it_creates_anything_or_sends_a_request(self):
        # Skip the JevBench suites so that the only reason to stop is pyarrow (their files are not in the repository).
        # Creating the run directory is the first thing a run does once it has checked its prerequisites.
        with mock.patch.object(D, "have_pyarrow", return_value=False), \
                mock.patch.object(os, "makedirs", side_effect=AssertionError("the run started")), \
                mock.patch.object(net, "request", side_effect=AssertionError("a request was sent")):
            with self.assertRaises(SystemExit) as stop:
                self.bm.cmd_run(self._args(skip=["jev_native", "jev_systemone"]))
        self.assertIn("pyarrow is not installed", str(stop.exception))
        self.assertNotIn("missing files", str(stop.exception))

    def test_plan_prints_the_same_hint(self):
        rc, out = self._plan(installed=False)
        self.assertEqual(rc, 1)
        self.assertIn("pyarrow is not installed, and these suites read parquet datasets: di_catchall, rag_dev", out)

    def test_skip_and_only_leave_the_parquet_suites_out(self):
        # No assertion on the return code: without JevBench synced, the files hint of the other suites makes it 1.
        for kw in ({"skip": ["di_catchall", "rag_dev"]}, {"only": ["massive_spot", "calibration"]}):
            self.assertNotIn("pyarrow", self._plan(installed=False, **kw)[1])


if __name__ == "__main__":
    unittest.main()
