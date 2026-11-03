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
"""Label sets and answer templates against the served DiffusionGemma tokenizer.

Runs structured_server.py's own parse_schema / resolve_template / system_text with the real tokenizer (no fakes):
- every label set resolves to single-token labels at one shared slot per question, in "lines" and "indexed";
- the default "az" templates are token-for-token identical to a reference server file (DGEM_REF_SERVER, e.g. the
  version before label sets existed), so opt-in label sets can't change default requests;
- prompt lines "AA: option" keep each label as one token (its id differs from the slot's, which carries a leading
  space, exactly as for A-Z).

Needs a directory with tokenizer.json (DGEM_TOKENIZER_DIR) and transformers. Skips otherwise.
  DGEM_TOKENIZER_DIR=/path/to/tok python3 -m unittest deploy/cloudrun/server/test_label_sets.py
"""

import importlib.util
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.environ.get("DGEM_TOKENIZER_DIR", "")


def _load(path, name):
    if "pybase64" not in sys.modules:
        try:
            import pybase64  # noqa: F401
        except ImportError:
            import base64

            sys.modules["pybase64"] = base64
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOK_DIR)


def choice(qid, n):
    return {"id": qid, "type": "choice", "instructions": f"Pick {qid}.",
            "options": [{"name": f"option_{i:02d}", "description": f"desc {i}"} for i in range(n)]}


FIXTURES = {
    "one_choice": {"questions": [choice("intent", 12)]},
    "mixed_3": {"questions": [{"id": "urgent", "type": "noul", "instructions": "Urgent?"}, choice("team", 26),
                              {"id": "tone", "type": "score", "instructions": "Tone", "levels": ["1", "2", "3", "4", "5"]}]},
    "ten": {"questions": [choice(f"q{i}", 4) for i in range(10)]},
    "twelve_indexed": {"questions": [choice(f"q{i}", 6) for i in range(12)]},
    "score_letters": {"questions": [{"id": "grade", "type": "score", "instructions": "g", "levels": [str(i) for i in range(12)]}]},
}


@unittest.skipUnless(TOK_DIR and os.path.exists(os.path.join(TOK_DIR, "tokenizer.json")), "DGEM_TOKENIZER_DIR not set")
class LabelSetTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            tok = _tokenizer()
        except ImportError as e:
            raise unittest.SkipTest(f"transformers not installed: {e}")
        cls.ss = _load(os.path.join(HERE, "structured_server.py"), "ss_current")
        cls.ss.init_tokenizer(tok)
        ref = os.environ.get("DGEM_REF_SERVER", "")
        cls.ref = None
        if ref and os.path.exists(ref):
            cls.ref = _load(ref, "ss_reference")
            cls.ref.init_tokenizer(tok)

    def template(self, mod, schema):
        s = mod.parse_schema(schema)
        return mod.resolve_template(s["questions"], mod.SCAFFOLD, "", s["format"]), s

    def test_default_az_identical_to_reference(self):
        if self.ref is None:
            self.skipTest("DGEM_REF_SERVER not set")
        for name, schema in FIXTURES.items():
            for layout in ("document_first", "schema_first"):
                sch = dict(schema, layout=layout)
                (t_new, s_new), _ = self.template(self.ss, sch)
                (t_ref, s_ref), _ = self.template(self.ref, sch)
                self.assertEqual(t_new, t_ref, f"{name}/{layout}: template ids differ")
                self.assertEqual(s_new, s_ref, f"{name}/{layout}: slots differ")
                self.assertEqual(str(self.ss.system_text(self.ss.parse_schema(sch))),
                                 str(self.ref.system_text(self.ref.parse_schema(sch))), f"{name}/{layout}: prompt differs")

    def check_wide(self, labels, schema):
        (template, slots), s = self.template(self.ss, dict(schema, labels=labels))
        for q, slot in zip(s["questions"], slots):
            self.assertEqual(len(slot["label_ids"]), len(q["labels"]))
            self.assertEqual(len(set(slot["label_ids"])), len(q["labels"]), f"{q['id']}: label ids collide")
            for lab, tid in zip(q["labels"], slot["label_ids"]):
                self.assertEqual(self.ss.TOK.decode([tid]).strip(), lab, f"{q['id']}: slot token for {lab!r}")
        self.assertLessEqual(len(self.ss.label_id_union(slots)), self.ss.LOGPROB_TOKEN_IDS_MAX)
        return s, slots

    def test_az_aa_52_options_lines_and_indexed(self):
        self.check_wide("az_aa", {"questions": [choice("intent", 52)]})
        self.check_wide("az_aa", {"questions": [choice("intent", 52), {"id": "urgent", "type": "noul"}]})
        for pos in (0, 6, 11):  # first, middle, last of a 12-question (indexed) schema
            qs = [choice(f"q{i}", 4) for i in range(12)]
            qs[pos] = choice(f"q{pos}", 52)
            s, _ = self.check_wide("az_aa", {"questions": qs})
            self.assertEqual(s["format"], "indexed")

    def test_az_aa_three_wide_questions_fit_128_ids(self):
        _, slots = self.check_wide("az_aa", {"questions": [choice(f"w{i}", 52) for i in range(3)]})
        self.assertLessEqual(len(self.ss.label_id_union(slots)), 128)

    def test_az_aa_refuses_53(self):
        with self.assertRaises(self.ss.SchemaError):
            self.ss.parse_schema({"labels": "az_aa", "questions": [choice("intent", 53)]})

    def test_az_lower_lines_only(self):
        self.check_wide("az_lower", {"questions": [choice("intent", 52)]})
        self.check_wide("az_lower", {"questions": [choice(f"q{i}", 40 if i == 9 else 4) for i in range(10)]})
        with self.assertRaises(self.ss.SchemaError):
            self.ss.parse_schema({"labels": "az_lower", "questions": [choice(f"q{i}", 4) for i in range(11)]})

    def test_prompt_lines_keep_labels_as_single_tokens(self):
        """In the rendered question list each 'AA: option' line keeps the label as one token."""
        for labels in ("az_aa", "az_lower"):
            s, slots = self.check_wide(labels, {"questions": [choice("intent", 52)]})
            ids = self.ss.TOK.encode(str(self.ss.system_text(s)), add_special_tokens=False)
            for lab, tid in zip(s["questions"][0]["labels"], slots[0]["label_ids"]):
                line = self.ss.TOK.encode(f"  {lab}: option_", add_special_tokens=False)
                lab_tokens = [t for t in line if self.ss.TOK.decode([t]).strip() == lab]
                self.assertTrue(lab_tokens, f"{labels}: {lab!r} is not one token in its prompt line")
                self.assertTrue(any(t in ids for t in lab_tokens), f"{labels}: {lab!r} token missing from prompt")


if __name__ == "__main__":
    unittest.main()
