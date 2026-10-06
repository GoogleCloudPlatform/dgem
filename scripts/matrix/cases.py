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

"""Cases for the /v1/systemone suites, and scoring of their answers.

case = {"id", "suite", "subset", "tier", "state", "qs": [q, ...]}
q    = {"qid", "type" (noul|choice|score), "instructions", "criteria", "labels", "expected",
        "values" (score levels), "gold_soft" (optional {label: p}), "expected_value" (optional)}

The same request body works on any server that speaks the /v1/systemone decision protocol. Protocols for the
public sets follow the published ones they come from (MASSIVE: 20 options = gold + 19 distractors, Random(13)
per language; option-order suites: one shuffle per case from Random(99)).
"""
import json
import math
import os
import random

from . import datasets as ds

REPO = ds.REPO
EPS = 1e-6


# ---------------------------------------------------------------- development suites
JEVBENCH = os.path.join(REPO, "benchmarks", "jevbench", "jevbench_public.jsonl")
JEVBENCH_SYNC = "./bin/dgem bench-jev --sync"


def jevbench():
    """JevBench public items (MIT). Not shipped in the repository: fetch them with `dgem bench-jev --sync`."""
    if not os.path.exists(JEVBENCH):
        raise RuntimeError(f"{os.path.relpath(JEVBENCH, REPO)} is missing: run `{JEVBENCH_SYNC}` first")
    out = []
    with open(JEVBENCH) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        q = r["question"]
        t = q["type"]
        qq = {"qid": "decision", "type": t, "instructions": q.get("instructions", ""), "criteria": q.get("criteria"),
              "values": None}
        exp = r["expected"]
        if t == "noul":
            qq["labels"] = ["yes", "no"]
            qq["expected"] = "yes" if str(exp).lower() in ("yes", "true", "1") else "no"
        elif t == "choice":
            qq["labels"] = list(q["criteria"].keys())
            qq["expected"] = str(exp)
        else:
            qq["labels"] = [str(i) for i in range(len(q["criteria"]))]
            qq["values"] = list(range(len(q["criteria"])))
            qq["expected"] = str(int(exp))
        out.append({"id": r["id"], "suite": "jev_systemone", "subset": r.get("family", ""), "tier": r.get("tier"),
                    "state": r["state"], "qs": [qq]})
    return out


# ---------------------------------------------------------------- MASSIVE
def _massive_case(lang, i, row, labels, rng, suite):
    gold = row["label_text"]
    pool = [x for x in labels if x != gold]
    keys = [gold] + rng.sample(pool, 19)
    rng.shuffle(keys)
    q = {"qid": "intent", "type": "choice", "instructions": "What is the user asking for in `utterance`?",
         "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}, "labels": keys, "expected": gold,
         "values": None}
    return {"id": f"{suite}-{lang}-{i:03d}", "suite": suite, "subset": lang, "tier": None,
            "state": {"utterance": row["text"]}, "qs": [q]}


def massive(split="test", langs=None, n=100, suite="massive"):
    out = []
    for lang in langs or ds.massive_langs():
        test = ds.jsonl_gz(ds.MASSIVE, f"test/{lang}.json.gz")
        labels = sorted({r["label_text"] for r in test})  # the option pool is the language's test label set
        rows = test if split == "test" else ds.jsonl_gz(ds.MASSIVE, f"{split}/{lang}.json.gz")
        rng = random.Random(13)
        out += [_massive_case(lang, i, r, labels, rng, suite) for i, r in enumerate(rows[:n])]
    return out


SPOT_LANGS = ["ru", "th", "hi", "ja", "es"]


def massive_wide(n=100):
    """MASSIVE validation (English) with the language's FULL label set as options (~60): every item goes through
    bracket routing in `dgem systemone serve`. Validation split, so it never touches the frozen test set."""
    test = ds.jsonl_gz(ds.MASSIVE, "test/en.json.gz")
    labels = sorted({r["label_text"] for r in test})
    rows = ds.jsonl_gz(ds.MASSIVE, "validation/en.json.gz")[:n]
    rng = random.Random(13)
    out = []
    for i, r in enumerate(rows):
        keys = list(labels)
        rng.shuffle(keys)
        q = {"qid": "intent", "type": "choice", "instructions": "What is the user asking for in `utterance`?",
             "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys}, "labels": keys,
             "expected": r["label_text"], "values": None}
        out.append({"id": f"di_wide-en-{i:03d}", "suite": "di_wide", "subset": "en", "tier": None,
                    "state": {"utterance": r["text"]}, "qs": [q]})
    return out


def massive_spot():
    """Multilingual smoke check: MASSIVE validation split (never the frozen test split), 5 languages x 20."""
    return massive("validation", SPOT_LANGS, 20, suite="massive_spot")


# ---------------------------------------------------------------- XNLI
XNLI_CRIT = {"entailment": "the premise implies the hypothesis is true",
             "neutral": "the premise neither implies nor contradicts the hypothesis",
             "contradiction": "the premise implies the hypothesis is false"}


def xnli(langs=None, n=300, suite="xnli"):
    names = ds.xnli_label_names()
    out = []
    for lang in langs or ds.xnli_langs():
        rows = ds.parquet(ds.XNLI, f"{lang}/test-00000-of-00001.parquet")[:n]
        for i, r in enumerate(rows):
            q = {"qid": "relation", "type": "choice",
                 "instructions": "What is the relationship between `premise` and `hypothesis`?",
                 "criteria": dict(XNLI_CRIT), "labels": list(XNLI_CRIT), "expected": names[r["label"]], "values": None}
            out.append({"id": f"{suite}-{lang}-{i:03d}", "suite": suite, "subset": lang, "tier": None,
                        "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "qs": [q]})
    return out


# ---------------------------------------------------------------- typed-decisions (multi-question cases)
def typed():
    out = []
    for idx, r in enumerate(ds.parquet(ds.TYPED, "all/test-00000-of-00001.parquet")):
        qs_raw, gold = json.loads(r["questions"]), json.loads(r["gold"])
        state = json.loads(r["state"]) if isinstance(r["state"], str) else r["state"]
        qs = []
        for qid, qd in qs_raw.items():
            g, t = gold[qid], qd["type"]
            pr = g.get("probabilities") or {}
            q = {"qid": qid, "type": t, "instructions": qd.get("instructions", ""), "criteria": qd.get("criteria"),
                 "values": None}
            if t == "noul":
                pt = pr.get("true", g.get("noul", 0.5))
                q.update(labels=["yes", "no"], gold_soft={"yes": pt, "no": 1 - pt},
                         expected="yes" if str(g["label"]).lower() == "true" else "no")
            elif t == "choice":
                keys = list(qd["criteria"].keys())
                q.update(labels=keys, gold_soft={k: pr.get(k, 0.0) for k in keys}, expected=str(g["label"]))
            else:
                k = len(qd["criteria"])
                q.update(labels=[str(i) for i in range(k)], values=list(range(k)),
                         gold_soft={str(i): pr.get(str(i), 0.0) for i in range(k)}, expected=str(int(g["label"])),
                         expected_value=float(g.get("score", g["label"])))
            s = sum(q["gold_soft"].values()) or 1.0
            q["gold_soft"] = {a: b / s for a, b in q["gold_soft"].items()}
            qs.append(q)
        out.append({"id": f"typed-{idx:03d}", "suite": "typed", "subset": r["workflow"], "tier": None,
                    "state": state, "qs": qs})
    return out


# ---------------------------------------------------------------- option-order suites
def order_cases():
    """MASSIVE-en 200 (20 options), dair-ai/emotion 200 (6), XNLI-en 200 (3), JevBench choice items."""
    out = massive("test", ["en"], 200, suite="order_massive_en")
    emo = ds.emotion_label_names()
    for i, r in enumerate(ds.parquet(ds.EMOTION, "split/test-00000-of-00001.parquet")[:200]):
        q = {"qid": "emotion", "type": "choice", "instructions": "Which emotion does the author of `text` express?",
             "criteria": {e: e for e in emo}, "labels": list(emo), "expected": emo[r["label"]], "values": None}
        out.append({"id": f"order_emotion-{i:03d}", "suite": "order_emotion", "subset": "en", "tier": None,
                    "state": {"text": r["text"]}, "qs": [q]})
    for c in xnli(["en"], 200, suite="order_xnli_en"):
        out.append(c)
    for c in jevbench():
        if c["qs"][0]["type"] == "choice":
            out.append({**c, "suite": "order_jev_choice"})
    return out


def permute(case, mode, rng=None):
    """Reorder choice criteria: none | random (one shuffle per case from a shared rng) | reverse."""
    if mode == "none":
        return case
    qs = []
    for q in case["qs"]:
        if q["type"] != "choice":
            qs.append(q)
            continue
        keys = list(q["criteria"].keys())
        if mode == "random":
            order = list(range(len(keys)))
            rng.shuffle(order)
            nk = [keys[i] for i in order]
        elif mode == "reverse":
            nk = keys[::-1]
        else:
            raise ValueError(mode)
        qs.append({**q, "criteria": {k: q["criteria"][k] for k in nk}, "shown_order": nk})
    return {**case, "qs": qs}


def body(case):
    return {"state": case["state"],
            "questions": {q["qid"]: {"type": q["type"], "instructions": q["instructions"],
                                     **({"criteria": q["criteria"]} if q["criteria"] is not None else {})}
                          for q in case["qs"]}}


# ---------------------------------------------------------------- answers -> distributions and per-item scores
def distribution(q, answer):
    """Normalize a /v1/systemone answer to {label: p} over q["labels"]. Never uses the server's `confidence`
    field (its meaning differs between servers); confidence is recomputed as max p."""
    if not isinstance(answer, dict):
        return None
    if q["type"] == "noul":
        p = answer.get("noul")
        if p is None:
            pr = answer.get("probabilities") or {}
            p = pr.get("true", pr.get("yes"))
        if p is None:
            return None
        p = min(max(float(p), 0.0), 1.0)
        return {"yes": p, "no": 1.0 - p}
    pr = answer.get("probabilities") or {}
    if q["type"] == "score":
        d = {lab: float(pr.get(str(i), 0.0)) for i, lab in enumerate(q["labels"])}
    else:
        d = {lab: float(pr.get(lab, 0.0)) for lab in q["labels"]}
    s = sum(d.values())
    if s <= 0:
        ch = answer.get("choice")
        return {k: (1.0 if k == ch else 0.0) for k in d} if ch in d else None
    return {k: v / s for k, v in d.items()}


def score(q, dist):
    labels = q["labels"]
    actual = max(labels, key=lambda l: dist[l])
    p_gold = dist.get(q["expected"], 0.0)
    h = -sum(p * math.log(p) for p in dist.values() if p > 1e-12)
    out = {"actual": actual, "accurate": actual == q["expected"], "confidence": dist[actual], "p_gold": p_gold,
           "nll": -math.log(max(p_gold, EPS)),
           "brier": sum((dist[l] - (1.0 if l == q["expected"] else 0.0)) ** 2 for l in labels),
           "entropy": h, "normalized_entropy": h / math.log(len(labels)) if len(labels) > 1 else 0.0}
    if q.get("values"):
        vals = dict(zip(labels, q["values"]))
        ev = sum(dist[l] * vals[l] for l in labels)
        out["abs_err_ev"] = abs(ev - q.get("expected_value", vals[q["expected"]]))
    gs = q.get("gold_soft")
    if gs:
        out["soft_acc"] = sum(dist[l] * gs.get(l, 0.0) for l in labels)
        out["soft_brier"] = sum((dist[l] - gs.get(l, 0.0)) ** 2 for l in labels)
        out["gold_soft_max"] = max(gs.values())
    return out


# ---------------------------------------------------------------- Decision Index adapter probes
# Synthetic requests with known answers that exercise `dgem systemone serve` paths a Decision Index run depends on:
# bracket routing at awkward option counts (K % 20 == 1 left a one-option bracket), slot batching (> 8 questions),
# and capacity refusals that must come back as HTTP 422 with a kit-recognised marker (else the kit records "error").
KIT_MARKERS = ("options per choice", "the canvas holds", "maximum context length", "context window", "too many tokens")


def _wide_probe(k):
    opts = {f"opt_{i:03d}": f"unrelated topic number {i}" for i in range(k - 1)}
    opts["refund_request"] = "the customer asks for their money back"
    keys = sorted(opts)
    return {"state": {"message": "I was charged twice, please give me my money back."},
            "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                                     "criteria": {x: opts[x] for x in keys}}}}


_BATCH_QS = [("Is the customer asking for a refund?", "yes"), ("Is the message in English?", "yes"),
             ("Does the message mention a double charge?", "yes"), ("Is the customer threatening legal action?", "no"),
             ("Is the customer asking about shipping?", "no"), ("Is the message polite?", "yes"),
             ("Does the message mention a password?", "no"), ("Is this about billing?", "yes"),
             ("Does the message include a phone number?", "no"), ("Is the customer asking to cancel?", "no"),
             ("Is money involved?", "yes"), ("Is the message about a software bug?", "no")]


def _catchall_probe(k):
    opts = {f"opt_{i:03d}": f"unrelated intent number {i}" for i in range(k - 2)}
    opts["refund_request"] = "the customer asks for their money back"
    opts["out_of_scope"] = "out of scope: none of the listed intents"
    keys = sorted(opts)
    return {"state": {"message": "I was charged twice, please give me my money back."},
            "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?",
                                     "criteria": {x: opts[x] for x in keys}}}}


def di_probes():
    """-> list of probes {name, body, expect: {status, answers?, complete?}}."""
    out = []
    for k in (27, 41, 61, 101, 151, 255):
        out.append({"name": f"wide_{k}", "body": _wide_probe(k),
                    "expect": {"status": 200, "answers": {"intent": "refund_request"}, "complete": k}})
    out.append({"name": "batch_12q",
                "body": {"state": {"message": "Hi, I was charged twice for my order. Could you please refund one of "
                                              "the charges? Thanks!"},
                         "questions": {f"q{i:02d}": {"type": "noul", "instructions": ins}
                                       for i, (ins, _) in enumerate(_BATCH_QS)}},
                "expect": {"status": 200, "answers": {f"q{i:02d}": a for i, (_, a) in enumerate(_BATCH_QS)}}})
    # Non-string option descriptions (objects, arrays): the Decision Index sends them for POP909, ChessBench, cfcolor.
    out.append({"name": "criteria_objects",
                "body": {"state": {"swatch": "a bright pure red square"},
                         "questions": {"color": {"type": "choice", "instructions": "Which colour value matches `swatch`?",
                                                 "criteria": {"red": [255, 0, 0], "blue": [0, 0, 255], "green": {"rgb": [0, 255, 0]}}}}},
                "expect": {"status": 200, "answers": {"color": "red"}, "complete": 3}})
    # noul with true/false descriptions that carry the meaning (the claim is only defined there). Informational: the
    # model often answers it right even when the adapter drops the criteria, so it cannot gate that bug reliably.
    out.append({"name": "noul_criteria",
                "body": {"state": {"context": "The meeting is on Tuesday at 10am in room 4.",
                                   "response": "The meeting is on Friday at 3pm in room 9."},
                         "questions": {"q": {"type": "noul", "instructions": "Hallucination check.",
                                             "criteria": {"true": "The response contains content not supported by the context.",
                                                          "false": "All content of the response is supported by the context."}}}},
                "expect": {"status": 200, "answers": {"q": "yes"}}, "info_only": True})
    # Catch-all among >26 options: the catch-all is correct inside its own bracket, so bracket finals can over-pick it
    # on in-scope items. This easy synthetic case passes today; it is informational until the adapter handles
    # catch-alls and a harder, validation-split case replaces it.
    out.append({"name": "wide_catchall_151", "body": _catchall_probe(151),
                "expect": {"status": 200, "answers": {"intent": "refund_request"}, "complete": 151}, "info_only": True})
    # Long input that fits a 32k canary but not a 4k server: must answer, or refuse with 422 + marker.
    mid = " ".join(["The quarterly report discusses revenue, costs and outlook in detail."] * 700)  # ~9k tokens
    out.append({"name": "long_9k",
                "body": {"state": {"document": mid + " The total headcount at year end was 412 employees. " + mid},
                         "questions": {"q": {"type": "noul", "instructions": "Does the document state the year-end headcount?"}}},
                "expect": {"status_in": [200, 422], "marker_if_422": True, "answers_if_200": {"q": "yes"}}})
    long_doc = " ".join(["The quarterly report discusses revenue, costs and outlook in detail."] * 2400)  # ~31k tokens
    out.append({"name": "context_refusal",
                "body": {"state": {"document": long_doc},
                         "questions": {"q": {"type": "noul", "instructions": "Does the document mention revenue?"}}},
                "expect": {"status_in": [200, 422], "marker_if_422": True}})
    return out


# ---------------------------------------------------------------- development sets for adapter weaknesses
CATCHALL_LABEL = "out of scope: none of the listed intents"


def di_catchall(n_in=80, n_oos=20):
    """CLINC150 validation (never the test split) in the Decision Index request format: 151 options keyed option_N in
    sorted label order, the utterance in the instructions, an empty state, and an 'out of scope' catch-all option.
    Tracks how often in-scope requests are answered 'out of scope' by the adapter's bracket routing."""
    f = "plus/validation-00000-of-00001.parquet"
    names = ds.parquet_labels(ds.CLINC, f, "intent")
    order = sorted(names)
    crit = {f"option_{i}": (CATCHALL_LABEL if nm == "oos" else nm.replace("_", " ")) for i, nm in enumerate(order)}
    key = {nm: f"option_{i}" for i, nm in enumerate(order)}
    rows = ds.parquet(ds.CLINC, f)
    rng = random.Random(20261002)
    ins = [r for r in rows if names[r["intent"]] != "oos"]
    oos = [r for r in rows if names[r["intent"]] == "oos"]
    rng.shuffle(ins)
    rng.shuffle(oos)
    out = []
    for i, r in enumerate(ins[:n_in] + oos[:n_oos]):
        nm = names[r["intent"]]
        q = {"qid": "q", "type": "choice",
             "instructions": "Classify the intent of this user request, or choose out of scope if none applies:\n" + r["text"],
             "criteria": crit, "labels": list(crit), "expected": key[nm], "values": None}
        out.append({"id": f"di_catchall-{i:03d}", "suite": "di_catchall", "subset": "oos" if nm == "oos" else "in",
                    "tier": None, "state": {}, "qs": [q]})
    return out


RAG_INS = "The response contains content that is not supported by the context in the prompt."


def rag_dev(n=200):
    """RAGTruth train split (never test), the Decision Index hallucination question wording with true/false criteria.
    Tracks the yes/no 'no' bias (F1 on the hallucinated class vs the always-flag baseline)."""
    import ast
    rows = ds.parquet(ds.RAGTRUTH, "data/train-00000-of-00001.parquet")
    rng = random.Random(20261002)
    by = {}
    for r in rows:
        by.setdefault(r["task_type"], []).append(r)
    out = []
    per = n // max(1, len(by))
    for task in sorted(by):
        rs = by[task]
        rng.shuffle(rs)
        for r in rs[:per]:
            lab = r["hallucination_labels_processed"]
            if not isinstance(lab, dict):
                try:
                    lab = ast.literal_eval(lab)
                except Exception:
                    lab = json.loads(str(lab).replace("'", '"'))
            hall = (int(lab.get("evident_conflict", 0)) + int(lab.get("baseless_info", 0))) > 0
            prompt = r["query"] + "\n" + r["context"] if r["context"] else r["query"]
            q = {"qid": "q", "type": "noul", "instructions": RAG_INS,
                 "criteria": {"true": RAG_INS, "false": "All content of the response is supported by the context in the prompt."},
                 "labels": ["yes", "no"], "expected": "yes" if hall else "no", "values": None}
            out.append({"id": f"rag_dev-{task}-{r['id']}", "suite": "rag_dev", "subset": task, "tier": None,
                        "state": {"prompt": prompt, "response": r["output"]}, "qs": [q]})
    return out


SUITES = {
    "di_catchall": di_catchall,
    "rag_dev": rag_dev,
    "di_wide": massive_wide,
    "jev_systemone": jevbench,
    "massive_spot": massive_spot,
    "massive": massive,
    "xnli": xnli,
    "typed": typed,
    "order": order_cases,
}
