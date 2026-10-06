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

"""Side-by-side serving contract probe for two or more dgemma structured servers.

Sends the same set of requests (valid decisions, edge cases and error cases) to every target and prints
a table comparing HTTP status, response shape, answers and probabilities. Use it before promoting a new
serving image: any row marked DIFF needs an explanation.

  python3 scripts/contract_diff.py --target old=<base-url> --target new=<base-url> [-o out.json]

<base-url> is the server root: a Cloud Run URL (https://<service>.run.app) or a Vertex dedicated
endpoint invoke base (https://<dns>/v1/projects/<p>/locations/<r>/endpoints/<id>/invoke). Tokens come
from Application Default Credentials (access token for Vertex, ID token otherwise).
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import serving_speed as ss  # noqa: E402  (reuses auth, schemas and the test image)
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

TICKET = "We were charged twice for our annual renewal and the API returns 500 on every invoice call."


def opts(n):
    return [{"name": f"opt_{i:02d}", "description": f"option {i}"} for i in range(n)]


CASES = {
    # name: (path, schema-or-body, kind)
    "triage_s1": ("chat", {"instructions": "Triage the ticket.", "samples": 1, "think": 0, "questions": ss.Q}, None),
    "triage_s4": ("chat", {"instructions": "Triage the ticket.", "samples": 4, "think": 0, "questions": ss.Q}, None),
    "samples_string": ("chat", {"instructions": "Triage the ticket.", "samples": "4", "think": 0, "questions": ss.Q}, None),
    "default_samples": ("chat", {"instructions": "Triage the ticket.", "think": 0, "questions": ss.Q}, None),
    "mirror_s1": ("chat", {"instructions": "Triage the ticket.", "samples": 1, "think": 0, "questions": ss.Q + ss.MIRROR}, None),
    "think120": ("chat", {"instructions": "Route the request.", "samples": 1, "think": 120, "questions": ss.TAXONOMY}, None),
    "image_s1": ("chat", {"instructions": "Inspect the screenshot.", "samples": 1, "think": 0, "questions": ss.IMAGE_Q}, "image"),
    "choice_26": ("chat", {"instructions": "Pick one.", "samples": 1, "think": 0,
                           "questions": [{"id": "pick", "type": "choice", "instructions": "Pick the best option.", "options": opts(26)}]}, None),
    "choice_27_error": ("chat", {"instructions": "Pick one.", "samples": 1, "think": 0,
                                 "questions": [{"id": "pick", "type": "choice", "instructions": "Pick.", "options": opts(27)}]}, None),
    "depends_on_2reads": ("chat", {"instructions": "Triage the ticket.", "samples": 1, "think": 0, "questions": [
        {"id": "billing_issue", "type": "boolean", "instructions": "Is this a billing issue?"},
        {"id": "refund", "type": "boolean", "instructions": "Should a refund be issued?", "depends_on": ["billing_issue"],
         "ask_if": {"billing_issue": ["yes"]}}]}, None),
    "unknown_type_error": ("chat", {"instructions": "x", "samples": 1, "questions": [{"id": "a", "type": "banana", "instructions": "?"}]}, None),
    "empty_questions_error": ("chat", {"instructions": "x", "samples": 1, "questions": []}, None),
    "systemone": ("systemone", {"model": "dgemma", "state": {"ticket": TICKET}, "questions": {
        "urgent": {"type": "noul", "instructions": "Does this issue require immediate same-day escalation?"},
        "billing": {"type": "noul", "instructions": "Is this a billing problem?"}}}, None),
    "invalid_json_error": ("rawbody", b"{not json", None),
}

# Multilingual /v1/systemone cases with a JSON-object state (the path that once escaped non-ASCII text to
# \uXXXX, so the model read hex codes instead of Russian or Thai). Each has a known answer: a wrong label
# on any target is reported as WRONG, not just as a difference between targets.
INTENT = {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
          "criteria": {"set_alarm": "set an alarm", "weather": "ask about the weather",
                       "play_music": "play music", "send_email": "send an email"}}
MULTILINGUAL = {
    "ml_ru": "Разбуди меня завтра в семь утра",
    "ml_th": "ตั้งนาฬิกาปลุกตอนเจ็ดโมงเช้า",
    "ml_hi": "कल सुबह सात बजे का अलार्म लगाओ",
    "ml_ja": "明日の朝7時にアラームをセットして",
    "ml_es": "Pon una alarma a las siete de la mañana",
}
EXPECT = {}
for _name, _text in MULTILINGUAL.items():
    CASES[_name] = ("systemone", {"model": "dgemma", "state": {"utterance": _text}, "questions": {"intent": INTENT}}, None)
    EXPECT[_name] = {"intent": "set_alarm"}
CASES["ml_el_noul"] = ("systemone", {"model": "dgemma", "state": {"message": "Χρεώθηκα δύο φορές για την ίδια παραγγελία."},
                                     "questions": {"billing": {"type": "noul", "instructions": "Is this a billing problem?"}}}, None)
EXPECT["ml_el_noul"] = {"billing": "yes"}


class Target:
    def __init__(self, name, base):
        self.name, self.base = name, base.rstrip("/")
        self.vertex = "prediction.vertexai.goog" in base

    def token(self):
        at, it = ss.TOK.get()
        return at if self.vertex else it

    def request(self, path, body=None, method="POST"):
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token()}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)
        except Exception as e:  # network errors are recorded, not raised
            return None, repr(e).encode(), {}


def _label(v):
    if not isinstance(v, dict):
        return v
    if v.get("label") is not None:
        return v["label"]
    if v.get("choice") is not None:
        return v["choice"]
    if v.get("noul") is not None:  # /v1/systemone noul answers carry only a probability
        return "yes" if v["noul"] >= 0.5 else "no"
    return v.get("level")


def summarize(status, raw, headers=None):
    out = {"status": status, "server_timing": {k.lower(): v for k, v in (headers or {}).items()}.get("server-timing")}
    try:
        body = json.loads(raw)
    except Exception:
        out["body"] = raw[:160].decode(errors="replace")
        return out
    if status != 200:
        err = body.get("error", body) if isinstance(body, dict) else body
        out["error"] = (err.get("message") if isinstance(err, dict) else str(err))[:160]
        return out
    env = body
    if "choices" in body:
        out["top_keys"] = sorted(body)
        try:
            env = json.loads(body["choices"][0]["message"]["content"])
        except Exception as e:
            out["parse_error"] = repr(e)[:120]
            return out
    out["envelope_keys"] = sorted(env) if isinstance(env, dict) else type(env).__name__
    answers = env.get("answers", {}) if isinstance(env, dict) else {}
    diag = env.get("diagnostics", {}) if isinstance(env, dict) else {}
    out["answers"] = answers
    out["diagnostic_keys"] = sorted(diag)
    t = diag.get("timing", {}) or {}
    out["reads"] = t.get("reads")
    out["denoise_ms"] = t.get("total_ms")
    out["probabilities"] = {k: v.get("probabilities") for k, v in answers.items() if isinstance(v, dict)}
    out["labels"] = {k: _label(v) for k, v in answers.items()}
    return out


def run_case(t, name):
    path, body, kind = CASES[name]
    if path == "chat":
        if kind == "image":
            ss.IMG = ss.IMG or ss.image_data_uri()
            user = [{"type": "text", "text": json.dumps({"note": TICKET})}, {"type": "image_url", "image_url": {"url": ss.IMG}}]
        else:
            user = json.dumps({"ticket": TICKET})
        payload = {"model": "dgemma", "messages": [{"role": "system", "content": json.dumps(body)}, {"role": "user", "content": user}],
                   "logprobs": True, "top_logprobs": 5}
        return summarize(*t.request("/v1/chat/completions", payload))
    if path == "systemone":
        return summarize(*t.request("/v1/systemone", body))
    if path == "rawbody":
        return summarize(*t.request("/v1/chat/completions", body))
    raise ValueError(path)


def top(probs):
    return {k: max(v, key=v.get) for k, v in (probs or {}).items() if isinstance(v, dict) and v}


def max_delta(a, b):
    d = 0.0
    for q in set(a or {}) & set(b or {}):
        pa, pb = a[q] or {}, b[q] or {}
        if isinstance(pa, dict) and isinstance(pb, dict):
            for k in set(pa) | set(pb):
                d = max(d, abs(float(pa.get(k, 0)) - float(pb.get(k, 0))))
    return round(d, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", required=True, help="name=base-url (repeat)")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    targets = [Target(*s.split("=", 1)) for s in a.target]
    res = {"health": {}, "cases": {}}
    for t in targets:
        st, raw, _ = t.request("/health", None, method="GET")
        try:
            h = json.loads(raw)
        except Exception:
            h = raw[:200].decode(errors="replace")
        res["health"][t.name] = {"status": st, "body": h}
        print(f"health {t.name:10s} {st} {json.dumps(h)[:220]}")
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {(t.name, c): ex.submit(run_case, t, c) for t in targets for c in a.cases}
        for (tn, c), f in futs.items():
            res["cases"].setdefault(c, {})[tn] = f.result()
    names = [t.name for t in targets]
    print(f"\n{'case':22s} " + " ".join(f"{n:>34s}" for n in names) + "  verdict")
    for c in a.cases:
        row = res["cases"][c]
        cells, sig = [], []
        for n in names:
            r = row[n]
            if r["status"] == 200:
                cell = f"200 r={r.get('reads')} " + ",".join(str(v) for v in (r.get("labels") or {}).values())[:26]
            else:
                cell = f"{r['status']} {(r.get('error') or r.get('body') or '')[:28]}"
            cells.append(cell)
            sig.append((r["status"], json.dumps(r.get("labels"), sort_keys=True), r.get("reads"),
                        tuple(r.get("envelope_keys") or []), tuple(r.get("diagnostic_keys") or [])))
        base = sig[0]
        notes = []
        for s in sig[1:]:
            if s[0] != base[0]:
                notes.append("status")
            if s[1] != base[1]:
                notes.append("answers")
            if s[2] != base[2]:
                notes.append("reads")
            if s[3] != base[3]:
                notes.append("envelope")
            if s[4] != base[4]:
                notes.append("diag-keys")
        if row[names[0]]["status"] == 200 and len(names) > 1:
            notes.append(f"maxΔp={max_delta(row[names[0]].get('probabilities'), row[names[1]].get('probabilities'))}")
        wrong = [n for n in names if c in EXPECT and row[n].get("labels") != EXPECT[c]]
        if wrong:
            notes.append("WRONG(" + ",".join(wrong) + ")")
        verdict = "same" if not [n for n in notes if not n.startswith("maxΔp")] else "DIFF " + ",".join(notes)
        if verdict == "same" and notes:
            verdict += " " + notes[-1]
        print(f"{c:22s} " + " ".join(f"{x:>34s}" for x in cells) + f"  {verdict}")
    for c in a.cases:
        row = res["cases"][c]
        ks = {n: (row[n].get("diagnostic_keys"), row[n].get("envelope_keys")) for n in names}
        if len({json.dumps(v) for v in ks.values()}) > 1:
            print(f"\n{c}: key differences")
            for n, v in ks.items():
                print(f"  {n}: envelope={v[1]} diagnostics={v[0]}")
    if a.output:
        json.dump(res, open(a.output, "w"), indent=1)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
