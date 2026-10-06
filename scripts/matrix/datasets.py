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

"""Runtime datasets for the regression matrix, pinned by Hugging Face commit and SHA-256.

Files are listed in benchmarks/matrix/datasets.lock.json and cached under $DGEM_MATRIX_CACHE
(default ~/.cache/dgem-matrix). A file whose hash does not match is deleted and the run stops. MASSIVE
(.json.gz) needs only the standard library; XNLI, typed-decisions and emotion are parquet and need pyarrow
(`pip install -r scripts/requirements-matrix.txt`).
"""
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCK = os.path.join(REPO, "benchmarks", "matrix", "datasets.lock.json")
CACHE = os.environ.get("DGEM_MATRIX_CACHE", os.path.expanduser("~/.cache/dgem-matrix"))

MASSIVE = "mteb/amazon_massive_intent"
XNLI = "facebook/xnli"
TYPED = "LocalLLaMA/typed-decisions"
EMOTION = "dair-ai/emotion"
CLINC = "clinc/clinc_oos"
RAGTRUTH = "wandb/RAGTruth-processed"


def lock():
    with open(LOCK) as f:
        return json.load(f)["datasets"]


def path(repo, file):
    """Local path of a locked file, downloading and verifying it if needed."""
    ent = lock()[repo]
    meta = ent["files"].get(file)
    if meta is None:
        raise KeyError(f"{repo}/{file} is not in {os.path.relpath(LOCK, REPO)}")
    local = os.path.join(CACHE, repo.replace("/", "__") + "@" + ent["revision"][:12], file)
    if os.path.exists(local) and _sha(local) == meta["sha256"]:
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{ent['revision']}/{file}"
    data = None
    for attempt in range(6):  # the Hub rate-limits anonymous downloads (HTTP 429), notably from cloud egress
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
                                         if os.environ.get("HF_TOKEN") else {})
            data = urllib.request.urlopen(req, timeout=300).read()
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == 5:
                raise
            time.sleep(10 * 2 ** attempt)
    got = hashlib.sha256(data).hexdigest()
    if got != meta["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {repo}/{file}@{ent['revision'][:7]}: {got} != {meta['sha256']}")
    tmp = local + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, local)
    return local


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_gz(repo, file):
    with gzip.open(path(repo, file), "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parquet(repo, file):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("this suite reads parquet: pip install -r scripts/requirements-matrix.txt") from e
    return pq.read_table(path(repo, file)).to_pylist()


def fetch(repos=None):
    """Download and verify every locked file (or those of the given repos). -> list of (repo, file, ok)."""
    out = []
    for repo, ent in lock().items():
        if repos and repo not in repos:
            continue
        for file in ent["files"]:
            try:
                path(repo, file)
                out.append((repo, file, True))
            except Exception as e:
                out.append((repo, file, str(e)[:200]))
    return out


def massive_langs():
    return sorted(f[len("test/"):-len(".json.gz")] for f in lock()[MASSIVE]["files"] if f.startswith("test/"))


def xnli_langs():
    return sorted(f.split("/")[0] for f in lock()[XNLI]["files"] if f.endswith("/test-00000-of-00001.parquet"))


def xnli_label_names():
    return ["entailment", "neutral", "contradiction"]  # facebook/xnli ClassLabel order


def emotion_label_names():
    return ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion ClassLabel order


def parquet_labels(repo, file, column):
    """ClassLabel names for a parquet column, from the Hugging Face schema metadata."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("this suite reads parquet: pip install -r scripts/requirements-matrix.txt") from e
    meta = pq.read_schema(path(repo, file)).metadata or {}
    import json as _json
    feats = _json.loads(meta.get(b"huggingface", b"{}")).get("info", {}).get("features", {})
    return feats.get(column, {}).get("names")
