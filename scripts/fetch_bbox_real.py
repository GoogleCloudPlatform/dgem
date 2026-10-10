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

"""Build the real-image bounding-box suite (seeded sample) and fetch its images.

Sources (images are downloaded, never committed; the manifest records each image's SHA-256):
  refcoco     RefCOCO validation referring expressions (Hugging Face jxu124/refcoco) on COCO train2014
              photos from images.cocodataset.org. One expression per image.
  screenspot  ScreenSpot GUI grounding test set (Hugging Face bevaya/ScreenSpot, Apache-2.0), shards 0-1:
              Windows, macOS, iOS, Android and GitLab screenshots, text and icon targets.

  pip install -r scripts/requirements-vision.txt
  python3 scripts/fetch_bbox_real.py build      # sample, download, write third_party/bbox_real/bbox_real.jsonl
  python3 scripts/fetch_bbox_real.py fetch      # re-download the images of an existing manifest and verify hashes
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HF = "https://huggingface.co/datasets"
REFCOCO_VAL = f"{HF}/jxu124/refcoco/resolve/main/data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet"
SCREENSPOT = [f"{HF}/bevaya/ScreenSpot/resolve/main/data/test-0000{i}-of-00003.parquet" for i in (0, 1)]
COCO_IMG = "http://images.cocodataset.org/train2014/{}"
TEMPLATE = "templates/multimodal/bbox_localization.json.tmpl"


def get(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "dgem-prop18"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def cached(url, cache: Path):
    p = cache / url.rsplit("/", 1)[-1]
    if not p.exists():
        print(f"  downloading {url}", file=sys.stderr)
        p.write_bytes(get(url))
    return p


def size_bucket(b):
    area = (b[2] - b[0]) * (b[3] - b[1])  # percent x percent
    return "small" if area < 75 else ("medium" if area < 400 else "large")


def pct(x0, y0, x1, y1, w, h):
    return [round(100 * y0 / h, 3), round(100 * x0 / w, 3), round(100 * y1 / h, 3), round(100 * x1 / w, 3)]


def build(args):
    import pyarrow.parquet as pq
    from PIL import Image

    rng = random.Random(args.seed)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / args.out_dir
    rows = []

    # RefCOCO: one ref per image, seeded sample.
    t = pq.read_table(cached(REFCOCO_VAL, cache)).to_pylist()
    by_img = defaultdict(list)
    for r in t:
        by_img[r["image_id"]].append(r)
    imgs = sorted(by_img)
    rng.shuffle(imgs)
    (out_dir / "refcoco").mkdir(parents=True, exist_ok=True)
    for image_id in imgs[: args.refcoco]:
        r = rng.choice(by_img[image_id])
        info = json.loads(r["raw_image_info"])
        fname = info["file_name"]
        data = get(COCO_IMG.format(fname), timeout=120)
        path = out_dir / "refcoco" / fname
        path.write_bytes(data)
        w, h = info["width"], info["height"]
        x0, y0, x1, y1 = r["bbox"]  # xyxy pixels (raw_anns.bbox is COCO xywh)
        gt = pct(x0, y0, x1, y1, w, h)
        sent = r["sentences"][0]["sent"]
        rows.append({
            "id": f"refcoco-{r['ref_id']}", "tier": "refcoco", "template": TEMPLATE,
            "image_path": str(path.relative_to(REPO)), "target": sent, "object_present": True,
            "gt_box_continuous": gt, "occluded_edge": "none",
            "tags": {"set": "refcoco", "category_id": str(r["category_id"]), "size": size_bucket(gt),
                     "orientation": "landscape" if w >= h else "portrait"},
            "image_sha256": hashlib.sha256(data).hexdigest(), "source_url": COCO_IMG.format(fname),
            "notes": f"RefCOCO val ref {r['ref_id']}, ann {r['ann_id']}",
        })

    # ScreenSpot: stratified by platform x element type.
    ss = []
    for url in SCREENSPOT:
        tbl = pq.read_table(cached(url, cache))
        meta = tbl.select([c for c in tbl.column_names if c != "image"]).to_pylist()
        for i, m in enumerate(meta):
            ss.append((url, i, m))
    strata = defaultdict(list)
    for item in ss:
        m = item[2]
        strata[(m["data_source"], m["data_type"])].append(item)
    keys = sorted(strata)
    per = max(1, args.screenspot // len(keys))
    chosen = []
    for k in keys:
        pool = strata[k][:]
        rng.shuffle(pool)
        chosen.extend(pool[:per])
    (out_dir / "screenspot").mkdir(parents=True, exist_ok=True)
    tables = {}
    for url, i, m in chosen:
        if url not in tables:
            tables[url] = pq.read_table(cached(url, cache), columns=["image"])
        data = tables[url].slice(i, 1).to_pylist()[0]["image"]["bytes"]
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        path = out_dir / "screenspot" / m["file_name"]
        path.write_bytes(data)
        b = m["bbox"]  # normalized xyxy
        gt = pct(b[0] * w, b[1] * h, b[2] * w, b[3] * h, w, h)
        rows.append({
            # One screenshot can carry several instructions, so the id is the source shard and row.
            "id": f"screenspot-{SCREENSPOT.index(url)}-{i:03d}", "tier": "screenspot", "template": TEMPLATE,
            "image_path": str(path.relative_to(REPO)), "target": m["instruction"], "object_present": True,
            "gt_box_continuous": gt, "occluded_edge": "none",
            "tags": {"set": "screenspot", "platform": m["data_source"], "element": m["data_type"], "size": size_bucket(gt),
                     "orientation": "landscape" if w >= h else "portrait"},
            "image_sha256": hashlib.sha256(data).hexdigest(), "source_url": url, "source_row": i,
            "notes": "ScreenSpot test (Apache-2.0)",
        })

    man = REPO / args.manifest
    with open(man, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items to {man}; images in {args.out_dir}/")


def fetch(args):
    import pyarrow.parquet as pq

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tables, bad = {}, 0
    for line in open(REPO / args.manifest):
        r = json.loads(line)
        path = REPO / r["image_path"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == r["image_sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if r["tier"] == "refcoco":
            data = get(r["source_url"], timeout=120)
        else:
            url = r["source_url"]
            if url not in tables:
                tables[url] = pq.read_table(cached(url, cache), columns=["image"])
            data = tables[url].slice(r["source_row"], 1).to_pylist()[0]["image"]["bytes"]
        path.write_bytes(data)
        if hashlib.sha256(data).hexdigest() != r["image_sha256"]:
            bad += 1
            print(f"hash mismatch: {r['id']}", file=sys.stderr)
    print("all images present and verified" if not bad else f"{bad} hash mismatches")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "fetch"])
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--refcoco", type=int, default=120)
    ap.add_argument("--screenspot", type=int, default=120)
    ap.add_argument("--out-dir", default="fixtures/bbox_real")
    ap.add_argument("--manifest", default="third_party/bbox_real/bbox_real.jsonl")
    ap.add_argument("--cache", default="/tmp/dgem-bbox-real-cache")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else fetch(args)


if __name__ == "__main__":
    main()
