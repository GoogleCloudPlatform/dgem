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

"""Build multi-aspect ground truth for the real-image suite (bench-vision on RefCOCO + ScreenSpot).

Reads benchmarks/bbox_real.jsonl (images from scripts/fetch_bbox_real.py) and writes
benchmarks/bbox_real_aspects.jsonl, one row per question set, with "aspects" ground truth and
"aspect_sources" saying where each label came from:

  grid_cell  geometry: 3x3 cell of the annotated box centre
  relation   geometry: target vs another annotated target in the same image (RefCOCO ref / ScreenSpot
             instruction), only when one axis clearly dominates; the other target's text is the reference
  present    "yes" rows: the annotated target. "no" rows (negatives): the same image queried with a target
             text from a different image (different COCO category / different platform), kept only when a
             Gemini 3.x judge confirms it is not visible (label source "gemini_confirmed")
  occluded   Gemini 3.x judge on the image with the annotated box drawn ("is part of the outlined target
             hidden or cut off?"); validated against hand labels (--handcheck)

  python3 scripts/build_bbox_real_aspects.py build --project <PROJECT> [--model gemini-3.8-flash]
  python3 scripts/build_bbox_real_aspects.py sheet --n 40          # contact sheets for hand labelling
  python3 scripts/build_bbox_real_aspects.py handcheck <labels.json>  # agreement of Gemini with hand labels
  python3 scripts/build_bbox_real_aspects.py masks     # RefCOCO segmentation polygons -> benchmarks/bbox_real_masks.jsonl

Needs scripts/requirements-vision.txt and `gcloud` (Application Default Credentials) for the Gemini calls.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/dgem-bbox-real-cache")
TEMPLATE = "templates/multimodal/vision_aspects_generic.json.tmpl"


def adc_token():
    return subprocess.check_output(["gcloud", "auth", "application-default", "print-access-token"], text=True).strip()


_TOKEN = {"t": None, "at": 0.0}


def gemini_json(project, model, png_bytes, prompt, schema):
    if not _TOKEN["t"] or time.time() - _TOKEN["at"] > 1800:
        _TOKEN["t"], _TOKEN["at"] = adc_token(), time.time()
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png_bytes).decode()}},
            {"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema},
    }
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {_TOKEN['t']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return json.loads(d["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + 3 * attempt * attempt)
    raise RuntimeError(f"gemini call failed: {err}")


def png_of(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay(path, box):
    from PIL import Image, ImageDraw
    img = Image.open(REPO / path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    t = max(2, round(min(w, h) / 250))
    d.rectangle((box[1] / 100 * w, box[0] / 100 * h, box[3] / 100 * w, box[2] / 100 * h), outline=(255, 0, 255), width=t)
    return img


def grid_cell(b):
    cy, cx = (b[0] + b[2]) / 200, (b[1] + b[3]) / 200
    return ["top", "middle", "bottom"][min(2, int(cy * 3))] + "_" + ["left", "center", "right"][min(2, int(cx * 3))]


def relation(t, r):
    dx = (t[1] + t[3]) / 2 - (r[1] + r[3]) / 2
    dy = (t[0] + t[2]) / 2 - (r[0] + r[2]) / 2
    if abs(dx) > 1.5 * abs(dy) and abs(dx) > 5:
        return "left_of" if dx < 0 else "right_of"
    if abs(dy) > 1.5 * abs(dx) and abs(dy) > 5:
        return "above" if dy < 0 else "below"
    return None


def others_in_image(man):
    """For each manifest row: other annotated targets in the same image, as (text, box_pct)."""
    import pyarrow.parquet as pq
    from PIL import Image
    out = {}
    rc = pq.read_table(CACHE / "validation-00000-of-00001-bfeafdc84ca37aa2.parquet").to_pylist()
    by_img = defaultdict(list)
    for r in rc:
        by_img[r["image_id"]].append(r)
    ss_rows = {}
    for i in (0, 1):
        t = pq.read_table(CACHE / f"test-0000{i}-of-00003.parquet", columns=["file_name", "bbox", "instruction"]).to_pylist()
        for j, r in enumerate(t):
            ss_rows[(i, j)] = r
    ss_by_file = defaultdict(list)
    for k, r in ss_rows.items():
        ss_by_file[r["file_name"]].append((k, r))
    for m in man:
        if m["tier"] == "refcoco":
            ref_id = int(m["notes"].split("ref ")[1].split(",")[0])
            info = None
            cands = []
            for rows in by_img.values():
                if any(r["ref_id"] == ref_id for r in rows):
                    for r in rows:
                        if r["ref_id"] != ref_id:
                            iw = json.loads(r["raw_image_info"])
                            x0, y0, x1, y1 = r["bbox"]
                            cands.append((r["sentences"][0]["sent"], [100 * y0 / iw["height"], 100 * x0 / iw["width"],
                                                                     100 * y1 / iw["height"], 100 * x1 / iw["width"]], r["category_id"]))
                    break
            out[m["id"]] = cands
        else:
            shard, row = int(m["id"].split("-")[1]), int(m["id"].split("-")[2])
            fn = ss_rows[(shard, row)]["file_name"]
            cands = []
            for (k, r) in ss_by_file[fn]:
                if k != (shard, row):
                    b = r["bbox"]
                    cands.append((r["instruction"], [100 * b[1], 100 * b[0], 100 * b[3], 100 * b[2]], None))
            out[m["id"]] = cands
    return out


def build(args):
    rng = random.Random(args.seed)
    man = [json.loads(l) for l in open(REPO / "benchmarks/bbox_real.jsonl")]
    others = others_in_image(man)
    pos_schema = {"type": "OBJECT", "properties": {"occluded": {"type": "STRING", "enum": ["yes", "no"]}}, "required": ["occluded"]}
    neg_schema = {"type": "OBJECT", "properties": {"visible": {"type": "STRING", "enum": ["yes", "no", "unsure"]}}, "required": ["visible"]}

    def occl(m):
        p = ("The magenta rectangle outlines the target \"%s\". Is part of the target hidden behind another object, "
             "or cut off by the image border? Answer yes only if a visible part of the target is clearly missing.") % m["target"]
        return gemini_json(args.project, args.model, png_of(overlay(m["image_path"], m["gt_box_continuous"])), p, pos_schema)["occluded"]

    # Negative queries: a target text from another item of a different category / platform.
    negs = []
    for m in man:
        pool = [o for o in man if o["image_path"] != m["image_path"] and o["tier"] == m["tier"] and
                o["tags"].get("category_id", o["tags"].get("platform")) != m["tags"].get("category_id", m["tags"].get("platform"))]
        negs.append((m, rng.choice(pool)["target"]))

    def neg_ok(pair):
        m, text = pair
        from PIL import Image
        img = Image.open(REPO / m["image_path"]).convert("RGB")
        p = (f"Is there anything in this image that matches the description \"{text}\"? Answer \"no\" only if you are "
             f"confident nothing in the image matches it; answer \"unsure\" if something might.")
        return gemini_json(args.project, args.model, png_of(img), p, neg_schema)["visible"]

    with ThreadPoolExecutor(args.workers) as ex:
        occ = list(ex.map(occl, man))
        print("occlusion labels done", file=sys.stderr)
        negv = list(ex.map(neg_ok, negs))
        print("negative checks done", file=sys.stderr)

    rows = []
    for m, oc in zip(man, occ):
        asp = {"present": "yes", "grid_cell": grid_cell(m["gt_box_continuous"]), "occluded": oc}
        src = {"present": "annotation", "grid_cell": "geometry", "occluded": f"gemini:{args.model}"}
        ref = None
        cands = [c for c in others[m["id"]] if relation(m["gt_box_continuous"], c[1])]
        if cands:
            text, box, _ = rng.choice(cands)
            asp["relation"], ref, src["relation"] = relation(m["gt_box_continuous"], box), text, "geometry"
        r = {"id": m["id"], "tier": m["tier"], "image_path": m["image_path"], "target": m["target"], "aspects": asp,
             "aspect_sources": src, "tags": dict(m["tags"], polarity="positive"), "gt_box_continuous": m["gt_box_continuous"]}
        if ref:
            r["reference"] = ref
        rows.append(r)
    kept = 0
    for (m, text), v in zip(negs, negv):
        if v != "no":
            continue
        kept += 1
        rows.append({"id": m["id"] + "-neg", "tier": m["tier"], "image_path": m["image_path"], "target": text,
                     "aspects": {"present": "no"}, "aspect_sources": {"present": f"gemini_confirmed:{args.model}"},
                     "tags": dict(m["tags"], polarity="negative")})
    with open(REPO / args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({len(man)} positive, {kept}/{len(negs)} negatives kept) to {args.out}")


def sheet(args):
    """Contact sheets of the first N positives (seeded order) with the box drawn, for hand labelling."""
    from PIL import Image, ImageDraw
    rows = [json.loads(l) for l in open(REPO / args.out) if '"positive"' in l]
    rng = random.Random(args.seed + 1)
    rng.shuffle(rows)
    rows = rows[: args.n]
    out = Path(args.sheet_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = 4
    for s in range(0, len(rows), per):
        tiles = []
        for r in rows[s: s + per]:
            im = overlay(r["image_path"], r["gt_box_continuous"])
            im.thumbnail((640, 640))
            canvas = Image.new("RGB", (640, 680), "white")
            canvas.paste(im, (0, 40))
            ImageDraw.Draw(canvas).text((6, 6), f'{r["id"]}: {r["target"][:70]}', fill="black")
            tiles.append(canvas)
        sheet_img = Image.new("RGB", (640 * 2, 680 * 2), "white")
        for i, t in enumerate(tiles):
            sheet_img.paste(t, ((i % 2) * 640, (i // 2) * 680))
        sheet_img.save(out / f"sheet_{s // per:02d}.png")
    json.dump([r["id"] for r in rows], open(out / "ids.json", "w"), indent=1)
    print(f"{len(rows)} items in {(len(rows) + per - 1) // per} sheets under {out}")


def handcheck(args):
    labels = json.load(open(args.labels))
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(REPO / args.out)}
    n = agree = 0
    conf = defaultdict(int)
    for rid, hand in labels["occluded"].items():
        g = rows[rid]["aspects"]["occluded"]
        n += 1
        agree += g == hand
        conf[f"hand={hand} gemini={g}"] += 1
    po = agree / n
    ph = sum(1 for v in labels["occluded"].values() if v == "yes") / n
    pg = sum(1 for k in labels["occluded"] if rows[k]["aspects"]["occluded"] == "yes") / n
    pe = ph * pg + (1 - ph) * (1 - pg)
    print(json.dumps({"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0,
                      "confusion": dict(conf)}, indent=1))


def masks(args):
    """Ground-truth instance masks for the RefCOCO items: COCO segmentation polygons in pixel coordinates."""
    import pyarrow.parquet as pq
    man = [json.loads(l) for l in open(REPO / "benchmarks/bbox_real.jsonl") if '"refcoco"' in l]
    want = {int(m["notes"].split("ref ")[1].split(",")[0]): m for m in man}
    rows = []
    for r in pq.read_table(CACHE / "validation-00000-of-00001-bfeafdc84ca37aa2.parquet").to_pylist():
        if r["ref_id"] in want:
            ann, info = json.loads(r["raw_anns"]), json.loads(r["raw_image_info"])
            seg = ann["segmentation"]
            if not isinstance(seg, list):  # RLE (crowd) annotations are not used
                continue
            rows.append({"id": want[r["ref_id"]]["id"], "width": info["width"], "height": info["height"],
                         "polygons": [[round(v, 1) for v in poly] for poly in seg]})
    with open(REPO / "benchmarks/bbox_real_masks.jsonl", "w") as f:
        for r in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} masks to benchmarks/bbox_real_masks.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "sheet", "handcheck", "masks"])
    ap.add_argument("labels", nargs="?")
    ap.add_argument("--project", default="")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="benchmarks/bbox_real_aspects.jsonl")
    ap.add_argument("--sheet-dir", default="/tmp/dgem-handcheck")
    args = ap.parse_args()
    {"build": build, "sheet": sheet, "handcheck": handcheck, "masks": masks}[args.cmd](args)


if __name__ == "__main__":
    main()
