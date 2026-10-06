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

"""Generate the synthetic bounding-box sweep (seeded, deterministic).

Each set varies one factor while the rest stay fixed, so effects can be read directly:

  geometry     aspect ratio (1:1, 16:9, 9:16, 4:3) x 10% grid lines (on/off), random placement,
               size and 0-2 distractors. The original fixtures all carry the grid.
  occlusion    10 placements x occluder coverage of one edge (0, 25, 50, 75% of the target's extent).
               The same placement repeats at every coverage, so each placement is its own control.
               gt_box_continuous is the full (amodal) box; visible_box_continuous is what is visible.
  degradation  10 placements x none / blur / noise / JPEG / low contrast.
  absent       scenes with distractors only; the requested target type is not drawn.

Output: fixtures/bbox_sweep/*.png and benchmarks/bbox_sweep.jsonl (bench-bbox suite format, with "tags").

  pip install -r scripts/requirements-vision.txt
  python3 scripts/generate_bbox_sweep.py [--seed 18] [--out-dir fixtures/bbox_sweep]
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = "templates/multimodal/bbox_localization.json.tmpl"
BG = (248, 250, 252)
ASPECTS = {"1x1": (1000, 1000), "16x9": (1600, 900), "9x16": (900, 1600), "4x3": (1200, 900)}

# type -> (width range %, height range % of the shorter side, label texts)
TYPES = {
    "primary_cta_button": ((10, 35), (5, 10), ["Checkout", "Submit", "Continue", "Pay now"]),
    "secondary_cancel_button": ((10, 30), (5, 10), ["Cancel", "Back", "Not now"]),
    "status_alert_badge": ((8, 18), (4, 7), ["3 alerts", "New", "12"]),
    "search_input_field": ((25, 60), (5, 9), ["Search...", "Find a product"]),
    "user_avatar_icon": ((6, 14), None, [""]),
    "info_card": ((25, 55), (18, 40), ["Order summary", "Shipping", "Account"]),
}


def font(size: int):
    try:
        return ImageFont.load_default(size=max(10, size))
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_element(d: ImageDraw.ImageDraw, kind: str, box_px, text: str):
    x0, y0, x1, y1 = box_px
    h = y1 - y0
    f = font(int(h * 0.45) if kind != "info_card" else int(min(h * 0.12, (x1 - x0) * 0.08)))
    if kind == "primary_cta_button":
        d.rounded_rectangle(box_px, radius=int(h * 0.25), fill=(37, 99, 235))
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, fill="white", font=f, anchor="mm")
    elif kind == "secondary_cancel_button":
        d.rounded_rectangle(box_px, radius=int(h * 0.25), fill=(255, 255, 255), outline=(100, 116, 139), width=max(2, h // 15))
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, fill=(51, 65, 85), font=f, anchor="mm")
    elif kind == "status_alert_badge":
        d.rounded_rectangle(box_px, radius=h // 2, fill=(220, 38, 38))
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, fill="white", font=f, anchor="mm")
    elif kind == "search_input_field":
        d.rectangle(box_px, fill=(255, 255, 255), outline=(148, 163, 184), width=max(2, h // 15))
        d.text((x0 + h * 0.4, (y0 + y1) / 2), text, fill=(148, 163, 184), font=f, anchor="lm")
    elif kind == "user_avatar_icon":
        d.ellipse(box_px, fill=(16, 185, 129))
        d.ellipse((x0 + (x1 - x0) * 0.33, y0 + h * 0.18, x1 - (x1 - x0) * 0.33, y0 + h * 0.5), fill=(236, 253, 245))
    elif kind == "info_card":
        d.rounded_rectangle(box_px, radius=max(4, h // 20), fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        pad = (x1 - x0) * 0.06
        d.text((x0 + pad, y0 + pad), text, fill=(15, 23, 42), font=f)
        for i in range(3):
            ly = y0 + h * (0.42 + 0.17 * i)
            d.rectangle((x0 + pad, ly, x1 - pad - (i * (x1 - x0) * 0.15), ly + max(3, h * 0.05)), fill=(226, 232, 240))


def sample_box(rng: random.Random, kind: str, W: int, H: int):
    """A box in pixels (x0, y0, x1, y1) for an element of this kind, fully inside the image."""
    wr, hr, _ = TYPES[kind]
    short = min(W, H)
    w = rng.uniform(*wr) / 100 * W
    if hr is None:  # circle: square in pixels
        w = rng.uniform(*wr) / 100 * short
        h = w
    else:
        h = rng.uniform(*hr) / 100 * short
        if kind == "info_card":
            h = rng.uniform(*hr) / 100 * H
    w, h = min(w, W * 0.9), min(h, H * 0.9)
    m = 0.02
    x0 = rng.uniform(W * m, W * (1 - m) - w)
    y0 = rng.uniform(H * m, H * (1 - m) - h)
    return (round(x0), round(y0), round(x0 + w), round(y0 + h))


def overlaps(a, b, margin=12):
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def place(rng, kind, W, H, taken, tries=200):
    for _ in range(tries):
        b = sample_box(rng, kind, W, H)
        if not any(overlaps(b, t) for t in taken):
            return b
    return None


def base_canvas(W, H, grid: bool):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if grid:
        for step in range(10, 100, 10):
            px, py = int(W * step / 100), int(H * step / 100)
            d.line([(px, 0), (px, H)], fill=(226, 232, 240), width=1)
            d.line([(0, py), (W, py)], fill=(226, 232, 240), width=1)
    return img, d


def pct_box(b, W, H):
    x0, y0, x1, y1 = b
    return [round(100 * y0 / H, 3), round(100 * x0 / W, 3), round(100 * y1 / H, 3), round(100 * x1 / W, 3)]


def scene(rng, W, H, grid, target_kind, n_distractors, draw_target=True):
    img, d = base_canvas(W, H, grid)
    taken = []
    tbox = place(rng, target_kind, W, H, taken) if draw_target else None
    if tbox:
        taken.append(tbox)
    others = [k for k in TYPES if k != target_kind]
    distractors = []
    for _ in range(n_distractors):
        k = rng.choice(others)
        b = place(rng, k, W, H, taken)
        if b:
            taken.append(b)
            distractors.append((k, b))
    for k, b in distractors:
        draw_element(d, k, b, rng.choice(TYPES[k][2]))
    if tbox:
        draw_element(d, target_kind, tbox, rng.choice(TYPES[target_kind][2]))
    LAST_DISTRACTORS[:] = distractors  # read by aspects(); consumes no randomness
    return img, d, tbox, [k for k, _ in distractors]


LAST_DISTRACTORS: list = []


def aspects(tbox, W, H, n_elements, quality=None, occluded=None):
    """Ground-truth answers for the bench-vision questions; None = not scored."""
    a = {"present": "yes" if tbox else "no", "element_count": str(min(n_elements, 4))}
    ref = None
    if tbox:
        cx, cy = (tbox[0] + tbox[2]) / 2 / W, (tbox[1] + tbox[3]) / 2 / H
        a["grid_cell"] = ["top", "middle", "bottom"][min(2, int(cy * 3))] + "_" + ["left", "center", "right"][min(2, int(cx * 3))]
        if LAST_DISTRACTORS:
            k, b = LAST_DISTRACTORS[0]
            dx = (tbox[0] + tbox[2]) / 2 - (b[0] + b[2]) / 2
            dy = (tbox[1] + tbox[3]) / 2 - (b[1] + b[3]) / 2
            # Relation only when one axis clearly dominates.
            if abs(dx) > 1.5 * abs(dy):
                a["relation"] = "left_of" if dx < 0 else "right_of"
            elif abs(dy) > 1.5 * abs(dx):
                a["relation"] = "above" if dy < 0 else "below"
            ref = k
    if quality is not None:
        a["image_quality"] = quality
    if occluded is not None:
        a["occluded"] = "yes" if occluded else "no"
    return a, ref


def occlude(d, tbox, edge, coverage, W, H):
    """Opaque panel hiding `coverage` of the target's extent from `edge`, running past the edge to the
    image border. Returns the visible box in pixels."""
    x0, y0, x1, y1 = tbox
    if coverage <= 0:
        return tbox
    color = (100, 116, 139)
    padx, pady = (x1 - x0) * 0.4, (y1 - y0) * 0.4
    if edge == "xmax":
        cut = x1 - coverage * (x1 - x0)
        d.rectangle((cut, y0 - pady, W, y1 + pady), fill=color)
        return (x0, y0, cut, y1)
    if edge == "xmin":
        cut = x0 + coverage * (x1 - x0)
        d.rectangle((0, y0 - pady, cut, y1 + pady), fill=color)
        return (cut, y0, x1, y1)
    if edge == "ymax":
        cut = y1 - coverage * (y1 - y0)
        d.rectangle((x0 - padx, cut, x1 + padx, H), fill=color)
        return (x0, y0, x1, cut)
    cut = y0 + coverage * (y1 - y0)
    d.rectangle((x0 - padx, 0, x1 + padx, cut), fill=color)
    return (x0, cut, x1, y1)


def degrade(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    short = min(img.size)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=short / 160))
    if kind == "noise":
        a = np.asarray(img).astype(np.float32)
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        a += nrng.normal(0, 40, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=8)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "low_contrast":
        return Image.blend(img, Image.new("RGB", img.size, (200, 200, 200)), 0.8)
    return img


def item(iid, set_name, img_rel, target, present, gt, tags, occluded_edge="none", visible=None, notes="", aspect=None):
    row = {
        "id": iid, "tier": set_name, "template": TEMPLATE, "image_path": img_rel, "target": target,
        "object_present": present, "gt_box_continuous": gt if present else [0.0, 0.0, 0.0, 0.0],
        "occluded_edge": occluded_edge, "tags": {k: str(v) for k, v in tags.items()}, "notes": notes,
    }
    if visible:
        row["visible_box_continuous"] = visible
    if aspect:
        row["aspects"], ref = aspect
        if ref:
            row["reference"] = ref
    return row


def size_bucket(gt):
    area = (gt[2] - gt[0]) * (gt[3] - gt[1])  # percent x percent: a 10% x 10% box is 100
    return "small" if area < 75 else ("medium" if area < 400 else "large")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--out-dir", default="fixtures/bbox_sweep")
    ap.add_argument("--manifest", default="benchmarks/bbox_sweep.jsonl")
    ap.add_argument("--geometry-per-cell", type=int, default=15)
    ap.add_argument("--placements", type=int, default=10)
    ap.add_argument("--absent", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(img, name):
        p = out / f"{name}.png"
        img.save(p, format="PNG", optimize=True)
        return str(p.relative_to(REPO))

    kinds = list(TYPES)
    # Geometry: aspect x grid.
    for aspect, (W, H) in ASPECTS.items():
        for grid in (True, False):
            for i in range(args.geometry_per_cell):
                kind = rng.choice(kinds)
                nd = rng.randint(0, 2)
                img, _, tbox, dist = scene(rng, W, H, grid, kind, nd)
                gt = pct_box(tbox, W, H)
                name = f"geo-{aspect}-{'grid' if grid else 'plain'}-{i:02d}"
                rows.append(item(name, "geometry", save(img, name), kind, True, gt,
                                 {"set": "geometry", "aspect": aspect, "grid": grid, "type": kind,
                                  "distractors": len(dist), "size": size_bucket(gt)},
                                 aspect=aspects(tbox, W, H, 1 + len(dist))))

    # Occlusion dose-response: the same placement at every coverage.
    W, H = ASPECTS["1x1"]
    for p in range(args.placements):
        kind = rng.choice([k for k in kinds if k != "user_avatar_icon"])
        edge = rng.choice(["ymin", "xmin", "ymax", "xmax"])
        nd = rng.randint(0, 1)
        state = rng.getstate()
        for cov in (0.0, 0.25, 0.5, 0.75):
            rng.setstate(state)  # identical scene at every coverage
            img, d, tbox, dist = scene(rng, W, H, False, kind, nd)
            vis = occlude(d, tbox, edge, cov, W, H)
            gt = pct_box(tbox, W, H)
            name = f"occ-{p:02d}-{edge}-{int(cov * 100):02d}"
            rows.append(item(name, "occlusion", save(img, name), kind, True, gt,
                             {"set": "occlusion", "placement": p, "coverage": int(cov * 100), "edge": edge, "type": kind},
                             occluded_edge=edge if cov > 0 else "none", visible=pct_box(vis, W, H),
                             notes=f"pair_id groups the coverages of one placement",
                             aspect=aspects(tbox, W, H, 1 + len(dist), occluded=cov > 0)).__or__(
                                 {"pair_id": f"occ-{p:02d}"} if cov in (0.0, 0.5) else {}))
        rng.setstate(state)
        rng.random()

    # Degradation: the same placement at every level.
    for p in range(args.placements):
        kind = rng.choice(kinds)
        nd = rng.randint(0, 2)
        state = rng.getstate()
        for level in ("none", "blur", "noise", "jpeg", "low_contrast"):
            rng.setstate(state)
            img, _, tbox, dist = scene(rng, W, H, False, kind, nd)
            img = degrade(img, level, random.Random(args.seed * 1000 + p))
            gt = pct_box(tbox, W, H)
            name = f"deg-{p:02d}-{level}"
            quality = {"none": "clean", "blur": "blurred", "noise": "noisy", "jpeg": "compressed", "low_contrast": "low_contrast"}[level]
            rows.append(item(name, "degradation", save(img, name), kind, True, gt,
                             {"set": "degradation", "placement": p, "degradation": level, "type": kind},
                             aspect=aspects(tbox, W, H, 1 + len(dist), quality=quality)))
        rng.setstate(state)
        rng.random()

    # Absent targets: distractors only.
    for i in range(args.absent):
        aspect = rng.choice(list(ASPECTS))
        W, H = ASPECTS[aspect]
        kind = rng.choice(kinds)
        img, _, _, dist = scene(rng, W, H, False, kind, rng.randint(1, 3), draw_target=False)
        name = f"abs-{i:02d}"
        rows.append(item(name, "absent", save(img, name), kind, False, None,
                         {"set": "absent", "aspect": aspect, "type": kind, "distractors": len(dist)},
                         aspect=aspects(None, W, H, len(dist))))

    man = REPO / args.manifest
    with open(man, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items to {man.relative_to(REPO)} and {out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
