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

"""
generate_bbox_fixtures.py — Synthetic Spatial Grounding & Occlusion Suite Generator

Generates:
  1. 12 synthetic 1000x1000 test images in `fixtures/bbox/*.png` (and `.svg` vector twins)
     with exact, noiseless pixel coordinates in [0.0, 100.0] percentage space.
  2. `benchmarks/bbox_suite.jsonl` containing the 12 stratified test items across:
       - Tier 1 (`subbin_precision`): Off-grid midpoints (e.g. 17.5%, 62.5%) to compare
         Discrete Argmax vs. Continuous Softmax Expectation (DFL / Integral Regression).
       - Tier 2 (`paired_occlusion`): Unoccluded vs. Single-Edge-Occluded twins to test
         whether Per-Edge Normalized Entropy (H_norm = H / ln(21)) isolates the exact
         occluded edge (`xmax` or `ymax`).
       - Tier 3 (`detr_and_negative`): Multi-object DETR set-prediction scenes and
         negative (`object_present: false`) controls for `depends_on` DAG gating.

Usage:
  python3 scripts/generate_bbox_fixtures.py
  python3 scripts/generate_bbox_fixtures.py --verify-math
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

BINS_5PCT = [f"{i:02d}" if i < 100 else "100" for i in range(0, 105, 5)]
BIN_VALUES = [float(i) for i in range(0, 105, 5)]  # 21 options: 0.0, 5.0, ..., 100.0


def nearest_5pct_bin(val: float) -> str:
    idx = int(round(val / 5.0))
    idx = max(0, min(len(BINS_5PCT) - 1, idx))
    return BINS_5PCT[idx]


def nearest_10pct_bin(val: float) -> str:
    idx = int(min(9, max(0, math.floor(val / 10.0))))
    return f"{idx * 10:02d}"


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute 2D Intersection-over-Union (IoU) for [ymin, xmin, ymax, xmax] in [0, 100]."""
    ya1, xa1, ya2, xa2 = box_a
    yb1, xb1, yb2, xb2 = box_b

    inter_ymin = max(ya1, yb1)
    inter_xmin = max(xa1, xb1)
    inter_ymax = min(ya2, yb2)
    inter_xmax = min(xa2, xb2)

    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w

    area_a = max(0.0, ya2 - ya1) * max(0.0, xa2 - xa1)
    area_b = max(0.0, yb2 - yb1) * max(0.0, xb2 - xb1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def softmax_expectation_and_entropy(prob_map: Dict[str, float]) -> Tuple[str, float, float]:
    """
    Given a probability distribution over the 21 coordinate bins ('00'..'100'),
    returns:
      - argmax_bin: discrete winner label (e.g. '15')
      - expected_coord: continuous softmax expectation sum_k v_k * p_k (e.g. 17.48)
      - normalized_entropy: H / ln(K) in [0.0, 1.0]
    """
    total_p = sum(prob_map.values()) or 1.0
    norm_probs = {k: v / total_p for k, v in prob_map.items()}

    argmax_bin = max(norm_probs.items(), key=lambda kv: kv[1])[0]
    expected_coord = sum(float(k) * p for k, p in norm_probs.items())

    raw_h = -sum(p * math.log(p) for p in norm_probs.values() if p > 1e-12)
    k_card = max(2, len(BINS_5PCT))
    norm_h = raw_h / math.log(k_card)
    return argmax_bin, expected_coord, norm_h


def write_minimal_png(
    path: Path,
    width: int,
    height: int,
    rects: List[Tuple[int, int, int, int, Tuple[int, int, int]]],
    bg_rgb: Tuple[int, int, int] = (248, 250, 252),
) -> None:
    """
    Write a valid RGB PNG image using pure Python stdlib (zlib + struct) or Pillow if installed.
    Downscales internal pixel buffer to 250x250 (1px = 0.4%) for fast pure-Python generation
    while preserving exact geometric ratios.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore

        img = Image.new("RGB", (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)
        # Draw subtle 10% coordinate grid lines
        for step in range(10, 100, 10):
            px = int(width * step / 100.0)
            py = int(height * step / 100.0)
            draw.line([(px, 0), (px, height)], fill=(226, 232, 240), width=1)
            draw.line([(0, py), (width, py)], fill=(226, 232, 240), width=1)
        for ymin_px, xmin_px, ymax_px, xmax_px, rgb in rects:
            draw.rectangle([xmin_px, ymin_px, xmax_px, ymax_px], fill=rgb, outline=(15, 23, 42), width=2)
        img.save(path, format="PNG")
        return
    except ImportError:
        pass

    # Pure-Python stdlib PNG fallback (250x250 resolution)
    scale = 4
    w, h = width // scale, height // scale
    scaled_rects = [
        (ymin // scale, xmin // scale, ymax // scale, xmax // scale, rgb)
        for (ymin, xmin, ymax, xmax, rgb) in rects
    ]
    rows = []
    for y in range(h):
        row = bytearray([0])  # Filter type 0 (None)
        for x in range(w):
            color = bg_rgb
            for r_ymin, r_xmin, r_ymax, r_xmax, rgb in scaled_rects:
                if r_ymin <= y <= r_ymax and r_xmin <= x <= r_xmax:
                    color = rgb
            row.extend(color)
        rows.append(bytes(row))

    raw_data = b"".join(rows)

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack("!I", len(data)) + chunk + struct.pack("!I", zlib.crc32(chunk) & 0xFFFFFFFF)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw_data, 6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)


def write_svg_twin(
    path: Path,
    width: int,
    height: int,
    rects: List[Tuple[int, int, int, int, Tuple[int, int, int], str]],
) -> None:
    """Write a crisp vector SVG companion showing the 5% grid lines, ruler ticks, and annotated rectangles."""
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <rect width="100%" height="100%" fill="#f8fafc"/>',
    ]
    for step in range(5, 100, 5):
        pos = int(width * step / 100.0)
        stroke = "#94a3b8" if step % 10 == 0 else "#e2e8f0"
        sw = "1.5" if step % 10 == 0 else "1"
        lines.append(f'  <line x1="{pos}" y1="0" x2="{pos}" y2="{height}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines.append(f'  <line x1="0" y1="{pos}" x2="{width}" y2="{pos}" stroke="{stroke}" stroke-width="{sw}"/>')
        if step % 10 == 0:
            lines.append(
                f'  <text x="{pos}" y="18" font-family="monospace" font-size="14" font-weight="bold" fill="#475569" text-anchor="middle">{step:02d}</text>'
            )
            lines.append(
                f'  <text x="18" y="{pos + 5}" font-family="monospace" font-size="14" font-weight="bold" fill="#475569" text-anchor="middle">{step:02d}</text>'
            )

    for ymin, xmin, ymax, xmax, rgb, label in rects:
        hex_c = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        w = max(1, xmax - xmin)
        h = max(1, ymax - ymin)
        lines.append(
            f'  <rect x="{xmin}" y="{ymin}" width="{w}" height="{h}" fill="{hex_c}" stroke="#0f172a" stroke-width="3" rx="6"/>'
        )
        cx = xmin + w // 2
        cy = ymin + h // 2
        lines.append(
            f'  <text x="{cx}" y="{cy}" font-family="monospace" font-size="20" font-weight="bold" fill="#ffffff" text-anchor="middle" dominant-baseline="middle">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # If rsvg-convert is installed, render the exact 1000x1000 SVG into the companion PNG
    import shutil
    import subprocess
    rsvg = shutil.which("rsvg-convert") or "rsvg-convert"
    if Path(rsvg).exists():
        png_out = path.with_suffix(".png")
        subprocess.run([rsvg, "-w", str(width), "-h", str(height), str(path), "-o", str(png_out)], check=False)


SUITE_SPEC: List[Dict[str, Any]] = [
    # --- TIER 1: SUB-BIN OFF-GRID PRECISION (5 cases) ---
    {
        "id": "bbox-t1-01-offgrid-cta",
        "tier": "subbin_precision",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "primary_cta_button",
        "object_present": True,
        "gt_box_continuous": [17.5, 32.5, 42.5, 67.5],
        "occluded_edge": "none",
        "notes": "All 4 edges sit exactly halfway between 5% bins (17.5% between 15 & 20; 32.5% between 30 & 35).",
    },
    {
        "id": "bbox-t1-02-offgrid-badge",
        "tier": "subbin_precision",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "status_alert_badge",
        "object_present": True,
        "gt_box_continuous": [12.5, 62.5, 27.5, 87.5],
        "occluded_edge": "none",
        "notes": "Upper-right alert badge centered on 2.5% half-steps.",
    },
    {
        "id": "bbox-t1-03-offgrid-card",
        "tier": "subbin_precision",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "checkout_summary_card",
        "object_present": True,
        "gt_box_continuous": [22.5, 12.5, 77.5, 52.5],
        "occluded_edge": "none",
        "notes": "Large left-pane card with off-grid boundaries.",
    },
    {
        "id": "bbox-t1-04-ongrid-control",
        "tier": "subbin_precision",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "primary_cta_button",
        "object_present": True,
        "gt_box_continuous": [20.0, 30.0, 45.0, 70.0],
        "occluded_edge": "none",
        "notes": "Exact 5%-grid-aligned baseline control (0 quantization error).",
    },
    {
        "id": "bbox-t1-05-micro-pill",
        "tier": "subbin_precision",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "notification_counter_pill",
        "object_present": True,
        "gt_box_continuous": [47.5, 47.5, 57.5, 57.5],
        "occluded_edge": "none",
        "notes": "Small 10%x10% target shifted by +2.5% off-grid where a 1-bin argmax snap loses 31% IoU.",
    },
    # --- TIER 2: PAIRED UNOCCLUDED VS. SINGLE-EDGE-OCCLUDED TWINS (4 cases) ---
    {
        "id": "bbox-t2-01a-unoccluded-twin",
        "tier": "paired_occlusion",
        "pair_id": "pair_right_occlusion",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "primary_cta_button",
        "object_present": True,
        "gt_box_continuous": [25.0, 20.0, 55.0, 65.0],
        "occluded_edge": "none",
        "notes": "Unoccluded baseline twin for pair_right_occlusion; all 4 edges sharply visible.",
    },
    {
        "id": "bbox-t2-01b-right-edge-occluded",
        "tier": "paired_occlusion",
        "pair_id": "pair_right_occlusion",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "primary_cta_button",
        "object_present": True,
        "gt_box_continuous": [25.0, 20.0, 55.0, 65.0],
        "occluder_box": [15.0, 55.0, 75.0, 85.0],
        "occluded_edge": "xmax",
        "notes": "Identical target to t2-01a, but a gray modal overlay obscures the right border (xmax) from 55% to 85%.",
    },
    {
        "id": "bbox-t2-02a-unoccluded-twin",
        "tier": "paired_occlusion",
        "pair_id": "pair_bottom_occlusion",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "warning_modal_banner",
        "object_present": True,
        "gt_box_continuous": [30.0, 25.0, 70.0, 75.0],
        "occluded_edge": "none",
        "notes": "Unoccluded baseline twin for pair_bottom_occlusion; all 4 edges sharply visible.",
    },
    {
        "id": "bbox-t2-02b-bottom-edge-occluded",
        "tier": "paired_occlusion",
        "pair_id": "pair_bottom_occlusion",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "warning_modal_banner",
        "object_present": True,
        "gt_box_continuous": [30.0, 25.0, 70.0, 75.0],
        "occluder_box": [60.0, 15.0, 90.0, 85.0],
        "occluded_edge": "ymax",
        "notes": "Identical target to t2-02a, but a bottom cookie-sheet drawer obscures the bottom border (ymax) from 60% to 90%.",
    },
    # --- TIER 3: MULTI-OBJECT DETR QUERIES & NEGATIVE GATE CONTROLS (3 cases) ---
    {
        "id": "bbox-t3-01-detr-dual-buttons",
        "tier": "detr_and_negative",
        "template": "templates/multimodal/bbox_multi_object_detr.json.tmpl",
        "target": "dual_dialog_controls",
        "object_present": True,
        "gt_box_continuous": [60.0, 15.0, 78.0, 45.0],
        "secondary_box_continuous": [60.0, 55.0, 78.0, 85.0],
        "secondary_class": "secondary_cancel_button",
        "occluded_edge": "none",
        "notes": "Two side-by-side buttons testing parallel DETR Object Query 1 (left CTA) and Object Query 2 (right Cancel).",
    },
    {
        "id": "bbox-t3-02-detr-stacked-banner-cta",
        "tier": "detr_and_negative",
        "template": "templates/multimodal/bbox_multi_object_detr.json.tmpl",
        "target": "banner_and_cta",
        "object_present": True,
        "gt_box_continuous": [12.0, 20.0, 28.0, 80.0],
        "secondary_box_continuous": [52.0, 30.0, 68.0, 70.0],
        "secondary_class": "primary_cta_button",
        "occluded_edge": "none",
        "notes": "Top warning banner (Query 1) + bottom primary CTA button (Query 2) resolved in 1 forward pass.",
    },
    {
        "id": "bbox-t3-03-absent-target-gate",
        "tier": "detr_and_negative",
        "template": "templates/multimodal/bbox_localization.json.tmpl",
        "target": "primary_cta_button",
        "object_present": False,
        "gt_box_continuous": [0.0, 0.0, 0.0, 0.0],
        "occluded_edge": "none",
        "notes": "Blank canvas with no target object; tests `object_present: false` conditional DAG gate pruning.",
    },
]


def build_suite_and_images(repo_root: Path) -> List[Dict[str, Any]]:
    fixtures_dir = repo_root / "fixtures" / "bbox"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir = repo_root / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    width, height = 1000, 1000

    for item in SUITE_SPEC:
        case_id = item["id"]
        png_rel = f"fixtures/bbox/{case_id}.png"
        svg_rel = f"fixtures/bbox/{case_id}.svg"
        png_path = repo_root / png_rel
        svg_path = repo_root / svg_rel

        rects_png: List[Tuple[int, int, int, int, Tuple[int, int, int]]] = []
        rects_svg: List[Tuple[int, int, int, int, Tuple[int, int, int], str]] = []

        if item["object_present"]:
            ymin, xmin, ymax, xmax = item["gt_box_continuous"]
            r_coords = (
                int(ymin * 10),
                int(xmin * 10),
                int(ymax * 10),
                int(xmax * 10),
            )
            primary_rgb = (37, 99, 235)  # Vibrant royal blue (#2563eb)
            rects_png.append((*r_coords, primary_rgb))
            rects_svg.append((*r_coords, primary_rgb, item["target"]))

        if "secondary_box_continuous" in item:
            s_ymin, s_xmin, s_ymax, s_xmax = item["secondary_box_continuous"]
            s_coords = (
                int(s_ymin * 10),
                int(s_xmin * 10),
                int(s_ymax * 10),
                int(s_xmax * 10),
            )
            sec_rgb = (16, 185, 129)  # Emerald green (#10b981)
            rects_png.append((*s_coords, sec_rgb))
            rects_svg.append((*s_coords, sec_rgb, item.get("secondary_class", "obj2")))

        if "occluder_box" in item:
            o_ymin, o_xmin, o_ymax, o_xmax = item["occluder_box"]
            o_coords = (
                int(o_ymin * 10),
                int(o_xmin * 10),
                int(o_ymax * 10),
                int(o_xmax * 10),
            )
            occ_rgb = (100, 116, 139)  # Slate gray occluder (#64748b)
            rects_png.append((*o_coords, occ_rgb))
            rects_svg.append((*o_coords, occ_rgb, f"OCCLUDER ({item['occluded_edge']})"))

        write_minimal_png(png_path, width, height, rects_png)
        write_svg_twin(svg_path, width, height, rects_svg)

        ymin, xmin, ymax, xmax = item["gt_box_continuous"]
        expected_bins = {
            "object_present": item["object_present"],
            "ymin": nearest_5pct_bin(ymin) if item["object_present"] else None,
            "xmin": nearest_5pct_bin(xmin) if item["object_present"] else None,
            "ymax": nearest_5pct_bin(ymax) if item["object_present"] else None,
            "xmax": nearest_5pct_bin(xmax) if item["object_present"] else None,
        }
        if "secondary_box_continuous" in item:
            sy1, sx1, sy2, sx2 = item["secondary_box_continuous"]
            expected_bins = {
                "obj1_class": "warning_modal_banner" if "banner" in item["target"] else "primary_cta_button",
                "obj1_ymin": nearest_10pct_bin(ymin),
                "obj1_xmin": nearest_10pct_bin(xmin),
                "obj1_ymax": nearest_10pct_bin(ymax),
                "obj1_xmax": nearest_10pct_bin(xmax),
                "obj2_class": item["secondary_class"],
                "obj2_ymin": nearest_10pct_bin(sy1),
                "obj2_xmin": nearest_10pct_bin(sx1),
                "obj2_ymax": nearest_10pct_bin(sy2),
                "obj2_xmax": nearest_10pct_bin(sx2),
            }

        record = {
            "id": case_id,
            "tier": item["tier"],
            "pair_id": item.get("pair_id"),
            "template": item["template"],
            "image_path": png_rel,
            "svg_path": svg_rel,
            "target": item["target"],
            "object_present": item["object_present"],
            "gt_box_continuous": item["gt_box_continuous"],
            "secondary_box_continuous": item.get("secondary_box_continuous"),
            "occluded_edge": item["occluded_edge"],
            "expected_discrete_slots": expected_bins,
            "notes": item["notes"],
        }
        records.append(record)

    suite_path = benchmarks_dir / "bbox_suite.jsonl"
    with suite_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return records


def run_math_verification(records: List[Dict[str, Any]]) -> None:
    """
    Demonstrates why Softmax Expectation (DFL / Integral Regression) outperforms
    Discrete Argmax on Tier-1 off-grid coordinates, and how Per-Edge Normalized
    Entropy isolates occluded boundaries on Tier-2 paired twins.
    """
    print("\n=== Mathematical Verification: Argmax vs. Softmax Expectation ===")
    t1_records = [r for r in records if r["tier"] == "subbin_precision"]
    argmax_ious: List[float] = []
    expect_ious: List[float] = []

    for r in t1_records:
        gt = r["gt_box_continuous"]
        argmax_box: List[float] = []
        expect_box: List[float] = []
        for coord in gt:
            # Construct a realistic bimodal Gaussian-like logprob distribution centered near `coord`
            probs = {
                b: math.exp(-((float(b) - coord) ** 2) / (2.0 * (2.6**2)))
                for b in BINS_5PCT
            }
            # Slight asymmetry break so argmax snaps to nearest 5% bin
            nearest_b = nearest_5pct_bin(coord)
            probs[nearest_b] *= 1.08
            top_b, exp_val, _ = softmax_expectation_and_entropy(probs)
            argmax_box.append(float(top_b))
            expect_box.append(exp_val)

        iou_arg = compute_iou(gt, argmax_box)
        iou_exp = compute_iou(gt, expect_box)
        argmax_ious.append(iou_arg)
        expect_ious.append(iou_exp)
        print(
            f"  {r['id']:26s} | GT={gt} | Argmax IoU={iou_arg:.4f} | Softmax-Expectation IoU={iou_exp:.4f} (+{(iou_exp - iou_arg)*100:.1f}%)"
        )

    print(
        f"  --> Tier-1 Mean IoU (mIoU): Discrete Argmax = {sum(argmax_ious)/len(argmax_ious):.4f} vs. Softmax Expectation = {sum(expect_ious)/len(expect_ious):.4f}"
    )

    print("\n=== Per-Edge Normalized Entropy (H_norm = H / ln 21) on Occluded Twins ===")
    for edge_name, is_occ in [("ymin (visible)", False), ("xmin (visible)", False), ("ymax (visible)", False), ("xmax (OCCLUDED)", True)]:
        if not is_occ:
            probs = {b: (0.88 if b == "25" else 0.06 if b in ("20", "30") else 1e-4) for b in BINS_5PCT}
        else:
            # Spread probability across occluded region 55%..80%
            probs = {b: (0.16 if b in ("55", "60", "65", "70", "75", "80") else 0.002) for b in BINS_5PCT}
        _, exp_c, h_norm = softmax_expectation_and_entropy(probs)
        flag = "🚨 ESCALATE / OCCLUDED EDGE" if h_norm >= 0.35 else "✅ SHARP VISIBLE EDGE"
        print(f"  Slot {edge_name:17s} | E[coord]={exp_c:5.1f}% | H_norm={h_norm:.4f} | {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bounding-box fixtures and JSONL suite.")
    parser.add_argument("--verify-math", action="store_true", help="Run offline IoU and Per-Edge Entropy verification")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    records = build_suite_and_images(repo_root)
    print(f"Generated {len(records)} synthetic PNG+SVG fixtures in {repo_root / 'fixtures' / 'bbox'}")
    print(f"Wrote benchmark dataset to {repo_root / 'benchmarks' / 'bbox_suite.jsonl'}")

    if args.verify_math:
        run_math_verification(records)


if __name__ == "__main__":
    main()
