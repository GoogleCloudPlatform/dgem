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

"""Experiment D2: Set-of-Marks (`az_aa`, <= 52) conformal shortlist cascade."""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import exp_stats_common as stats_lib  # noqa: E402

TRACK_NAME: str = "d2_som"
DEFAULT_LOCKFILE: pathlib.Path = (
    REPO_ROOT / "benchmarks" / "matrix" / "locks" / "dgem_exp_splits_v1.json"
)
TOTAL_APP_DOMAINS: int = 80
ITEMS_PER_APP_DOMAIN: int = 30
GRADUATION_ACCURACY_BAR: float = 0.90
TANGO_NON_INFERIORITY_MARGIN: float = 0.010
MAX_SHORTLIST_MARKS: int = 4


def build_az_aa_labels(count: int = 52) -> tuple[str, ...]:
  """Generate `az_aa` option labels (`A..Z`, `AA..AZ`) up to 52 marks.

  Args:
    count: Number of Set-of-Marks identifiers (`1 <= count <= 52`).

  Returns:
    Tuple of `count` uppercase `az_aa` codes.

  Raises:
    stats_lib.StatisticalValidationError: If `count` lies outside `[1, 52]`.
  """
  if not 1 <= count <= 52:
    raise stats_lib.StatisticalValidationError(
        f"az_aa count must lie in [1, 52], got {count}"
    )
  alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  labels = list(alphabet) + [f"A{letter}" for letter in alphabet]
  return tuple(labels[:count])


AZ_AA_52: tuple[str, ...] = build_az_aa_labels(52)


@dataclass(frozen=True)
class CandidateBoxMark:
  """Single interactive UI candidate mark in [0, 100] normalized coordinates."""

  code: str
  box: tuple[float, float, float, float]
  role_label: str


@dataclass(frozen=True)
class D2SoMCase:
  """Set-of-Marks UI localization benchmark case."""

  item_id: str
  app_domain: str
  target_instruction: str
  num_marks: int
  marks: tuple[CandidateBoxMark, ...]
  gold_code: str
  gold_box: tuple[float, float, float, float]
  provenance: str
  split_item: stats_lib.BenchmarkSplitItem


def compute_box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
  """Compute 2D IoU over `[y_min, x_min, y_max, x_max]` boxes in `[0, 100]`.

  Args:
    box_a: First `[y_min, x_min, y_max, x_max]` bounding box.
    box_b: Second `[y_min, x_min, y_max, x_max]` bounding box.

  Returns:
    Intersection-over-union in `[0.0, 1.0]`.

  Raises:
    ValueError: Never raised; inverted boxes return `0.0` area.
  """
  inter_h = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
  inter_w = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
  inter_area = inter_h * inter_w
  area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
  area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
  union_area = area_a + area_b - inter_area
  if union_area <= 0.0:
    return 0.0
  return inter_area / union_area


def is_localization_hit(
    predicted_box: tuple[float, float, float, float],
    gold_box: tuple[float, float, float, float],
) -> bool:
  """Evaluate UI grounding success (`IoU >= 0.5` or predicted center in box).

  Args:
    predicted_box: Selected candidate `[y_min, x_min, y_max, x_max]`.
    gold_box: Ground-truth target `[y_min, x_min, y_max, x_max]`.

  Returns:
    True when `IoU >= 0.5` or the center of `predicted_box` falls inside
    `gold_box`.

  Raises:
    ValueError: Never raised.
  """
  if compute_box_iou(predicted_box, gold_box) >= 0.5:
    return True
  center_y = 0.5 * (predicted_box[0] + predicted_box[2])
  center_x = 0.5 * (predicted_box[1] + predicted_box[3])
  return (
      gold_box[0] <= center_y <= gold_box[2]
      and gold_box[1] <= center_x <= gold_box[3]
  )


def _load_bbox_seed_records() -> list[dict[str, object]]:
  """Load committed bounding-box benchmark rows from `benchmarks/` and `third_party/`."""
  sources = (
      REPO_ROOT / "benchmarks" / "bbox_suite.jsonl",
      REPO_ROOT / "benchmarks" / "bbox_sweep.jsonl",
      REPO_ROOT / "third_party" / "bbox_real" / "bbox_real.jsonl",
  )
  rows: list[dict[str, object]] = []
  for src_path in sources:
    if not src_path.is_file():
      continue
    rel_name = str(src_path.relative_to(REPO_ROOT))
    for raw_line in src_path.read_text(encoding="utf-8").splitlines():
      stripped = raw_line.strip()
      if not stripped:
        continue
      parsed = json.loads(stripped)
      if not parsed.get("object_present", True):
        continue
      gt_box = parsed.get("gt_box_continuous")
      if not isinstance(gt_box, list) or len(gt_box) != 4:
        continue
      rows.append({
          "source": rel_name,
          "seed_id": str(parsed.get("id", "seed")),
          "target": str(parsed.get("target", "primary_button")),
          "gt_box": tuple(float(coord) for coord in gt_box),
          "image_sha256": str(parsed.get("image_sha256", "")),
      })
  return rows


def _synthesize_screen_marks(
    item_id: str,
    gold_box: tuple[float, float, float, float],
    num_marks: int,
    gold_idx: int,
) -> tuple[CandidateBoxMark, ...]:
  """Place `num_marks` non-identical UI candidate boxes including `gold_box`."""
  codes = build_az_aa_labels(num_marks)
  roles = (
      "button",
      "icon_button",
      "tab_link",
      "menu_trigger",
      "filter_chip",
      "input_field",
      "card_action",
      "toolbar_toggle",
  )
  marks: list[CandidateBoxMark] = []
  for idx in range(num_marks):
    code = codes[idx]
    role_name = roles[
        stats_lib.hmac_uint64("dgem-d2-role", item_id, code) % len(roles)
    ]
    if idx == gold_idx:
      marks.append(
          CandidateBoxMark(code=code, box=gold_box, role_label=role_name)
      )
      continue
    # Place realistic distractor controls including 2 nearby spatial neighbors.
    if idx in ((gold_idx + 1) % num_marks, (gold_idx + 2) % num_marks):
      shift_y = (10.0 if idx % 2 == 0 else -10.0)
      shift_x = (14.0 if idx % 2 == 1 else -14.0)
      y_min = max(1.0, min(86.0, gold_box[0] + shift_y))
      x_min = max(1.0, min(84.0, gold_box[1] + shift_x))
      height = max(5.0, min(14.0, gold_box[2] - gold_box[0]))
      width = max(7.0, min(18.0, gold_box[3] - gold_box[1]))
      cand_box = (
          round(y_min, 2),
          round(x_min, 2),
          round(min(99.0, y_min + height), 2),
          round(min(99.0, x_min + width), 2),
      )
    else:
      grid_r = idx % 6
      grid_c = (idx // 6) % 9
      jitter_y = 3.0 * stats_lib.hmac_uniform_float64(
          "dgem-d2-box", item_id, f"y:{idx}"
      )
      jitter_x = 2.5 * stats_lib.hmac_uniform_float64(
          "dgem-d2-box", item_id, f"x:{idx}"
      )
      y_min = round(min(89.0, 4.0 + grid_r * 15.0 + jitter_y), 2)
      x_min = round(min(88.0, 3.0 + grid_c * 10.5 + jitter_x), 2)
      cand_box = (y_min, x_min, round(y_min + 7.5, 2), round(x_min + 9.0, 2))
      if compute_box_iou(cand_box, gold_box) >= 0.35:
        shifted_y = round((y_min + 45.0) % 85.0, 2)
        cand_box = (
            shifted_y,
            x_min,
            round(shifted_y + 7.5, 2),
            round(x_min + 9.0, 2),
        )
    marks.append(
        CandidateBoxMark(code=code, box=cand_box, role_label=role_name)
    )
  return tuple(marks)


def build_d2_corpus() -> list[D2SoMCase]:
  """Construct the 2,400-case (`80` app_domains x `30`) Set-of-Marks corpus.

  Returns:
    Deterministic list of `2,400` `D2SoMCase` items with unique SHA-256 hashes.

  Raises:
    stats_lib.SplitLeakageError: If content hash computation fails.
  """
  seed_rows = _load_bbox_seed_records()
  mark_counts = (18, 24, 32, 40, 46, 52)
  corpus: list[D2SoMCase] = []
  seed_idx = 0

  for app_idx in range(TOTAL_APP_DOMAINS):
    app_domain = f"app_domain_{app_idx:02d}"
    for local_idx in range(ITEMS_PER_APP_DOMAIN):
      global_idx = app_idx * ITEMS_PER_APP_DOMAIN + local_idx
      item_id = f"d2-{app_idx:02d}-{local_idx:02d}"
      num_marks = mark_counts[(app_idx + local_idx) % len(mark_counts)]
      gold_idx = int(
          stats_lib.hmac_uint64("dgem-d2-gold", item_id, "idx") % num_marks
      )

      if seed_idx < len(seed_rows):
        seed_entry = seed_rows[seed_idx]
        seed_idx += 1
        raw_box = seed_entry["gt_box"]
        gold_box = (
            float(raw_box[0]),  # type: ignore[index]
            float(raw_box[1]),  # type: ignore[index]
            float(raw_box[2]),  # type: ignore[index]
            float(raw_box[3]),  # type: ignore[index]
        )
        target_instruction = (
            f"[{app_domain}] ({seed_entry['source']}:{seed_entry['seed_id']}) "
            f"Locate UI control '{seed_entry['target']}' among {num_marks} "
            f"az_aa marks (case #{global_idx})."
        )
        media_hash = str(seed_entry["image_sha256"]) or hashlib.sha256(
            f"{item_id}:{gold_box}".encode("utf-8")
        ).hexdigest()
        provenance = "repo_bbox_seed"
      else:
        base_y = round(
            8.0
            + 72.0
            * stats_lib.hmac_uniform_float64("dgem-d2-gt", item_id, "ymin"),
            2,
        )
        base_x = round(
            6.0
            + 74.0
            * stats_lib.hmac_uniform_float64("dgem-d2-gt", item_id, "xmin"),
            2,
        )
        gold_box = (
            base_y,
            base_x,
            round(min(98.0, base_y + 9.5), 2),
            round(min(98.0, base_x + 14.0), 2),
        )
        control_names = (
            "save_changes_cta",
            "export_csv_icon",
            "billing_period_dropdown",
            "revoke_token_action",
            "confirm_deploy_button",
            "audit_log_filter",
        )
        target_label = control_names[(app_idx + local_idx) % len(control_names)]
        target_instruction = (
            f"[{app_domain}] Locate interactive element '{target_label}' in "
            f"viewport #{global_idx} across {num_marks} az_aa badges."
        )
        media_hash = stats_lib.compute_content_sha256(
            f"screen:{app_domain}:{global_idx}:{gold_box}"
        )
        provenance = "deterministic_ui_som_expansion"

      marks = _synthesize_screen_marks(item_id, gold_box, num_marks, gold_idx)
      gold_code = marks[gold_idx].code
      content_sha = stats_lib.compute_content_sha256(
          target_instruction, media_digest=media_hash
      )
      split_item = stats_lib.BenchmarkSplitItem(
          item_id=item_id,
          cluster_id=app_domain,
          content_text=target_instruction,
          content_sha256=content_sha,
      )
      corpus.append(
          D2SoMCase(
              item_id=item_id,
              app_domain=app_domain,
              target_instruction=target_instruction,
              num_marks=num_marks,
              marks=marks,
              gold_code=gold_code,
              gold_box=gold_box,
              provenance=provenance,
              split_item=split_item,
          )
      )
  return corpus


def _gaussian_draw(seed_key: str, item_key: str, tag: str) -> float:
  """Generate deterministic standard normal variate via Box-Muller."""
  u_one = max(
      1e-12, stats_lib.hmac_uniform_float64(seed_key, item_key, f"{tag}:a")
  )
  u_two = stats_lib.hmac_uniform_float64(seed_key, item_key, f"{tag}:b")
  return math.sqrt(-2.0 * math.log(u_one)) * math.cos(2.0 * math.pi * u_two)


def _simulate_som_stage1_probs(case: D2SoMCase) -> dict[str, float]:
  """Simulate dgem single-pass `az_aa` mark probabilities on dense UI screens.

  Captures the empirical PR #34 behavior:
    - Single-pass Stage-1 hits ~66-70% top-1 on 18..52 marks, with errors
      concentrated on the 2 spatial neighbor marks (gold remains inside top 3-4
      ~96% of the time).
    - Raw logits are overconfident (`sharpness = 1.95`), so `H_norm = H/ln(M)`
      regularly drops below `0.25` even when an adjacent neighbor beats gold!
  """
  density_penalty = 0.018 * float(case.num_marks - 18)
  app_difficulty = 0.22 * _gaussian_draw(
      "dgem-d2-app", case.app_domain, "screen_diff"
  )
  logits: dict[str, float] = {}
  gold_pos = next(
      idx for idx, m in enumerate(case.marks) if m.code == case.gold_code
  )
  neighbor_one = case.marks[(gold_pos + 1) % case.num_marks].code
  neighbor_two = case.marks[(gold_pos + 2) % case.num_marks].code

  logits[case.gold_code] = (
      1.62
      - density_penalty
      + app_difficulty
      + 0.72 * _gaussian_draw("dgem-d2-s1", case.item_id, "gold")
  )
  for mark in case.marks:
    if mark.code == case.gold_code:
      continue
    if mark.code == neighbor_one:
      logits[mark.code] = 1.12 + 0.72 * _gaussian_draw(
          "dgem-d2-s1", case.item_id, "n1"
      )
    elif mark.code == neighbor_two:
      logits[mark.code] = 0.58 + 0.62 * _gaussian_draw(
          "dgem-d2-s1", case.item_id, "n2"
      )
    else:
      logits[mark.code] = -2.45 + 0.45 * _gaussian_draw(
          "dgem-d2-s1", case.item_id, f"bg:{mark.code}"
      )

  raw_sharpness = 1.95
  scaled = {code: val * raw_sharpness for code, val in logits.items()}
  max_val = max(scaled.values())
  exps = {code: math.exp(val - max_val) for code, val in scaled.items()}
  total = sum(exps.values())
  return {code: val / total for code, val in exps.items()}


def _query_live_som_probs(
    endpoint_url: str, case: D2SoMCase
) -> dict[str, float]:
  """Query a live `/v1/systemone` endpoint with `az_aa` UI mark options."""
  criteria = {
      mark.code: f"{mark.role_label} at {mark.box}" for mark in case.marks
  }
  headers: dict[str, str] = {}
  server_key = os.environ.get("DGEM_SERVER_KEY", "").split(",")[0].strip()
  if server_key:
    headers["X-DGem-Key"] = server_key
  gateway_key = os.environ.get("DGEM_GATEWAY_KEY", "").strip()
  if gateway_key:
    headers["X-DGem-Api-Key"] = gateway_key
  payload: dict[str, object] = {
      "state": case.target_instruction,
      "labels": "az_aa",
      "questions": {
          "target_mark": {
              "type": "choice",
              "instructions": "Select the target Set-of-Marks UI element code.",
              "criteria": criteria,
          }
      },
  }
  response_obj = stats_lib.post_json_secure(
      endpoint_url, payload, headers=headers, timeout_sec=30.0
  )
  answers = response_obj.get("answers")
  if isinstance(answers, Mapping):
    slot = answers.get("target_mark")
    if isinstance(slot, Mapping):
      probs = slot.get("probabilities")
      if isinstance(probs, Mapping) and probs:
        return stats_lib.normalize_probabilities(
            {str(k): float(v) for k, v in probs.items()}  # type: ignore[arg-type]
        )
  raise stats_lib.StatisticalValidationError(
      "live endpoint missing target_mark probabilities"
  )


def _stage2_select_mark(
    case: D2SoMCase,
    candidate_codes: Sequence[str],
    stage2_model: str,
) -> tuple[str, float]:
  """Resolve candidate UI mark via Stage-2 (`<= 4` shortlist vs `<= 52` full).

  Returns `(chosen_code, latency_ms)`.
  """
  num_cands = len(candidate_codes)
  base_latency = 410.0 if stage2_model == "gemini-3.8-flash" else 265.0
  latency_ms = (
      base_latency
      + 5.8 * float(num_cands)
      + 18.0
      * abs(_gaussian_draw("dgem-d2-lat", case.item_id, f"c={num_cands}"))
  )
  if case.gold_code not in candidate_codes:
    return candidate_codes[0], latency_ms

  # Dense 36..52 badge overlays cause occasional badge-occlusion confusion;
  # corner-anchored <=4 conformal shortlists eliminate badge crowding.
  crowding_loss = 0.022 * float(max(0, num_cands - 4))
  model_snr = (2.15 if stage2_model == "gemini-3.8-flash" else 1.85) - crowding_loss
  draw_z = _gaussian_draw(
      f"dgem-d2-s2:{stage2_model}", case.item_id, f"mode={'full' if num_cands > 4 else 'short'}"
  )
  if model_snr + 0.95 * draw_z > 0.0:
    return case.gold_code, latency_ms
  wrong = [code for code in candidate_codes if code != case.gold_code]
  return (wrong[0] if wrong else case.gold_code), latency_ms


def verify_four_surface_locate_isolation() -> dict[str, bool]:
  """Verify Set-of-Marks locate stays off CLI, HTTP gateway, and MCP surfaces.

  Returns:
    Dictionary reporting Set-of-Marks surface isolation across
    `cmd/locate_cmd.go`, `cmd/serve.go`, `cmd/mcp.go`, and `pkg/decisionindex/`.

  Raises:
    stats_lib.StatisticalValidationError: If Set-of-Marks leaked onto surfaces.
  """
  som_pkg_absent = not (REPO_ROOT / "pkg" / "decisionindex" / "som.go").exists()
  locate_cmd_text = (REPO_ROOT / "cmd" / "locate_cmd.go").read_text(
      encoding="utf-8"
  )
  serve_text = (REPO_ROOT / "cmd" / "serve.go").read_text(encoding="utf-8")
  mcp_text = (REPO_ROOT / "cmd" / "mcp.go").read_text(encoding="utf-8")

  locate_cmd_clean = (
      "--marks" not in locate_cmd_text
      and "--som-max-h" not in locate_cmd_text
      and "LocateMarkInput" not in locate_cmd_text
  )
  serve_clean = "ui_pathfinder_som" not in serve_text
  mcp_clean = "som_max_h" not in mcp_text and "Set-of-Marks" not in mcp_text

  if not (som_pkg_absent and locate_cmd_clean and serve_clean and mcp_clean):
    raise stats_lib.StatisticalValidationError(
        "4-surface isolation violated: Set-of-Marks locate registered before graduation!"
    )
  return {
      "pkg_som_absent": som_pkg_absent,
      "cmd_locate_som_absent": locate_cmd_clean,
      "cmd_serve_som_absent": serve_clean,
      "cmd_mcp_som_absent": mcp_clean,
  }


def run_d2_suite(
    lockfile_path: pathlib.Path,
    dgem_url: str = "",
    gemini_model: str = "gemini-3.8-flash",
) -> dict[str, object]:
  """Execute Experiment D2 across the four paired Set-of-Marks arms.

  Args:
    lockfile_path: Path to `dgem_exp_splits_v1.json`.
    dgem_url: Optional live HTTPS `/v1/systemone` endpoint URL.
    gemini_model: Validated GA Stage-2 Gemini model identifier.

  Returns:
    Structured JSON evaluation dictionary with accuracy, latency, McNemar, and
    Tango non-inferiority (`margin = 0.010`) tests.

  Raises:
    stats_lib.SplitLeakageError: If split lockfile verification fails.
    stats_lib.StatisticalValidationError: If surface isolation or inputs fail.
  """
  surface_isolation = verify_four_surface_locate_isolation()
  validated_stage2 = stats_lib.validate_ga_gemini_model(gemini_model)
  safe_dgem_url = stats_lib.validate_endpoint_url(dgem_url) if dgem_url else ""

  corpus = build_d2_corpus()
  by_id = {entry.item_id: entry for entry in corpus}
  lock_payload = json.loads(
      stats_lib.validate_local_file_path(lockfile_path, must_exist=True).read_text(
          encoding="utf-8"
      )
  )
  verified_folds = stats_lib.verify_split_lockfile(
      lock_payload, TRACK_NAME, [entry.split_item for entry in corpus]
  )

  cal_cases = [by_id[row.item_id] for row in verified_folds["cal"]]
  val_cases = [by_id[row.item_id] for row in verified_folds["val"]]
  test_cases = [by_id[row.item_id] for row in verified_folds["test"]]

  def _get_probs(cases_seq: Sequence[D2SoMCase]) -> list[dict[str, float]]:
    if safe_dgem_url:
      return [_query_live_som_probs(safe_dgem_url, c) for c in cases_seq]
    return [_simulate_som_stage1_probs(c) for c in cases_seq]

  cal_raw = _get_probs(cal_cases)
  val_raw = _get_probs(val_cases)
  test_raw = _get_probs(test_cases)

  # Fit 5-fold OOF temperature and alpha=0.05 RAPS quantile strictly on D_cal.
  global_temp, oof_cal_probs, fold_temps = stats_lib.fit_oof_temperature_nll(
      cal_raw,
      [c.gold_code for c in cal_cases],
      [c.app_domain for c in cal_cases],
      num_folds=5,
  )
  raps_params = stats_lib.ConformalParameters(
      method="raps",
      alpha=0.05,
      k_reg=1,
      lambda_reg=0.01,
      randomized=True,
      seed_key="dgem-d2-som",
  )
  som_fit = stats_lib.calibrate_conformal_quantile(
      oof_cal_probs,
      [c.gold_code for c in cal_cases],
      raps_params,
      item_keys=[c.item_id for c in cal_cases],
  )

  val_cal = [stats_lib.apply_temperature(p, global_temp) for p in val_raw]
  test_cal = [stats_lib.apply_temperature(p, global_temp) for p in test_raw]

  # Select singleton confidence cutoff tau_val strictly on D_val so accepted
  # singletons achieve >= 95% precision on D_val.
  candidate_taus = (0.55, 0.65, 0.72, 0.78, 0.84, 0.90)
  selected_tau_val = 0.78
  for cand_tau in candidate_taus:
    accepted_hits: list[bool] = []
    for idx, v_case in enumerate(val_cases):
      c_res = stats_lib.conformal_prediction_set(
          val_cal[idx],
          som_fit.q_hat,
          stats_lib.ConformalParameters(
              method="raps",
              alpha=0.05,
              k_reg=1,
              lambda_reg=0.01,
              randomized=True,
              seed_key="dgem-d2-som",
              item_key=v_case.item_id,
          ),
      )
      if c_res.singleton and c_res.top_probability >= cand_tau:
        box_map = {m.code: m.box for m in v_case.marks}
        accepted_hits.append(
            is_localization_hit(box_map[c_res.top_option], v_case.gold_box)
        )
    if accepted_hits and (
        sum(1 for ok in accepted_hits if ok) / float(len(accepted_hits)) >= 0.95
    ):
      selected_tau_val = cand_tau
      break

  # Evaluate all 4 paired arms once on locked D_test (N = 1,200).
  arm_names = (
      "arm_direct_stage2",
      "arm_dgem_only",
      "arm_dgem_uncalibrated_skip",
      "arm_dgem_conformal_shortlist",
  )
  hits_by_arm: dict[str, list[bool]] = {name: [] for name in arm_names}
  latencies_by_arm: dict[str, list[float]] = {name: [] for name in arm_names}
  escalated_by_arm: dict[str, int] = {name: 0 for name in arm_names}
  shortlist_sizes: list[int] = []
  shortlist_recall_hits = 0

  for idx, t_case in enumerate(test_cases):
    box_lookup = {m.code: m.box for m in t_case.marks}
    all_codes = [m.code for m in t_case.marks]
    s1_lat_ms = 58.0 + 0.45 * float(t_case.num_marks)

    # 1. arm_direct_stage2
    direct_code, direct_lat = _stage2_select_mark(
        t_case, all_codes, validated_stage2
    )
    hits_by_arm["arm_direct_stage2"].append(
        is_localization_hit(box_lookup[direct_code], t_case.gold_box)
    )
    latencies_by_arm["arm_direct_stage2"].append(direct_lat)
    escalated_by_arm["arm_direct_stage2"] += 1

    # 2. arm_dgem_only
    raw_top_code = max(test_raw[idx].items(), key=lambda kv: (kv[1], kv[0]))[0]
    s1_hit = is_localization_hit(box_lookup[raw_top_code], t_case.gold_box)
    hits_by_arm["arm_dgem_only"].append(s1_hit)
    latencies_by_arm["arm_dgem_only"].append(s1_lat_ms)

    # 3. arm_dgem_uncalibrated_skip (skip Stage-2 whenever H_norm < 0.25)
    raw_hnorm = stats_lib.normalized_entropy(test_raw[idx])
    if raw_hnorm < 0.25:
      hits_by_arm["arm_dgem_uncalibrated_skip"].append(s1_hit)
      latencies_by_arm["arm_dgem_uncalibrated_skip"].append(s1_lat_ms)
    else:
      escalated_by_arm["arm_dgem_uncalibrated_skip"] += 1
      skip_s2_code, skip_s2_lat = _stage2_select_mark(
          t_case, all_codes, validated_stage2
      )
      hits_by_arm["arm_dgem_uncalibrated_skip"].append(
          is_localization_hit(box_lookup[skip_s2_code], t_case.gold_box)
      )
      latencies_by_arm["arm_dgem_uncalibrated_skip"].append(
          s1_lat_ms + skip_s2_lat
      )

    # 4. arm_dgem_conformal_shortlist (|C(x)| <= 4)
    c_set = stats_lib.conformal_prediction_set(
        test_cal[idx],
        som_fit.q_hat,
        stats_lib.ConformalParameters(
            method="raps",
            alpha=0.05,
            k_reg=1,
            lambda_reg=0.01,
            randomized=True,
            seed_key="dgem-d2-som",
            item_key=t_case.item_id,
        ),
    )
    ordered_cal_codes = [
        code
        for code, _ in sorted(
            test_cal[idx].items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]
    if c_set.singleton and c_set.top_probability >= selected_tau_val:
      hits_by_arm["arm_dgem_conformal_shortlist"].append(
          is_localization_hit(box_lookup[c_set.top_option], t_case.gold_box)
      )
      latencies_by_arm["arm_dgem_conformal_shortlist"].append(s1_lat_ms)
      shortlist_sizes.append(1)
      if t_case.gold_code == c_set.top_option:
        shortlist_recall_hits += 1
    else:
      escalated_by_arm["arm_dgem_conformal_shortlist"] += 1
      short_cands = list(c_set.prediction_set[:MAX_SHORTLIST_MARKS])
      if len(short_cands) < 2:
        short_cands = ordered_cal_codes[: min(MAX_SHORTLIST_MARKS, len(ordered_cal_codes))]
      shortlist_sizes.append(len(short_cands))
      if t_case.gold_code in short_cands:
        shortlist_recall_hits += 1
      short_code, short_lat = _stage2_select_mark(
          t_case, short_cands, validated_stage2
      )
      hits_by_arm["arm_dgem_conformal_shortlist"].append(
          is_localization_hit(box_lookup[short_code], t_case.gold_box)
      )
      latencies_by_arm["arm_dgem_conformal_shortlist"].append(
          s1_lat_ms + short_lat
      )

  test_clusters = [c.app_domain for c in test_cases]
  arm_summaries: dict[str, dict[str, object]] = {}
  for arm_name in arm_names:
    hit_floats = [1.0 if ok else 0.0 for ok in hits_by_arm[arm_name]]
    acc_ci = stats_lib.cluster_bootstrap_mean_ci(
        test_clusters, hit_floats, num_draws=2000, seed=20262201
    )
    sorted_lats = sorted(latencies_by_arm[arm_name])
    p50_lat = sorted_lats[len(sorted_lats) // 2]
    p95_lat = sorted_lats[int(0.95 * (len(sorted_lats) - 1))]
    arm_summaries[arm_name] = {
        "n_test": len(test_cases),
        "localization_accuracy": round(acc_ci.estimate, 4),
        "accuracy_ci95": [round(acc_ci.ci_lower, 4), round(acc_ci.ci_upper, 4)],
        "stage2_escalation_rate": round(
            escalated_by_arm[arm_name] / float(len(test_cases)), 4
        ),
        "mean_latency_ms": round(
            sum(sorted_lats) / float(len(sorted_lats)), 2
        ),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "meets_90pct_graduation_bar": acc_ci.ci_lower >= GRADUATION_ACCURACY_BAR,
    }

  # Hypothesis tests: Paired McNemar + Tango non-inferiority (margin = 0.010).
  shortlist_hits = hits_by_arm["arm_dgem_conformal_shortlist"]
  comparisons: dict[str, dict[str, object]] = {}
  raw_p_list: list[float] = []
  comp_names: list[str] = []

  for ref_arm in (
      "arm_direct_stage2",
      "arm_dgem_only",
      "arm_dgem_uncalibrated_skip",
  ):
    mcnemar = stats_lib.paired_mcnemar_exact(hits_by_arm[ref_arm], shortlist_hits)
    diff_ci = stats_lib.paired_cluster_bootstrap_ci(
        test_clusters,
        [1.0 if ok else 0.0 for ok in hits_by_arm[ref_arm]],
        [1.0 if ok else 0.0 for ok in shortlist_hits],
        num_draws=2000,
        seed=20262301,
    )
    tango = stats_lib.tango_non_inferiority_test(
        hits_by_arm[ref_arm],
        shortlist_hits,
        margin=TANGO_NON_INFERIORITY_MARGIN,
    )
    key = f"arm_dgem_conformal_shortlist_vs_{ref_arm}"
    comp_names.append(key)
    raw_p_list.append(mcnemar.p_value_exact)
    comparisons[key] = {
        "delta_accuracy": round(mcnemar.delta_accuracy, 4),
        "delta_accuracy_ci95": [
            round(diff_ci.ci_lower, 4),
            round(diff_ci.ci_upper, 4),
        ],
        "n_10_regressions": mcnemar.n_10,
        "n_01_improvements": mcnemar.n_01,
        "mcnemar_chi2_cc": round(mcnemar.chi2_continuity, 4),
        "mcnemar_p_value_exact": mcnemar.p_value_exact,
        "tango_margin": TANGO_NON_INFERIORITY_MARGIN,
        "tango_z_score": round(tango.z_score, 4),
        "tango_p_value_one_sided": tango.p_value_one_sided,
        "tango_non_inferior_at_005": tango.non_inferior_at_005,
    }

  # Also check whether uncalibrated skip fails Tango NI vs arm_direct_stage2.
  uncal_vs_direct_tango = stats_lib.tango_non_inferiority_test(
      hits_by_arm["arm_direct_stage2"],
      hits_by_arm["arm_dgem_uncalibrated_skip"],
      margin=TANGO_NON_INFERIORITY_MARGIN,
  )

  holm_p = stats_lib.holm_bonferroni_adjust(raw_p_list)
  for key, adj_p in zip(comp_names, holm_p):
    comparisons[key]["mcnemar_p_value_holm"] = adj_p

  seed_count = sum(1 for c in corpus if c.provenance == "repo_bbox_seed")
  execution_mode = (
      "live_http_endpoint" if safe_dgem_url else "calibrated_parametric_fixture"
  )
  return {
      "experiment_id": "D2",
      "title": "Set-of-Marks (az_aa, <= 52) Candidate Shortlist vs Fast-Skip",
      "execution_mode": execution_mode,
      "claim_bucket": (
          "Live Endpoint Measurement"
          if safe_dgem_url
          else "Modeled / Calibrated Offline Benchmark Fixture"
      ),
      "stage2_model": validated_stage2,
      "four_surface_locate_isolation": surface_isolation,
      "corpus_summary": {
          "total_cases": len(corpus),
          "repo_bbox_seed_cases": seed_count,
          "deterministic_som_expanded_cases": len(corpus) - seed_count,
          "clustering_unit": "app_domain",
          "total_clusters": TOTAL_APP_DOMAINS,
          "splits": {
              "cal": len(cal_cases),
              "val": len(val_cases),
              "test": len(test_cases),
          },
      },
      "calibration_summary": {
          "oof_temperature": round(global_temp, 4),
          "fold_temperatures": [round(t, 4) for t in fold_temps],
          "raps_q_hat_alpha_005": round(som_fit.q_hat, 4),
          "singleton_prob_cutoff_tau_val": selected_tau_val,
          "shortlist_cap": MAX_SHORTLIST_MARKS,
          "test_mean_forwarded_marks": round(
              sum(shortlist_sizes) / float(len(shortlist_sizes)), 3
          ),
          "test_shortlist_gold_recall": round(
              shortlist_recall_hits / float(len(test_cases)), 4
          ),
      },
      "arms": arm_summaries,
      "comparisons": comparisons,
      "uncalibrated_skip_vs_direct_tango": {
          "delta_accuracy": round(uncal_vs_direct_tango.delta_accuracy, 4),
          "tango_z_score": round(uncal_vs_direct_tango.z_score, 4),
          "tango_p_value_one_sided": uncal_vs_direct_tango.p_value_one_sided,
          "tango_non_inferior_at_005": (
              uncal_vs_direct_tango.non_inferior_at_005
          ),
      },
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for Experiment D2."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--lockfile", type=pathlib.Path, default=DEFAULT_LOCKFILE)
  parser.add_argument("--verify-splits-only", action="store_true")
  parser.add_argument("--dgem-url", default="")
  parser.add_argument("--gemini-model", default="gemini-3.8-flash")
  parser.add_argument("--output-json", type=pathlib.Path, default=None)
  parsed = parser.parse_args(argv)

  if parsed.verify_splits_only:
    corpus = build_d2_corpus()
    lock_payload = json.loads(
        stats_lib.validate_local_file_path(
            parsed.lockfile, must_exist=True
        ).read_text(encoding="utf-8")
    )
    folds = stats_lib.verify_split_lockfile(
        lock_payload, TRACK_NAME, [entry.split_item for entry in corpus]
    )
    print(
        json.dumps({
            "track": TRACK_NAME,
            "verified": True,
            "counts": {k: len(v) for k, v in folds.items()},
        })
    )
    return 0

  summary = run_d2_suite(
      lockfile_path=parsed.lockfile,
      dgem_url=parsed.dgem_url,
      gemini_model=parsed.gemini_model,
  )
  serialized = json.dumps(summary, indent=2)
  if parsed.output_json is not None:
    out_path = stats_lib.validate_local_file_path(
        parsed.output_json, must_exist=False
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized + "\n", encoding="utf-8")
  print(serialized)
  return 0


if __name__ == "__main__":
  sys.exit(main())
