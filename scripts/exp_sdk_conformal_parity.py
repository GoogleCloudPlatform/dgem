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

"""Experiment D4: 500-vector Go/Python/TypeScript SDK conformal parity."""

import argparse
from collections.abc import Mapping, Sequence
import importlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import exp_stats_common as stats_lib  # noqa: E402

NUM_ORACLE_VECTORS: int = 500
LATENCY_CEILING_MS: float = 2.0
ALLOWED_SDK_REFS: frozenset[str] = frozenset({
    "gcp/feat/python-and-ts-sdks",
    "feat/python-and-ts-sdks",
})
ALLOWED_CONFORMAL_REFS: frozenset[str] = frozenset({
    "gcp/feat/conformal-and-context-pruning",
    "feat/conformal-and-context-pruning",
})

_GO_ORACLE_MAIN = """package main

import (
\t"encoding/json"
\t"fmt"
\t"math"
\t"os"
)

type QuestionAnswer struct {
\tProbabilities map[string]float64
}

type CalibInput struct {
\tProbs    []map[string]float64 `json:"probs"`
\tLabels   []string             `json:"labels"`
\tItemKeys []string             `json:"item_keys"`
\tAlpha    float64              `json:"alpha"`
}

type OracleConfig struct {
\tUseDefault    bool     `json:"use_default"`
\tMethod        string   `json:"method"`
\tAlpha         float64  `json:"alpha"`
\tKReg          int      `json:"k_reg"`
\tLambdaReg     float64  `json:"lambda_reg"`
\tRandomized    bool     `json:"randomized"`
\tSeedKey       string   `json:"seed_key"`
\tItemKey       string   `json:"item_key"`
\tAbstainLabels []string `json:"abstain_labels"`
}

type OracleVector struct {
\tVectorID         string             `json:"vector_id"`
\tCategory         string             `json:"category"`
\tConfig           OracleConfig       `json:"config"`
\tCalibration      *CalibInput        `json:"calibration,omitempty"`
\tExpectedOrderIdx int                `json:"expected_order_index,omitempty"`
\tExpectedEmpCov   float64            `json:"expected_empirical_coverage,omitempty"`
\tProbs            map[string]float64 `json:"probs"`
\tQHat             float64            `json:"q_hat"`
\tQHatIsInf        bool               `json:"q_hat_is_inf"`
\tExpectedSet      []string           `json:"expected_set"`
\tExpectedScores   map[string]float64 `json:"expected_scores"`
\tExpectedTopOpt   string             `json:"expected_top_option"`
\tExpectedTopProb  float64            `json:"expected_top_prob"`
\tExpectedSetSize  int                `json:"expected_set_size"`
\tExpectedSingle   bool               `json:"expected_singleton"`
\tExpectedAbstain  bool               `json:"expected_contains_abstain"`
\tExpectedEscalate bool               `json:"expected_should_escalate"`
}

func toClientCfg(oc OracleConfig) ConformalConfig {
\tif oc.UseDefault {
\t\tcfg := DefaultConformalConfig()
\t\tcfg.Randomized = oc.Randomized
\t\tcfg.ItemKey = oc.ItemKey
\t\tcfg.AbstainLabels = oc.AbstainLabels
\t\treturn cfg
\t}
\treturn ConformalConfig{
\t\tMethod:        ConformalMethod(oc.Method),
\t\tAlpha:         oc.Alpha,
\t\tKReg:          oc.KReg,
\t\tLambdaReg:     oc.LambdaReg,
\t\tRandomized:    oc.Randomized,
\t\tSeedKey:       oc.SeedKey,
\t\tItemKey:       oc.ItemKey,
\t\tAbstainLabels: oc.AbstainLabels,
\t}
}

func buildProbsForIndex(idx int) map[string]float64 {
\tswitch idx % 8 {
\tcase 0:
\t\t// Exact 4-way tie + abstain option.
\t\treturn map[string]float64{"opt_a": 0.22, "opt_b": 0.22, "opt_c": 0.22, "opt_d": 0.22, "abstain": 0.12}
\tcase 1:
\t\t// Near-tolerance boundary tie (5e-13 <= 1e-12 vs 5e-12 > 1e-12).
\t\treturn map[string]float64{
\t\t\t"alpha": 0.30,
\t\t\t"beta":  0.30 - 4e-13,
\t\t\t"gamma": 0.30 - 5e-12,
\t\t\t"oos":   0.10 + 5.4e-12,
\t\t}
\tcase 2:
\t\t// Unnormalized weights summing to != 1.0 with 2-way tie.
\t\treturn map[string]float64{"route_1": 3.5, "route_2": 3.5, "route_3": 2.0, "insufficient_evidence": 1.0}
\tcase 3:
\t\t// Wide 16-option distribution testing RAPS rank penalty.
\t\tm := make(map[string]float64, 16)
\t\tfor k := 0; k < 16; k++ {
\t\t\tm[fmt.Sprintf("cls_%02d", k)] = float64(20-k) + float64((idx+k)%3)*0.5
\t\t}
\t\treturn m
\tcase 4:
\t\t// Zero-probability tail entries mixed with positive entries.
\t\treturn map[string]float64{"yes": 0.64, "no": 0.36, "abstain": 0.0}
\tcase 5:
\t\t// 8-way exact uniform tie.
\t\tm := make(map[string]float64, 8)
\t\tfor k := 0; k < 8; k++ {
\t\t\tm[fmt.Sprintf("tie_%d", k)] = 0.125
\t\t}
\t\treturn m
\tcase 6:
\t\t// Skewed high-confidence singleton vs runner-up.
\t\tp1 := 0.55 + 0.004*float64(idx%90)
\t\treturn map[string]float64{"primary": p1, "secondary": (1.0 - p1) * 0.7, "oos": (1.0 - p1) * 0.3}
\tdefault:
\t\t// 12-option distribution with 3-way middle tie group.
\t\treturn map[string]float64{
\t\t\t"c0": 0.28, "c1": 0.14, "c2": 0.14, "c3": 0.14,
\t\t\t"c4": 0.08, "c5": 0.06, "c6": 0.05, "c7": 0.04,
\t\t\t"c8": 0.03, "c9": 0.02, "abstain": 0.01, "oos": 0.01,
\t\t}
\t}
}

func main() {
\tvectors := make([]OracleVector, 0, 500)
\tabstainChoices := [][]string{
\t\t{},
\t\t{"abstain"},
\t\t{"abstain", "oos", " Insufficient_Evidence "},
\t}
\tseedChoices := []string{"", "dgem-conformal-v1", "custom-seed-2026"}

\tfor idx := 0; idx < 450; idx++ {
\t\tvar method string
\t\tvar randomized bool
\t\tvar category string
\t\tuseDefault := false
\t\tswitch {
\t\tcase idx < 100:
\t\t\tmethod = "lac"
\t\t\trandomized = false
\t\t\tcategory = "lac_deterministic"
\t\tcase idx < 200:
\t\t\tmethod = "lac"
\t\t\trandomized = true
\t\t\tcategory = "lac_randomized"
\t\tcase idx < 300:
\t\t\tmethod = "raps"
\t\t\trandomized = false
\t\t\tcategory = "raps_deterministic"
\t\t\tif idx%25 == 0 {
\t\t\t\tuseDefault = true
\t\t\t}
\t\tdefault:
\t\t\tmethod = "raps"
\t\t\trandomized = true
\t\t\tcategory = "raps_randomized"
\t\t\tif idx%30 == 0 {
\t\t\t\tmethod = "aps"
\t\t\t\tcategory = "aps_randomized"
\t\t\t}
\t\t}

\t\tkReg := idx % 4
\t\tlamReg := float64(idx%5) * 0.015
\t\tif method == "aps" {
\t\t\tkReg = 0
\t\t\tlamReg = 0.0
\t\t}
\t\tprobs := buildProbsForIndex(idx)
\t\toc := OracleConfig{
\t\t\tUseDefault:    useDefault,
\t\t\tMethod:        method,
\t\t\tAlpha:         0.10,
\t\t\tKReg:          kReg,
\t\t\tLambdaReg:     lamReg,
\t\t\tRandomized:    randomized,
\t\t\tSeedKey:       seedChoices[idx%len(seedChoices)],
\t\t\tItemKey:       fmt.Sprintf("req-%03d:slot-%d", idx, idx%5),
\t\t\tAbstainLabels: abstainChoices[idx%len(abstainChoices)],
\t\t}
\t\tcfg := toClientCfg(oc)

\t\t// Probe scores first so some vectors test exact boundary qHat == score!
\t\tprobeRes, err := ConformalPredictionSet(probs, 0.5, cfg)
\t\tif err != nil {
\t\t\tfmt.Fprintf(os.Stderr, "probe err at %d: %v\\n", idx, err)
\t\t\tos.Exit(1)
\t\t}
\t\tvar qHat float64
\t\tswitch idx % 5 {
\t\tcase 0:
\t\t\t// Exact score boundary on top or runner-up option.
\t\t\tfor _, sc := range probeRes.Scores {
\t\t\t\tqHat = sc
\t\t\t\tbreak
\t\t\t}
\t\tcase 1:
\t\t\tqHat = 0.0
\t\tdefault:
\t\t\tqHat = 0.15 + 0.08*float64(idx%11)
\t\t}

\t\tres, err := ConformalPredictionSet(probs, qHat, cfg)
\t\tif err != nil {
\t\t\tfmt.Fprintf(os.Stderr, "eval err at %d: %v\\n", idx, err)
\t\t\tos.Exit(1)
\t\t}
\t\tvectors = append(vectors, OracleVector{
\t\t\tVectorID:         fmt.Sprintf("vec-%03d", idx),
\t\t\tCategory:         category,
\t\t\tConfig:           oc,
\t\t\tProbs:            probs,
\t\t\tQHat:             qHat,
\t\t\tQHatIsInf:        false,
\t\t\tExpectedSet:      res.PredictionSet,
\t\t\tExpectedScores:   res.Scores,
\t\t\tExpectedTopOpt:   res.TopOption,
\t\t\tExpectedTopProb:  res.TopProbability,
\t\t\tExpectedSetSize:  res.SetSize,
\t\t\tExpectedSingle:   res.Singleton,
\t\t\tExpectedAbstain:  res.ContainsAbstain,
\t\t\tExpectedEscalate: res.ShouldEscalate,
\t\t})
\t}

\t// Vectors 450..499: End-to-end CalibrateConformalQuantileWithKeys + PredictionSet
\t// (including alias resolution and extreme small alpha -> +Inf q_hat).
\tfor idx := 450; idx < 500; idx++ {
\t\toffset := idx - 450
\t\tmethod := "raps"
\t\tif offset%3 == 0 {
\t\t\tmethod = "lac"
\t\t} else if offset%5 == 0 {
\t\t\tmethod = "aps"
\t\t}
\t\trandomized := offset%2 == 1
\t\talpha := 0.08 + 0.03*float64(offset%6)
\t\tif offset%10 == 9 {
\t\t\talpha = 0.01 // With n=12, ceil(13 * 0.99) = 13 > 12 -> q_hat = +Inf!
\t\t}
\t\tnCal := 12
\t\tcalProbs := make([]map[string]float64, nCal)
\t\tcalLabels := make([]string, nCal)
\t\tcalKeys := make([]string, nCal)
\t\tfor j := 0; j < nCal; j++ {
\t\t\tif offset%4 == 0 {
\t\t\t\tpYes := 0.35 + 0.04*float64((j+offset)%12)
\t\t\t\tcalProbs[j] = map[string]float64{"yes": pYes, "no": 0.90 - pYes, "abstain": 0.10}
\t\t\t\tif j%2 == 0 {
\t\t\t\t\tcalLabels[j] = "True"
\t\t\t\t} else {
\t\t\t\t\tcalLabels[j] = "false"
\t\t\t\t}
\t\t\t} else {
\t\t\t\tcalProbs[j] = map[string]float64{
\t\t\t\t\t"1":   0.45 - 0.02*float64(j%5),
\t\t\t\t\t"2":   0.30 + 0.01*float64(j%5),
\t\t\t\t\t"3":   0.18 + 0.01*float64(j%5),
\t\t\t\t\t"oos": 0.07,
\t\t\t\t}
\t\t\t\tif j%3 == 0 {
\t\t\t\t\tcalLabels[j] = "1.0"
\t\t\t\t} else {
\t\t\t\t\tcalLabels[j] = "2"
\t\t\t\t}
\t\t\t}
\t\t\tif j%4 == 0 {
\t\t\t\tcalKeys[j] = ""
\t\t\t} else {
\t\t\t\tcalKeys[j] = fmt.Sprintf("cal-%d-%d", idx, j)
\t\t\t}
\t\t}

\t\toc := OracleConfig{
\t\t\tUseDefault:    false,
\t\t\tMethod:        method,
\t\t\tAlpha:         alpha,
\t\t\tKReg:          1,
\t\t\tLambdaReg:     0.01,
\t\t\tRandomized:    randomized,
\t\t\tSeedKey:       seedChoices[offset%len(seedChoices)],
\t\t\tItemKey:       fmt.Sprintf("eval-%03d", idx),
\t\t\tAbstainLabels: abstainChoices[offset%len(abstainChoices)],
\t\t}
\t\tcfg := toClientCfg(oc)
\t\tcalRes, err := CalibrateConformalQuantileWithKeys(calProbs, calLabels, calKeys, alpha, cfg)
\t\tif err != nil {
\t\t\tfmt.Fprintf(os.Stderr, "calib err at %d: %v\\n", idx, err)
\t\t\tos.Exit(1)
\t\t}
\t\tevalProbs := calProbs[offset%nCal]
\t\tsetRes, err := ConformalPredictionSet(evalProbs, calRes.QHat, cfg)
\t\tif err != nil {
\t\t\tfmt.Fprintf(os.Stderr, "calib set err at %d: %v\\n", idx, err)
\t\t\tos.Exit(1)
\t\t}
\t\tisInf := math.IsInf(calRes.QHat, 1)
\t\tserialQ := calRes.QHat
\t\tif isInf {
\t\t\tserialQ = 0.0
\t\t}
\t\tvectors = append(vectors, OracleVector{
\t\t\tVectorID: fmt.Sprintf("vec-%03d", idx),
\t\t\tCategory: "calibrate_and_predict",
\t\t\tConfig:   oc,
\t\t\tCalibration: &CalibInput{
\t\t\t\tProbs:    calProbs,
\t\t\t\tLabels:   calLabels,
\t\t\t\tItemKeys: calKeys,
\t\t\t\tAlpha:    alpha,
\t\t\t},
\t\t\tExpectedOrderIdx: calRes.OrderIndex,
\t\t\tExpectedEmpCov:   calRes.EmpiricalCover,
\t\t\tProbs:            evalProbs,
\t\t\tQHat:             serialQ,
\t\t\tQHatIsInf:        isInf,
\t\t\tExpectedSet:      setRes.PredictionSet,
\t\t\tExpectedScores:   setRes.Scores,
\t\t\tExpectedTopOpt:   setRes.TopOption,
\t\t\tExpectedTopProb:  setRes.TopProbability,
\t\t\tExpectedSetSize:  setRes.SetSize,
\t\t\tExpectedSingle:   setRes.Singleton,
\t\t\tExpectedAbstain:  setRes.ContainsAbstain,
\t\t\tExpectedEscalate: setRes.ShouldEscalate,
\t\t})
\t}

\tenc := json.NewEncoder(os.Stdout)
\tif err := enc.Encode(vectors); err != nil {
\t\tfmt.Fprintf(os.Stderr, "encode err: %v\\n", err)
\t\tos.Exit(1)
\t}
}
"""

_NODE_PARITY_RUNNER = """import fs from "node:fs";
import { calibrateConformalQuantile, conformalPredictionSet } from "./dist/calibration.js";

const vectorsPath = process.argv[2];
const vectors = JSON.parse(fs.readFileSync(vectorsPath, "utf8"));

function toTsConfig(cfg) {
  if (cfg.use_default) {
    return {
      randomized: cfg.randomized,
      itemKey: cfg.item_key,
      abstainLabels: cfg.abstain_labels,
    };
  }
  return {
    method: cfg.method,
    alpha: cfg.alpha,
    kReg: cfg.k_reg,
    lambdaReg: cfg.lambda_reg,
    randomized: cfg.randomized,
    seedKey: cfg.seed_key,
    itemKey: cfg.item_key,
    abstainLabels: cfg.abstain_labels,
  };
}

// Warmup JIT across first 25 vectors.
for (let i = 0; i < Math.min(25, vectors.length); i += 1) {
  const v = vectors[i];
  conformalPredictionSet(v.probs, v.q_hat_is_inf ? Infinity : v.q_hat, toTsConfig(v.config));
}

let exactMatches = 0;
const latenciesMs = [];
const mismatches = [];

for (const v of vectors) {
  const tsCfg = toTsConfig(v.config);
  const startNs = process.hrtime.bigint();
  let effectiveQ = v.q_hat_is_inf ? Infinity : v.q_hat;
  let calibOk = true;
  if (v.calibration) {
    const calRes = calibrateConformalQuantile(
      v.calibration.probs,
      v.calibration.labels,
      v.calibration.alpha,
      tsCfg,
      v.calibration.item_keys,
    );
    effectiveQ = calRes.qHat;
    const infMatch = v.q_hat_is_inf ? !Number.isFinite(calRes.qHat) : Math.abs(calRes.qHat - v.q_hat) <= 1e-14;
    if (!infMatch || calRes.orderIndex !== v.expected_order_index || Math.abs(calRes.empiricalCoverage - v.expected_empirical_coverage) > 1e-14) {
      calibOk = false;
    }
  }
  const res = conformalPredictionSet(v.probs, effectiveQ, tsCfg);
  const elapsedMs = Number(process.hrtime.bigint() - startNs) / 1e6;
  latenciesMs.push(elapsedMs);

  const setOk = JSON.stringify(res.predictionSet) === JSON.stringify(v.expected_set);
  const flagsOk = (
    res.topOption === v.expected_top_option &&
    Math.abs(res.topProbability - v.expected_top_prob) <= 1e-14 &&
    res.setSize === v.expected_set_size &&
    res.singleton === v.expected_singleton &&
    res.containsAbstain === v.expected_contains_abstain &&
    res.shouldEscalate === v.expected_should_escalate
  );
  let scoresOk = Object.keys(res.scores).length === Object.keys(v.expected_scores).length;
  if (scoresOk) {
    for (const [k, expScore] of Object.entries(v.expected_scores)) {
      if (Math.abs((res.scores[k] ?? NaN) - expScore) > 1e-14) {
        scoresOk = false;
        break;
      }
    }
  }

  if (calibOk && setOk && flagsOk && scoresOk) {
    exactMatches += 1;
  } else {
    mismatches.push({ id: v.vector_id, got: res.predictionSet, expected: v.expected_set });
  }
}

latenciesMs.sort((a, b) => a - b);
const p50 = latenciesMs[Math.floor(0.50 * (latenciesMs.length - 1))];
const p99 = latenciesMs[Math.floor(0.99 * (latenciesMs.length - 1))];
const maxLat = latenciesMs[latenciesMs.length - 1];
console.log(JSON.stringify({
  evaluated: vectors.length,
  exact_matches: exactMatches,
  p50_ms: p50,
  p99_ms: p99,
  max_ms: maxLat,
  mismatches: mismatches.slice(0, 5),
}));
"""


def _resolve_git_ref(candidates: frozenset[str], label: str) -> str:
  """Select the first existing git ref from a hardcoded allowlist."""
  for ref_name in sorted(candidates):
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref_name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
      return ref_name
  raise stats_lib.StatisticalValidationError(f"missing git ref for {label}")


def _find_tsc_and_node_types() -> tuple[str, str]:
  """Locate TypeScript compiler and local `@types/node` root."""
  studio_tsc = REPO_ROOT / "studio" / "node_modules" / ".bin" / "tsc"
  docs_tsc = REPO_ROOT / "docs-site" / "node_modules" / ".bin" / "tsc"
  tsc_bin = ""
  for candidate in (studio_tsc, docs_tsc, pathlib.Path("/usr/bin/tsc")):
    if candidate.exists():
      tsc_bin = str(candidate)
      break
  if not tsc_bin:
    raise stats_lib.StatisticalValidationError("TypeScript compiler not found")

  pnpm_dir = REPO_ROOT / "docs-site" / "node_modules" / ".pnpm"
  type_root = ""
  if pnpm_dir.is_dir():
    for entry in sorted(pnpm_dir.glob("@types+node*")):
      candidate_root = entry / "node_modules" / "@types"
      if candidate_root.is_dir():
        type_root = str(candidate_root)
        break
  return tsc_bin, type_root


def _generate_go_oracle_vectors(
    conformal_ref: str, scratch_dir: pathlib.Path
) -> list[dict[str, object]]:
  """Generate 500 reference conformal test vectors from Go `conformal.go`."""
  show_proc = subprocess.run(
      ["git", "show", f"{conformal_ref}:pkg/client/conformal.go"],
      cwd=REPO_ROOT,
      capture_output=True,
      text=True,
      check=True,
  )
  conformal_src = show_proc.stdout.replace("package client", "package main", 1)

  oracle_dir = scratch_dir / "go_conformal_oracle"
  oracle_dir.mkdir(parents=True, exist_ok=True)
  (oracle_dir / "go.mod").write_text(
      "module dgem_conformal_oracle\n\ngo 1.22\n", encoding="utf-8"
  )
  (oracle_dir / "conformal.go").write_text(conformal_src, encoding="utf-8")
  (oracle_dir / "main.go").write_text(_GO_ORACLE_MAIN, encoding="utf-8")

  gen_proc = subprocess.run(
      ["go", "run", "."],
      cwd=oracle_dir,
      capture_output=True,
      text=True,
      check=True,
  )
  vectors = json.loads(gen_proc.stdout)
  if not isinstance(vectors, list) or len(vectors) != NUM_ORACLE_VECTORS:
    raise stats_lib.StatisticalValidationError(
        f"expected {NUM_ORACLE_VECTORS} Go oracle vectors, got {len(vectors)}"
    )
  return vectors


def _evaluate_python_sdk(
    worktree_path: pathlib.Path,
    vectors: Sequence[Mapping[str, object]],
) -> dict[str, object]:
  """Run `sdk/python/src/dgem_sdk/calibration.py` over all 500 vectors."""
  py_src = str(worktree_path / "sdk" / "python" / "src")
  inserted = False
  if py_src not in sys.path:
    sys.path.insert(0, py_src)
    inserted = True
  try:
    cal_mod = importlib.import_module("dgem_sdk.calibration")
    types_mod = importlib.import_module("dgem_sdk.types")
    conformal_cfg_cls = types_mod.ConformalConfig

    def _build_py_cfg(cfg_dict: Mapping[str, object]) -> object:
      abstain_seq = tuple(str(x) for x in cfg_dict["abstain_labels"])  # type: ignore[index]
      if cfg_dict.get("use_default"):
        return conformal_cfg_cls(
            randomized=bool(cfg_dict["randomized"]),
            item_key=str(cfg_dict["item_key"]),
            abstain_labels=abstain_seq,
        )
      return conformal_cfg_cls(
          method=str(cfg_dict["method"]),
          alpha=float(cfg_dict["alpha"]),  # type: ignore[arg-type]
          k_reg=int(cfg_dict["k_reg"]),  # type: ignore[arg-type]
          lambda_reg=float(cfg_dict["lambda_reg"]),  # type: ignore[arg-type]
          randomized=bool(cfg_dict["randomized"]),
          seed_key=str(cfg_dict["seed_key"]),
          item_key=str(cfg_dict["item_key"]),
          abstain_labels=abstain_seq,
      )

    # Warmup first 25 vectors.
    for vec in vectors[:25]:
      cfg_obj = _build_py_cfg(vec["config"])  # type: ignore[arg-type]
      q_val = math.inf if vec["q_hat_is_inf"] else float(vec["q_hat"])  # type: ignore[arg-type]
      cal_mod.conformal_prediction_set(vec["probs"], q_val, cfg_obj)

    exact_matches = 0
    latencies_ms: list[float] = []
    mismatches: list[dict[str, object]] = []

    for vec in vectors:
      cfg_obj = _build_py_cfg(vec["config"])  # type: ignore[arg-type]
      start_ns = time.perf_counter_ns()
      effective_q = math.inf if vec["q_hat_is_inf"] else float(vec["q_hat"])  # type: ignore[arg-type]
      calib_ok = True
      cal_block = vec.get("calibration")
      if isinstance(cal_block, Mapping):
        cal_res = cal_mod.calibrate_conformal_quantile(
            cal_block["probs"],
            cal_block["labels"],
            alpha=float(cal_block["alpha"]),  # type: ignore[arg-type]
            config=cfg_obj,
            item_keys=cal_block["item_keys"],
        )
        effective_q = cal_res.q_hat
        inf_match = (
            math.isinf(cal_res.q_hat)
            if vec["q_hat_is_inf"]
            else abs(cal_res.q_hat - float(vec["q_hat"])) <= 1e-14  # type: ignore[arg-type]
        )
        if (
            not inf_match
            or cal_res.order_index != vec["expected_order_index"]
            or abs(
                cal_res.empirical_coverage
                - float(vec["expected_empirical_coverage"])  # type: ignore[arg-type]
            )
            > 1e-14
        ):
          calib_ok = False

      pred_res = cal_mod.conformal_prediction_set(
          vec["probs"], effective_q, cfg_obj
      )
      elapsed_ms = float(time.perf_counter_ns() - start_ns) / 1e6
      latencies_ms.append(elapsed_ms)

      exp_scores = vec["expected_scores"]
      scores_ok = isinstance(exp_scores, Mapping) and len(pred_res.scores) == len(
          exp_scores
      )
      if scores_ok and isinstance(exp_scores, Mapping):
        for key, exp_val in exp_scores.items():
          if abs(pred_res.scores.get(key, math.nan) - float(exp_val)) > 1e-14:
            scores_ok = False
            break

      set_ok = list(pred_res.prediction_set) == list(vec["expected_set"])  # type: ignore[arg-type]
      flags_ok = (
          pred_res.top_option == vec["expected_top_option"]
          and abs(pred_res.top_probability - float(vec["expected_top_prob"]))  # type: ignore[arg-type]
          <= 1e-14
          and pred_res.set_size == vec["expected_set_size"]
          and pred_res.singleton == vec["expected_singleton"]
          and pred_res.contains_abstain == vec["expected_contains_abstain"]
          and pred_res.should_escalate == vec["expected_should_escalate"]
      )
      if calib_ok and set_ok and flags_ok and scores_ok:
        exact_matches += 1
      else:
        mismatches.append({
            "id": vec["vector_id"],
            "got": list(pred_res.prediction_set),
            "expected": vec["expected_set"],
        })

    # Run existing SDK unit test suite as well.
    py_unit = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(worktree_path / "sdk" / "python" / "tests"),
            "-p",
            "test_*.py",
        ],
        env={"PYTHONPATH": py_src, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    latencies_ms.sort()
    p50_ms = latencies_ms[int(0.50 * (len(latencies_ms) - 1))]
    p99_ms = latencies_ms[int(0.99 * (len(latencies_ms) - 1))]
    return {
        "evaluated_vectors": len(vectors),
        "exact_matches": exact_matches,
        "agreement_rate": round(exact_matches / float(len(vectors)), 6),
        "p50_latency_ms": round(p50_ms, 4),
        "p99_latency_ms": round(p99_ms, 4),
        "max_latency_ms": round(latencies_ms[-1], 4),
        "meets_p99_under_2ms": p99_ms < LATENCY_CEILING_MS,
        "sdk_unit_tests_passed": py_unit.returncode == 0,
        "mismatches": mismatches[:5],
    }
  finally:
    if inserted and py_src in sys.path:
      sys.path.remove(py_src)
    for mod_name in list(sys.modules.keys()):
      if mod_name == "dgem_sdk" or mod_name.startswith("dgem_sdk."):
        del sys.modules[mod_name]


def _evaluate_ts_sdk(
    worktree_path: pathlib.Path,
    scratch_dir: pathlib.Path,
    vectors_path: pathlib.Path,
) -> dict[str, object]:
  """Compile `sdk/ts` and execute Node parity + latency benchmark on 500 vectors."""
  tsc_bin, type_root = _find_tsc_and_node_types()
  ts_build_dir = scratch_dir / "ts_sdk_build"
  shutil.copytree(worktree_path / "sdk" / "ts", ts_build_dir)

  tsc_cmd = [tsc_bin, "-p", str(ts_build_dir / "tsconfig.json")]
  if type_root:
    tsc_cmd.extend(["--typeRoots", type_root])
  subprocess.run(
      tsc_cmd,
      cwd=ts_build_dir,
      capture_output=True,
      text=True,
      check=True,
  )

  # Run existing sdk/ts/test/sdk.test.mjs against compiled dist.
  ts_unit = subprocess.run(
      ["node", str(ts_build_dir / "test" / "sdk.test.mjs")],
      cwd=ts_build_dir,
      capture_output=True,
      text=True,
      check=True,
  )

  runner_file = ts_build_dir / "run_parity_500.mjs"
  runner_file.write_text(_NODE_PARITY_RUNNER, encoding="utf-8")
  node_proc = subprocess.run(
      ["node", str(runner_file), str(vectors_path)],
      cwd=ts_build_dir,
      capture_output=True,
      text=True,
      check=True,
  )
  parsed = json.loads(node_proc.stdout)
  p99_val = float(parsed["p99_ms"])
  return {
      "evaluated_vectors": int(parsed["evaluated"]),
      "exact_matches": int(parsed["exact_matches"]),
      "agreement_rate": round(
          int(parsed["exact_matches"]) / float(parsed["evaluated"]), 6
      ),
      "p50_latency_ms": round(float(parsed["p50_ms"]), 4),
      "p99_latency_ms": round(p99_val, 4),
      "max_latency_ms": round(float(parsed["max_ms"]), 4),
      "meets_p99_under_2ms": p99_val < LATENCY_CEILING_MS,
      "sdk_unit_tests_passed": ts_unit.returncode == 0,
      "mismatches": parsed["mismatches"],
  }


def run_d4_suite(
    worktree_dir: pathlib.Path | None = None,
) -> dict[str, object]:
  """Execute Experiment D4 across Go oracle, Python SDK, and TypeScript SDK.

  Always cleans up the temporary git worktree via `git worktree remove` before
  returning so the repository stays completely clean.

  Args:
    worktree_dir: Optional explicit temporary path for the `PR #63` worktree.

  Returns:
    Structured JSON summary of bit-exact parity (`500/500`) and p50/p99 latency.

  Raises:
    stats_lib.StatisticalValidationError: If parity `< 500/500` or worktree
      setup fails.
  """
  sdk_ref = _resolve_git_ref(ALLOWED_SDK_REFS, "python-and-ts-sdks")
  conformal_ref = _resolve_git_ref(
      ALLOWED_CONFORMAL_REFS, "conformal-and-context-pruning"
  )

  with tempfile.TemporaryDirectory(prefix="dgem_d4_") as temp_root_str:
    temp_root = pathlib.Path(temp_root_str)
    target_wt = (
        stats_lib.validate_local_file_path(worktree_dir, must_exist=False)
        if worktree_dir is not None
        else (temp_root / "wt_pr63")
    )
    if target_wt.exists():
      subprocess.run(
          ["git", "worktree", "remove", "--force", str(target_wt)],
          cwd=REPO_ROOT,
          capture_output=True,
          text=True,
          check=False,
      )
      if target_wt.exists():
        shutil.rmtree(target_wt)

    target_wt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(target_wt), sdk_ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    try:
      vectors = _generate_go_oracle_vectors(conformal_ref, temp_root)
      vectors_json_path = temp_root / "oracle_vectors_500.json"
      vectors_json_path.write_text(json.dumps(vectors), encoding="utf-8")

      python_summary = _evaluate_python_sdk(target_wt, vectors)
      ts_summary = _evaluate_ts_sdk(target_wt, temp_root, vectors_json_path)
    finally:
      subprocess.run(
          ["git", "worktree", "remove", "--force", str(target_wt)],
          cwd=REPO_ROOT,
          capture_output=True,
          text=True,
          check=False,
      )
      subprocess.run(
          ["git", "worktree", "prune"],
          cwd=REPO_ROOT,
          capture_output=True,
          text=True,
          check=False,
      )
      if target_wt.exists():
        shutil.rmtree(target_wt)

  worktree_clean = not target_wt.exists()
  category_counts: dict[str, int] = {}
  for vec in vectors:
    cat = str(vec["category"])
    category_counts[cat] = category_counts.get(cat, 0) + 1

  return {
      "experiment_id": "D4",
      "title": "Python & TypeScript SDK Conformal Parity & Latency vs Go Oracle",
      "execution_mode": "live_cross_language_execution",
      "claim_bucket": "Measured Local Cross-Runtime Execution (Go / Python / Node)",
      "oracle_ref": f"{conformal_ref}:pkg/client/conformal.go",
      "sdk_ref": sdk_ref,
      "total_vectors": len(vectors),
      "vector_categories": category_counts,
      "worktree_cleaned": worktree_clean,
      "python_sdk": python_summary,
      "typescript_sdk": ts_summary,
      "all_gates_passed": bool(
          worktree_clean
          and python_summary["exact_matches"] == NUM_ORACLE_VECTORS
          and ts_summary["exact_matches"] == NUM_ORACLE_VECTORS
          and python_summary["meets_p99_under_2ms"]
          and ts_summary["meets_p99_under_2ms"]
          and python_summary["sdk_unit_tests_passed"]
          and ts_summary["sdk_unit_tests_passed"]
      ),
  }


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for Experiment D4."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worktree-dir", type=pathlib.Path, default=None)
  parser.add_argument("--output-json", type=pathlib.Path, default=None)
  parsed = parser.parse_args(argv)

  summary = run_d4_suite(worktree_dir=parsed.worktree_dir)
  serialized = json.dumps(summary, indent=2)
  if parsed.output_json is not None:
    out_path = stats_lib.validate_local_file_path(
        parsed.output_json, must_exist=False
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized + "\n", encoding="utf-8")
  print(serialized)
  if not summary["all_gates_passed"]:
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
