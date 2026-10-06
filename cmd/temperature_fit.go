// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"math"
	"math/big"
	"strings"
)

// TemperatureFit records how the temperature behind a report's calibration figures was chosen.
//
//	kfold      (--auto-temperature, default): each case is scaled by the T fitted on the other folds, so the
//	           reported ECE/Brier are out of sample. Per-case T is in each case's temperature_applied.
//	in_sample  (--auto-temperature-in-sample): one T fitted on every case and applied to every case (optimistic,
//	           the behaviour before 2026-10-04).
//	fixed      (--temperature-scale): the given T.
type TemperatureFit struct {
	Method              string    `json:"method"`
	Folds               int       `json:"folds,omitempty"`
	FoldTemperatures    []float64 `json:"fold_temperatures,omitempty"`
	InSampleTemperature float64   `json:"in_sample_temperature"`
	ECERaw              float64   `json:"ece_raw"`
	ECEApplied          float64   `json:"ece_applied"`
	BrierRaw            float64   `json:"brier_raw"`
}

// stampTemperatureMethod records the fit on the parity summary; for k-fold its "raw" ECE/Brier (computed from the
// already-scaled cases) are replaced by the true T=1 values.
func stampTemperatureMethod(jp *JevParitySummary, f *TemperatureFit) {
	if jp == nil || f == nil {
		return
	}
	jp.TemperatureMethod = f.Method
	if f.Method == "kfold" {
		jp.RawT1ECE10Bin, jp.RawT1BrierMean = f.ECERaw, f.BrierRaw
	}
}

func meanBrier(cs []CalibrationCaseResult) float64 {
	if len(cs) == 0 {
		return 0
	}
	var s float64
	for _, c := range cs {
		s += c.BrierScore
	}
	return s / float64(len(cs))
}

// temperatureFolds is the fold count of the held-out fit; folds are assigned like the regression matrix's T-cal
// (scripts/matrix/metrics.py _fold: int(sha1(id).hexdigest(), 16) % k), so both report the same held-out split.
const temperatureFolds = 5

func temperatureFold(id string, k int) int {
	sum := sha1.Sum([]byte(id))
	n := new(big.Int)
	n.SetString(hex.EncodeToString(sum[:]), 16)
	return int(new(big.Int).Mod(n, big.NewInt(int64(k))).Int64())
}

// heldOutTemperatures fits T on every fold's complement and returns the T for each item (by its fold) and the T of
// each fold. fit receives the indices of the training items; a fold with no training items gets T = 1.
func heldOutTemperatures(ids []string, k int, fit func(train []int) float64) (perItem []float64, foldT []float64) {
	fold := make([]int, len(ids))
	for i, id := range ids {
		fold[i] = temperatureFold(id, k)
	}
	foldT = make([]float64, k)
	for f := 0; f < k; f++ {
		var train []int
		for i := range ids {
			if fold[i] != f {
				train = append(train, i)
			}
		}
		foldT[f] = 1.0
		if len(train) > 0 {
			foldT[f] = fit(train)
		}
	}
	perItem = make([]float64, len(ids))
	for i := range ids {
		perItem[i] = foldT[fold[i]]
	}
	return perItem, foldT
}

// unscaleTemperature inverts a temperature a receipt was saved with (per case when recorded, else the report-wide T),
// so replays start from T = 1 and are idempotent.
func unscaleTemperature(c CalibrationCaseResult, perCase, reportWide float64) CalibrationCaseResult {
	t := perCase
	if t <= 0 {
		t = reportWide
	}
	if t <= 0 || math.Abs(t-1.0) < 1e-6 {
		return c
	}
	c.TopProbabilities, c.Confidence, c.Entropy, c.NormalizedEntropy = scaleCaseDistribution(c, 1.0/t)
	return c
}

// applyCalibrationTemperature returns the cases scaled as the flags ask, the report-wide T to pass to the report
// builder (1 when cases carry their own T), and the fit record. Input cases must be at T = 1.
func applyCalibrationTemperature(cases []CalibrationCaseResult, auto, inSample bool, fixed float64) ([]CalibrationCaseResult, float64, *TemperatureFit) {
	base := enrichAndScaleCaseResults(cases, 1.0)
	eceRaw, _ := compute10BinECE(base)
	fit := &TemperatureFit{InSampleTemperature: findOptimalTemperature(cases), ECERaw: eceRaw, BrierRaw: meanBrier(base)}
	switch {
	case auto && !inSample:
		ids := make([]string, len(cases))
		for i, c := range cases {
			ids[i] = c.ID
		}
		perItem, foldT := heldOutTemperatures(ids, temperatureFolds, func(train []int) float64 {
			sub := make([]CalibrationCaseResult, len(train))
			for j, i := range train {
				sub[j] = cases[i]
			}
			return findOptimalTemperature(sub)
		})
		out := make([]CalibrationCaseResult, len(cases))
		for i, c := range cases {
			out[i] = enrichAndScaleCaseResults([]CalibrationCaseResult{c}, perItem[i])[0]
			out[i].TemperatureApplied = perItem[i]
		}
		fit.Method, fit.Folds, fit.FoldTemperatures = "kfold", temperatureFolds, foldT
		fit.ECEApplied, _ = compute10BinECE(out)
		return out, 1.0, fit
	case auto:
		fit.Method = "in_sample"
		sc := enrichAndScaleCaseResults(cases, fit.InSampleTemperature)
		fit.ECEApplied, _ = compute10BinECE(sc)
		return cases, fit.InSampleTemperature, fit
	default:
		if fixed <= 0 {
			fixed = 1.0
		}
		fit.Method = "fixed"
		sc := enrichAndScaleCaseResults(cases, fixed)
		fit.ECEApplied, _ = compute10BinECE(sc)
		return cases, fixed, fit
	}
}

func describeTemperatureFit(f *TemperatureFit, applied float64) string {
	if f == nil {
		return fmt.Sprintf("%.2f", applied)
	}
	switch f.Method {
	case "kfold":
		ts := make([]string, len(f.FoldTemperatures))
		for i, t := range f.FoldTemperatures {
			ts[i] = fmt.Sprintf("%.2f", t)
		}
		return fmt.Sprintf("held-out %d-fold (fold T: %s; in-sample T*=%.2f) — ECE10 %.4f raw -> %.4f held-out",
			f.Folds, strings.Join(ts, ", "), f.InSampleTemperature, f.ECERaw, f.ECEApplied)
	case "in_sample":
		return fmt.Sprintf("%.2f fitted in-sample (optimistic) — ECE10 %.4f raw -> %.4f", applied, f.ECERaw, f.ECEApplied)
	default:
		return fmt.Sprintf("%.2f (fixed; in-sample T*=%.2f)", applied, f.InSampleTemperature)
	}
}
