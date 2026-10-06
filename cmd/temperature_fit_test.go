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
	"math"
	"testing"
)

// Folds must match scripts/matrix/metrics.py _fold so the matrix and the CLI hold out the same items.
func TestTemperatureFoldMatchesMatrix(t *testing.T) {
	for id, want := range map[string]int{"easy-intent-00": 2, "hard-opus-a-long_policy-01": 4, "x": 0} {
		if got := temperatureFold(id, 5); got != want {
			t.Fatalf("%s: fold %d, want %d", id, got, want)
		}
	}
}

func TestHeldOutTemperaturesNeverSeeOwnFold(t *testing.T) {
	ids := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	per, foldT := heldOutTemperatures(ids, 5, func(train []int) float64 {
		return float64(len(train)) // T = training-set size identifies which items the fit received
	})
	for i, id := range ids {
		f := temperatureFold(id, 5)
		n := 0
		for _, other := range ids {
			if temperatureFold(other, 5) != f {
				n++
			}
		}
		if per[i] != float64(n) || foldT[f] != float64(n) {
			t.Fatalf("%s: T %v, want %d (training size of the other folds)", id, per[i], n)
		}
	}
}

func TestUnscaleTemperatureRoundTrip(t *testing.T) {
	c := CalibrationCaseResult{ID: "x", VocabCardinality: 3, TopProbabilities: map[string]float64{"a": 0.7, "b": 0.2, "c": 0.1}, Confidence: 0.7}
	scaled := enrichAndScaleCaseResults([]CalibrationCaseResult{c}, 1.8)[0]
	back := unscaleTemperature(scaled, 1.8, 0)
	for k, p := range c.TopProbabilities {
		if math.Abs(back.TopProbabilities[k]-p) > 1e-9 {
			t.Fatalf("%s: %v != %v", k, back.TopProbabilities[k], p)
		}
	}
	if unscaleTemperature(c, 0, 1.0).TopProbabilities["a"] != 0.7 {
		t.Fatal("T=1 must be a no-op")
	}
}

func TestApplyCalibrationTemperatureKFold(t *testing.T) {
	var cases []CalibrationCaseResult
	for i := 0; i < 60; i++ { // overconfident: 0.95 on everything, 70% right
		exp, act := "yes", "yes"
		if i%10 >= 7 {
			act = "no"
		}
		cases = append(cases, CalibrationCaseResult{ID: string(rune('A'+i%26)) + string(rune('a'+i/26)), Metric: "x/y-check",
			Expected: exp, Actual: act, Accurate: exp == act, VocabCardinality: 2, Confidence: 0.95,
			TopProbabilities: map[string]float64{act: 0.95, map[string]string{"yes": "no", "no": "yes"}[act]: 0.05}})
	}
	out, tReport, fit := applyCalibrationTemperature(cases, true, false, 1.0)
	if fit.Method != "kfold" || tReport != 1.0 || len(fit.FoldTemperatures) != 5 {
		t.Fatalf("fit %+v, report T %v", fit, tReport)
	}
	for i, c := range out {
		if c.TemperatureApplied != fit.FoldTemperatures[temperatureFold(cases[i].ID, 5)] {
			t.Fatalf("case %d: T %v not its fold's", i, c.TemperatureApplied)
		}
	}
	if !(fit.ECEApplied < fit.ECERaw) {
		t.Fatalf("held-out T should reduce ECE on a uniformly overconfident set: %v -> %v", fit.ECERaw, fit.ECEApplied)
	}
	_, tIn, fitIn := applyCalibrationTemperature(cases, true, true, 1.0)
	if fitIn.Method != "in_sample" || tIn != fitIn.InSampleTemperature {
		t.Fatalf("in-sample: %+v %v", fitIn, tIn)
	}
}
