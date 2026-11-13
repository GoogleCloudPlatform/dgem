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
	"fmt"
	"math/rand"
	"testing"
)

func TestApplyConformalCalibrationDisabledByDefault(t *testing.T) {
	cases := []CalibrationCaseResult{
		{ID: "c1", Expected: "true", Actual: "true", VocabCardinality: 2, TopProbabilities: map[string]float64{"yes": 0.9, "no": 0.1}},
	}
	gotCases, summary, err := applyConformalCalibration(cases, "", 0.10)
	if err != nil {
		t.Fatalf("applyConformalCalibration disabled error: %v", err)
	}
	if summary != nil || len(gotCases[0].PredictionSet) != 0 {
		t.Fatalf("got summary=%+v set=%v, want nil and empty when disabled", summary, gotCases[0].PredictionSet)
	}
}

func TestApplyConformalCalibrationFiveFoldCoverageAndTailCompletion(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	var cases []CalibrationCaseResult
	for i := 0; i < 150; i++ {
		probs := map[string]float64{"a": 0.72, "b": 0.20, "c": 0.08}
		draw := rng.Float64()
		gold := "a"
		switch {
		case draw >= 0.92:
			gold = "c"
		case draw >= 0.72:
			gold = "b"
		}
		cases = append(cases, CalibrationCaseResult{
			ID:               fmt.Sprintf("item-%03d", i),
			Metric:           "calibration/anli-entailment-select",
			Expected:         gold,
			Actual:           "a",
			Accurate:         gold == "a",
			VocabCardinality: 3,
			Confidence:       0.72,
			TopProbabilities: probs,
		})
	}
	// Also append a 26-way top-5 truncated item whose gold label sits in the unseen tail.
	cases = append(cases, CalibrationCaseResult{
		ID:               "banking77-tail-01",
		Metric:           "calibration/banking77-intent-select",
		Expected:         "card_swallowed",
		Actual:           "card_arrival",
		Accurate:         false,
		VocabCardinality: 26,
		Confidence:       0.60,
		TopProbabilities: map[string]float64{
			"card_arrival": 0.60, "card_linking": 0.15, "change_pin": 0.10, "atm_support": 0.08, "compromised_card": 0.05,
		},
	})

	for _, method := range []string{"lac", "aps", "raps"} {
		out, summary, err := applyConformalCalibration(cases, method, 0.10)
		if err != nil {
			t.Fatalf("applyConformalCalibration(%s) error: %v", method, err)
		}
		if summary == nil || summary.EvaluatedCases != len(cases) || len(summary.FoldQuantiles) != 5 {
			t.Fatalf("method %s summary = %+v, want EvaluatedCases=%d Folds=5", method, summary, len(cases))
		}
		if summary.EmpiricalCoverage < 0.84 {
			t.Fatalf("method %s EmpiricalCoverage = %.3f, want >= 0.84 at alpha=0.10", method, summary.EmpiricalCoverage)
		}
		last := out[len(out)-1]
		if last.ConformalCovered == nil || last.PredictionSetSize < 1 {
			t.Fatalf("method %s tail case missing conformal evaluation: %+v", method, last)
		}
	}

	if _, _, err := applyConformalCalibration(cases, "invalid_method", 0.10); err == nil {
		t.Fatal("applyConformalCalibration(invalid_method) error = nil, want non-nil")
	}
	if _, _, err := applyConformalCalibration(cases, "lac", 1.0); err == nil {
		t.Fatal("applyConformalCalibration(alpha=1.0) error = nil, want non-nil")
	}

	var oosCases []CalibrationCaseResult
	for i := 0; i < 20; i++ {
		oosCases = append(oosCases, CalibrationCaseResult{
			ID:               fmt.Sprintf("clinc-%02d", i),
			Metric:           "calibration/clinc-oos-intent-select",
			Expected:         "oos",
			Actual:           "oos",
			Accurate:         true,
			VocabCardinality: 2,
			Confidence:       0.97,
			TopProbabilities: map[string]float64{"oos": 0.97, "balance": 0.03},
		})
	}
	_, defaultSummary, err := applyConformalCalibration(oosCases, "lac", 0.10)
	if err != nil || defaultSummary == nil || defaultSummary.EscalationRate != 0 {
		t.Fatalf("default oos summary = %+v err=%v, want EscalationRate=0", defaultSummary, err)
	}
	_, optInSummary, err := applyConformalCalibration(oosCases, "lac", 0.10, "oos")
	if err != nil || optInSummary == nil || optInSummary.EscalationRate != 1.0 {
		t.Fatalf("opt-in oos summary = %+v err=%v, want EscalationRate=1.0", optInSummary, err)
	}
}
