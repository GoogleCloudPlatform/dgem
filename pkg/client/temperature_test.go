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

package client

import (
	"encoding/json"
	"math"
	"testing"
)

func TestTemperature(t *testing.T) {
	var tt Temperature
	if err := json.Unmarshal([]byte(`{"boolean": 3, "choice": 1.5}`), &tt); err != nil || tt.For("noul") != 3 || tt.For("choice") != 1.5 || tt.For("score") != 1 {
		t.Fatalf("object form: %+v %v", tt, err)
	}
	if err := json.Unmarshal([]byte(`2`), &tt); err != nil || tt.For("score") != 2 {
		t.Fatalf("number form: %+v %v", tt, err)
	}
	if err := json.Unmarshal([]byte(`{"colour": 2}`), &Temperature{}); err == nil {
		t.Fatal("unknown type must be rejected")
	}
	r := &StructuredDecisionResponse{Answers: map[string]QuestionAnswer{
		"a": {Type: "noul", Label: "yes", Noul: 0.9, Probabilities: map[string]float64{"yes": 0.9, "no": 0.1}},
		"c": {Type: "choice", Choice: "x", Probabilities: map[string]float64{"x": 0.8, "y": 0.2}},
	}, Diagnostics: Diagnostics{Questions: map[string]QuestionDiagnostic{"a": {}}}}
	r.ApplyTemperature(Temperature{ByType: map[string]float64{"noul": 2}})
	a := r.Answers["a"]
	want := 3.0 / 4.0 // 0.9^0.5 / (0.9^0.5 + 0.1^0.5)
	if math.Abs(a.Noul-want) > 1e-9 || math.Abs(a.Confidence-want) > 1e-9 || a.Entropy <= 0 || r.Diagnostics.Questions["a"].Entropy != a.Entropy {
		t.Fatalf("noul tempered wrong: %+v", a)
	}
	if r.Answers["c"].Probabilities["x"] != 0.8 {
		t.Fatalf("choice must be unchanged without a choice temperature")
	}
	s, _ := ParseSchemaTemperature(`{"questions": [], "temperature": {"noul": 3}}`)
	if s.For("noul") != 3 {
		t.Fatalf("schema temperature not read: %+v", s)
	}
}

func TestApplyCalibrationDiagLogprobsAndPriorBias(t *testing.T) {
	// 1. Saturated float probabilities (1.0 vs 0.0) recover exact log-odds from Diagnostics.Questions[id].Logprobs.
	sat := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"finding": {
				Type:          "noul",
				Label:         "yes",
				Noul:          1.0,
				Confidence:    1.0,
				Probabilities: map[string]float64{"yes": 1.0, "no": 0.0},
			},
		},
		Diagnostics: Diagnostics{
			Questions: map[string]QuestionDiagnostic{
				"finding": {Logprobs: map[string]float64{"yes": 0.0, "no": -40.0}},
			},
		},
	}
	sat.ApplyTemperature(Temperature{All: 10.0})
	// Exact: 1 / (1 + exp(-4.0)) = 0.98201379 (vs legacy 1e-12 clamp: 1 / (1 + exp(-2.7631)) = 0.94065).
	wantExact := 1.0 / (1.0 + math.Exp(-4.0))
	if math.Abs(sat.Answers["finding"].Noul-wantExact) > 1e-6 {
		t.Fatalf("expected lossless logprob scaling %.6f, got %.6f", wantExact, sat.Answers["finding"].Noul)
	}

	// 2. PriorBias (Platt intercept shift) flips a base-rate-inflated false positive across 0.5,
	// whereas pure Temperature scaling preserves argmax >= 0.5.
	decoy := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"finding": {
				Type:          "noul",
				Label:         "yes",
				Noul:          0.880797078,
				Confidence:    0.880797078,
				Probabilities: map[string]float64{"yes": 0.880797078, "no": 0.119202922},
			},
			"route": {
				Type:          "choice",
				Choice:        "escalate",
				Confidence:    0.7,
				Probabilities: map[string]float64{"escalate": 0.7, "allow": 0.3},
			},
		},
		Diagnostics: Diagnostics{
			Questions: map[string]QuestionDiagnostic{
				"finding": {Logprobs: map[string]float64{"yes": 0.0, "no": -2.0}},
			},
		},
	}
	pb, err := ParseSchemaPriorBias(`{"prior_bias": {"finding": {"true": -2.0}, "route": {"allow": 1.5}}}`)
	if err != nil {
		t.Fatalf("ParseSchemaPriorBias failed: %v", err)
	}
	decoy.ApplyCalibration(Temperature{All: 2.0}, pb)
	// finding: base yes=0, no=-2 -> scaled yes=0/2 - 2.0 = -2.0, no=-2/2 + 0 = -1.0 -> p(yes) = sigmoid(-1.0) = 0.26894142
	wantPlatt := 1.0 / (1.0 + math.Exp(1.0))
	gotFinding := decoy.Answers["finding"]
	if gotFinding.Label != "no" || math.Abs(gotFinding.Noul-wantPlatt) > 1e-6 || math.Abs(gotFinding.Confidence-(1.0-wantPlatt)) > 1e-6 {
		t.Fatalf("Platt prior bias did not flip decoy cleanly: %+v", gotFinding)
	}
	if decoy.Answers["route"].Choice != "allow" {
		t.Fatalf("choice prior bias did not flip route to allow: %+v", decoy.Answers["route"])
	}

	// 3. Guard-branch validation rejects out-of-range and malformed prior_bias values.
	for _, bad := range []string{
		`{"prior_bias": {"finding": {"yes": 25}}}`,
		`{"prior_bias": {"finding": {"yes": -25}}}`,
		`{"prior_bias": {"": {"yes": 1}}}`,
		`{"prior_bias": {"finding": {"": 1}}}`,
		`{"prior_bias": "not-an-object"}`,
	} {
		if _, err := ParseSchemaPriorBias(bad); err == nil {
			t.Fatalf("expected error for malformed prior_bias %s", bad)
		}
	}
}
