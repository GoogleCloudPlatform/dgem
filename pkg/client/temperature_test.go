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

func TestApplyTemperatureLosslessDiagLogprobs(t *testing.T) {
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

	// 2. Boolean "true"/"false" probability keys map losslessly to "yes"/"no" diagnostic logprob keys.
	boolKeys := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"safe": {
				Type:          "noul",
				Label:         "yes",
				Noul:          1.0,
				Confidence:    1.0,
				Probabilities: map[string]float64{"true": 1.0, "false": 0.0},
			},
		},
		Diagnostics: Diagnostics{
			Questions: map[string]QuestionDiagnostic{
				"safe": {Logprobs: map[string]float64{"yes": 0.0, "no": -30.0}},
			},
		},
	}
	boolKeys.ApplyTemperature(Temperature{All: 10.0})
	wantBool := 1.0 / (1.0 + math.Exp(-3.0))
	if math.Abs(boolKeys.Answers["safe"].Noul-wantBool) > 1e-6 {
		t.Fatalf("expected boolean alias logprob scaling %.6f, got %.6f", wantBool, boolKeys.Answers["safe"].Noul)
	}

	// 3. Partial or non-finite diagnostic logprobs fall back safely to clamped Probabilities.
	partial := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"route": {
				Type:          "choice",
				Choice:        "allow",
				Confidence:    0.9,
				Probabilities: map[string]float64{"allow": 0.9, "deny": 0.1},
			},
		},
		Diagnostics: Diagnostics{
			Questions: map[string]QuestionDiagnostic{
				"route": {Logprobs: map[string]float64{"allow": 0.0}}, // missing "deny"
			},
		},
	}
	partial.ApplyTemperature(Temperature{All: 2.0})
	if math.Abs(partial.Answers["route"].Probabilities["allow"]-0.75) > 1e-9 {
		t.Fatalf("expected fallback probability 0.75, got %.6f", partial.Answers["route"].Probabilities["allow"])
	}
}
