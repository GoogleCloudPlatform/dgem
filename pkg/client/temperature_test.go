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
