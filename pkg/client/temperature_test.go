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

func TestApplyTemperatureSystemOneScoreLegendAndNamedLevels(t *testing.T) {
	// 1. /v1/systemone score answers key Probabilities by "0".."K-1" and Diagnostics.Logprobs by Legend level name,
	// with 0-indexed expected score sum(i * p[strconv.Itoa(i)]).
	sysOne := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"mood": {
				Type:          "score",
				Score:         0.0,
				Confidence:    1.0,
				Legend:        map[string]string{"0": "calm", "1": "annoyed", "2": "furious"},
				Probabilities: map[string]float64{"0": 1.0, "1": 0.0, "2": 0.0},
			},
		},
		Diagnostics: Diagnostics{
			Questions: map[string]QuestionDiagnostic{
				"mood": {Logprobs: map[string]float64{"calm": 0.0, "annoyed": -20.0, "furious": -40.0}},
			},
		},
	}
	sysOne.ApplyTemperature(Temperature{All: 10.0})
	z := 1.0 + math.Exp(-2.0) + math.Exp(-4.0)
	wantP0 := 1.0 / z
	wantP1 := math.Exp(-2.0) / z
	wantP2 := math.Exp(-4.0) / z
	wantSysScore := 0.0*wantP0 + 1.0*wantP1 + 2.0*wantP2
	gotSys := sysOne.Answers["mood"]
	if math.Abs(gotSys.Probabilities["0"]-wantP0) > 1e-9 || math.Abs(gotSys.Probabilities["1"]-wantP1) > 1e-9 {
		t.Fatalf("expected lossless Legend logprob lookup [%.8f, %.8f], got %+v", wantP0, wantP1, gotSys.Probabilities)
	}
	if math.Abs(gotSys.Score-wantSysScore) > 1e-9 {
		t.Fatalf("expected 0-indexed systemone score %.8f, got %.8f", wantSysScore, gotSys.Score)
	}
	if _, ok := sysOne.Diagnostics.Questions["mood"].Logprobs["calm"]; !ok {
		t.Fatalf("expected systemone diagnostic logprobs to retain level-name keys, got %+v", sysOne.Diagnostics.Questions["mood"].Logprobs)
	}

	// 2. /v1/chat/completions named score levels (e.g. ["calm", "annoyed", "furious"], which are NOT alphabetical)
	// preserve exact wire order via QuestionAnswer.UnmarshalJSON and recompute 1-indexed sum((i+1) * p).
	wireJSON := `{
		"answers": {
			"mood": {
				"type": "score",
				"label": "1",
				"level": "calm",
				"score": 1.4,
				"confidence": 0.7,
				"probabilities": {"calm": 0.7, "annoyed": 0.2, "furious": 0.1}
			}
		},
		"diagnostics": {
			"questions": {
				"mood": {
					"entropy": 0.8018,
					"logprobs": {"calm": 0.0, "annoyed": -1.252762968495368, "furious": -1.9459101490553135}
				}
			}
		}
	}`
	parsed, err := ParseStructuredContent(wireJSON)
	if err != nil {
		t.Fatalf("ParseStructuredContent failed: %v", err)
	}
	parsed.ApplyTemperature(Temperature{All: 2.0})
	gotWire := parsed.Answers["mood"]
	wantNamedScore := 1.0*gotWire.Probabilities["calm"] + 2.0*gotWire.Probabilities["annoyed"] + 3.0*gotWire.Probabilities["furious"]
	if math.Abs(gotWire.Score-wantNamedScore) > 1e-9 {
		t.Fatalf("expected 1-indexed named score %.8f, got %.8f", wantNamedScore, gotWire.Score)
	}

	// 3. In-memory Go struct literal with named levels (without JSON unmarshal) also recomputes 1-indexed score accurately.
	inMem := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"mood": {
				Type:          "score",
				Label:         "1",
				Level:         "calm",
				Score:         1.4,
				Confidence:    0.7,
				Probabilities: map[string]float64{"calm": 0.7, "annoyed": 0.2, "furious": 0.1},
			},
		},
	}
	inMem.ApplyTemperature(Temperature{All: 2.0})
	if math.Abs(inMem.Answers["mood"].Score-wantNamedScore) > 1e-9 {
		t.Fatalf("expected in-memory named score %.8f, got %.8f", wantNamedScore, inMem.Answers["mood"].Score)
	}

	// 4. /v1/chat/completions numeric levels "0".."4" (no Legend) keep 1-indexed sum((i+1)*p) and Level confidence.
	zeroNumeric := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"severity": {
				Type:          "score",
				Label:         "1",
				Level:         "0",
				Score:         1.35,
				Confidence:    0.80,
				Probabilities: map[string]float64{"0": 0.80, "1": 0.10, "2": 0.05, "3": 0.05},
			},
		},
	}
	zeroNumeric.ApplyTemperature(Temperature{All: 2.0})
	gotNum := zeroNumeric.Answers["severity"]
	wantNumScore := 1.0*gotNum.Probabilities["0"] + 2.0*gotNum.Probabilities["1"] + 3.0*gotNum.Probabilities["2"] + 4.0*gotNum.Probabilities["3"]
	if math.Abs(gotNum.Score-wantNumScore) > 1e-9 {
		t.Fatalf("expected 1-indexed numeric score %.8f, got %.8f", wantNumScore, gotNum.Score)
	}
	if math.Abs(gotNum.Confidence-gotNum.Probabilities["0"]) > 1e-9 {
		t.Fatalf("expected confidence to track winning Level '0' (%.6f), got %.6f", gotNum.Probabilities["0"], gotNum.Confidence)
	}
}

func TestApplyTemperatureSequentialComposition(t *testing.T) {
	makeFixture := func() *StructuredDecisionResponse {
		return &StructuredDecisionResponse{
			Answers: map[string]QuestionAnswer{
				"gate": {
					Type:          "noul",
					Label:         "yes",
					Noul:          1.0,
					Confidence:    1.0,
					Probabilities: map[string]float64{"yes": 1.0, "no": 0.0},
				},
				"route": {
					Type:          "choice",
					Label:         "A",
					Choice:        "billing",
					Confidence:    0.75,
					Probabilities: map[string]float64{"billing": 0.75, "support": 0.20, "sales": 0.05},
				},
				"mood": {
					Type:          "score",
					Label:         "1",
					Level:         "calm",
					Score:         1.35,
					Confidence:    0.75,
					Levels:        []string{"calm", "annoyed", "furious"},
					Probabilities: map[string]float64{"calm": 0.75, "annoyed": 0.15, "furious": 0.10},
				},
				"sys_score": {
					Type:          "score",
					Score:         0.35,
					Confidence:    0.75,
					Legend:        map[string]string{"0": "calm", "1": "annoyed", "2": "furious"},
					Probabilities: map[string]float64{"0": 0.75, "1": 0.15, "2": 0.10},
				},
			},
			Diagnostics: Diagnostics{
				Questions: map[string]QuestionDiagnostic{
					"gate":      {Logprobs: map[string]float64{"yes": 0.0, "no": -36.0}},
					"route":     {Logprobs: map[string]float64{"billing": math.Log(0.75), "support": math.Log(0.20), "sales": math.Log(0.05)}},
					"mood":      {Logprobs: map[string]float64{"calm": math.Log(0.75), "annoyed": math.Log(0.15), "furious": math.Log(0.10)}},
					"sys_score": {Logprobs: map[string]float64{"calm": 0.0, "annoyed": -18.0, "furious": -30.0}},
				},
			},
		}
	}

	twoStep := makeFixture()
	twoStep.ApplyTemperature(Temperature{All: 2.0})
	twoStep.ApplyTemperature(Temperature{All: 1.5})

	oneStep := makeFixture()
	oneStep.ApplyTemperature(Temperature{All: 3.0})

	for qID, got := range twoStep.Answers {
		want := oneStep.Answers[qID]
		if math.Abs(got.Confidence-want.Confidence) > 1e-12 ||
			math.Abs(got.Entropy-want.Entropy) > 1e-12 ||
			math.Abs(got.Noul-want.Noul) > 1e-12 ||
			math.Abs(got.Score-want.Score) > 1e-12 {
			t.Fatalf("slot %s scalar mismatch between 2.0*1.5 and 3.0: got %+v, want %+v", qID, got, want)
		}
		for k, gp := range got.Probabilities {
			if math.Abs(gp-want.Probabilities[k]) > 1e-12 {
				t.Fatalf("slot %s prob[%s] mismatch: got %.15f, want %.15f", qID, k, gp, want.Probabilities[k])
			}
		}
		gotLP := twoStep.Diagnostics.Questions[qID].Logprobs
		wantLP := oneStep.Diagnostics.Questions[qID].Logprobs
		for k, glp := range gotLP {
			if math.Abs(glp-wantLP[k]) > 1e-12 {
				t.Fatalf("slot %s logprob[%s] mismatch: got %.15f, want %.15f", qID, k, glp, wantLP[k])
			}
		}
	}

	// Sharpening (T=0.05, pushing runner-up below 1e-12) followed by softening (T=20.0) recovers T=1.0 losslessly.
	roundTrip := &StructuredDecisionResponse{
		Answers: map[string]QuestionAnswer{
			"check": {
				Type:          "noul",
				Label:         "yes",
				Noul:          0.9,
				Confidence:    0.9,
				Probabilities: map[string]float64{"yes": 0.9, "no": 0.1},
			},
		},
	}
	roundTrip.ApplyTemperature(Temperature{All: 0.05})
	roundTrip.ApplyTemperature(Temperature{All: 20.0})
	if math.Abs(roundTrip.Answers["check"].Noul-0.9) > 1e-9 {
		t.Fatalf("expected lossless round-trip Noul 0.9, got %.9f", roundTrip.Answers["check"].Noul)
	}
}
