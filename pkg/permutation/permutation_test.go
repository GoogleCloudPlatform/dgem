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

package permutation

import (
	"math"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestBALDDecompositionAndPriorDeBiasing(t *testing.T) {
	keys := []string{"opt_A", "opt_B", "opt_C"}

	// Two permutations that strongly disagree due to positional 'A' bias:
	// Pass 1 (fwd): opt_A=0.90, opt_B=0.05, opt_C=0.05 (low single-pass entropy!)
	// Pass 2 (rev): opt_A=0.05, opt_B=0.05, opt_C=0.90 (also low single-pass entropy, but opposite winner!)
	d1 := map[string]float64{"opt_A": 0.90, "opt_B": 0.05, "opt_C": 0.05}
	d2 := map[string]float64{"opt_A": 0.05, "opt_B": 0.05, "opt_C": 0.90}

	ens, winner, meanH, totalH, jsd := ComputeBALDDecomposition([]map[string]float64{d1, d2}, keys)
	if math.Abs((meanH+jsd)-totalH) > 1e-6 {
		t.Fatalf("BALD identity violated: meanH (%.6f) + jsd (%.6f) != totalH (%.6f)", meanH, jsd, totalH)
	}
	if jsd < 0.45 {
		t.Errorf("expected high JSD (>0.45 nats) when permutations flip between 0.90 and 0.05, got %.4f", jsd)
	}
	if math.Abs(ens["opt_A"]-0.475) > 1e-6 || math.Abs(ens["opt_C"]-0.475) > 1e-6 {
		t.Errorf("unexpected ensemble distribution: %v (winner=%s)", ens, winner)
	}

	// Test Prior De-Biasing cancels a 2:1 first-slot bias
	np := &NullPrior{
		ByCardinality: map[int][]float64{
			2: {0.70, 0.30}, // Strong slot-0 ('A') bias
		},
	}
	opts := []OptionItem{
		{Name: "distractor_in_slot_A"},
		{Name: "true_answer_in_slot_B"},
	}
	// Suppose raw probabilities slightly favor slot A (0.58 vs 0.42) solely because of the 0.70 vs 0.30 prior
	raw := map[string]float64{
		"distractor_in_slot_A":  0.58,
		"true_answer_in_slot_B": 0.42,
	}
	deb, debWinner := ApplyPriorDeBiasing(raw, opts, np, 0.85)
	if debWinner != "true_answer_in_slot_B" {
		t.Errorf("expected ApplyPriorDeBiasing to recover true_answer_in_slot_B, got %s (probs=%v)", debWinner, deb)
	}
}

func TestPostProcessDecisionResponseRefreshesDiagnosticsLogprobsForApplyTemperature(t *testing.T) {
	// 1. Null-prior debiasing flips primacy-biased Slot-0 winner ("opt_A" -> "opt_B").
	// Subsequent ApplyTemperature must scale the debiased distribution rather than reverting to stale single-pass Logprobs.
	slotOpts := map[string][]OptionItem{
		"route": {
			{Name: "opt_A"},
			{Name: "opt_B"},
		},
	}
	resp := &client.StructuredDecisionResponse{
		Answers: map[string]client.QuestionAnswer{
			"route": {
				Type:          "choice",
				Label:         "opt_A",
				Choice:        "opt_A",
				Confidence:    0.54,
				Probabilities: map[string]float64{"opt_A": 0.54, "opt_B": 0.46},
			},
		},
		Diagnostics: client.Diagnostics{
			Questions: map[string]client.QuestionDiagnostic{
				"route": {
					ArgmaxIsLabel: true,
					ArgmaxToken:   "opt_A",
					Entropy:       0.69,
					LabelMass:     0.54,
					Logprobs:      map[string]float64{"opt_A": 0.0, "opt_B": math.Log(0.46 / 0.54)},
				},
			},
		},
	}

	PostProcessDecisionResponse(resp, slotOpts, false, true, 0.90)
	debiasedA := resp.Answers["route"].Probabilities["opt_A"]
	debiasedB := resp.Answers["route"].Probabilities["opt_B"]
	if resp.Answers["route"].Choice != "opt_B" || debiasedB <= debiasedA {
		t.Fatalf("expected null-prior debiasing to flip winner to opt_B, got %+v", resp.Answers["route"])
	}

	resp.ApplyTemperature(client.Temperature{All: 1.5})
	z := math.Pow(debiasedA, 1.0/1.5) + math.Pow(debiasedB, 1.0/1.5)
	wantB := math.Pow(debiasedB, 1.0/1.5) / z
	got := resp.Answers["route"]
	if got.Probabilities["opt_B"] <= got.Probabilities["opt_A"] {
		t.Fatalf("ApplyTemperature reverted debiased winner order using stale Diagnostics.Logprobs: %+v", got)
	}
	if math.Abs(got.Probabilities["opt_B"]-wantB) > 1e-9 || math.Abs(got.Confidence-wantB) > 1e-9 {
		t.Fatalf("expected tempered debiased opt_B=%.9f, got prob=%.9f conf=%.9f", wantB, got.Probabilities["opt_B"], got.Confidence)
	}

	// 2. Dual-mirror interior winner flip ("opt_A" -> "opt_B") cleans up mirror diagnostics and preserves debiased order under ApplyTemperature.
	mirrorOpts := map[string][]OptionItem{
		"team": {
			{Name: "opt_A"},
			{Name: "opt_B"},
			{Name: "opt_C"},
		},
	}
	mirrorResp := &client.StructuredDecisionResponse{
		Answers: map[string]client.QuestionAnswer{
			"team": {
				Type:          "choice",
				Choice:        "opt_A",
				Confidence:    0.55,
				Probabilities: map[string]float64{"opt_A": 0.55, "opt_B": 0.40, "opt_C": 0.05},
			},
			"team" + MirrorSlotSuffix: {
				Type:          "choice",
				Choice:        "item_2", // reversed interior slot -> opt_B
				Confidence:    0.90,
				Probabilities: map[string]float64{"item_1": 0.08, "item_2": 0.90, "item_3": 0.02},
			},
		},
		Diagnostics: client.Diagnostics{
			Questions: map[string]client.QuestionDiagnostic{
				"team":                    {Logprobs: map[string]float64{"opt_A": 0.0, "opt_B": -0.32, "opt_C": -2.40}},
				"team" + MirrorSlotSuffix: {Logprobs: map[string]float64{"item_1": -2.42, "item_2": 0.0, "item_3": -3.81}},
			},
		},
	}
	PostProcessDecisionResponse(mirrorResp, mirrorOpts, true, false, 0.50)
	if mirrorResp.Answers["team"].Choice != "opt_B" {
		t.Fatalf("expected dual-mirror interior flip to opt_B, got %+v", mirrorResp.Answers["team"])
	}
	if _, stillHasMirrorDiag := mirrorResp.Diagnostics.Questions["team"+MirrorSlotSuffix]; stillHasMirrorDiag {
		t.Fatalf("expected mirror diagnostic slot %s to be cleaned up", "team"+MirrorSlotSuffix)
	}
	mirrorResp.ApplyTemperature(client.Temperature{All: 2.0})
	afterMirrorTemp := mirrorResp.Answers["team"]
	if afterMirrorTemp.Probabilities["opt_B"] <= afterMirrorTemp.Probabilities["opt_A"] ||
		afterMirrorTemp.Probabilities["opt_B"] <= afterMirrorTemp.Probabilities["opt_C"] {
		t.Fatalf("expected opt_B to remain strict argmax after ApplyTemperature, got %+v", afterMirrorTemp)
	}
}
