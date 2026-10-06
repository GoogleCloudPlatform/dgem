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
