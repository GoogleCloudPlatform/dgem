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

package decisionindex

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/template"
)

func TestCheckTransitiveTrianglesDetectsContradictory3Cycles(t *testing.T) {
	edges := []PairwiseEdge{
		// Consistent equivalence triangle (sku_1, sku_2, sku_3).
		{NodeA: "sku_1", NodeB: "sku_2", EquivProb: 0.94},
		{NodeA: "sku_2", NodeB: "sku_3", EquivProb: 0.91},
		{NodeA: "sku_1", NodeB: "sku_3", EquivProb: 0.88},

		// Contradictory triangle (sku_a, sku_b, sku_c): A==B (0.92), B==C (0.86), but A!=C (0.09).
		{NodeA: "sku_a", NodeB: "sku_b", EquivProb: 0.92},
		{NodeA: "sku_b", NodeB: "sku_c", EquivProb: 0.86},
		{NodeA: "sku_a", NodeB: "sku_c", EquivProb: 0.09},

		// Severe contradictory triangle (sku_x, sku_y, sku_z): X==Y (0.98), Y==Z (0.95), X!=Z (0.02).
		{NodeA: "sku_x", NodeB: "sku_y", EquivProb: 0.98},
		{NodeA: "sku_y", NodeB: "sku_z", EquivProb: 0.95},
		{NodeA: "sku_x", NodeB: "sku_z", EquivProb: 0.02},
	}

	contradictions := CheckTransitiveTriangles(edges, 0.80, 0.25)
	if len(contradictions) != 2 {
		t.Fatalf("expected 2 triangle contradictions, got %d: %+v", len(contradictions), contradictions)
	}
	// Highest severity first: (sku_x, sku_y, sku_z) has minStrong=0.95, weakProb=0.02 -> severity=0.93.
	first := contradictions[0]
	if first.Nodes != [3]string{"sku_x", "sku_y", "sku_z"} {
		t.Fatalf("unexpected first triangle nodes: %v", first.Nodes)
	}
	if first.WeakEdge != [2]string{"sku_x", "sku_z"} || math.Abs(first.Severity-0.93) > 1e-6 {
		t.Fatalf("unexpected first triangle weak edge/severity: %+v", first)
	}
	second := contradictions[1]
	if second.Nodes != [3]string{"sku_a", "sku_b", "sku_c"} || second.WeakEdge != [2]string{"sku_a", "sku_c"} {
		t.Fatalf("unexpected second triangle: %+v", second)
	}
}

func TestCheckTransitiveTrianglesOppositePolarityPairsAndTemplates(t *testing.T) {
	// Opposite-polarity slot pairs (e.g., args_grounded="yes" vs premature="yes"):
	// when both slots agree with high probability (pSameJoint >= 0.75), 1 - pSameJoint <= 0.25
	// forms a 3-cycle contradiction against the opposite-polarity invariant anchor.
	pairs := [][2]float64{
		{0.92, 0.90}, // collapsed both-yes -> pSameJoint = 0.828 + 0.008 = 0.836 -> contradiction
		{0.94, 0.05}, // clean opposite polarity -> pSameJoint = 0.047 + 0.057 = 0.104 -> consistent
		{0.10, 0.12}, // collapsed both-no -> pSameJoint = 0.012 + 0.792 = 0.804 -> contradiction
	}
	var edges []PairwiseEdge
	for idx, p := range pairs {
		n0 := fmt.Sprintf("case_%d:args_grounded", idx)
		n1 := fmt.Sprintf("case_%d:premature", idx)
		anchor := fmt.Sprintf("case_%d:opposite_polarity_invariant", idx)
		pSame := p[0]*p[1] + (1.0-p[0])*(1.0-p[1])
		edges = append(edges,
			PairwiseEdge{NodeA: n0, NodeB: anchor, EquivProb: 0.95},
			PairwiseEdge{NodeA: n1, NodeB: anchor, EquivProb: 0.95},
			PairwiseEdge{NodeA: n0, NodeB: n1, EquivProb: 1.0 - pSame},
		)
	}
	contradictions := CheckTransitiveTriangles(edges, 0.80, 0.25)
	if len(contradictions) != 2 {
		t.Fatalf("expected 2 opposite-polarity 3-cycle contradictions, got %d", len(contradictions))
	}

	eng := template.NewEngine()
	for _, rel := range []string{
		filepath.Join("..", "..", "templates", "multimodal", "entity_resolution_dedup.json.tmpl"),
		filepath.Join("..", "..", "templates", "live_checklist_verify.json.tmpl"),
		filepath.Join("..", "..", "templates", "swarm_lock_arbiter.json.tmpl"),
	} {
		if _, err := os.Stat(rel); err != nil {
			t.Fatalf("stat %s: %v", rel, err)
		}
		rendered, err := eng.RenderFile(rel, map[string]interface{}{})
		if err != nil {
			t.Fatalf("RenderFile(%s): %v", rel, err)
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(rendered), &parsed); err != nil {
			t.Fatalf("rendered %s is not valid JSON: %v", rel, err)
		}
	}
}

func TestParsePackMultiplierAndConsistency(t *testing.T) {
	cases := []struct {
		title string
		want  int
	}{
		{"San Pellegrino Sparkling Water 6x330ml Cans", 6},
		{"Coca-Cola Zero Sugar (Pack of 12, 355ml)", 12},
		{"Monster Energy Ultra 24-Pack", 24},
		{"Clif Bar Crunchy Peanut Butter 10 ct Box", 10},
		{"Single 500ml Glass Bottle Olive Oil", 1},
	}
	for _, tc := range cases {
		if got := ParsePackMultiplier(tc.title); got != tc.want {
			t.Fatalf("ParsePackMultiplier(%q)=%d, want %d", tc.title, got, tc.want)
		}
	}

	if ValidatePackQuantityConsistency(
		"San Pellegrino 6x330ml Cans",
		"San Pellegrino Single 330ml Can",
		"exact_duplicate",
		"identical_pack_and_volume",
	) {
		t.Fatalf("expected pack-quantity mismatch to invalidate exact_duplicate prediction")
	}

	if !ValidatePackQuantityConsistency(
		"San Pellegrino 6x330ml Cans",
		"San PellegrinoSingle 330ml Can",
		"sibling_variant",
		"multipack_vs_single_mismatch",
	) {
		t.Fatalf("expected sibling_variant + multipack_vs_single_mismatch to be consistent")
	}
}

func TestArbitrateSwarmLocksCollisionAndHesitation(t *testing.T) {
	proposals := []SwarmAgentProposal{
		{
			AgentID:     "agent_1",
			TargetPaths: []string{"pkg/decisionindex/engine.go", "pkg/decisionindex/som.go"},
			Confidence:  0.94,
			Hesitation:  0.05,
		},
		{
			AgentID:     "agent_2",
			TargetPaths: []string{"pkg/decisionindex/engine.go"},
			Confidence:  0.81,
			Hesitation:  0.09,
		},
		{
			AgentID:     "agent_3",
			TargetPaths: []string{"docs/experiments/README.md"},
			Confidence:  0.88,
			Hesitation:  0.04,
		},
		{
			AgentID:     "agent_4",
			TargetPaths: []string{"pkg/decisionindex/engine.go"},
			Confidence:  0.55,
			Hesitation:  0.29, // exceeds 0.16 hesitation threshold
		},
	}

	dec := ArbitrateSwarmLocks(proposals, 0.16)
	if !dec.WriteCollision {
		t.Fatalf("expected WriteCollision=true")
	}
	if dec.WinningAgent != "agent_1" {
		t.Fatalf("WinningAgent=%q, want agent_1", dec.WinningAgent)
	}
	if dec.AgentActions["agent_1"] != "grant_lock" {
		t.Fatalf("agent_1 action=%q, want grant_lock", dec.AgentActions["agent_1"])
	}
	if dec.AgentActions["agent_2"] != "serialize_after_peer" {
		t.Fatalf("agent_2 action=%q, want serialize_after_peer", dec.AgentActions["agent_2"])
	}
	if dec.AgentActions["agent_3"] != "grant_lock" {
		t.Fatalf("non-colliding agent_3 action=%q, want grant_lock", dec.AgentActions["agent_3"])
	}
	if dec.AgentActions["agent_4"] != "escalate_conflict" || !dec.EscalateToHuman {
		t.Fatalf("hesitant colliding agent_4 action=%q EscalateToHuman=%v, want escalate_conflict/true",
			dec.AgentActions["agent_4"], dec.EscalateToHuman)
	}
}

func TestScoreProbabilisticTrianglesAndSwarmLocks_Probabilistic(t *testing.T) {
	// Deterministic check on near-boundary contradiction (0.79, 0.85, 0.22):
	// missed by hard highThresh=0.80, caught by continuous MaxViolationMass = 0.79*0.85*0.78 = 0.52377.
	nearBoundary := []PairwiseEdge{
		{NodeA: "n1", NodeB: "n2", EquivProb: 0.79},
		{NodeA: "n2", NodeB: "n3", EquivProb: 0.85},
		{NodeA: "n1", NodeB: "n3", EquivProb: 0.22},
	}
	if hard := CheckTransitiveTriangles(nearBoundary, 0.80, 0.25); len(hard) != 0 {
		t.Fatalf("expected hard threshold 0.80 to miss 0.79 edge, got %+v", hard)
	}
	soft := ScoreProbabilisticTriangles(nearBoundary, 0.35)
	if len(soft) != 1 || soft[0].WeakEdge != [2]string{"n1", "n3"} || math.Abs(soft[0].MaxViolationMass-0.52377) > 1e-4 {
		t.Fatalf("expected ScoreProbabilisticTriangles to catch (n1,n3) with mass ~0.5238, got %+v", soft)
	}

	// Part 1: 400-trial probabilistic noisy entity-graph evaluation comparing continuous violation mass AUROC
	// against hard-threshold contradiction detection.
	rng := rand.New(rand.NewSource(20261026))
	const numGraphTrials = 400
	var (
		softScores, hardScores []float64
		isContradiction        []bool
	)
	sigmoid := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

	for trial := 0; trial < numGraphTrials; trial++ {
		a := fmt.Sprintf("t%d_a", trial)
		b := fmt.Sprintf("t%d_b", trial)
		c := fmt.Sprintf("t%d_c", trial)
		var pab, pbc, pac float64
		injectedFlip := false
		switch trial % 4 {
		case 0:
			// Consistent 3-clique (all same entity)
			pab = sigmoid(2.1 + 0.65*rng.NormFloat64())
			pbc = sigmoid(2.1 + 0.65*rng.NormFloat64())
			pac = sigmoid(2.1 + 0.65*rng.NormFloat64())
		case 1:
			// All 3 distinct entities
			pab = sigmoid(-2.1 + 0.65*rng.NormFloat64())
			pbc = sigmoid(-2.1 + 0.65*rng.NormFloat64())
			pac = sigmoid(-2.1 + 0.65*rng.NormFloat64())
		case 2:
			// Two equivalent (a==b), third distinct (c!=a, c!=b) -> consistent!
			pab = sigmoid(2.1 + 0.65*rng.NormFloat64())
			pbc = sigmoid(-2.1 + 0.65*rng.NormFloat64())
			pac = sigmoid(-2.1 + 0.65*rng.NormFloat64())
		case 3:
			// Intransitive contradiction: two positive (including near-boundary ~0.72..0.88), third negative
			pab = sigmoid(1.35 + 0.45*rng.NormFloat64())
			pbc = sigmoid(1.55 + 0.45*rng.NormFloat64())
			pac = sigmoid(-1.55 + 0.45*rng.NormFloat64())
			injectedFlip = true
		}

		triEdges := []PairwiseEdge{
			{NodeA: a, NodeB: b, EquivProb: pab},
			{NodeA: b, NodeB: c, EquivProb: pbc},
			{NodeA: a, NodeB: c, EquivProb: pac},
		}
		scored := ScoreProbabilisticTriangles(triEdges, 1e-9)
		mass := 0.0
		if len(scored) == 1 {
			mass = scored[0].MaxViolationMass
		}
		hardHit := 0.0
		if len(CheckTransitiveTriangles(triEdges, 0.80, 0.25)) > 0 {
			hardHit = 1.0
		}
		softScores = append(softScores, mass)
		hardScores = append(hardScores, hardHit)
		isContradiction = append(isContradiction, injectedFlip)
	}

	aurocSoft := computeHesitationAUROC(softScores, isContradiction)
	aurocHard := computeHesitationAUROC(hardScores, isContradiction)
	if aurocSoft < 0.94 || aurocSoft <= aurocHard {
		t.Fatalf("expected continuous triangle mass AUROC (%.4f) >= 0.94 and strictly above hard threshold AUROC (%.4f)", aurocSoft, aurocHard)
	}

	// Part 2: 400-trial random swarm lock contention stress test verifying zero double-grants on shared paths.
	filePool := []string{"cmd/serve.go", "pkg/decisionindex/engine.go", "pkg/client/client.go", "docs/README.md", "Makefile"}
	for round := 0; round < 400; round++ {
		var props []SwarmAgentProposal
		for a := 0; a < 4; a++ {
			p1 := filePool[rng.Intn(len(filePool))]
			p2 := filePool[rng.Intn(len(filePool))]
			conf := 0.55 + 0.44*rng.Float64()
			hes := 0.02 + 0.24*rng.Float64()
			props = append(props, SwarmAgentProposal{
				AgentID:     fmt.Sprintf("worker_%d", a),
				TargetPaths: []string{p1, p2},
				Confidence:  conf,
				Hesitation:  hes,
			})
		}
		dec := ArbitrateSwarmLocks(props, 0.16)
		grantedOwner := make(map[string]string)
		for _, p := range props {
			act := dec.AgentActions[p.AgentID]
			if act == "grant_lock" {
				if p.Hesitation >= 0.16 {
					t.Fatalf("round %d: granted lock to hesitant agent %s (H=%.3f)", round, p.AgentID, p.Hesitation)
				}
				seenLocal := make(map[string]bool)
				for _, path := range p.TargetPaths {
					if seenLocal[path] {
						continue
					}
					seenLocal[path] = true
					if prev, exists := grantedOwner[path]; exists && prev != p.AgentID {
						t.Fatalf("round %d: safety violation: path %s double-granted to %s and %s", round, path, prev, p.AgentID)
					}
					grantedOwner[path] = p.AgentID
				}
			}
		}
	}
}
