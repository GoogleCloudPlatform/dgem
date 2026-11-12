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

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/firebase/genkit/go/genkit"
)

func computeBinaryAUROC(scores []float64, labels []bool) float64 {
	type pair struct {
		score float64
		pos   bool
	}
	items := make([]pair, len(scores))
	var nPos, nNeg int
	for i := range scores {
		items[i] = pair{score: scores[i], pos: labels[i]}
		if labels[i] {
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0.5
	}
	sort.Slice(items, func(i, j int) bool { return items[i].score < items[j].score })
	var rankSum float64
	for i := 0; i < len(items); {
		j := i + 1
		for j < len(items) && items[j].score == items[i].score {
			j++
		}
		avgRank := 0.5 * float64((i+1)+j)
		for k := i; k < j; k++ {
			if items[k].pos {
				rankSum += avgRank
			}
		}
		i = j
	}
	return (rankSum - float64(nPos*(nPos+1))/2.0) / float64(nPos*nNeg)
}

func TestCheckTransitiveTrianglesDetectsContradictory3Cycles(t *testing.T) {
	edges := []PairwiseEdge{
		{NodeA: "sku_1", NodeB: "sku_2", EquivProb: 0.94},
		{NodeA: "sku_2", NodeB: "sku_3", EquivProb: 0.91},
		{NodeA: "sku_1", NodeB: "sku_3", EquivProb: 0.88},

		{NodeA: "sku_a", NodeB: "sku_b", EquivProb: 0.92},
		{NodeA: "sku_b", NodeB: "sku_c", EquivProb: 0.86},
		{NodeA: "sku_a", NodeB: "sku_c", EquivProb: 0.09},

		{NodeA: "sku_x", NodeB: "sku_y", EquivProb: 0.98},
		{NodeA: "sku_y", NodeB: "sku_z", EquivProb: 0.95},
		{NodeA: "sku_x", NodeB: "sku_z", EquivProb: 0.02},
	}

	contradictions := CheckTransitiveTriangles(edges, 0.80, 0.25)
	if len(contradictions) != 2 {
		t.Fatalf("len(contradictions) = %d (%+v), want 2", len(contradictions), contradictions)
	}
	first := contradictions[0]
	if first.Nodes != [3]string{"sku_x", "sku_y", "sku_z"} {
		t.Fatalf("first.Nodes = %v, want [sku_x sku_y sku_z]", first.Nodes)
	}
	if first.WeakEdge != [2]string{"sku_x", "sku_z"} || math.Abs(first.Severity-0.93) > 1e-6 {
		t.Fatalf("first = %+v, want WeakEdge=[sku_x sku_z] Severity=0.93", first)
	}
	second := contradictions[1]
	if second.Nodes != [3]string{"sku_a", "sku_b", "sku_c"} || second.WeakEdge != [2]string{"sku_a", "sku_c"} {
		t.Fatalf("second = %+v, want [sku_a sku_b sku_c] WeakEdge=[sku_a sku_c]", second)
	}
}

func TestParsePackMultiplierAndConsistency(t *testing.T) {
	cases := []struct {
		title string
		want  int
	}{
		{"San Pellegrino Sparkling Water 6x330ml Cans", 6},
		{"Mineral Sparkling Water 2 x 500ml", 2},
		{"Isotonic Sports Drink 500ml x 4", 4},
		{"6x Brand Product 330ml", 6},
		{"12 x Sparkling Mineral Water 330ml", 12},
		{"Coca-Cola Zero Sugar (Pack of 12, 355ml)", 12},
		{"Monster Energy Ultra 24-Pack", 24},
		{"Organic Oat Barista Milk 6-pack 1L", 6},
		{"Clif Bar Crunchy Peanut Butter 10 ct Box", 10},
		{"Sensitive Toothpaste 75ml Twin-Pack", 2},
		{"Dishwasher Rinse Aid 250ml Triple Pack", 3},
		{"Dark Roast Espresso 2 x 20 pods", 2},
		{"Dark Roast Espresso 20 pods", 1},
		{"Vitamin D3 60 capsules", 1},
		{"Single 500ml Glass Bottle Olive Oil", 1},
		{"Whole Bean Espresso 250g Bag", 1},
		{"Still Spring Water 1.5L", 1},
		{"27-Inch IPS Monitor 1920x1080 144Hz", 1},
		{"Remote Control 4x4 Offroad Truck", 1},
	}
	for _, tc := range cases {
		if got := ParsePackMultiplier(tc.title); got != tc.want {
			t.Fatalf("ParsePackMultiplier(%q) = %d, want %d", tc.title, got, tc.want)
		}
	}

	// Verify linear O(N) runtime on 50,000-character unclosed/repetitive inputs (no polynomial backtracking).
	adversarialInputs := []string{
		"6x" + strings.Repeat(" ", 50000) + "500",
		"12" + strings.Repeat("-", 50000) + "unclosed",
		"pack" + strings.Repeat(" ", 50000) + "of",
		strings.Repeat("9", 50000) + "x330ml",
	}
	start := time.Now()
	for _, input := range adversarialInputs {
		if got := ParsePackMultiplier(input); got != 1 {
			t.Fatalf("ParsePackMultiplier(len=%d) = %d, want 1", len(input), got)
		}
	}
	if elapsed := time.Since(start); elapsed > 250*time.Millisecond {
		t.Fatalf("50,000-char ParsePackMultiplier suite took %v, want <= 250ms", elapsed)
	}

	if ValidatePackQuantityConsistency(
		"San Pellegrino 6x330ml Cans",
		"San Pellegrino Single 330ml Can",
		"exact_duplicate",
		"identical_pack_and_volume",
	) {
		t.Fatalf("ValidatePackQuantityConsistency = true, want false on 6-pack vs single mismatch")
	}

	if !ValidatePackQuantityConsistency(
		"San Pellegrino 6x330ml Cans",
		"San Pellegrino Single 330ml Can",
		"sibling_variant",
		"multipack_vs_single_mismatch",
	) {
		t.Fatalf("ValidatePackQuantityConsistency = false, want true on sibling_variant")
	}
}

func TestNormalizedOrdinalVarianceAndSelectiveEscalation(t *testing.T) {
	varCases := []struct {
		name  string
		probs []float64
		want  float64
	}{
		{"empty_fails_closed_zero", nil, 0.0},
		{"singleton_zero", []float64{1.0}, 0.0},
		{"point_mass_5pt", []float64{0, 0, 1, 0, 0}, 0.0},
		{"bimodal_polar_5pt_reaches_popoviciu_bound", []float64{0.5, 0, 0, 0, 0.5}, 1.0},
		{"bimodal_polar_4pt_scale_invariant", []float64{0.5, 0, 0, 0.5}, 1.0},
		{"benign_adjacent_5pt", []float64{0, 0.5, 0.5, 0, 0}, 0.0625},
		{"uniform_5pt", []float64{0.2, 0.2, 0.2, 0.2, 0.2}, 0.5},
		{"negative_entry_fails_closed_zero", []float64{0.6, -0.1, 0.5}, 0.0},
		{"nan_entry_fails_closed_zero", []float64{0.5, math.NaN(), 0.5}, 0.0},
		{"inf_entry_fails_closed_zero", []float64{0.5, math.Inf(1), 0.5}, 0.0},
	}
	for _, tc := range varCases {
		got := NormalizedOrdinalVariance(tc.probs)
		if math.Abs(got-tc.want) > 1e-12 {
			t.Fatalf("%s: NormalizedOrdinalVariance(%v) = %.12f, want %.12f", tc.name, tc.probs, got, tc.want)
		}
	}

	sidak := SidakSlotAlpha(0.10, 8)
	if math.Abs(sidak-0.0130842) > 1e-5 {
		t.Fatalf("SidakSlotAlpha(0.10, 8) = %.7f, want ~0.0130842", sidak)
	}

	// Invalid probability inputs fail closed to escalate=true.
	if !ShouldEscalateOrdinalSlot(nil, 0.10, sidak) {
		t.Fatalf("ShouldEscalateOrdinalSlot(nil) = false, want true")
	}
	if !ShouldEscalateOrdinalSlot([]float64{0.5, math.NaN(), 0.5}, 0.10, sidak) {
		t.Fatalf("ShouldEscalateOrdinalSlot(NaN) = false, want true")
	}

	// Benign adjacent split [0, 0.5, 0.5, 0, 0] has H_norm = 0.4307 (>= 0.16) yet Popoviciu variance = 0.0625 <= 0.10:
	// selective slot gating keeps it without escalating the bundle, whereas bimodal [0.5, 0, 0, 0, 0.5] escalates.
	levels5 := []string{"l0", "l1", "l2", "l3", "l4"}
	bundle := []BundleSlotGateInput{
		{
			SlotID:        "benign_adjacent_score",
			OrderedLevels: levels5,
			Probabilities: map[string]float64{"l0": 0.0, "l1": 0.5, "l2": 0.5, "l3": 0.0, "l4": 0.0},
			Hesitation:    0.43,
		},
		{
			SlotID:        "bimodal_polar_score",
			OrderedLevels: levels5,
			Probabilities: map[string]float64{"l0": 0.5, "l1": 0.0, "l2": 0.0, "l3": 0.0, "l4": 0.5},
			Hesitation:    0.43,
		},
		{
			SlotID:        "confident_nominal",
			Probabilities: map[string]float64{"yes": 0.97, "no": 0.03},
			Hesitation:    0.04,
		},
		{
			SlotID:        "hesitant_nominal",
			Probabilities: map[string]float64{"yes": 0.55, "no": 0.45},
			Hesitation:    0.29,
		},
	}
	escalated := SelectEscalatedBundleSlots(bundle, 0.16, 0.10, sidak)
	wantEscalated := []string{"bimodal_polar_score", "hesitant_nominal"}
	if fmt.Sprintf("%v", escalated) != fmt.Sprintf("%v", wantEscalated) {
		t.Fatalf("SelectEscalatedBundleSlots = %v, want %v", escalated, wantEscalated)
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
			Hesitation:  0.29,
		},
	}

	dec := ArbitrateSwarmLocks(proposals, 0.16)
	if !dec.WriteCollision {
		t.Fatalf("WriteCollision = false, want true")
	}
	if dec.WinningAgent != "agent_1" {
		t.Fatalf("WinningAgent = %q, want \"agent_1\"", dec.WinningAgent)
	}
	if dec.AgentActions["agent_1"] != "grant_lock" {
		t.Fatalf("agent_1 action = %q, want \"grant_lock\"", dec.AgentActions["agent_1"])
	}
	if dec.AgentActions["agent_2"] != "serialize_after_peer" {
		t.Fatalf("agent_2 action = %q, want \"serialize_after_peer\"", dec.AgentActions["agent_2"])
	}
	if dec.AgentActions["agent_3"] != "grant_lock" {
		t.Fatalf("agent_3 action = %q, want \"grant_lock\"", dec.AgentActions["agent_3"])
	}
	if dec.AgentActions["agent_4"] != "escalate_conflict" || !dec.EscalateToHuman {
		t.Fatalf("agent_4 action = %q EscalateToHuman = %v, want \"escalate_conflict\" / true",
			dec.AgentActions["agent_4"], dec.EscalateToHuman)
	}
}

func TestScoreProbabilisticTrianglesAndSwarmLockSafety(t *testing.T) {
	nearBoundary := []PairwiseEdge{
		{NodeA: "n1", NodeB: "n2", EquivProb: 0.79},
		{NodeA: "n2", NodeB: "n3", EquivProb: 0.85},
		{NodeA: "n1", NodeB: "n3", EquivProb: 0.22},
	}
	if hard := CheckTransitiveTriangles(nearBoundary, 0.80, 0.25); len(hard) != 0 {
		t.Fatalf("CheckTransitiveTriangles(nearBoundary) = %+v, want empty", hard)
	}
	soft := ScoreProbabilisticTriangles(nearBoundary, 0.35)
	if len(soft) != 1 || soft[0].WeakEdge != [2]string{"n1", "n3"} || math.Abs(soft[0].MaxViolationMass-0.52377) > 1e-4 {
		t.Fatalf("ScoreProbabilisticTriangles(nearBoundary) = %+v, want 1 hit (n1,n3) with mass ~0.5238", soft)
	}

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
			pab = sigmoid(2.1 + 0.65*rng.NormFloat64())
			pbc = sigmoid(2.1 + 0.65*rng.NormFloat64())
			pac = sigmoid(2.1 + 0.65*rng.NormFloat64())
		case 1:
			pab = sigmoid(-2.1 + 0.65*rng.NormFloat64())
			pbc = sigmoid(-2.1 + 0.65*rng.NormFloat64())
			pac = sigmoid(-2.1 + 0.65*rng.NormFloat64())
		case 2:
			pab = sigmoid(2.1 + 0.65*rng.NormFloat64())
			pbc = sigmoid(-2.1 + 0.65*rng.NormFloat64())
			pac = sigmoid(-2.1 + 0.65*rng.NormFloat64())
		case 3:
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

	aurocSoft := computeBinaryAUROC(softScores, isContradiction)
	aurocHard := computeBinaryAUROC(hardScores, isContradiction)
	if aurocSoft < 0.94 || aurocSoft <= aurocHard {
		t.Fatalf("aurocSoft = %.4f, aurocHard = %.4f; want aurocSoft >= 0.94 and > aurocHard", aurocSoft, aurocHard)
	}

	filePool := []string{"cmd/serve.go", "pkg/decisionindex/engine.go", "pkg/client/client.go", "docs/README.md", "Makefile"}
	for round := 0; round < 400; round++ {
		var props []SwarmAgentProposal
		for a := 0; a < 4; a++ {
			p1 := filePool[rng.Intn(len(filePool))]
			p2 := filePool[rng.Intn(len(filePool))]
			props = append(props, SwarmAgentProposal{
				AgentID:     fmt.Sprintf("worker_%d", a),
				TargetPaths: []string{p1, p2},
				Confidence:  0.55 + 0.44*rng.Float64(),
				Hesitation:  0.02 + 0.24*rng.Float64(),
			})
		}
		dec := ArbitrateSwarmLocks(props, 0.16)
		grantedOwner := make(map[string]string)
		for _, p := range props {
			if dec.AgentActions[p.AgentID] != "grant_lock" {
				continue
			}
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
					t.Fatalf("round %d: path %s double-granted to %s and %s", round, path, prev, p.AgentID)
				}
				grantedOwner[path] = p.AgentID
			}
		}
	}
}

func TestGenkitPipelinesEndToEndWithTemplates(t *testing.T) {
	// Mock upstream /v1/chat/completions server that answers any rendered dgem schema deterministically.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		var schema struct {
			Questions []struct {
				ID      string   `json:"id"`
				Type    string   `json:"type"`
				Levels  []string `json:"levels"`
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range body.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &schema)
			}
		}
		answers := make(map[string]client.QuestionAnswer, len(schema.Questions))
		for _, q := range schema.Questions {
			switch q.Type {
			case "boolean", "noul":
				answers[q.ID] = client.QuestionAnswer{
					Type:          "noul",
					Label:         "yes",
					Choice:        "yes",
					Confidence:    0.96,
					Probabilities: map[string]float64{"yes": 0.96, "no": 0.04},
				}
			case "score":
				lv := q.Levels[0]
				probs := make(map[string]float64, len(q.Levels))
				for i, k := range q.Levels {
					if i == 0 {
						probs[k] = 0.94
					} else {
						probs[k] = 0.06 / float64(len(q.Levels)-1)
					}
				}
				answers[q.ID] = client.QuestionAnswer{
					Type:          "score",
					Level:         lv,
					Choice:        lv,
					Confidence:    0.94,
					Probabilities: probs,
				}
			default:
				win := q.Options[0].Name
				probs := make(map[string]float64, len(q.Options))
				for i, o := range q.Options {
					if i == 0 {
						probs[o.Name] = 0.97
					} else {
						probs[o.Name] = 0.03 / float64(len(q.Options)-1)
					}
				}
				answers[q.ID] = client.QuestionAnswer{
					Type:          "choice",
					Choice:        win,
					Confidence:    0.97,
					Probabilities: probs,
				}
			}
		}
		contentBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "diffgemma-26b-a4b-it-q4",
			"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": string(contentBytes)}}},
		})
	}))
	defer upstream.Close()

	cfg := dgemgenkit.EndpointConfig{URL: upstream.URL + "/v1", ModelID: "diffgemma-26b-a4b-it-q4"}
	plugin, cleanup := dgemgenkit.NewSystemOnePlugin(cfg)
	defer cleanup()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	modelName := cfg.GenkitModelName()
	templatesDir := filepath.Join("..", "..", "templates")

	// 1. Entity pair where mock predicts exact_duplicate, but titles are 6x330ml vs Single -> guardrail quarantines!
	ent, err := EvaluateEntityPairWithGuardrail(ctx, g, modelName, templatesDir,
		"sku_1", "Sparkling Water 6x330ml Cans",
		"sku_2", "Sparkling Water 330ml Single Can",
	)
	if err != nil {
		t.Fatalf("EvaluateEntityPairWithGuardrail error: %v", err)
	}
	if ent.PackConsistent || ent.MergeAction != "quarantine_for_human_review" {
		t.Fatalf("ent = %+v, want PackConsistent=false MergeAction=\"quarantine_for_human_review\"", ent)
	}

	// 2. Swarm lock proposals where agent_1 and agent_2 collide on engine.go -> agent_1 granted, agent_2 serialized.
	workers := [4]SwarmAgentProposal{
		{AgentID: "agent_1", Proposal: "Edit engine.go", TargetPaths: []string{"pkg/decisionindex/engine.go"}},
		{AgentID: "agent_2", Proposal: "Refactor engine.go", TargetPaths: []string{"pkg/decisionindex/engine.go"}},
		{AgentID: "agent_3", Proposal: "Update README", TargetPaths: []string{"README.md"}},
		{AgentID: "agent_4", Proposal: "Update Makefile", TargetPaths: []string{"Makefile"}},
	}
	lockDec, _, err := EvaluateSwarmLockProposals(ctx, g, modelName, templatesDir, workers, 0.16)
	if err != nil {
		t.Fatalf("EvaluateSwarmLockProposals error: %v", err)
	}
	if !lockDec.WriteCollision || lockDec.AgentActions["agent_1"] != "grant_lock" || lockDec.AgentActions["agent_2"] != "serialize_after_peer" {
		t.Fatalf("lockDec = %+v, want WriteCollision=true agent_1=grant_lock agent_2=serialize_after_peer", lockDec)
	}
	if lockDec.RollbackSeverity != "clean_fast_forward" || lockDec.RollbackOrdinalVar >= 0.121 || fmt.Sprintf("%v", lockDec.EscalatedSlots) != "[file_write_collision]" {
		t.Fatalf("lockDec ordinal gating = %+v, want RollbackSeverity=clean_fast_forward RollbackOrdinalVar<0.121 EscalatedSlots=[file_write_collision]", lockDec)
	}

	// 3. Live checklist window.
	chk, err := EvaluateChecklistWindow(ctx, g, modelName, templatesDir, "win-01", "Operator verified badge and consent.")
	if err != nil {
		t.Fatalf("EvaluateChecklistWindow error: %v", err)
	}
	if !chk.CommitmentRecorded || chk.InterruptUrgency != "silent_no_alert" || chk.UrgencyOrdinalVar >= 0.121 || fmt.Sprintf("%v", chk.EscalatedSlots) != "[speaker_commitment_present]" || chk.SidakSlotAlpha <= 0 {
		t.Fatalf("chk = %+v, want CommitmentRecorded=true InterruptUrgency=silent_no_alert UrgencyOrdinalVar<0.121 EscalatedSlots=[speaker_commitment_present]", chk)
	}
}
