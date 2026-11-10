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
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/firebase/genkit/go/genkit"
)

func TestContextPruningValidationAndTieBreakDeterminism(t *testing.T) {
	validProbs := map[string]map[string]float64{
		"b0": {"full": 0.6, "stub": 0.3, "drop": 0.1},
		"b1": {"full": 0.5, "stub": 0.3, "drop": 0.2},
	}

	invalidBlockSlices := [][]ContextBlockSpec{
		{{ID: "neg_stub", FullTokens: 100, StubTokens: -1}},
		{{ID: "inverted", FullTokens: 40, StubTokens: 80}},
		{{ID: "neg_weight", FullTokens: 100, StubTokens: 40, ImportanceWeight: -0.5}},
		{{ID: "nan_weight", FullTokens: 100, StubTokens: 40, ImportanceWeight: math.NaN()}},
		{{ID: "dup", FullTokens: 100, StubTokens: 40}, {ID: "dup", FullTokens: 120, StubTokens: 30}},
	}
	for idx, badBlocks := range invalidBlockSlices {
		if _, err := SelectContextPruningActions(badBlocks, validProbs, ContextPruneConfig{TokenBudget: 500}); !errors.Is(err, ErrInvalidContextBlock) {
			t.Fatalf("case %d SelectContextPruningActions err = %v, want %v", idx, err, ErrInvalidContextBlock)
		}
	}

	nineBlocks := make([]ContextBlockSpec, 9)
	for i := range nineBlocks {
		nineBlocks[i] = ContextBlockSpec{FullTokens: 50, StubTokens: 10}
	}
	if _, err := SelectContextPruningActions(nineBlocks, validProbs, ContextPruneConfig{TokenBudget: 500}); !errors.Is(err, ErrTooManyContextBlocks) {
		t.Fatalf("SelectContextPruningActions(9 blocks) err = %v, want %v", err, ErrTooManyContextBlocks)
	}
	if _, err := SelectContextPruningActions(nil, nil, ContextPruneConfig{TokenBudget: -1}); !errors.Is(err, ErrNegativeTokenBudget) {
		t.Fatalf("SelectContextPruningActions(budget=-1) err = %v, want %v", err, ErrNegativeTokenBudget)
	}

	oneBlock := []ContextBlockSpec{{ID: "blk_a", FullTokens: 100, StubTokens: 40}}
	for _, badCfg := range []ContextPruneConfig{
		{TokenBudget: 200, StubUtilityRatio: -0.1},
		{TokenBudget: 200, StubUtilityRatio: 1.2},
		{TokenBudget: 200, StubUtilityRatio: math.NaN()},
		{TokenBudget: 200, UncertaintyStubBonus: math.Inf(1)},
	} {
		if _, err := SelectContextPruningActions(oneBlock, map[string]map[string]float64{"blk_a": {"full": 0.5, "stub": 0.3, "drop": 0.2}}, badCfg); !errors.Is(err, ErrInvalidContextBlock) {
			t.Fatalf("SelectContextPruningActions(%+v) err = %v, want %v", badCfg, err, ErrInvalidContextBlock)
		}
	}

	badSlotProbs := []map[string]map[string]float64{
		nil,
		{},
		{"other": {"full": 0.5, "stub": 0.3, "drop": 0.2}},
		{"blk_a": {}},
		{"blk_a": {"full": -0.2, "stub": 0.7, "drop": 0.5}},
		{"blk_a": {"full": math.NaN(), "stub": 0.5, "drop": 0.5}},
		{"blk_a": {"full": 0.0, "stub": 0.0, "drop": 0.0}},
	}
	for idx, sp := range badSlotProbs {
		if _, err := SelectContextPruningActions(oneBlock, sp, ContextPruneConfig{TokenBudget: 200}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("slotProbs case %d err = %v, want %v", idx, err, ErrInvalidProbabilities)
		}
	}

	symBlocks := []ContextBlockSpec{
		{ID: "b0", FullTokens: 100, StubTokens: 50},
		{ID: "b1", FullTokens: 100, StubTokens: 50},
	}
	symProbs := map[string]map[string]float64{
		"b0": {"full": 0.75, "stub": 0.25, "drop": 0.0},
		"b1": {"full": 0.75, "stub": 0.25, "drop": 0.0},
	}
	symRes, err := SelectContextPruningActions(symBlocks, symProbs, ContextPruneConfig{TokenBudget: 150})
	if err != nil {
		t.Fatalf("SelectContextPruningActions symmetric tie error: %v", err)
	}
	if symRes.Decisions[0].Action != ContextActionFull || symRes.Decisions[1].Action != ContextActionStub {
		t.Fatalf("got actions [%s, %s], want [full, stub]", symRes.Decisions[0].Action, symRes.Decisions[1].Action)
	}

	eightBlocks := []ContextBlockSpec{
		{ID: "b0", FullTokens: 320, StubTokens: 80, Pinned: true},
		{ID: "b1", FullTokens: 260, StubTokens: 65, Pinned: true},
		{ID: "b2", FullTokens: 480, StubTokens: 110},
		{ID: "b3", FullTokens: 390, StubTokens: 95},
		{ID: "b4", FullTokens: 510, StubTokens: 120},
		{ID: "b5", FullTokens: 290, StubTokens: 70},
		{ID: "b6", FullTokens: 610, StubTokens: 140},
		{ID: "b7", FullTokens: 350, StubTokens: 85},
	}
	eightProbs := map[string]map[string]float64{
		"b0": {"full": 0.72, "stub": 0.20, "drop": 0.08},
		"b1": {"full": 0.35, "stub": 0.50, "drop": 0.15},
		"b2": {"full": 0.64, "stub": 0.26, "drop": 0.10},
		"b3": {"full": 0.28, "stub": 0.52, "drop": 0.20},
		"b4": {"full": 0.81, "stub": 0.14, "drop": 0.05},
		"b5": {"full": 0.18, "stub": 0.32, "drop": 0.50},
		"b6": {"full": 0.55, "stub": 0.30, "drop": 0.15},
		"b7": {"full": 0.12, "stub": 0.22, "drop": 0.66},
	}
	if _, err := SelectContextPruningActions(eightBlocks, eightProbs, ContextPruneConfig{TokenBudget: 144}); !errors.Is(err, ErrInfeasiblePinnedBudget) {
		t.Fatalf("SelectContextPruningActions(budget=144) err = %v, want %v", err, ErrInfeasiblePinnedBudget)
	}
	prevUtil := -1.0
	for step := 0; step < 30; step++ {
		bgt := 145 + step*105
		r, err := SelectContextPruningActions(eightBlocks, eightProbs, ContextPruneConfig{TokenBudget: bgt})
		if err != nil {
			t.Fatalf("step %d (budget=%d) error: %v", step, bgt, err)
		}
		if !r.BudgetSatisfied || r.RetainedTokens > bgt {
			t.Fatalf("step %d retained = %d, want <= %d", step, r.RetainedTokens, bgt)
		}
		if r.Decisions[0].Action == ContextActionDrop || r.Decisions[1].Action == ContextActionDrop {
			t.Fatalf("step %d pinned block dropped: b0=%s b1=%s", step, r.Decisions[0].Action, r.Decisions[1].Action)
		}
		if r.ExpectedUtility+1e-12 < prevUtil {
			t.Fatalf("step %d utility = %.9f, want >= %.9f", step, r.ExpectedUtility, prevUtil)
		}
		prevUtil = r.ExpectedUtility
	}
}

func TestEvaluateAndPruneContextBlocksEndToEnd(t *testing.T) {
	forwardPasses := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwardPasses++
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var sysPayload struct {
			Questions []struct {
				ID string `json:"id"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}
		if len(sysPayload.Questions) > MaxContextSlotsPerPass {
			http.Error(w, "exceeded 8-slot canvas limit", http.StatusUnprocessableEntity)
			return
		}

		answers := make(map[string]client.QuestionAnswer, len(sysPayload.Questions))
		for idx, q := range sysPayload.Questions {
			var probs map[string]float64
			var chosen string
			switch {
			case idx < 2:
				chosen = "full"
				probs = map[string]float64{"full": 0.88, "stub": 0.09, "drop": 0.03}
			case idx < 4:
				chosen = "stub"
				probs = map[string]float64{"full": 0.35, "stub": 0.55, "drop": 0.10}
			default:
				chosen = "drop"
				probs = map[string]float64{"full": 0.04, "stub": 0.08, "drop": 0.88}
			}
			answers[q.ID] = client.QuestionAnswer{
				Type:          "choice",
				Choice:        chosen,
				Confidence:    probs[chosen],
				Probabilities: probs,
			}
		}
		contentBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "diffgemma-26b-a4b-it-q4",
			"choices": []map[string]any{
				{"index": 0, "message": map[string]any{"role": "assistant", "content": string(contentBytes)}},
			},
		})
	}))
	defer upstream.Close()

	cfg := dgemgenkit.EndpointConfig{URL: upstream.URL + "/v1", ModelID: "diffgemma-26b-a4b-it-q4"}
	plugin, cleanup := dgemgenkit.NewSystemOnePlugin(cfg)
	defer cleanup()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	templatesDir := filepath.Join("..", "..", "templates")

	blocks := sampleTrajectoryBlocks()
	pruned, err := EvaluateAndPruneContextBlocks(
		ctx,
		g,
		cfg.GenkitModelName(),
		templatesDir,
		"Fix data race in worker_pool.go Shutdown() reported by TestWorkerPoolDrain.",
		6,
		blocks,
		ContextPruneConfig{TokenBudget: 1200, UncertaintyStubBonus: 0.15},
	)
	if err != nil {
		t.Fatalf("EvaluateAndPruneContextBlocks error: %v", err)
	}
	if forwardPasses != 1 {
		t.Fatalf("forwardPasses = %d, want 1", forwardPasses)
	}
	if !pruned.BudgetSatisfied || pruned.RetainedTokens > 1200 || pruned.SavedTokens <= 0 {
		t.Fatalf("pruned = %+v, want BudgetSatisfied=true RetainedTokens<=1200 SavedTokens>0", pruned)
	}
	if pruned.Decisions[0].Action == ContextActionDrop || pruned.Decisions[1].Action == ContextActionDrop {
		t.Fatalf("pinned blocks dropped: %+v", pruned.Decisions[:2])
	}
}
