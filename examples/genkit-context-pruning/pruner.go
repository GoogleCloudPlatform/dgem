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
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/firebase/genkit/go/genkit"
)

const (
	// MaxContextSlotsPerPass matches the 8-slot single-pass dgem canvas limit.
	MaxContextSlotsPerPass = 8
	// ContextActionFull retains the complete trajectory block verbatim.
	ContextActionFull = "full"
	// ContextActionStub replaces the block body with its compact summary or signature stub.
	ContextActionStub = "stub"
	// ContextActionDrop omits the block from the active context window.
	ContextActionDrop = "drop"

	paretoUtilityTolerance = 1e-12
)

var (
	// ErrTooManyContextBlocks reports that more than MaxContextSlotsPerPass blocks were passed.
	ErrTooManyContextBlocks = errors.New("context block count exceeds 8-slot single-pass limit")
	// ErrNegativeTokenBudget reports a negative ContextPruneConfig.TokenBudget.
	ErrNegativeTokenBudget = errors.New("token budget must be non-negative")
	// ErrInvalidProbabilities reports an empty or non-finite action probability map.
	ErrInvalidProbabilities = errors.New("action probability map must be non-empty with finite non-negative values summing above zero")
	// ErrInvalidContextBlock reports negative/inverted token counts, invalid weights, or duplicate block IDs.
	ErrInvalidContextBlock = errors.New("invalid context block specification")
	// ErrInfeasiblePinnedBudget reports that TokenBudget cannot cover the minimum stub sum of pinned blocks.
	ErrInfeasiblePinnedBudget = errors.New("token budget below minimum pinned block stub cost")
)

// ContextBlockSpec describes one candidate trajectory block (up to 8 blocks per pass).
type ContextBlockSpec struct {
	ID               string  `json:"id"`
	Kind             string  `json:"kind,omitempty"`
	Summary          string  `json:"summary,omitempty"`
	FullText         string  `json:"full_text,omitempty"`
	StubText         string  `json:"stub_text,omitempty"`
	FullTokens       int     `json:"full_tokens"`
	StubTokens       int     `json:"stub_tokens"`
	Pinned           bool    `json:"pinned,omitempty"`
	ImportanceWeight float64 `json:"importance_weight,omitempty"`
}

// ContextPruneConfig controls expected-utility Multiple-Choice Knapsack (MCKP) context pruning.
type ContextPruneConfig struct {
	TokenBudget          int     `json:"token_budget"`
	StubUtilityRatio     float64 `json:"stub_utility_ratio,omitempty"`
	UncertaintyStubBonus float64 `json:"uncertainty_stub_bonus,omitempty"`
}

// ContextBlockDecision records the selected retention action, token accounting, and probabilities for one block.
type ContextBlockDecision struct {
	BlockID           string             `json:"block_id"`
	SlotID            string             `json:"slot_id"`
	Action            string             `json:"action"`
	RetainedText      string             `json:"retained_text,omitempty"`
	RetainedTokens    int                `json:"retained_tokens"`
	SavedTokens       int                `json:"saved_tokens"`
	NormalizedEntropy float64            `json:"normalized_entropy"`
	Probabilities     map[string]float64 `json:"probabilities"`
}

// ContextPruneResult aggregates per-block pruning decisions and token savings.
type ContextPruneResult struct {
	Decisions       []ContextBlockDecision `json:"decisions"`
	OriginalTokens  int                    `json:"original_tokens"`
	RetainedTokens  int                    `json:"retained_tokens"`
	SavedTokens     int                    `json:"saved_tokens"`
	TokenBudget     int                    `json:"token_budget"`
	BudgetSatisfied bool                   `json:"budget_satisfied"`
	ExpectedUtility float64                `json:"expected_utility"`
	WallTimeMs      int64                  `json:"wall_time_ms,omitempty"`
}

type preparedBlock struct {
	slotID  string
	blockID string
	costs   [3]int
	utils   [3]float64
	probs   [3]float64
	hNorm   float64
	pinned  bool
}

type mckpState struct {
	tokens  int
	utility float64
	choices [MaxContextSlotsPerPass]int
	depth   int
}

// NormalizedShannonEntropy computes H(p) / ln(K) in [0, 1] for K non-negative probabilities.
func NormalizedShannonEntropy(probs []float64) float64 {
	k := len(probs)
	if k <= 1 {
		return 0
	}
	sum := 0.0
	for _, p := range probs {
		if !math.IsNaN(p) && !math.IsInf(p, 0) && p > 0 {
			sum += p
		}
	}
	if sum <= 0 {
		return 0
	}
	h := 0.0
	for _, p := range probs {
		if !math.IsNaN(p) && !math.IsInf(p, 0) && p > 0 {
			pn := p / sum
			h -= pn * math.Log(pn)
		}
	}
	normH := h / math.Log(float64(k))
	if normH < 0 {
		return 0
	}
	if normH > 1 {
		return 1
	}
	return normH
}

func validateContextBlockSpec(b ContextBlockSpec, idx int, seenIDs map[string]bool) (string, string, error) {
	slotID := fmt.Sprintf("b%d", idx)
	blockID := strings.TrimSpace(b.ID)
	if blockID == "" {
		blockID = slotID
	}
	if seenIDs[blockID] {
		return "", "", fmt.Errorf("%w: duplicate block ID %q", ErrInvalidContextBlock, blockID)
	}
	seenIDs[blockID] = true
	if b.StubTokens < 0 || b.FullTokens < 0 || b.FullTokens < b.StubTokens {
		return "", "", fmt.Errorf("%w: block %q requires full_tokens (%d) >= stub_tokens (%d) >= 0", ErrInvalidContextBlock, blockID, b.FullTokens, b.StubTokens)
	}
	if math.IsNaN(b.ImportanceWeight) || math.IsInf(b.ImportanceWeight, 0) || b.ImportanceWeight < 0 {
		return "", "", fmt.Errorf("%w: block %q has invalid importance_weight %v", ErrInvalidContextBlock, blockID, b.ImportanceWeight)
	}
	return slotID, blockID, nil
}

func validateActionTriplet(blockID string, raw map[string]float64) (float64, float64, float64, error) {
	if len(raw) == 0 {
		return 0, 0, 0, fmt.Errorf("%w: missing action probabilities for block %q", ErrInvalidProbabilities, blockID)
	}
	for k, v := range raw {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return 0, 0, 0, fmt.Errorf("%w: block %q action %q has invalid probability %v", ErrInvalidProbabilities, blockID, k, v)
		}
	}
	vFull := raw[ContextActionFull]
	vStub := raw[ContextActionStub]
	vDrop := raw[ContextActionDrop]
	sum := vFull + vStub + vDrop
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return 0, 0, 0, fmt.Errorf("%w: block %q action probabilities sum to %v", ErrInvalidProbabilities, blockID, sum)
	}
	return vFull / sum, vStub / sum, vDrop / sum, nil
}

// SelectContextPruningActions solves the exact Multiple-Choice Knapsack Problem across up to 8 blocks under cfg.TokenBudget.
func SelectContextPruningActions(blocks []ContextBlockSpec, slotProbs map[string]map[string]float64, cfg ContextPruneConfig) (ContextPruneResult, error) {
	if len(blocks) > MaxContextSlotsPerPass {
		return ContextPruneResult{}, fmt.Errorf("%w: got %d", ErrTooManyContextBlocks, len(blocks))
	}
	if cfg.TokenBudget < 0 {
		return ContextPruneResult{}, fmt.Errorf("%w: got %d", ErrNegativeTokenBudget, cfg.TokenBudget)
	}
	if math.IsNaN(cfg.StubUtilityRatio) || math.IsInf(cfg.StubUtilityRatio, 0) || cfg.StubUtilityRatio < 0 || cfg.StubUtilityRatio > 1 {
		return ContextPruneResult{}, fmt.Errorf("%w: stub_utility_ratio %v must lie in [0, 1]", ErrInvalidContextBlock, cfg.StubUtilityRatio)
	}
	if math.IsNaN(cfg.UncertaintyStubBonus) || math.IsInf(cfg.UncertaintyStubBonus, 0) {
		return ContextPruneResult{}, fmt.Errorf("%w: uncertainty_stub_bonus %v must be finite", ErrInvalidContextBlock, cfg.UncertaintyStubBonus)
	}
	if len(blocks) == 0 {
		return ContextPruneResult{
			Decisions:       []ContextBlockDecision{},
			TokenBudget:     cfg.TokenBudget,
			BudgetSatisfied: true,
		}, nil
	}

	m := len(blocks)
	prep := make([]preparedBlock, m)
	seenIDs := make(map[string]bool, m)
	origTokens := 0
	minPinnedStubTokens := 0

	for i, b := range blocks {
		slotID, blockID, err := validateContextBlockSpec(b, i, seenIDs)
		if err != nil {
			return ContextPruneResult{}, err
		}
		origTokens += b.FullTokens
		if b.Pinned {
			minPinnedStubTokens += b.StubTokens
		}

		rawMap, ok := lookupBlockProbs(slotProbs, b.ID, slotID)
		if !ok {
			return ContextPruneResult{}, fmt.Errorf("%w: missing slotProbs entry for block %q (%s)", ErrInvalidProbabilities, blockID, slotID)
		}

		pFull, pStub, pDrop, pErr := validateActionTriplet(blockID, rawMap)
		if pErr != nil {
			return ContextPruneResult{}, pErr
		}
		hNorm := NormalizedShannonEntropy([]float64{pFull, pStub, pDrop})

		w := b.ImportanceWeight
		if w == 0 {
			w = 1.0
		}

		var uFull, uStub float64
		if cfg.StubUtilityRatio == 0 && cfg.UncertaintyStubBonus == 0 {
			uFull = w * (1.0*pFull + 0.5*pStub)
			uStub = w * (0.6*pFull + 1.0*pStub)
		} else {
			rho := cfg.StubUtilityRatio
			if rho == 0 {
				rho = 0.6
			}
			beta := max(0.0, cfg.UncertaintyStubBonus)
			uFull = w * (1.0*pFull + rho*pStub)
			uStub = w * (rho*pFull + 1.0*pStub + beta*hNorm*(1.0-pDrop))
		}

		prep[i] = preparedBlock{
			slotID:  slotID,
			blockID: blockID,
			costs:   [3]int{b.FullTokens, b.StubTokens, 0},
			utils:   [3]float64{uFull, uStub, 0},
			probs:   [3]float64{pFull, pStub, pDrop},
			hNorm:   hNorm,
			pinned:  b.Pinned,
		}
	}

	if minPinnedStubTokens > cfg.TokenBudget {
		return ContextPruneResult{}, fmt.Errorf("%w: budget %d < minimum pinned stub tokens %d", ErrInfeasiblePinnedBudget, cfg.TokenBudget, minPinnedStubTokens)
	}

	frontier := []mckpState{{tokens: 0, utility: 0}}
	for i := 0; i < m; i++ {
		maxAct := 3
		if prep[i].pinned {
			maxAct = 2
		}
		expanded := make([]mckpState, 0, len(frontier)*maxAct)
		for _, st := range frontier {
			for act := 0; act < maxAct; act++ {
				nt := st.tokens + prep[i].costs[act]
				if nt <= cfg.TokenBudget {
					next := mckpState{
						tokens:  nt,
						utility: st.utility + prep[i].utils[act],
						choices: st.choices,
						depth:   i + 1,
					}
					next.choices[i] = act
					expanded = append(expanded, next)
				}
			}
		}
		if len(expanded) == 0 {
			return ContextPruneResult{}, fmt.Errorf("%w: no feasible assignment within budget %d", ErrInfeasiblePinnedBudget, cfg.TokenBudget)
		}
		frontier = paretoFilterMCKPStates(expanded, 4096)
	}

	best := frontier[0]
	for _, st := range frontier[1:] {
		uDiff := math.Round(st.utility*1e12) - math.Round(best.utility*1e12)
		if uDiff > 0 ||
			(uDiff == 0 && st.tokens < best.tokens) ||
			(uDiff == 0 && st.tokens == best.tokens && compareMCKPChoices(st.choices, best.choices, m) > 0) {
			best = st
		}
	}
	return buildContextPruneResult(blocks, prep, best.choices[:m], origTokens, best.tokens, cfg.TokenBudget, true, best.utility), nil
}

func lookupBlockProbs(slotProbs map[string]map[string]float64, rawID, slotID string) (map[string]float64, bool) {
	if trimmed := strings.TrimSpace(rawID); trimmed != "" {
		if pm, ok := slotProbs[trimmed]; ok {
			return pm, true
		}
	}
	if rawID != "" {
		if pm, ok := slotProbs[rawID]; ok {
			return pm, true
		}
	}
	pm, ok := slotProbs[slotID]
	return pm, ok
}

func compareMCKPChoices(a, b [MaxContextSlotsPerPass]int, depth int) int {
	asciiOrder := [3]int{1, 2, 0}
	for i := 0; i < depth; i++ {
		ra, rb := asciiOrder[a[i]], asciiOrder[b[i]]
		if ra < rb {
			return -1
		}
		if ra > rb {
			return 1
		}
	}
	return 0
}

func paretoFilterMCKPStates(states []mckpState, maxStates int) []mckpState {
	sort.Slice(states, func(i, j int) bool {
		if states[i].tokens != states[j].tokens {
			return states[i].tokens < states[j].tokens
		}
		if states[i].utility != states[j].utility {
			return states[i].utility > states[j].utility
		}
		return compareMCKPChoices(states[i].choices, states[j].choices, states[i].depth) < 0
	})

	pareto := make([]mckpState, 0, len(states))
	bestU := -1.0
	for _, st := range states {
		if st.utility > bestU+paretoUtilityTolerance {
			pareto = append(pareto, st)
			bestU = st.utility
		}
	}
	if len(pareto) <= maxStates {
		return pareto
	}

	minT := pareto[0].tokens
	maxT := pareto[len(pareto)-1].tokens
	if maxT <= minT {
		return pareto[:1]
	}
	span := float64(maxT - minT)
	buckets := make([]mckpState, maxStates)
	hasBucket := make([]bool, maxStates)
	for _, st := range pareto {
		bIdx := int((float64(st.tokens-minT) / span) * float64(maxStates-1))
		if bIdx >= maxStates {
			bIdx = maxStates - 1
		}
		if !hasBucket[bIdx] || st.utility > buckets[bIdx].utility || (st.utility == buckets[bIdx].utility && st.tokens < buckets[bIdx].tokens) {
			buckets[bIdx] = st
			hasBucket[bIdx] = true
		}
	}
	compacted := make([]mckpState, 0, maxStates)
	for i := 0; i < maxStates; i++ {
		if hasBucket[i] {
			compacted = append(compacted, buckets[i])
		}
	}
	return compacted
}

func buildContextPruneResult(blocks []ContextBlockSpec, meta []preparedBlock, actions []int, origTokens, retainedTokens, budget int, satisfied bool, expectedUtil float64) ContextPruneResult {
	actionNames := [3]string{ContextActionFull, ContextActionStub, ContextActionDrop}
	decisions := make([]ContextBlockDecision, len(blocks))
	for i, b := range blocks {
		actIdx := actions[i]
		actName := actionNames[actIdx]
		retTok := meta[i].costs[actIdx]
		var retText string
		switch actIdx {
		case 0:
			retText = b.FullText
		case 1:
			retText = b.StubText
		}
		decisions[i] = ContextBlockDecision{
			BlockID:           meta[i].blockID,
			SlotID:            meta[i].slotID,
			Action:            actName,
			RetainedText:      retText,
			RetainedTokens:    retTok,
			SavedTokens:       meta[i].costs[0] - retTok,
			NormalizedEntropy: meta[i].hNorm,
			Probabilities: map[string]float64{
				ContextActionFull: meta[i].probs[0],
				ContextActionStub: meta[i].probs[1],
				ContextActionDrop: meta[i].probs[2],
			},
		}
	}
	return ContextPruneResult{
		Decisions:       decisions,
		OriginalTokens:  origTokens,
		RetainedTokens:  retainedTokens,
		SavedTokens:     origTokens - retainedTokens,
		TokenBudget:     budget,
		BudgetSatisfied: satisfied,
		ExpectedUtility: expectedUtil,
	}
}

// EvaluateAndPruneContextBlocks executes templates/agent_context_pruning.json.tmpl via Genkit and solves the exact MCKP under cfg.TokenBudget.
func EvaluateAndPruneContextBlocks(ctx context.Context, g *genkit.Genkit, modelName, templatesDir, taskGoal string, turnIndex int, blocks []ContextBlockSpec, cfg ContextPruneConfig) (ContextPruneResult, error) {
	if len(blocks) > MaxContextSlotsPerPass {
		return ContextPruneResult{}, fmt.Errorf("%w: got %d", ErrTooManyContextBlocks, len(blocks))
	}
	if len(blocks) == 0 {
		return SelectContextPruningActions(blocks, nil, cfg)
	}

	seenIDs := make(map[string]bool, len(blocks))
	blockPayload := make([]map[string]any, len(blocks))
	for i, b := range blocks {
		slotID, blockID, err := validateContextBlockSpec(b, i, seenIDs)
		if err != nil {
			return ContextPruneResult{}, err
		}
		blockPayload[i] = map[string]any{
			"id":          slotID,
			"block_id":    blockID,
			"kind":        b.Kind,
			"full_tokens": b.FullTokens,
			"stub_tokens": b.StubTokens,
			"pinned":      b.Pinned,
			"summary":     b.Summary,
		}
	}

	tmplPath := filepath.Join(templatesDir, "agent_context_pruning.json.tmpl")
	dec, err := dgemgenkit.DecideWithTemplate(ctx, g, modelName, tmplPath, map[string]any{
		"task_goal":    taskGoal,
		"turn_index":   turnIndex,
		"token_budget": cfg.TokenBudget,
		"blocks":       blockPayload,
		"samples":      1,
	})
	if err != nil {
		return ContextPruneResult{}, err
	}

	slotProbs := make(map[string]map[string]float64, len(blocks))
	for i := range blocks {
		slotID := fmt.Sprintf("b%d", i)
		ans := dec.Answers[slotID]
		probs := ans.Probabilities
		if len(probs) == 0 && ans.Choice != "" {
			probs = map[string]float64{ans.Choice: 1.0}
		}
		slotProbs[slotID] = probs
	}

	result, err := SelectContextPruningActions(blocks, slotProbs, cfg)
	if err != nil {
		return ContextPruneResult{}, err
	}
	result.WallTimeMs = dec.WallTimeMs
	return result, nil
}
