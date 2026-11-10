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
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func setContainsLabel(set []string, target string) bool {
	for _, s := range set {
		if s == target {
			return true
		}
	}
	return false
}

func TestConformalStrictValidationAndEdgeCases(t *testing.T) {
	badProbMaps := []map[string]float64{
		nil,
		{},
		{"a": math.NaN(), "b": 0.5},
		{"a": math.Inf(1), "b": 0.5},
		{"a": math.Inf(-1), "b": 0.5},
		{"a": -0.1, "b": 1.1},
		{"a": 0.0, "b": 0.0},
	}
	for idx, badMap := range badProbMaps {
		if _, err := ConformalScore(badMap, "a", ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d ConformalScore err = %v, want %v", idx, err, ErrInvalidProbabilities)
		}
		if _, err := CalibrateConformalQuantile([]map[string]float64{badMap}, []string{"a"}, 0.10, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d CalibrateConformalQuantile err = %v, want %v", idx, err, ErrInvalidProbabilities)
		}
		if _, err := ConformalPredictionSet(badMap, 0.20, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d ConformalPredictionSet err = %v, want %v", idx, err, ErrInvalidProbabilities)
		}
		if _, err := EvaluateQuestionConformalSet(QuestionAnswer{Probabilities: badMap}, nil, 0.20, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d EvaluateQuestionConformalSet err = %v, want %v", idx, err, ErrInvalidProbabilities)
		}
	}

	validProbs := map[string]float64{"a": 0.7, "b": 0.3}
	for _, badQ := range []float64{-0.01, math.NaN(), math.Inf(-1)} {
		if _, err := ConformalPredictionSet(validProbs, badQ, ConformalConfig{}); !errors.Is(err, ErrInvalidConformalQHat) {
			t.Fatalf("qHat=%v err = %v, want %v", badQ, err, ErrInvalidConformalQHat)
		}
	}
	if _, err := ConformalPredictionSet(validProbs, 0.20, ConformalConfig{Method: "bogus"}); !errors.Is(err, ErrInvalidConformalMethod) {
		t.Fatalf("ConformalPredictionSet(bogus) err = %v, want %v", err, ErrInvalidConformalMethod)
	}
	for _, badCfg := range []ConformalConfig{
		{Method: ConformalRAPS, KReg: -1, LambdaReg: 0.02},
		{Method: ConformalRAPS, KReg: 2, LambdaReg: -0.05},
		{Method: ConformalRAPS, KReg: 2, LambdaReg: math.NaN()},
		{Method: ConformalRAPS, KReg: 2, LambdaReg: math.Inf(1)},
	} {
		if _, err := ConformalScore(validProbs, "a", badCfg); !errors.Is(err, ErrInvalidConformalParam) {
			t.Fatalf("ConformalScore(%+v) err = %v, want %v", badCfg, err, ErrInvalidConformalParam)
		}
	}

	defCfg := DefaultConformalConfig()
	if defCfg.Method != ConformalRAPS || defCfg.Alpha != 0.10 || defCfg.KReg != 1 || defCfg.LambdaReg != 0.01 {
		t.Fatalf("DefaultConformalConfig() = %+v, want raps/0.10/1/0.01", defCfg)
	}
	sixOpts := map[string]float64{"o1": 0.30, "o2": 0.25, "o3": 0.20, "o4": 0.12, "o5": 0.08, "o6": 0.05}
	for _, cfgUse := range []ConformalConfig{defCfg, {}} {
		s5, err := ConformalScore(sixOpts, "o5", cfgUse)
		if err != nil {
			t.Fatalf("ConformalScore o5 error: %v", err)
		}
		s6, err := ConformalScore(sixOpts, "o6", cfgUse)
		if err != nil {
			t.Fatalf("ConformalScore o6 error: %v", err)
		}
		if math.Abs(s5-0.91) > 1e-12 || math.Abs(s6-1.00) > 1e-12 {
			t.Fatalf("got (s5=%.12f, s6=%.12f), want (0.91, 1.00)", s5, s6)
		}
	}

	nineProbs := make([]map[string]float64, 9)
	nineLabels := make([]string, 9)
	for i := 0; i < 9; i++ {
		pTop := 0.95 - float64(i)*0.05
		nineProbs[i] = map[string]float64{"yes": pTop, "no": 1.0 - pTop}
		nineLabels[i] = "true" // boolean alias resolution
	}
	cal9, err := CalibrateConformalQuantile(nineProbs, nineLabels, 0.30, ConformalConfig{Method: ConformalLAC})
	if err != nil {
		t.Fatalf("CalibrateConformalQuantile(n=9, alpha=0.30) error: %v", err)
	}
	if cal9.OrderIndex != 7 {
		t.Fatalf("cal9.OrderIndex = %d, want 7", cal9.OrderIndex)
	}
}

func TestConformalPreFusionSafetyAndAbstainTriggers(t *testing.T) {
	// Pre-fusion probabilities are ambiguous (52/48), while caller prior inflated post-fusion to 96/4.
	qa := QuestionAnswer{
		Choice:        "approve",
		Probabilities: map[string]float64{"approve": 0.96, "deny": 0.04},
	}
	raw := map[string]float64{"approve": 0.52, "deny": 0.48}
	got, err := EvaluateQuestionConformalSet(qa, raw, 0.55, ConformalConfig{Method: ConformalLAC})
	if err != nil {
		t.Fatalf("EvaluateQuestionConformalSet error: %v", err)
	}
	if got.SetSize != 2 || !got.ShouldEscalate {
		t.Fatalf("got %+v, want SetSize=2 ShouldEscalate=true on pre-fusion ambiguity", got)
	}
	// Contrast: without raw probabilities, inflated post-fusion 0.96/0.04 collapses to SetSize=1.
	postOnly, err := EvaluateQuestionConformalSet(qa, nil, 0.55, ConformalConfig{Method: ConformalLAC})
	if err != nil || postOnly.SetSize != 1 || postOnly.ShouldEscalate {
		t.Fatalf("postOnly = %+v err=%v, want SetSize=1 ShouldEscalate=false", postOnly, err)
	}

	for _, absKey := range []string{"0", "abstain", "insufficient_evidence", "none", "none_of_the_above", "none_visible", "oos", "out_of_scope", "uncertain", "unknown", " OOS "} {
		absRes, absErr := ConformalPredictionSet(map[string]float64{absKey: 0.96, "valid_opt": 0.04}, 0.20, ConformalConfig{Method: ConformalLAC})
		if absErr != nil || !absRes.ContainsAbstain || !absRes.ShouldEscalate {
			t.Fatalf("abstain key %q got res=%+v err=%v, want ContainsAbstain=true ShouldEscalate=true", absKey, absRes, absErr)
		}
	}
	optOutRes, optOutErr := ConformalPredictionSet(map[string]float64{"oos": 0.96, "valid_opt": 0.04}, 0.20, ConformalConfig{Method: ConformalLAC, AbstainLabels: []string{}})
	if optOutErr != nil || optOutRes.ContainsAbstain || optOutRes.ShouldEscalate || !optOutRes.Singleton {
		t.Fatalf("opt-out got res=%+v err=%v, want Singleton=true ShouldEscalate=false", optOutRes, optOutErr)
	}
}

func TestStatelessHMACRandomizedCoverageAndConcurrency(t *testing.T) {
	evalRNG := rand.New(rand.NewSource(101))
	const numSplits = 12
	const calN = 200
	const testN = 200
	tieHits := 0
	singletonA, singletonB := 0, 0
	var sampleQHat float64

	for sp := 0; sp < numSplits; sp++ {
		tieCalProbs := make([]map[string]float64, calN)
		tieCalLabels := make([]string, calN)
		tieCalKeys := make([]string, calN)
		for i := 0; i < calN; i++ {
			tieCalProbs[i] = map[string]float64{"A": 0.5, "B": 0.5}
			tieCalKeys[i] = fmt.Sprintf("sp-%d-cal-%d", sp, i)
			if evalRNG.Float64() < 0.5 {
				tieCalLabels[i] = "A"
			} else {
				tieCalLabels[i] = "B"
			}
		}
		baseCfg := ConformalConfig{Method: ConformalAPS, Randomized: true, SeedKey: fmt.Sprintf("seed-%d", sp)}
		calTie, err := CalibrateConformalQuantileWithKeys(tieCalProbs, tieCalLabels, tieCalKeys, 0.25, baseCfg)
		if err != nil {
			t.Fatalf("CalibrateConformalQuantileWithKeys(split=%d) error: %v", sp, err)
		}
		sampleQHat = calTie.QHat
		for j := 0; j < testN; j++ {
			gold := "A"
			if evalRNG.Float64() < 0.5 {
				gold = "B"
			}
			itemCfg := baseCfg
			itemCfg.ItemKey = fmt.Sprintf("sp-%d-eval-%d", sp, j)
			res, err := ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, calTie.QHat, itemCfg)
			if err != nil {
				t.Fatalf("ConformalPredictionSet(split=%d, j=%d) error: %v", sp, j, err)
			}
			if setContainsLabel(res.PredictionSet, gold) {
				tieHits++
			}
			if res.SetSize == 1 {
				if res.PredictionSet[0] == "A" {
					singletonA++
				} else if res.PredictionSet[0] == "B" {
					singletonB++
				}
			}
		}
	}
	if cov := float64(tieHits) / float64(numSplits*testN); cov < 0.72 {
		t.Fatalf("marginal randomized tie coverage = %.3f, want >= 0.72 at alpha=0.25", cov)
	}
	if singletonA == 0 || singletonB == 0 {
		t.Fatalf("got singletons A=%d B=%d, want both > 0", singletonA, singletonB)
	}

	// Verify pure determinism across concurrent goroutines with zero shared mutable state.
	wantSet, err := ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, sampleQHat, ConformalConfig{
		Method:     ConformalAPS,
		Randomized: true,
		SeedKey:    "bench-seed-7",
		ItemKey:    "req-fixed-99",
	})
	if err != nil {
		t.Fatalf("ConformalPredictionSet error: %v", err)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < 50; k++ {
				got, cErr := ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, sampleQHat, ConformalConfig{
					Method:     ConformalAPS,
					Randomized: true,
					SeedKey:    "bench-seed-7",
					ItemKey:    "req-fixed-99",
				})
				if cErr != nil || len(got.PredictionSet) != len(wantSet.PredictionSet) {
					t.Errorf("concurrent prediction mismatch: got %+v err=%v, want %+v", got, cErr, wantSet)
					return
				}
			}
		}()
	}
	wg.Wait()
}
