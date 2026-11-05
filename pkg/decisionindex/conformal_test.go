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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
)

func TestCalibrateConformalQuantileValidation(t *testing.T) {
	if _, err := CalibrateConformalQuantile(nil, nil, 0.10, ConformalConfig{}); !errors.Is(err, ErrEmptyCalibration) {
		t.Fatalf("expected ErrEmptyCalibration, got %v", err)
	}

	sampleProb := []map[string]float64{{"a": 0.7, "b": 0.3}}
	if _, err := CalibrateConformalQuantile(sampleProb, []string{"a", "b"}, 0.10, ConformalConfig{}); !errors.Is(err, ErrCalibrationLenMismatch) {
		t.Fatalf("expected ErrCalibrationLenMismatch, got %v", err)
	}

	for _, badAlpha := range []float64{0.0, 1.0, -0.1, 1.2, math.NaN(), math.Inf(1)} {
		if _, err := CalibrateConformalQuantile(sampleProb, []string{"a"}, badAlpha, ConformalConfig{}); !errors.Is(err, ErrInvalidAlpha) {
			t.Fatalf("alpha=%v: expected ErrInvalidAlpha, got %v", badAlpha, err)
		}
	}

	if _, err := CalibrateConformalQuantile(sampleProb, []string{"missing"}, 0.10, ConformalConfig{}); !errors.Is(err, ErrUnknownCalibrationKey) {
		t.Fatalf("expected ErrUnknownCalibrationKey, got %v", err)
	}
	if _, err := ConformalScore(sampleProb[0], "missing", ConformalConfig{}); !errors.Is(err, ErrUnknownCalibrationKey) {
		t.Fatalf("ConformalScore: expected ErrUnknownCalibrationKey, got %v", err)
	}

	badCfg := ConformalConfig{Method: ConformalMethod("invalid_mode")}
	if _, err := CalibrateConformalQuantile(sampleProb, []string{"a"}, 0.10, badCfg); !errors.Is(err, ErrInvalidConformalMethod) {
		t.Fatalf("expected ErrInvalidConformalMethod, got %v", err)
	}
	if _, err := ConformalScore(sampleProb[0], "a", badCfg); !errors.Is(err, ErrInvalidConformalMethod) {
		t.Fatalf("ConformalScore: expected ErrInvalidConformalMethod, got %v", err)
	}

	// When ceil((n+1)*(1-alpha) - 1e-12) > n (for example n=5, alpha=0.05 -> ceil(5.7)=6 > 5),
	// QHat must be +Inf so finite-sample marginal coverage >= 1-alpha holds unconditionally.
	smallProbs := make([]map[string]float64, 5)
	smallLabels := make([]string, 5)
	for i := 0; i < 5; i++ {
		smallProbs[i] = map[string]float64{"opt_0": 0.6, "opt_1": 0.4}
		smallLabels[i] = "opt_0"
	}
	calInf, err := CalibrateConformalQuantile(smallProbs, smallLabels, 0.05, ConformalConfig{Method: ConformalLAC})
	if err != nil {
		t.Fatalf("unexpected error on small n calibration: %v", err)
	}
	if !math.IsInf(calInf.QHat, 1) || calInf.OrderIndex != 6 || calInf.EmpiricalCover != 1.0 {
		t.Fatalf("expected +Inf QHat and OrderIndex=6, got QHat=%v OrderIndex=%d Cover=%v", calInf.QHat, calInf.OrderIndex, calInf.EmpiricalCover)
	}
	setInf, err := ConformalPredictionSet(map[string]float64{"opt_0": 0.9, "opt_1": 0.1}, calInf.QHat, ConformalConfig{Method: ConformalLAC})
	if err != nil {
		t.Fatalf("ConformalPredictionSet under +Inf QHat failed: %v", err)
	}
	if setInf.SetSize != 2 || !setInf.ShouldEscalate {
		t.Fatalf("expected full 2-option set under +Inf QHat, got %+v", setInf)
	}
}

func drawExchangeableMiscalibratedBatch(rng *rand.Rand, n int, numClasses int, tempMiscal float64) ([]map[string]float64, []string) {
	probs := make([]map[string]float64, n)
	labels := make([]string, n)

	classNames := make([]string, numClasses)
	for c := 0; c < numClasses; c++ {
		classNames[c] = fmt.Sprintf("class_%02d", c)
	}

	for i := 0; i < n; i++ {
		trueLogits := make([]float64, numClasses)
		maxTrue := math.Inf(-1)
		for c := 0; c < numClasses; c++ {
			z := rng.NormFloat64() * 1.35
			trueLogits[c] = z
			if z > maxTrue {
				maxTrue = z
			}
		}
		trueProbs := make([]float64, numClasses)
		sumTrue := 0.0
		for c := 0; c < numClasses; c++ {
			v := math.Exp(trueLogits[c] - maxTrue)
			trueProbs[c] = v
			sumTrue += v
		}
		u := rng.Float64() * sumTrue
		acc := 0.0
		goldIdx := numClasses - 1
		for c := 0; c < numClasses; c++ {
			acc += trueProbs[c]
			if u <= acc {
				goldIdx = c
				break
			}
		}
		labels[i] = classNames[goldIdx]

		// Model observes noisy, overconfident (tempMiscal < 1) logits.
		modelLogits := make([]float64, numClasses)
		maxModel := math.Inf(-1)
		for c := 0; c < numClasses; c++ {
			ml := (trueLogits[c] + rng.NormFloat64()*0.45) / tempMiscal
			modelLogits[c] = ml
			if ml > maxModel {
				maxModel = ml
			}
		}
		pMap := make(map[string]float64, numClasses)
		sumModel := 0.0
		for c := 0; c < numClasses; c++ {
			v := math.Exp(modelLogits[c] - maxModel)
			pMap[classNames[c]] = v
			sumModel += v
		}
		for c := 0; c < numClasses; c++ {
			pMap[classNames[c]] /= sumModel
		}
		probs[i] = pMap
	}
	return probs, labels
}

func setContainsLabel(set []string, target string) bool {
	for _, s := range set {
		if s == target {
			return true
		}
	}
	return false
}

func TestSplitConformalFiniteSampleCoverage(t *testing.T) {
	configs := []struct {
		name string
		cfg  ConformalConfig
	}{
		{name: "lac", cfg: ConformalConfig{Method: ConformalLAC}},
		{name: "aps_det", cfg: ConformalConfig{Method: ConformalAPS, Randomized: false}},
		{name: "aps_rand", cfg: ConformalConfig{Method: ConformalAPS, Randomized: true, Seed: 42}},
		{name: "raps_det", cfg: ConformalConfig{Method: ConformalRAPS, KReg: 2, LambdaReg: 0.05, Randomized: false}},
		{name: "raps_rand", cfg: ConformalConfig{Method: ConformalRAPS, KReg: 2, LambdaReg: 0.05, Randomized: true, Seed: 42}},
	}

	for _, alpha := range []float64{0.05, 0.10} {
		for _, tc := range configs {
			rng := rand.New(rand.NewSource(20260417 + int64(alpha*1000)))
			// 1. Single large exchangeable evaluation with N_test = 1,500 (>= 1,000 trials).
			calProbs, calLabels := drawExchangeableMiscalibratedBatch(rng, 500, 8, 0.55)
			testProbs, testLabels := drawExchangeableMiscalibratedBatch(rng, 1500, 8, 0.55)

			cal, err := CalibrateConformalQuantile(calProbs, calLabels, alpha, tc.cfg)
			if err != nil {
				t.Fatalf("%s alpha=%.2f: calibration failed: %v", tc.name, alpha, err)
			}

			hits := 0
			for j := 0; j < len(testProbs); j++ {
				res, err := ConformalPredictionSet(testProbs[j], cal.QHat, tc.cfg)
				if err != nil {
					t.Fatalf("%s alpha=%.2f item %d: %v", tc.name, alpha, j, err)
				}
				if setContainsLabel(res.PredictionSet, testLabels[j]) {
					hits++
				}
			}
			empCover := float64(hits) / float64(len(testProbs))
			minRequired := 1.0 - alpha - 0.025
			if empCover < minRequired {
				t.Fatalf("%s alpha=%.2f: 1500-trial empirical coverage %.4f < %.4f (q_hat=%.4f)",
					tc.name, alpha, empCover, minRequired, cal.QHat)
			}

			// 2. Repeated split-conformal coverage across R=30 independent calibration/test splits.
			const numSplits = 30
			sumCover := 0.0
			for s := 0; s < numSplits; s++ {
				cP, cL := drawExchangeableMiscalibratedBatch(rng, 250, 8, 0.55)
				tP, tL := drawExchangeableMiscalibratedBatch(rng, 250, 8, 0.55)
				cRes, err := CalibrateConformalQuantile(cP, cL, alpha, tc.cfg)
				if err != nil {
					t.Fatalf("%s split %d: %v", tc.name, s, err)
				}
				splitHits := 0
				for j := 0; j < len(tP); j++ {
					pSet, err := ConformalPredictionSet(tP[j], cRes.QHat, tc.cfg)
					if err != nil {
						t.Fatalf("%s split %d item %d: %v", tc.name, s, j, err)
					}
					if setContainsLabel(pSet.PredictionSet, tL[j]) {
						splitHits++
					}
				}
				sumCover += float64(splitHits) / float64(len(tP))
			}
			meanCover := sumCover / float64(numSplits)
			if meanCover < 1.0-alpha-0.015 {
				t.Fatalf("%s alpha=%.2f: mean coverage across %d splits = %.4f < %.4f",
					tc.name, alpha, numSplits, meanCover, 1.0-alpha-0.015)
			}
		}
	}
}

func TestRAPSReducesPredictionSetSizeOnWideVocabulary(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const K = 20

	drawDiffuseWide := func(n int) ([]map[string]float64, []string) {
		probs := make([]map[string]float64, n)
		labels := make([]string, n)
		for i := 0; i < n; i++ {
			perm := rng.Perm(K)
			topA := fmt.Sprintf("opt_%02d", perm[0])
			topB := fmt.Sprintf("opt_%02d", perm[1])

			pMap := make(map[string]float64, K)
			var pA, pB, topMass float64
			if i%2 == 0 {
				// Asymmetric top-2 concentration (pA ~ 0.66..0.70, pB ~ 0.20..0.24).
				topMass = 0.90
				pA = 0.66 + rng.Float64()*0.04
				pB = topMass - pA
			} else {
				// Ambiguous top-2 with 44% noisy probability mass leaked thinly across 18 tail classes.
				topMass = 0.56
				pA = 0.29 + rng.Float64()*0.03
				pB = topMass - pA
			}
			pMap[topA] = pA
			pMap[topB] = pB

			tailMass := 1.0 - topMass
			tailRaw := make([]float64, K-2)
			tailSum := 0.0
			for j := 0; j < K-2; j++ {
				w := 0.5 + rng.Float64()
				tailRaw[j] = w
				tailSum += w
			}
			for j := 0; j < K-2; j++ {
				optName := fmt.Sprintf("opt_%02d", perm[2+j])
				pMap[optName] = tailMass * (tailRaw[j] / tailSum)
			}

			// True label is concentrated on {topA, topB} (98% mass), with 2% true tail noise.
			r := rng.Float64()
			switch {
			case i%2 == 0 && r < 0.68:
				labels[i] = topA
			case i%2 == 0 && r < 0.98:
				labels[i] = topB
			case i%2 == 1 && r < 0.52:
				labels[i] = topA
			case i%2 == 1 && r < 0.98:
				labels[i] = topB
			default:
				labels[i] = fmt.Sprintf("opt_%02d", perm[2+rng.Intn(K-2)])
			}
			probs[i] = pMap
		}
		return probs, labels
	}

	calProbs, calLabels := drawDiffuseWide(500)
	testProbs, testLabels := drawDiffuseWide(1000)

	for _, randomized := range []bool{false, true} {
		apsCfg := ConformalConfig{Method: ConformalAPS, Randomized: randomized, Seed: 42}
		rapsCfg := ConformalConfig{Method: ConformalRAPS, KReg: 2, LambdaReg: 0.14, Randomized: randomized, Seed: 42}

		apsCal, err := CalibrateConformalQuantile(calProbs, calLabels, 0.05, apsCfg)
		if err != nil {
			t.Fatalf("randomized=%v APS calibration failed: %v", randomized, err)
		}
		rapsCal, err := CalibrateConformalQuantile(calProbs, calLabels, 0.05, rapsCfg)
		if err != nil {
			t.Fatalf("randomized=%v RAPS calibration failed: %v", randomized, err)
		}

		apsSizeSum, rapsSizeSum := 0, 0
		rapsHits := 0
		for i := 0; i < len(testProbs); i++ {
			apsRes, err := ConformalPredictionSet(testProbs[i], apsCal.QHat, apsCfg)
			if err != nil {
				t.Fatalf("randomized=%v APS prediction set %d failed: %v", randomized, i, err)
			}
			rapsRes, err := ConformalPredictionSet(testProbs[i], rapsCal.QHat, rapsCfg)
			if err != nil {
				t.Fatalf("randomized=%v RAPS prediction set %d failed: %v", randomized, i, err)
			}
			apsSizeSum += apsRes.SetSize
			rapsSizeSum += rapsRes.SetSize
			if setContainsLabel(rapsRes.PredictionSet, testLabels[i]) {
				rapsHits++
			}
		}

		meanAPS := float64(apsSizeSum) / float64(len(testProbs))
		meanRAPS := float64(rapsSizeSum) / float64(len(testProbs))
		rapsCover := float64(rapsHits) / float64(len(testProbs))

		if meanRAPS >= meanAPS {
			t.Fatalf("randomized=%v: expected RAPS mean set size (%.3f) < APS mean set size (%.3f)", randomized, meanRAPS, meanAPS)
		}
		if rapsCover < 0.94 {
			t.Fatalf("randomized=%v: expected RAPS empirical coverage >= 0.94 at alpha=0.05, got %.4f", randomized, rapsCover)
		}
	}
}

func TestConformalCascadeEscalationAndPreFusionGate(t *testing.T) {
	cfg := ConformalConfig{Method: ConformalLAC}

	// 1. Singleton non-abstain set -> fast-path (ShouldEscalate == false).
	confident := map[string]float64{"approve": 0.95, "manual_review": 0.03, "abstain": 0.02}
	resSingle, err := ConformalPredictionSet(confident, 0.20, cfg)
	if err != nil {
		t.Fatalf("ConformalPredictionSet failed: %v", err)
	}
	if !resSingle.Singleton || resSingle.ContainsAbstain || resSingle.ShouldEscalate {
		t.Fatalf("expected singleton non-abstain set without escalation, got %+v", resSingle)
	}
	if resSingle.TopOption != "approve" || resSingle.SetSize != 1 {
		t.Fatalf("unexpected top option or set size: %+v", resSingle)
	}

	// 2. Singleton abstain set -> must escalate despite Singleton == true.
	for _, absOpt := range []string{"abstain", "insufficient_evidence"} {
		absProbs := map[string]float64{absOpt: 0.94, "approve": 0.06}
		resAbs, err := ConformalPredictionSet(absProbs, 0.20, cfg)
		if err != nil {
			t.Fatalf("abstain option %q failed: %v", absOpt, err)
		}
		if !resAbs.Singleton || !resAbs.ContainsAbstain || !resAbs.ShouldEscalate {
			t.Fatalf("abstain option %q: expected Singleton && ContainsAbstain && ShouldEscalate, got %+v", absOpt, resAbs)
		}
	}

	// 3. Multi-option ambiguous set and empty set -> both must escalate.
	ambig := map[string]float64{"approve": 0.52, "manual_review": 0.48}
	resMulti, err := ConformalPredictionSet(ambig, 0.55, cfg)
	if err != nil {
		t.Fatalf("resMulti failed: %v", err)
	}
	if resMulti.SetSize != 2 || resMulti.Singleton || !resMulti.ShouldEscalate {
		t.Fatalf("expected 2-option set to escalate, got %+v", resMulti)
	}

	resEmpty, err := ConformalPredictionSet(ambig, 0.10, cfg)
	if err != nil {
		t.Fatalf("resEmpty failed: %v", err)
	}
	if resEmpty.SetSize != 0 || resEmpty.Singleton || !resEmpty.ShouldEscalate {
		t.Fatalf("expected empty prediction set to escalate safely, got %+v", resEmpty)
	}

	// 4. Pre-prior-fusion gate must evaluate RawProbabilities rather than post-fusion Probabilities.
	qr := QuestionResult{
		Choice:           "opt_a",
		RawProbabilities: map[string]float64{"opt_a": 0.51, "opt_b": 0.49},
		Probabilities:    map[string]float64{"opt_a": 0.96, "opt_b": 0.04},
		PriorFused:       true,
	}
	postOnly, err := ConformalPredictionSet(qr.Probabilities, 0.52, cfg)
	if err != nil {
		t.Fatalf("postOnly failed: %v", err)
	}
	if postOnly.ShouldEscalate {
		t.Fatalf("sanity check: post-fusion distribution {0.96, 0.04} should be singleton at qHat=0.52")
	}
	preGate, err := EvaluatePreFusionConformalGate(qr, 0.52, cfg)
	if err != nil {
		t.Fatalf("EvaluatePreFusionConformalGate failed: %v", err)
	}
	if preGate.SetSize != 2 || !preGate.ShouldEscalate {
		t.Fatalf("expected EvaluatePreFusionConformalGate to use RawProbabilities and escalate, got %+v", preGate)
	}

	// 5. Verify SystemOneAnswer pre-fusion recovery from FusePriorWithGateMode("always").
	rawInput := map[string]float64{"opt_a": 0.51, "opt_b": 0.49}
	priors := map[string]float64{"opt_a": 3.5, "opt_b": -3.5}
	fusedProbs, winner, h, normH, rawH, rawNormH, fused := FusePriorWithGateMode(
		rawInput, priors, []string{"opt_a", "opt_b"}, 1.0, 0.5, 0.16, "always",
	)
	if !fused {
		t.Fatalf("expected FusePriorWithGateMode('always') to fuse priors")
	}
	ans := SystemOneAnswer{
		Type:           "choice",
		Choice:         winner,
		Probabilities:  fusedProbs,
		Entropy:        h,
		NormalizedH:    normH,
		RawEntropy:     rawH,
		RawNormalizedH: rawNormH,
		PriorFused:     fused,
	}
	ansGate, err := EvaluateSystemOnePreFusionGate(ans, priors, 0.5, 0.52, cfg)
	if err != nil {
		t.Fatalf("EvaluateSystemOnePreFusionGate failed: %v", err)
	}
	if ansGate.SetSize != 2 || !ansGate.ShouldEscalate {
		t.Fatalf("expected EvaluateSystemOnePreFusionGate to invert prior fusion and escalate, got %+v", ansGate)
	}
}

func TestSelectContextPruningActionsExactKnapsackAndStubRescue(t *testing.T) {
	nineBlocks := make([]ContextBlockSpec, 9)
	if _, _, err := BuildContextPruningQuestions(nineBlocks); !errors.Is(err, ErrTooManyContextBlocks) {
		t.Fatalf("expected ErrTooManyContextBlocks on 9 blocks, got %v", err)
	}
	if _, err := SelectContextPruningActions(nineBlocks, nil, ContextPruneConfig{TokenBudget: 1000}); !errors.Is(err, ErrTooManyContextBlocks) {
		t.Fatalf("expected ErrTooManyContextBlocks from SelectContextPruningActions, got %v", err)
	}
	if _, err := PruneAgentContextBlocks(nil, nil, ContextPruneConfig{TokenBudget: -5}); !errors.Is(err, ErrNegativeTokenBudget) {
		t.Fatalf("expected ErrNegativeTokenBudget, got %v", err)
	}

	// Verify Pinned == true invariance (minimum action = "stub", "drop" forbidden) and stub rescue.
	blocks := []ContextBlockSpec{
		{
			ID:         "sys_invariant",
			Kind:       "system_rules",
			FullText:   "PINNED SYSTEM INVARIANTS",
			StubText:   "PINNED STUB",
			FullTokens: 300,
			StubTokens: 60,
			Pinned:     true,
		},
		{
			ID:         "uncertain_trace",
			Kind:       "test_trace",
			FullText:   "FULL STACK TRACE BODY",
			StubText:   "panic at worker.go:84 (race on map write)",
			FullTokens: 600,
			StubTokens: 80,
		},
		{
			ID:         "stale_build_log",
			Kind:       "tool_output",
			FullText:   "VERBOSE OLD BUILD LOG",
			StubText:   "build succeeded 4 turns ago",
			FullTokens: 500,
			StubTokens: 70,
		},
	}
	probs := map[string]map[string]float64{
		"b0": {"full": 0.65, "stub": 0.10, "drop": 0.25},
		"b1": {"full": 0.42, "stub": 0.38, "drop": 0.20},
		"b2": {"full": 0.05, "stub": 0.10, "drop": 0.85},
	}

	res, err := PruneAgentContextBlocks(blocks, probs, ContextPruneConfig{TokenBudget: 420})
	if err != nil {
		t.Fatalf("PruneAgentContextBlocks failed: %v", err)
	}
	if !res.BudgetSatisfied || res.RetainedTokens > 420 {
		t.Fatalf("expected budget<=420 satisfied, got %+v", res)
	}
	if res.Decisions[0].Action != ContextActionFull || res.Decisions[0].RetainedText != "PINNED SYSTEM INVARIANTS" {
		t.Fatalf("expected pinned block b0 to remain full under 420 budget, got %+v", res.Decisions[0])
	}
	if res.Decisions[1].Action != ContextActionStub || res.Decisions[1].RetainedTokens != 80 {
		t.Fatalf("expected high-hesitation block b1 to be rescued as stub (80 tok), got %+v", res.Decisions[1])
	}
	if res.Decisions[2].Action != ContextActionDrop || res.Decisions[2].RetainedTokens != 0 {
		t.Fatalf("expected stale block b2 to be dropped, got %+v", res.Decisions[2])
	}

	// Under a tighter 200-token budget (< 300 full tokens for pinned b0, >= 60 stub tokens),
	// pinned b0 compresses to "stub" rather than being dropped, even if drop probability is 0.90.
	dropHeavyPinnedProbs := map[string]map[string]float64{
		"b0": {"full": 0.04, "stub": 0.06, "drop": 0.90},
		"b1": {"full": 0.42, "stub": 0.38, "drop": 0.20},
		"b2": {"full": 0.05, "stub": 0.10, "drop": 0.85},
	}
	resStubPinned, err := SelectContextPruningActions(blocks, dropHeavyPinnedProbs, ContextPruneConfig{TokenBudget: 200})
	if err != nil {
		t.Fatalf("unexpected error when pinned stub fits within budget: %v", err)
	}
	if !resStubPinned.BudgetSatisfied || resStubPinned.Decisions[0].Action != ContextActionStub || resStubPinned.Decisions[0].RetainedText != "PINNED STUB" {
		t.Fatalf("expected pinned block b0 to compress to stub under 200 budget, got %+v", resStubPinned)
	}

	// When TokenBudget < sum(pinned StubTokens) (40 < 60), SelectContextPruningActions returns ErrInfeasiblePinnedBudget.
	if _, err := SelectContextPruningActions(blocks, dropHeavyPinnedProbs, ContextPruneConfig{TokenBudget: 40}); !errors.Is(err, ErrInfeasiblePinnedBudget) {
		t.Fatalf("expected ErrInfeasiblePinnedBudget when budget < pinned stub cost, got %v", err)
	}

	// Verify exact MCKP optimality against exhaustive 3^8 enumeration across 50 random 8-block instances
	// for both default zero-config canonical utilities and custom StubUtilityRatio/UncertaintyStubBonus.
	rng := rand.New(rand.NewSource(12345))
	for trial := 0; trial < 50; trial++ {
		rBlocks := make([]ContextBlockSpec, MaxSlotsPerPass)
		rProbs := make(map[string]map[string]float64, MaxSlotsPerPass)
		totalFull := 0
		for i := 0; i < MaxSlotsPerPass; i++ {
			fullT := 150 + rng.Intn(1800)
			stubT := 20 + rng.Intn(fullT/2)
			totalFull += fullT
			rBlocks[i] = ContextBlockSpec{
				ID:               fmt.Sprintf("blk_%d", i),
				FullTokens:       fullT,
				StubTokens:       stubT,
				Pinned:           i == 0 && trial%3 == 0,
				ImportanceWeight: 0.5 + rng.Float64()*1.5,
			}
			a, b, c := rng.Float64()+0.05, rng.Float64()+0.05, rng.Float64()+0.05
			rProbs[fmt.Sprintf("b%d", i)] = map[string]float64{"full": a, "stub": b, "drop": c}
		}
		budget := rBlocks[0].StubTokens + rng.Intn(totalFull)

		// 1. Default zero-config canonical utility matrix: uFull = w*(pF + 0.5*pS), uStub = w*(0.6*pF + pS), uDrop = 0.
		gotDefault, err := SelectContextPruningActions(rBlocks, rProbs, ContextPruneConfig{TokenBudget: budget})
		if err != nil {
			t.Fatalf("trial %d default config: %v", trial, err)
		}
		if !gotDefault.BudgetSatisfied || gotDefault.RetainedTokens > budget {
			t.Fatalf("trial %d default config: budget violated: retained=%d > budget=%d", trial, gotDefault.RetainedTokens, budget)
		}

		// 2. Custom StubUtilityRatio + UncertaintyStubBonus with uDrop = 0.
		gotCustom, err := SelectContextPruningActions(rBlocks, rProbs, ContextPruneConfig{
			TokenBudget:          budget,
			StubUtilityRatio:     0.65,
			UncertaintyStubBonus: 0.25,
		})
		if err != nil {
			t.Fatalf("trial %d custom config: %v", trial, err)
		}
		if !gotCustom.BudgetSatisfied || gotCustom.RetainedTokens > budget {
			t.Fatalf("trial %d custom config: budget violated: retained=%d > budget=%d", trial, gotCustom.RetainedTokens, budget)
		}

		bestDefault := math.Inf(-1)
		bestCustom := math.Inf(-1)
		const totalStates = 6561
		for mask := 0; mask < totalStates; mask++ {
			tmp := mask
			tok := 0
			utilDefault := 0.0
			utilCustom := 0.0
			valid := true
			for i := 0; i < MaxSlotsPerPass; i++ {
				act := tmp % 3
				tmp /= 3
				if rBlocks[i].Pinned && act == 2 {
					valid = false
					break
				}
				d := gotDefault.Decisions[i]
				pF := d.Probabilities[ContextActionFull]
				pS := d.Probabilities[ContextActionStub]
				pD := d.Probabilities[ContextActionDrop]
				w := rBlocks[i].ImportanceWeight
				switch act {
				case 0:
					tok += rBlocks[i].FullTokens
					utilDefault += w * (1.0*pF + 0.5*pS)
					utilCustom += w * (pF + 0.65*pS)
				case 1:
					tok += rBlocks[i].StubTokens
					utilDefault += w * (0.6*pF + 1.0*pS)
					utilCustom += w * (0.65*pF + pS + 0.25*d.NormalizedEntropy*(1.0-pD))
				case 2:
					// u("drop") == 0.0
				}
			}
			if valid && tok <= budget {
				if utilDefault > bestDefault {
					bestDefault = utilDefault
				}
				if utilCustom > bestCustom {
					bestCustom = utilCustom
				}
			}
		}
		if math.Abs(gotDefault.ExpectedUtility-bestDefault) > 1e-9 {
			t.Fatalf("trial %d default: solver utility %.9f != brute-force optimum %.9f", trial, gotDefault.ExpectedUtility, bestDefault)
		}
		if math.Abs(gotCustom.ExpectedUtility-bestCustom) > 1e-9 {
			t.Fatalf("trial %d custom: solver utility %.9f != brute-force optimum %.9f", trial, gotCustom.ExpectedUtility, bestCustom)
		}
	}
}

func TestAgentContextPruningTemplateAndSinglePassExecution(t *testing.T) {
	tmplPath := filepath.Join("..", "..", "templates", "agent_context_pruning.json.tmpl")
	eng := template.NewEngine()

	// 1. Empty variable map renders valid JSON with defaults.
	emptyRendered, err := eng.RenderFile(tmplPath, map[string]interface{}{})
	if err != nil {
		t.Fatalf("RenderFile with empty data failed: %v", err)
	}
	var emptyParsed map[string]any
	if err := json.Unmarshal([]byte(emptyRendered), &emptyParsed); err != nil {
		t.Fatalf("rendered template with empty data is not valid JSON: %v", err)
	}

	// 2. Render with 8 trajectory blocks and validate via template.ParseStructuredPayload.
	blocks := make([]ContextBlockSpec, MaxSlotsPerPass)
	blockPayload := make([]map[string]any, MaxSlotsPerPass)
	for i := 0; i < MaxSlotsPerPass; i++ {
		blocks[i] = ContextBlockSpec{
			ID:         fmt.Sprintf("turn_block_%d", i),
			Kind:       "tool_output",
			Summary:    fmt.Sprintf("Tool output digest %d", i),
			FullText:   fmt.Sprintf("Complete content for block %d", i),
			StubText:   fmt.Sprintf("Stub digest %d", i),
			FullTokens: 400,
			StubTokens: 60,
			Pinned:     i == 0,
		}
		blockPayload[i] = map[string]any{
			"slot_id":     fmt.Sprintf("b%d", i),
			"id":          blocks[i].ID,
			"kind":        blocks[i].Kind,
			"full_tokens": blocks[i].FullTokens,
			"stub_tokens": blocks[i].StubTokens,
			"summary":     blocks[i].Summary,
		}
	}

	rendered, err := eng.RenderFile(tmplPath, map[string]interface{}{
		"task_goal":    "fix data race in worker pool shutdown",
		"turn_index":   4,
		"token_budget": 1200,
		"blocks":       blockPayload,
	})
	if err != nil {
		t.Fatalf("RenderFile with 8 blocks failed: %v", err)
	}

	schemaJSON, stateJSON, err := template.ParseStructuredPayload(rendered, nil)
	if err != nil {
		t.Fatalf("ParseStructuredPayload failed: %v", err)
	}

	var schemaObj struct {
		Instructions string `json:"instructions"`
		Samples      int    `json:"samples"`
		Questions    []struct {
			ID      string `json:"id"`
			Type    string `json:"type"`
			Options []struct {
				Name string `json:"name"`
			} `json:"options"`
		} `json:"questions"`
	}
	if err := json.Unmarshal([]byte(schemaJSON), &schemaObj); err != nil {
		t.Fatalf("unmarshaling normalized schema failed: %v", err)
	}
	if schemaObj.Samples != 1 {
		t.Fatalf("expected samples=1, got %d", schemaObj.Samples)
	}
	if len(schemaObj.Questions) != MaxSlotsPerPass {
		t.Fatalf("expected %d questions in template, got %d", MaxSlotsPerPass, len(schemaObj.Questions))
	}
	for i, q := range schemaObj.Questions {
		wantID := fmt.Sprintf("b%d", i)
		if q.ID != wantID || q.Type != "choice" {
			t.Fatalf("question %d: want id=%q type=choice, got id=%q type=%q", i, wantID, q.ID, q.Type)
		}
		if len(q.Options) != 3 || q.Options[0].Name != "full" || q.Options[1].Name != "stub" || q.Options[2].Name != "drop" {
			t.Fatalf("question %s: unexpected options %+v", q.ID, q.Options)
		}
	}

	var stateObj map[string]any
	if err := json.Unmarshal([]byte(stateJSON), &stateObj); err != nil {
		t.Fatalf("unmarshaling state JSON failed: %v", err)
	}
	if stateObj["task_goal"] != "fix data race in worker pool shutdown" {
		t.Fatalf("unexpected state task_goal: %v", stateObj["task_goal"])
	}

	// 3. Execute all 8 slots through ExecuteSystemOne against an in-process mock server and assert ForwardPasses == 1.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		if len(sysPayload.Questions) > MaxSlotsPerPass {
			http.Error(w, "the canvas holds at most 8 slots per pass", http.StatusUnprocessableEntity)
			return
		}

		answers := make(map[string]client.QuestionAnswer, len(sysPayload.Questions))
		for idx, q := range sysPayload.Questions {
			var probs map[string]float64
			var chosen string
			switch {
			case idx < 2:
				chosen = "full"
				probs = map[string]float64{"full": 0.85, "stub": 0.10, "drop": 0.05}
			case idx < 5:
				chosen = "full"
				probs = map[string]float64{"full": 0.44, "stub": 0.40, "drop": 0.16}
			default:
				chosen = "drop"
				probs = map[string]float64{"full": 0.05, "stub": 0.10, "drop": 0.85}
			}
			answers[q.ID] = client.QuestionAnswer{
				Choice:        chosen,
				Probabilities: probs,
			}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": string(envBytes)}},
			},
		})
	}))
	defer srv.Close()

	qs, slotIDs, err := BuildContextPruningQuestions(blocks)
	if err != nil {
		t.Fatalf("BuildContextPruningQuestions failed: %v", err)
	}
	if len(slotIDs) != MaxSlotsPerPass {
		t.Fatalf("expected %d slotIDs, got %d", MaxSlotsPerPass, len(slotIDs))
	}

	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)
	sysResp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State:     stateObj,
		Questions: qs,
	}, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}
	if sysResp.EvaluationTrace == nil || sysResp.EvaluationTrace.ForwardPasses != 1 {
		t.Fatalf("expected ForwardPasses == 1 for 8-slot context pruning, got %+v", sysResp.EvaluationTrace)
	}

	slotProbs := make(map[string]map[string]float64, len(sysResp.Answers))
	for slotID, ans := range sysResp.Answers {
		slotProbs[slotID] = ans.Probabilities
	}
	pruned, err := PruneAgentContextBlocks(blocks, slotProbs, ContextPruneConfig{TokenBudget: 1200})
	if err != nil {
		t.Fatalf("PruneAgentContextBlocks failed: %v", err)
	}
	if !pruned.BudgetSatisfied || pruned.RetainedTokens > 1200 || pruned.SavedTokens <= 0 {
		t.Fatalf("unexpected pruning summary: %+v", pruned)
	}
}

func TestConformalStrictValidationAndEdgeCases(t *testing.T) {
	// 1. Reject invalid probability maps across ConformalScore, CalibrateConformalQuantile,
	// ConformalPredictionSet, EvaluatePreFusionConformalGate, and EvaluateSystemOnePreFusionGate.
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
			t.Fatalf("case %d ConformalScore: expected ErrInvalidProbabilities, got %v", idx, err)
		}
		if _, err := CalibrateConformalQuantile([]map[string]float64{badMap}, []string{"a"}, 0.10, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d CalibrateConformalQuantile: expected ErrInvalidProbabilities, got %v", idx, err)
		}
		if _, err := ConformalPredictionSet(badMap, 0.20, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d ConformalPredictionSet: expected ErrInvalidProbabilities, got %v", idx, err)
		}
		if _, err := EvaluatePreFusionConformalGate(QuestionResult{Probabilities: badMap}, 0.20, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d EvaluatePreFusionConformalGate: expected ErrInvalidProbabilities, got %v", idx, err)
		}
		if _, err := EvaluateSystemOnePreFusionGate(SystemOneAnswer{Probabilities: badMap}, nil, 0, 0.20, ConformalConfig{}); !errors.Is(err, ErrInvalidProbabilities) {
			t.Fatalf("case %d EvaluateSystemOnePreFusionGate: expected ErrInvalidProbabilities, got %v", idx, err)
		}
	}

	// 2. Reject invalid qHat, invalid Method, and invalid RAPS regularization parameters.
	validProbs := map[string]float64{"a": 0.7, "b": 0.3}
	for _, badQ := range []float64{-0.01, math.NaN(), math.Inf(-1)} {
		if _, err := ConformalPredictionSet(validProbs, badQ, ConformalConfig{}); !errors.Is(err, ErrInvalidConformalQHat) {
			t.Fatalf("qHat=%v: expected ErrInvalidConformalQHat, got %v", badQ, err)
		}
	}
	if _, err := ConformalPredictionSet(validProbs, 0.20, ConformalConfig{Method: "bogus"}); !errors.Is(err, ErrInvalidConformalMethod) {
		t.Fatalf("expected ErrInvalidConformalMethod on ConformalPredictionSet, got %v", err)
	}
	for _, badCfg := range []ConformalConfig{
		{Method: ConformalRAPS, KReg: -1, LambdaReg: 0.02},
		{Method: ConformalRAPS, KReg: 2, LambdaReg: -0.05},
		{Method: ConformalRAPS, KReg: 2, LambdaReg: math.NaN()},
		{Method: ConformalRAPS, KReg: 2, LambdaReg: math.Inf(1)},
	} {
		if _, err := ConformalScore(validProbs, "a", badCfg); !errors.Is(err, ErrInvalidConformalParam) {
			t.Fatalf("cfg=%+v: expected ErrInvalidConformalParam from ConformalScore, got %v", badCfg, err)
		}
		if _, err := ConformalPredictionSet(validProbs, 0.20, badCfg); !errors.Is(err, ErrInvalidConformalParam) {
			t.Fatalf("cfg=%+v: expected ErrInvalidConformalParam from ConformalPredictionSet, got %v", badCfg, err)
		}
	}

	// DefaultConformalConfig() and zero-value ConformalConfig{} use RAPS with Alpha=0.10, KReg=1, LambdaReg=0.01,
	// while explicit ConformalConfig{Method: ConformalRAPS, KReg: ..., LambdaReg: ...} honors KReg=0 and LambdaReg=0.0 literally.
	defCfg := DefaultConformalConfig()
	if defCfg.Method != ConformalRAPS || defCfg.Alpha != 0.10 || defCfg.KReg != 1 || defCfg.LambdaReg != 0.01 {
		t.Fatalf("unexpected DefaultConformalConfig(): %+v", defCfg)
	}
	sixOpts := map[string]float64{
		"o1": 0.30, "o2": 0.25, "o3": 0.20, "o4": 0.12, "o5": 0.08, "o6": 0.05,
	}
	// DefaultConformalConfig / ConformalConfig{}: KReg=1, LambdaReg=0.01 -> rank 5 score = 0.87 + 0.04 = 0.91; rank 6 = 0.95 + 0.05 = 1.00.
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
			t.Fatalf("expected (s5=0.91, s6=1.00) for default RAPS config %+v, got (%.12f, %.12f)", cfgUse, s5, s6)
		}
	}
	// Explicit LambdaReg=0.0 reduces RAPS identically to APS across all ranks (< 1e-12).
	for _, kZeroTest := range []int{0, 1, 3} {
		rapsZero := ConformalConfig{Method: ConformalRAPS, KReg: kZeroTest, LambdaReg: 0.0}
		apsRef := ConformalConfig{Method: ConformalAPS}
		for optKey := range sixOpts {
			sr, err1 := ConformalScore(sixOpts, optKey, rapsZero)
			sa, err2 := ConformalScore(sixOpts, optKey, apsRef)
			if err1 != nil || err2 != nil || math.Abs(sr-sa) > 1e-12 {
				t.Fatalf("RAPS(LambdaReg=0, KReg=%d) score mismatch on %s: raps=%v (%v) vs aps=%v (%v)", kZeroTest, optKey, sr, err1, sa, err2)
			}
		}
	}
	// Explicit KReg=0 with LambdaReg>0 penalizes starting at rank 1 (LambdaReg=0.10 -> rank 1 = 0.10, rank 2 = 0.50; LambdaReg=0.20 -> rank 1 = 0.20, rank 2 = 0.70).
	s1K0Ten, err := ConformalScore(sixOpts, "o1", ConformalConfig{Method: ConformalRAPS, KReg: 0, LambdaReg: 0.10})
	if err != nil || math.Abs(s1K0Ten-0.10) > 1e-12 {
		t.Fatalf("expected rank-1 KReg=0 LambdaReg=0.10 score 0.10, got %v (err=%v)", s1K0Ten, err)
	}
	s2K0Ten, err := ConformalScore(sixOpts, "o2", ConformalConfig{Method: ConformalRAPS, KReg: 0, LambdaReg: 0.10})
	if err != nil || math.Abs(s2K0Ten-0.50) > 1e-12 {
		t.Fatalf("expected rank-2 KReg=0 LambdaReg=0.10 score 0.50, got %v (err=%v)", s2K0Ten, err)
	}
	s1K0, err := ConformalScore(sixOpts, "o1", ConformalConfig{Method: ConformalRAPS, KReg: 0, LambdaReg: 0.20})
	if err != nil || math.Abs(s1K0-0.20) > 1e-12 {
		t.Fatalf("expected rank-1 KReg=0 score 0.20, got %v (err=%v)", s1K0, err)
	}
	s2K0, err := ConformalScore(sixOpts, "o2", ConformalConfig{Method: ConformalRAPS, KReg: 0, LambdaReg: 0.20})
	if err != nil || math.Abs(s2K0-0.70) > 1e-12 {
		t.Fatalf("expected rank-2 KReg=0 score 0.70, got %v (err=%v)", s2K0, err)
	}
	// All 9 default abstain labels (case-insensitive trimmed) trigger ContainsAbstain and ShouldEscalate when AbstainLabels is nil.
	for _, absKey := range []string{"0", "abstain", "insufficient_evidence", "none", "none_of_the_above", "none_visible", "oos", "uncertain", "unknown", " OOS ", "UNCERTAIN"} {
		absRes, absErr := ConformalPredictionSet(map[string]float64{absKey: 0.96, "valid_opt": 0.04}, 0.20, ConformalConfig{Method: ConformalLAC})
		if absErr != nil || !absRes.ContainsAbstain || !absRes.ShouldEscalate {
			t.Fatalf("expected default abstain escalation for %q, got res=%+v err=%v", absKey, absRes, absErr)
		}
	}
	// Explicit empty AbstainLabels ([]string{}) opts out of default abstention.
	optOutRes, optOutErr := ConformalPredictionSet(map[string]float64{"oos": 0.96, "valid_opt": 0.04}, 0.20, ConformalConfig{Method: ConformalLAC, AbstainLabels: []string{}})
	if optOutErr != nil || optOutRes.ContainsAbstain || optOutRes.ShouldEscalate || !optOutRes.Singleton {
		t.Fatalf("expected explicit empty AbstainLabels []string{} to disable abstain escalation, got res=%+v err=%v", optOutRes, optOutErr)
	}

	// 3. Exact floating-point order-statistic boundary: n=9, alpha=0.30 -> (9+1)*(1-0.30) = 7.0 -> OrderIndex == 7.
	nineProbs := make([]map[string]float64, 9)
	nineLabels := make([]string, 9)
	for i := 0; i < 9; i++ {
		pTop := 0.95 - float64(i)*0.05
		nineProbs[i] = map[string]float64{"y": pTop, "n": 1.0 - pTop}
		nineLabels[i] = "y"
	}
	cal9, err := CalibrateConformalQuantile(nineProbs, nineLabels, 0.30, ConformalConfig{Method: ConformalLAC})
	if err != nil {
		t.Fatalf("n=9 alpha=0.30 calibration failed: %v", err)
	}
	if cal9.OrderIndex != 7 {
		t.Fatalf("expected OrderIndex=7 at n=9, alpha=0.30, got %d (qHat=%v)", cal9.OrderIndex, cal9.QHat)
	}

	// 4. Deterministic APS and RAPS on sharp/degenerate distributions always include top-1 (score 0.0, never empty).
	for _, m := range []ConformalMethod{ConformalAPS, ConformalRAPS} {
		cfgDet := ConformalConfig{Method: m, Randomized: false}
		for _, dist := range []map[string]float64{
			{"A": 1.0, "B": 0.0, "C": 0.0},
			{"A": 0.99, "B": 0.01},
		} {
			for _, qHat := range []float64{0.85, 0.0} {
				got, err := ConformalPredictionSet(dist, qHat, cfgDet)
				if err != nil {
					t.Fatalf("method=%s qHat=%v failed: %v", m, qHat, err)
				}
				if got.SetSize != 1 || got.PredictionSet[0] != "A" || !got.Singleton || got.ShouldEscalate {
					t.Fatalf("method=%s dist=%v qHat=%v: expected singleton [A] without escalation, got %+v", m, dist, qHat, got)
				}
			}
		}
	}

	// 5. Deterministic APS and RAPS on exact and near-exact 4-way ties assign identical tie-group scores
	// and never collapse to a lexical singleton.
	for _, m := range []ConformalMethod{ConformalAPS, ConformalRAPS} {
		cfgDet := ConformalConfig{Method: m, KReg: 2, LambdaReg: 0.05, Randomized: false}
		tie4 := map[string]float64{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
		for _, qHat := range []float64{0.0, 0.10} {
			got, err := ConformalPredictionSet(tie4, qHat, cfgDet)
			if err != nil {
				t.Fatalf("method=%s tie4 failed: %v", m, err)
			}
			if got.SetSize != 4 || got.Singleton || !got.ShouldEscalate {
				t.Fatalf("method=%s qHat=%v: expected 4-element tie set escalating, got %+v", m, qHat, got)
			}
			for _, k := range []string{"A", "B", "C", "D"} {
				if got.Scores[k] != 0.0 {
					t.Fatalf("method=%s: expected score 0.0 for tied key %q, got %v", m, k, got.Scores[k])
				}
			}
		}

		renamed := map[string]float64{"Z_first": 0.25 + 1e-13, "B": 0.25, "C": 0.25, "D": 0.25 - 1e-13}
		gotRenamed, err := ConformalPredictionSet(renamed, 0.0, cfgDet)
		if err != nil {
			t.Fatalf("method=%s renamed near-tie failed: %v", m, err)
		}
		if gotRenamed.SetSize != 4 || !gotRenamed.ShouldEscalate {
			t.Fatalf("method=%s: expected near-tie within 1e-12 tolerance to include all 4 keys, got %+v", m, gotRenamed)
		}
	}

	// 6. Randomized=true stateful RNG lockstep maintains marginal coverage on repeated identical distributions
	// (both uniform 2-way ties and sharp {"A": 0.99, "B": 0.01} under adversarial seeds) and is goroutine-safe.
	tieCalProbs := make([]map[string]float64, 200)
	tieCalLabels := make([]string, 200)
	for i := 0; i < 200; i++ {
		tieCalProbs[i] = map[string]float64{"A": 0.5, "B": 0.5}
		if i%2 == 0 {
			tieCalLabels[i] = "A"
		} else {
			tieCalLabels[i] = "B"
		}
	}
	randTieCfg := ConformalConfig{Method: ConformalAPS, Randomized: true, Seed: 7}
	calTie, err := CalibrateConformalQuantile(tieCalProbs, tieCalLabels, 0.25, randTieCfg)
	if err != nil {
		t.Fatalf("randomized tie calibration failed: %v", err)
	}
	tieHits := 0
	singletonA, singletonB := 0, 0
	evalRNG := rand.New(rand.NewSource(101))
	for j := 0; j < 1000; j++ {
		gold := "A"
		if evalRNG.Float64() < 0.5 {
			gold = "B"
		}
		res, err := ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, calTie.QHat, randTieCfg)
		if err != nil {
			t.Fatalf("randomized tie prediction %d failed: %v", j, err)
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
	if float64(tieHits)/1000.0 < 0.71 {
		t.Fatalf("randomized tie coverage %.3f < 0.71 at alpha=0.25", float64(tieHits)/1000.0)
	}
	if singletonA == 0 || singletonB == 0 {
		t.Fatalf("expected tie-group shuffle to yield both [A] and [B] singletons, got A=%d B=%d", singletonA, singletonB)
	}

	for _, seedVal := range []int64{2, 42} {
		sharpCalProbs := make([]map[string]float64, 200)
		sharpCalLabels := make([]string, 200)
		dataRNG := rand.New(rand.NewSource(9001 + seedVal))
		for i := 0; i < 200; i++ {
			sharpCalProbs[i] = map[string]float64{"A": 0.99, "B": 0.01}
			if dataRNG.Float64() < 0.99 {
				sharpCalLabels[i] = "A"
			} else {
				sharpCalLabels[i] = "B"
			}
		}
		sharpCfg := ConformalConfig{Method: ConformalAPS, Randomized: true, Seed: seedVal}
		calSharp, err := CalibrateConformalQuantile(sharpCalProbs, sharpCalLabels, 0.10, sharpCfg)
		if err != nil {
			t.Fatalf("seed=%d sharp calibration failed: %v", seedVal, err)
		}
		sharpHits := 0
		for j := 0; j < 1000; j++ {
			gold := "A"
			if dataRNG.Float64() >= 0.99 {
				gold = "B"
			}
			res, err := ConformalPredictionSet(map[string]float64{"A": 0.99, "B": 0.01}, calSharp.QHat, sharpCfg)
			if err != nil {
				t.Fatalf("seed=%d sharp prediction %d failed: %v", seedVal, j, err)
			}
			if setContainsLabel(res.PredictionSet, gold) {
				sharpHits++
			}
		}
		if float64(sharpHits)/1000.0 < 0.86 {
			t.Fatalf("seed=%d sharp randomized coverage %.3f < 0.86 at alpha=0.10 (qHat=%.4f)",
				seedVal, float64(sharpHits)/1000.0, calSharp.QHat)
		}
	}

	// For unseeded Seed=0 (shared wall-clock defaultConformalRNG), evaluate marginal coverage across
	// 12 independent calibration/test splits (400 cal + 400 test each = 4,800 draws, SE ~ 0.0061)
	// so single-split Beta(181, 20) tail variance cannot cause wall-clock flakiness.
	unseededDataRNG := rand.New(rand.NewSource(9001))
	unseededCfg := ConformalConfig{Method: ConformalAPS, Randomized: true, Seed: 0}
	unseededTotalHits := 0
	const unseededSplits = 12
	for sp := 0; sp < unseededSplits; sp++ {
		sharpCalProbs := make([]map[string]float64, 400)
		sharpCalLabels := make([]string, 400)
		for i := 0; i < 400; i++ {
			sharpCalProbs[i] = map[string]float64{"A": 0.99, "B": 0.01}
			if unseededDataRNG.Float64() < 0.99 {
				sharpCalLabels[i] = "A"
			} else {
				sharpCalLabels[i] = "B"
			}
		}
		calSharp, err := CalibrateConformalQuantile(sharpCalProbs, sharpCalLabels, 0.10, unseededCfg)
		if err != nil {
			t.Fatalf("seed=0 split=%d sharp calibration failed: %v", sp, err)
		}
		for j := 0; j < 400; j++ {
			gold := "A"
			if unseededDataRNG.Float64() >= 0.99 {
				gold = "B"
			}
			res, err := ConformalPredictionSet(map[string]float64{"A": 0.99, "B": 0.01}, calSharp.QHat, unseededCfg)
			if err != nil {
				t.Fatalf("seed=0 split=%d prediction %d failed: %v", sp, j, err)
			}
			if setContainsLabel(res.PredictionSet, gold) {
				unseededTotalHits++
			}
		}
	}
	unseededMeanCov := float64(unseededTotalHits) / float64(unseededSplits*400)
	if unseededMeanCov < 0.865 {
		t.Fatalf("seed=0 marginal sharp randomized coverage %.4f < 0.865 across %d splits at alpha=0.10",
			unseededMeanCov, unseededSplits)
	}

	// Verify explicit caller RNG stream and concurrent goroutine safety on shared/default RNG streams.
	explicitRNG := rand.New(rand.NewSource(55))
	explicitCfg := ConformalConfig{Method: ConformalRAPS, Randomized: true, RNG: explicitRNG}
	if _, err := CalibrateConformalQuantile(tieCalProbs[:20], tieCalLabels[:20], 0.20, explicitCfg); err != nil {
		t.Fatalf("explicit RNG calibration failed: %v", err)
	}
	if _, err := ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, 0.5, explicitCfg); err != nil {
		t.Fatalf("explicit RNG prediction failed: %v", err)
	}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			c := ConformalConfig{Method: ConformalAPS, Randomized: true, Seed: int64(id % 3)}
			for k := 0; k < 40; k++ {
				_, _ = ConformalPredictionSet(map[string]float64{"A": 0.5, "B": 0.5}, 0.4, c)
			}
		}(g)
	}
	wg.Wait()
}

func TestContextPruningValidationAndTieBreakDeterminism(t *testing.T) {
	validProbs := map[string]map[string]float64{
		"b0": {"full": 0.6, "stub": 0.3, "drop": 0.1},
		"b1": {"full": 0.5, "stub": 0.3, "drop": 0.2},
	}

	// 1. Reject negative tokens, inverted FullTokens < StubTokens, invalid ImportanceWeight, and duplicate IDs.
	invalidBlockSlices := [][]ContextBlockSpec{
		{{ID: "neg_stub", FullTokens: 100, StubTokens: -1}},
		{{ID: "inverted", FullTokens: 40, StubTokens: 80}},
		{{ID: "neg_weight", FullTokens: 100, StubTokens: 40, ImportanceWeight: -0.5}},
		{{ID: "nan_weight", FullTokens: 100, StubTokens: 40, ImportanceWeight: math.NaN()}},
		{{ID: "dup", FullTokens: 100, StubTokens: 40}, {ID: "dup", FullTokens: 120, StubTokens: 30}},
	}
	for idx, badBlocks := range invalidBlockSlices {
		if _, _, err := BuildContextPruningQuestions(badBlocks); !errors.Is(err, ErrInvalidContextBlock) {
			t.Fatalf("case %d BuildContextPruningQuestions: expected ErrInvalidContextBlock, got %v", idx, err)
		}
		if _, err := SelectContextPruningActions(badBlocks, validProbs, ContextPruneConfig{TokenBudget: 500}); !errors.Is(err, ErrInvalidContextBlock) {
			t.Fatalf("case %d SelectContextPruningActions: expected ErrInvalidContextBlock, got %v", idx, err)
		}
	}

	// 2. Reject invalid ContextPruneConfig ratios and malformed/missing slotProbs entries.
	oneBlock := []ContextBlockSpec{{ID: "blk_a", FullTokens: 100, StubTokens: 40}}
	for _, badCfg := range []ContextPruneConfig{
		{TokenBudget: 200, StubUtilityRatio: -0.1},
		{TokenBudget: 200, StubUtilityRatio: 1.2},
		{TokenBudget: 200, StubUtilityRatio: math.NaN()},
		{TokenBudget: 200, UncertaintyStubBonus: math.Inf(1)},
	} {
		if _, err := SelectContextPruningActions(oneBlock, map[string]map[string]float64{"blk_a": {"full": 0.5, "stub": 0.3, "drop": 0.2}}, badCfg); !errors.Is(err, ErrInvalidContextBlock) {
			t.Fatalf("cfg=%+v: expected ErrInvalidContextBlock, got %v", badCfg, err)
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
			t.Fatalf("slotProbs case %d: expected ErrInvalidProbabilities, got %v", idx, err)
		}
	}

	// 3. Deterministic Pareto-frontier tie-breaking on symmetric and zero-cost blocks.
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
		t.Fatalf("symmetric tie test failed: %v", err)
	}
	if symRes.Decisions[0].Action != ContextActionFull || symRes.Decisions[1].Action != ContextActionStub {
		t.Fatalf("expected deterministic tie-break [full, stub], got [%s, %s]", symRes.Decisions[0].Action, symRes.Decisions[1].Action)
	}

	zeroCost := []ContextBlockSpec{{ID: "z0", FullTokens: 0, StubTokens: 0}}
	zeroRes, err := SelectContextPruningActions(
		zeroCost,
		map[string]map[string]float64{"z0": {"full": 0.0, "stub": 0.0, "drop": 1.0}},
		ContextPruneConfig{TokenBudget: 100},
	)
	if err != nil {
		t.Fatalf("zero-cost tie test failed: %v", err)
	}
	if zeroRes.Decisions[0].Action != ContextActionDrop {
		t.Fatalf("expected zero-utility zero-cost tie to select drop, got %q", zeroRes.Decisions[0].Action)
	}

	// 4. Verify 8-block MCKP Pareto-frontier parity across 30 budgets (145 + i*105) with 2 pinned blocks.
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
		t.Fatalf("expected ErrInfeasiblePinnedBudget at budget=144 (< 80+65=145), got %v", err)
	}
	prevUtil := -1.0
	for step := 0; step < 30; step++ {
		bgt := 145 + step*105
		r, err := SelectContextPruningActions(eightBlocks, eightProbs, ContextPruneConfig{TokenBudget: bgt})
		if err != nil {
			t.Fatalf("step %d (budget=%d) failed: %v", step, bgt, err)
		}
		if !r.BudgetSatisfied || r.RetainedTokens > bgt {
			t.Fatalf("step %d: retained %d > budget %d", step, r.RetainedTokens, bgt)
		}
		if r.Decisions[0].Action == ContextActionDrop || r.Decisions[1].Action == ContextActionDrop {
			t.Fatalf("step %d: pinned block was dropped: b0=%s b1=%s", step, r.Decisions[0].Action, r.Decisions[1].Action)
		}
		if r.ExpectedUtility+1e-12 < prevUtil {
			t.Fatalf("step %d: expected monotone non-decreasing utility across budgets, got %.9f < %.9f", step, r.ExpectedUtility, prevUtil)
		}
		prevUtil = r.ExpectedUtility
	}
}

func TestDefaultConformalConfigZeroBoundaryAndAbstainOptOut(t *testing.T) {
	calibProbs := []map[string]float64{
		{"opt_a": 0.60, "opt_b": 0.35, "opt_c": 0.05},
		{"opt_a": 0.55, "opt_b": 0.30, "opt_c": 0.15},
		{"opt_a": 0.50, "opt_b": 0.45, "opt_c": 0.05},
		{"opt_a": 0.70, "opt_b": 0.20, "opt_c": 0.10},
	}
	calibLabels := []string{"opt_c", "opt_b", "opt_b", "opt_a"}
	testProb := map[string]float64{"opt_a": 0.52, "opt_b": 0.33, "opt_c": 0.15}

	// 1. DefaultConformalConfig(), zero-value ConformalConfig{}, and whitespace Method default to RAPS(KReg=1, LambdaReg=0.01).
	defCfg := DefaultConformalConfig()
	calDef, err := CalibrateConformalQuantile(calibProbs, calibLabels, defCfg.Alpha, defCfg)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantile(DefaultConformalConfig) failed: %v", err)
	}
	if calDef.Method != ConformalRAPS || math.Abs(calDef.Alpha-0.10) > 1e-12 {
		t.Fatalf("expected Method=raps Alpha=0.10, got %+v", calDef)
	}
	for _, blankCfg := range []ConformalConfig{{}, {Method: "   "}} {
		calZero, err := CalibrateConformalQuantile(calibProbs, calibLabels, 0.10, blankCfg)
		if err != nil {
			t.Fatalf("CalibrateConformalQuantile(%+v) failed: %v", blankCfg, err)
		}
		if calZero.Method != ConformalRAPS || math.Abs(calZero.QHat-calDef.QHat) > 1e-12 {
			t.Fatalf("expected blank config %+v to match DefaultConformalConfig QHat=%v, got %+v", blankCfg, calDef.QHat, calZero)
		}
		setDef, err1 := ConformalPredictionSet(testProb, calDef.QHat, defCfg)
		setZero, err2 := ConformalPredictionSet(testProb, calZero.QHat, blankCfg)
		if err1 != nil || err2 != nil || len(setDef.PredictionSet) != len(setZero.PredictionSet) {
			t.Fatalf("prediction set mismatch between DefaultConformalConfig and %+v: %+v vs %+v", blankCfg, setDef, setZero)
		}
		for i := range setDef.PredictionSet {
			if setDef.PredictionSet[i] != setZero.PredictionSet[i] {
				t.Fatalf("prediction set element %d mismatch: %v vs %v", i, setDef.PredictionSet, setZero.PredictionSet)
			}
		}
	}

	// 2. Explicit Method=ConformalRAPS with LambdaReg=0.0 matches ConformalAPS on calibration and prediction sets (deterministic and randomized).
	for _, randomized := range []bool{false, true} {
		for _, kVal := range []int{0, 1, 2} {
			apsCfg := ConformalConfig{Method: ConformalAPS, Randomized: randomized, Seed: 77}
			rapsZeroCfg := ConformalConfig{Method: ConformalRAPS, KReg: kVal, LambdaReg: 0.0, Randomized: randomized, Seed: 77}
			calAPS, err1 := CalibrateConformalQuantile(calibProbs, calibLabels, 0.25, apsCfg)
			calRAPS0, err2 := CalibrateConformalQuantile(calibProbs, calibLabels, 0.25, rapsZeroCfg)
			if err1 != nil || err2 != nil || math.Abs(calAPS.QHat-calRAPS0.QHat) > 1e-12 {
				t.Fatalf("randomized=%v k=%d: RAPS(LambdaReg=0) QHat=%v (%v) != APS QHat=%v (%v)",
					randomized, kVal, calRAPS0.QHat, err2, calAPS.QHat, err1)
			}
			setAPS, err1 := ConformalPredictionSet(testProb, calAPS.QHat, apsCfg)
			setRAPS0, err2 := ConformalPredictionSet(testProb, calRAPS0.QHat, rapsZeroCfg)
			if err1 != nil || err2 != nil || len(setAPS.PredictionSet) != len(setRAPS0.PredictionSet) {
				t.Fatalf("randomized=%v k=%d: RAPS(LambdaReg=0) set %+v != APS set %+v", randomized, kVal, setRAPS0, setAPS)
			}
			for i := range setAPS.PredictionSet {
				if setAPS.PredictionSet[i] != setRAPS0.PredictionSet[i] {
					t.Fatalf("randomized=%v k=%d: set item %d mismatch %v vs %v", randomized, kVal, i, setRAPS0.PredictionSet, setAPS.PredictionSet)
				}
			}
		}
	}

	// 3. Explicit Method=ConformalRAPS with KReg=0 and LambdaReg=0.10 penalizes starting at rank 1 (strictly +0.10 above KReg=1 across every rank >= 1).
	cfgK0 := ConformalConfig{Method: ConformalRAPS, KReg: 0, LambdaReg: 0.10}
	cfgK1 := ConformalConfig{Method: ConformalRAPS, KReg: 1, LambdaReg: 0.10}
	calK0, err := CalibrateConformalQuantile(calibProbs, calibLabels, 0.25, cfgK0)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantile(KReg=0) failed: %v", err)
	}
	calK1, err := CalibrateConformalQuantile(calibProbs, calibLabels, 0.25, cfgK1)
	if err != nil {
		t.Fatalf("CalibrateConformalQuantile(KReg=1) failed: %v", err)
	}
	if math.Abs((calK0.QHat-calK1.QHat)-0.10) > 1e-12 {
		t.Fatalf("expected KReg=0 QHat (%v) to exceed KReg=1 QHat (%v) by exact rank-1 shift 0.10", calK0.QHat, calK1.QHat)
	}

	// 4. Exhaustively verify all 9 default abstain keys (case-insensitive trimmed), explicit []string{} opt-out, and custom override list.
	allNineAbstain := []string{
		"0",
		"abstain",
		"insufficient_evidence",
		"none",
		"none_of_the_above",
		"none_visible",
		"oos",
		"uncertain",
		"unknown",
	}
	variants := append([]string{}, allNineAbstain...)
	variants = append(variants, " 0 ", "ABSTAIN", " None ", "None_Visible", " OOS\t", "UNCERTAIN", "Unknown")
	for _, key := range variants {
		dist := map[string]float64{key: 0.97, "concrete_option": 0.03}
		resNil, err := ConformalPredictionSet(dist, 0.20, ConformalConfig{Method: ConformalLAC})
		if err != nil {
			t.Fatalf("abstain probe %q failed: %v", key, err)
		}
		if !resNil.Singleton || !resNil.ContainsAbstain || !resNil.ShouldEscalate {
			t.Fatalf("abstain probe %q: expected Singleton && ContainsAbstain && ShouldEscalate, got %+v", key, resNil)
		}

		resOptOut, err := ConformalPredictionSet(dist, 0.20, ConformalConfig{Method: ConformalLAC, AbstainLabels: []string{}})
		if err != nil {
			t.Fatalf("opt-out probe %q failed: %v", key, err)
		}
		if !resOptOut.Singleton || resOptOut.ContainsAbstain || resOptOut.ShouldEscalate {
			t.Fatalf("opt-out probe %q: expected ContainsAbstain=false and ShouldEscalate=false with AbstainLabels=[]string{}, got %+v", key, resOptOut)
		}
	}

	customCfg := ConformalConfig{Method: ConformalLAC, AbstainLabels: []string{"needs_human"}}
	resCustomHit, err := ConformalPredictionSet(map[string]float64{" NEEDS_HUMAN ": 0.95, "ok": 0.05}, 0.20, customCfg)
	if err != nil || !resCustomHit.ContainsAbstain || !resCustomHit.ShouldEscalate {
		t.Fatalf("expected custom abstain label match, got %+v (err=%v)", resCustomHit, err)
	}
	resCustomNonMember, err := ConformalPredictionSet(map[string]float64{"oos": 0.95, "ok": 0.05}, 0.20, customCfg)
	if err != nil || resCustomNonMember.ContainsAbstain || resCustomNonMember.ShouldEscalate {
		t.Fatalf("expected default key 'oos' not to trigger when custom AbstainLabels excludes it, got %+v (err=%v)", resCustomNonMember, err)
	}
}
