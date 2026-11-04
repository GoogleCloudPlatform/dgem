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
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestFusePriorWithHesitationGate_Unit(t *testing.T) {
	optKeys := []string{"opt_a", "opt_b", "opt_c"}

	// Case 0: DefaultEngineOptions keeps PriorWeight == 0 (opt-in), so prior_logits are ignored by default.
	defOpts := DefaultEngineOptions()
	if defOpts.PriorWeight != 0 {
		t.Fatalf("expected DefaultEngineOptions().PriorWeight == 0 (opt-in), got %v", defOpts.PriorWeight)
	}
	rawConfident := map[string]float64{
		"opt_a": 0.97,
		"opt_b": 0.02,
		"opt_c": 0.01,
	}
	priors := map[string]float64{
		"opt_a": 1.2,
		"opt_b": -0.5,
		"opt_c": -1.0,
	}
	probs0, _, _, _, _, _, fused0 := FusePriorWithHesitationGate(
		rawConfident, priors, optKeys, 1.0, defOpts.PriorWeight, defOpts.HesitationGate,
	)
	if fused0 {
		t.Fatalf("expected PriorFused=false when PriorWeight=0")
	}
	if math.Abs(probs0["opt_a"]-0.97) > 1e-6 {
		t.Fatalf("expected raw probability 0.97 unchanged when PriorWeight=0, got %.6f", probs0["opt_a"])
	}

	// Case 1: Low-entropy raw distribution (normH < 0.16) with PriorWeight=0.35 fuses the prior.
	probs, winner, _, _, _, rawNormH, fused := FusePriorWithHesitationGate(
		rawConfident, priors, optKeys, 1.0, 0.35, 0.16,
	)
	if !fused {
		t.Fatalf("expected prior to be fused when rawNormH=%.4f < 0.16", rawNormH)
	}
	if winner != "opt_a" {
		t.Fatalf("expected winner opt_a, got %q", winner)
	}
	if probs["opt_a"] <= 0.97 {
		t.Fatalf("expected reinforcing prior to increase opt_a probability above 0.97, got %.6f", probs["opt_a"])
	}

	// Case 2: High-entropy raw distribution (normH >= 0.16) suppresses a conflicting prior
	// so raw hesitation is never masked by stage-1 retrieval confidence.
	rawUncertain := map[string]float64{
		"opt_a": 0.38,
		"opt_b": 0.35,
		"opt_c": 0.27,
	}
	conflictingPrior := map[string]float64{
		"opt_a": -2.0,
		"opt_b": 4.0,
		"opt_c": -2.0,
	}
	probs2, winner2, _, normH2, _, rawNormH2, fused2 := FusePriorWithHesitationGate(
		rawUncertain, conflictingPrior, optKeys, 1.0, 0.50, 0.16,
	)
	if fused2 {
		t.Fatalf("expected prior fusion to be suppressed when rawNormH=%.4f >= 0.16", rawNormH2)
	}
	if winner2 != "opt_a" {
		t.Fatalf("expected raw winner opt_a to be preserved when gate suppresses prior, got %q", winner2)
	}
	if math.Abs(normH2-rawNormH2) > 1e-9 {
		t.Fatalf("expected normalized entropy to match rawNormalizedH (%.6f != %.6f)", normH2, rawNormH2)
	}
	if math.Abs(probs2["opt_a"]-0.38) > 1e-4 {
		t.Fatalf("expected raw probability 0.38 to be preserved, got %.6f", probs2["opt_a"])
	}
}

func computeHesitationAUROC(scores []float64, isError []bool) float64 {
	var pos, neg []float64
	for i, s := range scores {
		if isError[i] {
			pos = append(pos, s)
		} else {
			neg = append(neg, s)
		}
	}
	if len(pos) == 0 || len(neg) == 0 {
		return 0.5
	}
	wins := 0.0
	for _, hp := range pos {
		for _, hn := range neg {
			if hp > hn {
				wins += 1.0
			} else if hp == hn {
				wins += 0.5
			}
		}
	}
	return wins / float64(len(pos)*len(neg))
}

func TestExecuteSystemOne_EvidencePromptValidation(t *testing.T) {
	cli := client.NewClient("http://127.0.0.1:0", "", 0)
	opts := DefaultEngineOptions()

	// 1. Missing or <2 options in evidence_criteria -> ErrInvalidRequest.
	_, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "test state",
		Questions: map[string]SystemOneQuestion{
			"q1": {
				Type:             "choice",
				Instructions:     "Pick one.",
				Criteria:         map[string]string{"a": "Option A", "b": "Option B"},
				EvidencePrompt:   "Check evidence.",
				EvidenceCriteria: map[string]string{"only_one": "Single option"},
			},
		},
	}, opts)
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest when evidence_criteria has <2 options, got %v", err)
	}

	// 2. Collision between <id>_evidence and an existing question key -> ErrInvalidRequest.
	_, err = ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "test state",
		Questions: map[string]SystemOneQuestion{
			"q1": {
				Type:             "choice",
				Instructions:     "Pick one.",
				Criteria:         map[string]string{"a": "Option A", "b": "Option B"},
				EvidencePrompt:   "Check evidence.",
				EvidenceCriteria: map[string]string{"ev_a": "Cue A", "ev_b": "Cue B"},
			},
			"q1_evidence": {
				Type:         "choice",
				Instructions: "Colliding question.",
				Criteria:     map[string]string{"x": "X", "y": "Y"},
			},
		},
	}, opts)
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest on <id>_evidence collision, got %v", err)
	}

	// 3. evidence_prompt on a wide question (K > 26) -> ErrInvalidRequest.
	wideCrit := make(map[string]string, 30)
	for i := 0; i < 30; i++ {
		wideCrit[fmt.Sprintf("opt_%02d", i)] = fmt.Sprintf("Option %d", i)
	}
	_, err = ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "test state",
		Questions: map[string]SystemOneQuestion{
			"wide_q": {
				Type:             "choice",
				Instructions:     "Pick one of 30.",
				Criteria:         wideCrit,
				EvidencePrompt:   "Check evidence.",
				EvidenceCriteria: map[string]string{"ev_a": "Cue A", "ev_b": "Cue B"},
			},
		},
	}, opts)
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest when evidence_prompt is used on a wide question, got %v", err)
	}
}

func TestExecuteSystemOne_EvidencePromptAndSlotBudget_RealCalibration(t *testing.T) {
	calibPath := filepath.Join("..", "..", "benchmarks", "calibration_suite.jsonl")
	f, err := os.Open(calibPath)
	if err != nil {
		t.Fatalf("failed to open %s: %v", calibPath, err)
	}
	defer f.Close()

	var firstQuery string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row struct {
			Fields map[string]any `json:"fields"`
		}
		if err := json.Unmarshal([]byte(line), &row); err == nil && len(row.Fields) > 0 {
			b, _ := json.Marshal(row.Fields)
			firstQuery = string(b)
			break
		}
	}
	if firstQuery == "" {
		t.Fatal("failed to read first calibration item")
	}

	var calls int32
	var sawEvidenceBeforeDecision bool
	var sawCallerEvidenceOptions bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		var sysPayload struct {
			Questions []struct {
				ID      string `json:"id"`
				Options []struct {
					Name        string `json:"name"`
					Description string `json:"description"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}
		qs := sysPayload.Questions
		if len(qs) >= 2 && qs[0].ID == "q1_evidence" && qs[1].ID == "q1" {
			sawEvidenceBeforeDecision = true
			if len(qs[0].Options) == 3 {
				sawCallerEvidenceOptions = true
			}
		}
		ans := make(map[string]client.QuestionAnswer, len(qs))
		for _, q := range qs {
			if strings.HasSuffix(q.ID, "_evidence") {
				ans[q.ID] = client.QuestionAnswer{
					Choice:        "present_clear",
					Probabilities: map[string]float64{"present_clear": 0.94, "occluded_or_blurry": 0.04, "missing_from_input": 0.02},
				}
			} else {
				ans[q.ID] = client.QuestionAnswer{
					Choice:        "entailment",
					Probabilities: map[string]float64{"entailment": 0.92, "neutral": 0.05, "contradiction": 0.03},
				}
			}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: ans})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": string(envBytes)}},
			},
		})
	}))
	defer srv.Close()

	// Build 5 questions where 4 have EvidencePrompt + EvidenceCriteria (total slots = 4*2 + 1 = 9 > MaxSlotsPerPass=8),
	// verifying caller-supplied evidence options, same-canvas placement, and slot-budget batch splitting into 2 passes.
	evCrit := map[string]string{
		"present_clear":      "The decisive premise fact is explicitly stated.",
		"occluded_or_blurry": "The premise is ambiguous or underspecified.",
		"missing_from_input": "The premise omits the required fact.",
	}
	questions := make(map[string]SystemOneQuestion, 5)
	for i := 1; i <= 5; i++ {
		qid := fmt.Sprintf("q%d", i)
		ev := ""
		var ec map[string]string
		if i <= 4 {
			ev = "Verify whether the premise states the numeric date required by the hypothesis."
			ec = evCrit
		}
		questions[qid] = SystemOneQuestion{
			Type:             "choice",
			Instructions:     "Select the NLI relation.",
			EvidencePrompt:   ev,
			EvidenceCriteria: ec,
			Criteria: map[string]string{
				"entailment":    "Premise establishes hypothesis.",
				"neutral":       "Neither proven nor disproven.",
				"contradiction": "Premise contradicts hypothesis.",
			},
		}
	}

	cli := client.NewClient(srv.URL, "", 0)
	resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State:     firstQuery,
		Questions: questions,
	}, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}
	if !sawEvidenceBeforeDecision {
		t.Fatal("expected q1_evidence to be placed immediately ahead of q1 on the canvas")
	}
	if !sawCallerEvidenceOptions {
		t.Fatal("expected q1_evidence to carry the caller-supplied 3 evidence_criteria options")
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("expected 9 total slots (4x2 + 1) to split into 2 passes at MaxSlotsPerPass=8, got %d passes", got)
	}
	if resp.Answers["q1"].Evidence != "present_clear" {
		t.Fatalf("expected q1.Evidence = 'present_clear', got %q", resp.Answers["q1"].Evidence)
	}
	if resp.Answers["q5"].Evidence != "" {
		t.Fatalf("expected q5.Evidence to be empty when EvidencePrompt is unset, got %q", resp.Answers["q5"].Evidence)
	}
}

func TestFusePriorWithGateMode_Schedules(t *testing.T) {
	optKeys := []string{"opt_a", "opt_b", "opt_c"}
	rawUncertain := map[string]float64{"opt_a": 0.46, "opt_b": 0.44, "opt_c": 0.10}
	tieBreakPrior := map[string]float64{"opt_a": -1.5, "opt_b": 3.0, "opt_c": -1.5}

	// "hesitant" fuses on rawNormH >= 0.16 to break near-ties while preserving rawNormH.
	probsH, winnerH, _, _, _, rawNormHH, fusedH := FusePriorWithGateMode(
		rawUncertain, tieBreakPrior, optKeys, 1.0, 0.35, 0.16, "hesitant",
	)
	if !fusedH {
		t.Fatalf("expected hesitant gate to fuse when rawNormH=%.4f >= 0.16", rawNormHH)
	}
	if winnerH != "opt_b" || probsH["opt_b"] <= probsH["opt_a"] {
		t.Fatalf("expected hesitant gate to flip near-tie winner to opt_b, got %q (%v)", winnerH, probsH)
	}

	// "hesitant" leaves confident answers (rawNormH < 0.16) untouched even under an adversarial prior.
	rawConfident := map[string]float64{"opt_a": 0.98, "opt_b": 0.015, "opt_c": 0.005}
	advPrior := map[string]float64{"opt_a": -4.0, "opt_b": 8.0, "opt_c": -4.0}
	probsHC, winnerHC, _, _, _, rawNormHC, fusedHC := FusePriorWithGateMode(
		rawConfident, advPrior, optKeys, 1.0, 0.80, 0.16, "hesitant",
	)
	if fusedHC {
		t.Fatalf("expected hesitant gate to skip confident item (rawNormH=%.4f < 0.16)", rawNormHC)
	}
	if winnerHC != "opt_a" || math.Abs(probsHC["opt_a"]-0.98) > 1e-6 {
		t.Fatalf("expected confident item unchanged under hesitant gate, got %q (%.6f)", winnerHC, probsHC["opt_a"])
	}

	// "proportional" scales effective weight by priorWeight * rawNormH.
	_, winnerP, _, _, _, _, fusedP := FusePriorWithGateMode(
		rawUncertain, tieBreakPrior, optKeys, 1.0, 0.50, 0.16, "proportional",
	)
	if !fusedP || winnerP != "opt_b" {
		t.Fatalf("expected proportional gate to fuse and pick opt_b on uncertain item, fused=%v winner=%q", fusedP, winnerP)
	}

	// Invalid PriorGateMode is rejected by ExecuteSystemOne.
	cli := client.NewClient("http://127.0.0.1:0", "", 0)
	badOpts := DefaultEngineOptions()
	badOpts.PriorGateMode = "invalid_mode"
	_, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "test",
		Questions: map[string]SystemOneQuestion{
			"q1": {Type: "choice", Instructions: "Pick", Criteria: map[string]string{"a": "A", "b": "B"}},
		},
	}, badOpts)
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for invalid PriorGateMode, got %v", err)
	}
}

func TestExecuteSystemOne_WideTournamentAndNoulPriorFusion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		var sysPayload struct {
			Questions []struct {
				ID      string `json:"id"`
				Type    string `json:"type"`
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}

		ans := make(map[string]client.QuestionAnswer, len(sysPayload.Questions))
		for _, q := range sysPayload.Questions {
			if q.Type == "boolean" || q.ID == "is_fraud" {
				ans[q.ID] = client.QuestionAnswer{
					Noul:          0.48,
					Entropy:       0.692,
					Probabilities: map[string]float64{"yes": 0.48, "no": 0.52},
				}
				continue
			}
			probs := make(map[string]float64, len(q.Options))
			has00, has01 := false, false
			for _, o := range q.Options {
				if o.Name == "opt_00" {
					has00 = true
				}
				if o.Name == "opt_01" {
					has01 = true
				}
			}
			winner := q.Options[0].Name
			if has00 && has01 {
				// Near-tie between opt_00 (0.49) and opt_01 (0.47), remaining mass spread thinly.
				rem := 0.04 / float64(maxInt(1, len(q.Options)-2))
				for _, o := range q.Options {
					switch o.Name {
					case "opt_00":
						probs[o.Name] = 0.49
					case "opt_01":
						probs[o.Name] = 0.47
					default:
						probs[o.Name] = rem
					}
				}
				winner = "opt_00"
			} else {
				u := 1.0 / float64(len(q.Options))
				for _, o := range q.Options {
					probs[o.Name] = u
				}
			}
			ans[q.ID] = client.QuestionAnswer{Choice: winner, Probabilities: probs}
		}

		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: ans})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": string(envBytes)}},
			},
		})
	}))
	defer srv.Close()

	wideCrit := make(map[string]string, 30)
	widePriors := make(map[string]float64, 30)
	for i := 0; i < 30; i++ {
		k := fmt.Sprintf("opt_%02d", i)
		wideCrit[k] = fmt.Sprintf("Option %02d", i)
		widePriors[k] = -1.0
	}
	widePriors["opt_01"] = 3.0 // Strong prior favouring runner-up opt_01 over opt_00.

	req := SystemOneRequest{
		State: "Evaluate both a wide choice question (K=30) and a yes/no noul question with prior_logits.",
		Questions: map[string]SystemOneQuestion{
			"wide_intent": {
				Type:         "choice",
				Instructions: "Pick the matching intent.",
				Criteria:     wideCrit,
				PriorLogits:  widePriors,
			},
			"is_fraud": {
				Type:         "noul",
				Instructions: "Is this transaction fraudulent?",
				PriorLogits:  map[string]float64{"yes": 2.0, "no": -2.0},
			},
		},
	}

	if FitsServingImage(SystemOneRequest{
		State: "s",
		Questions: map[string]SystemOneQuestion{
			"is_fraud": {
				Type:         "noul",
				Instructions: "Check",
				PriorLogits:  map[string]float64{"yes": 1.0, "no": -1.0},
			},
		},
	}) {
		t.Fatal("expected FitsServingImage=false when PriorLogits is populated")
	}

	cli := client.NewClient(srv.URL, "", 0)
	opts := DefaultEngineOptions()
	opts.PriorWeight = 0.50
	opts.PriorGateMode = "hesitant"
	opts.HesitationGate = 0.16

	resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}

	wideAns := resp.Answers["wide_intent"]
	if !wideAns.PriorFused {
		t.Fatalf("expected wide_intent PriorFused=true on K=30 tournament, got false (rawNormH=%.4f)", wideAns.RawNormalizedH)
	}
	if wideAns.Choice != "opt_01" {
		t.Fatalf("expected wide_intent winner to flip from opt_00 to opt_01 under hesitant prior fusion, got %q", wideAns.Choice)
	}
	if wideAns.RawNormalizedH < 0.16 {
		t.Fatalf("expected wide_intent RawNormalizedH >= 0.16 to be preserved, got %.4f", wideAns.RawNormalizedH)
	}

	noulAns := resp.Answers["is_fraud"]
	if !noulAns.PriorFused || noulAns.Noul == nil {
		t.Fatalf("expected is_fraud noul question to fuse prior_logits, got PriorFused=%v Noul=%v", noulAns.PriorFused, noulAns.Noul)
	}
	if *noulAns.Noul <= 0.50 {
		t.Fatalf("expected positive yes prior to push noul probability above 0.50 (raw 0.48), got %.4f", *noulAns.Noul)
	}
	if noulAns.RawNormalizedH < 0.99 {
		t.Fatalf("expected pre-fusion RawNormalizedH near 1.0 for 0.48/0.52 split, got %.4f", noulAns.RawNormalizedH)
	}
}

func TestFusePriorWithGateMode_ProbabilisticAdversarialAndLabelShift(t *testing.T) {
	rng := rand.New(rand.NewSource(20261024))
	optKeys := []string{"class_a", "class_b", "class_c", "class_d"}
	const numTrials = 500

	// Part 1: Adversarial prior stress test on post-fusion normalized_entropy vs raw_normalized_entropy.
	// Confident gate fuses on low-hesitation items (including over-confident wrong items whose runner-up or
	// predicted wrong class receives reinforcement), depressing post-fusion normalized_entropy on errors,
	// whereas hesitant gate leaves rawNormH < 0.16 untouched and RawNormalizedH stays invariant across all modes.
	var (
		confPostH, hesPostH, propPostH, rawPreH          []float64
		confIsErr, hesIsErr, propIsErr, rawBaselineIsErr []bool
	)
	for trial := 0; trial < numTrials; trial++ {
		gold := optKeys[trial%len(optKeys)]
		wrong := optKeys[(trial+1)%len(optKeys)]

		logits := make(map[string]float64, len(optKeys))
		switch {
		case trial%10 < 6:
			// Confident correct item
			logits[gold] = 3.4 + 0.35*rng.NormFloat64()
			logits[wrong] = -0.5 + 0.35*rng.NormFloat64()
		case trial%10 < 8:
			// Moderately confident error (rawNormH around 0.08..0.15)
			logits[wrong] = 2.7 + 0.25*rng.NormFloat64()
			logits[gold] = -0.3 + 0.25*rng.NormFloat64()
		default:
			// Uncertain item (rawNormH > 0.45)
			logits[gold] = 0.6 + 0.5*rng.NormFloat64()
			logits[wrong] = 0.5 + 0.5*rng.NormFloat64()
		}
		for _, k := range optKeys {
			if _, ok := logits[k]; !ok {
				logits[k] = -1.6 + 0.3*rng.NormFloat64()
			}
		}
		probs, _, _, _ := NormalizeAndScaleProbabilities(expMap(logits), optKeys, 1.0)
		advPrior := map[string]float64{wrong: 3.2, gold: -1.5}

		_, baseWin, _, _, _, rNormBase, _ := FusePriorWithGateMode(probs, nil, optKeys, 1.0, 0, 0.16, "confident")
		_, confWin, _, confNormH, _, rNormConf, _ := FusePriorWithGateMode(probs, advPrior, optKeys, 1.0, 0.45, 0.16, "confident")
		_, hesWin, _, hesNormH, _, rNormHes, _ := FusePriorWithGateMode(probs, advPrior, optKeys, 1.0, 0.35, 0.16, "hesitant")
		_, propWin, _, propNormH, _, rNormProp, _ := FusePriorWithGateMode(probs, advPrior, optKeys, 1.0, 0.35, 0.16, "proportional")

		if math.Abs(rNormBase-rNormConf) > 1e-12 || math.Abs(rNormBase-rNormHes) > 1e-12 || math.Abs(rNormBase-rNormProp) > 1e-12 {
			t.Fatalf("trial %d: RawNormalizedH drifted across gate modes", trial)
		}

		rawPreH = append(rawPreH, rNormBase)
		rawBaselineIsErr = append(rawBaselineIsErr, baseWin != gold)
		confPostH = append(confPostH, confNormH)
		confIsErr = append(confIsErr, confWin != gold)
		hesPostH = append(hesPostH, hesNormH)
		hesIsErr = append(hesIsErr, hesWin != gold)
		propPostH = append(propPostH, propNormH)
		propIsErr = append(propIsErr, propWin != gold)
	}

	aurocRaw := computeHesitationAUROC(rawPreH, rawBaselineIsErr)
	aurocConfPost := computeHesitationAUROC(confPostH, confIsErr)
	aurocHesRaw := computeHesitationAUROC(rawPreH, hesIsErr)
	aurocPropRaw := computeHesitationAUROC(rawPreH, propIsErr)

	if aurocConfPost >= aurocRaw-0.01 {
		t.Fatalf("expected confident post-fusion normalized_entropy AUROC (%.4f) to drop below raw AUROC (%.4f) under adversarial prior", aurocConfPost, aurocRaw)
	}
	if aurocHesRaw < 0.75 || aurocPropRaw < 0.75 {
		t.Fatalf("expected RawNormalizedH under hesitant/proportional gates to maintain high error AUROC (>=0.75), got %.4f / %.4f", aurocHesRaw, aurocPropRaw)
	}

	// Part 2: Induced base-rate label shift (90% prevalence on class_a, 10% across class_b..d).
	// Calibrated base-rate log-prior log(pi) improves accuracy and NLL on ambiguous trials under hesitant
	// and proportional gates while preserving confident minority-class predictions.
	shiftPrior := map[string]float64{
		"class_a": math.Log(0.90),
		"class_b": math.Log(0.10 / 3.0),
		"class_c": math.Log(0.10 / 3.0),
		"class_d": math.Log(0.10 / 3.0),
	}
	var baseCorrect, hesCorrect, propCorrect int
	var baseNLL, hesNLL, propNLL float64
	var minorityConfidentPreserved, minorityConfidentTotal int

	for trial := 0; trial < numTrials; trial++ {
		isMajority := (trial % 10) < 9
		gold := "class_a"
		if !isMajority {
			gold = optKeys[1+(trial%3)]
		}
		ambiguous := (trial % 5) < 3
		logits := make(map[string]float64, len(optKeys))
		for _, k := range optKeys {
			if k == gold {
				if ambiguous {
					logits[k] = 0.55 + 0.55*rng.NormFloat64()
				} else {
					logits[k] = 3.8 + 0.30*rng.NormFloat64()
				}
			} else {
				if ambiguous {
					logits[k] = 0.35 + 0.55*rng.NormFloat64()
				} else {
					logits[k] = -1.2 + 0.30*rng.NormFloat64()
				}
			}
		}
		probs, baseWin, _, rawNormH := NormalizeAndScaleProbabilities(expMap(logits), optKeys, 1.0)
		probsHes, hesWin, _, _, _, _, _ := FusePriorWithGateMode(probs, shiftPrior, optKeys, 1.0, 0.45, 0.16, "hesitant")
		probsProp, propWin, _, _, _, _, _ := FusePriorWithGateMode(probs, shiftPrior, optKeys, 1.0, 0.45, 0.16, "proportional")

		if baseWin == gold {
			baseCorrect++
		}
		if hesWin == gold {
			hesCorrect++
		}
		if propWin == gold {
			propCorrect++
		}
		baseNLL += -math.Log(math.Max(1e-6, probs[gold]))
		hesNLL += -math.Log(math.Max(1e-6, probsHes[gold]))
		propNLL += -math.Log(math.Max(1e-6, probsProp[gold]))

		if !isMajority && rawNormH < 0.16 && baseWin == gold {
			minorityConfidentTotal++
			if hesWin == gold {
				minorityConfidentPreserved++
			}
		}
	}

	if hesCorrect <= baseCorrect || propCorrect <= baseCorrect {
		t.Fatalf("expected hesitant (%d) and proportional (%d) gates to beat unshifted baseline (%d) under 90%% label shift", hesCorrect, propCorrect, baseCorrect)
	}
	if hesNLL >= baseNLL || propNLL >= baseNLL {
		t.Fatalf("expected hesitant (%.2f) and proportional (%.2f) NLL to improve over baseline (%.2f)", hesNLL, propNLL, baseNLL)
	}
	if minorityConfidentTotal == 0 || minorityConfidentPreserved != minorityConfidentTotal {
		t.Fatalf("expected hesitant gate to preserve 100%% of confident minority predictions (%d/%d)", minorityConfidentPreserved, minorityConfidentTotal)
	}
}

func expMap(logits map[string]float64) map[string]float64 {
	maxL := math.Inf(-1)
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	out := make(map[string]float64, len(logits))
	for k, v := range logits {
		out[k] = math.Exp(v - maxL)
	}
	return out
}
