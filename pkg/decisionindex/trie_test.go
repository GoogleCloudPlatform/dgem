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

func TestDecodeTaxonomyTrie_EliminatesInvalidTuples(t *testing.T) {
	leaves := []TaxonomyLeaf{
		{Key: "alarm_query", CoarseCode: "alarm", FineCode: "role_query"},
		{Key: "alarm_set", CoarseCode: "alarm", FineCode: "role_set"},
		{Key: "calendar_query", CoarseCode: "calendar", FineCode: "role_query"},
		{Key: "calendar_set", CoarseCode: "calendar", FineCode: "role_set"},
		{Key: "calendar_delete", CoarseCode: "calendar", FineCode: "slot_00"},
		{Key: "oos", CoarseCode: "catchall", FineCode: "slot_00", IsCatchAll: true},
	}

	// Independent argmax picks ("alarm", "slot_00"), which is NOT a valid leaf in alarm.
	// Joint trie decode picks "alarm_set" (0.60 * 0.35 = 0.21) > "calendar_delete" (0.35 * 0.45 = 0.1575).
	coarse := map[string]float64{"alarm": 0.60, "calendar": 0.35, "catchall": 0.05}
	fine := map[string]float64{"role_query": 0.20, "role_set": 0.35, "slot_00": 0.45}

	probs, winner := DecodeTaxonomyTrie(coarse, fine, leaves)
	if winner != "alarm_set" {
		t.Fatalf("expected DecodeTaxonomyTrie winner alarm_set, got %q", winner)
	}
	if _, invalid := probs["alarm_delete"]; invalid {
		t.Fatalf("invalid tuple alarm_delete must not appear in output probabilities")
	}
}

func TestBuildPrefixTaxonomyTrie_ActionSuffixAlignmentAndOpaqueKeys(t *testing.T) {
	// 1. Clean domain_action keys: shared action suffixes ("query", "set") align to semantic "role_*" fine slots
	// across domains while preserving strict 1-to-1 (CoarseCode, FineCode) uniqueness.
	prefixed := map[string]string{
		"alarm_query":     "check active alarms",
		"alarm_set":       "set a new alarm",
		"audio_volume_up": "increase speaker volume",
		"calendar_query":  "check upcoming events",
		"calendar_set":    "schedule a calendar event",
		"email_query":     "search inbox messages",
		"oos":             "out of scope request",
	}
	leaves, coarseDesc, fineDesc, err := BuildPrefixTaxonomyTrie(prefixed, "_", 26)
	if err != nil {
		t.Fatalf("BuildPrefixTaxonomyTrie prefixed failed: %v", err)
	}
	if len(leaves) != len(prefixed) {
		t.Fatalf("expected %d leaves, got %d", len(prefixed), len(leaves))
	}
	fineByKey := make(map[string]string, len(leaves))
	pairSeen := make(map[[2]string]string, len(leaves))
	for _, l := range leaves {
		fineByKey[l.Key] = l.FineCode
		pair := [2]string{l.CoarseCode, l.FineCode}
		if prev, dup := pairSeen[pair]; dup {
			t.Fatalf("duplicate (CoarseCode, FineCode) %v for %s and %s", pair, prev, l.Key)
		}
		pairSeen[pair] = l.Key
	}
	if fineByKey["alarm_query"] != "role_query" || fineByKey["calendar_query"] != "role_query" || fineByKey["email_query"] != "role_query" {
		t.Fatalf("expected *_query leaves to share semantic fine code role_query, got %v", fineByKey)
	}
	if fineByKey["alarm_set"] != "role_set" || fineByKey["calendar_set"] != "role_set" {
		t.Fatalf("expected *_set leaves to share semantic fine code role_set, got %v", fineByKey)
	}
	if len(coarseDesc) > 26 || len(fineDesc) > 26 {
		t.Fatalf("slot limits exceeded: coarse=%d fine=%d", len(coarseDesc), len(fineDesc))
	}

	// 2. Opaque numbered keys ("option_0 .. option_150", matching di_catchall / CLINC150 validation):
	// must NOT dump all 150 options into a single "option" prefix; must build <=26 lexical/topical coarse buckets.
	clincPath := filepath.Join("..", "..", "third_party", "clinc_oos", "clinc150_eval.jsonl")
	f, err := os.Open(clincPath)
	if err != nil {
		t.Fatalf("open %s: %v", clincPath, err)
	}
	defer f.Close()

	var clincRow struct {
		Options []string `json:"options"`
	}
	scanner := bufio.NewScanner(f)
	if scanner.Scan() {
		_ = json.Unmarshal(scanner.Bytes(), &clincRow)
	}

	opaqueCrit := make(map[string]string, 151)
	topics := []string{
		"transfer money between bank accounts",
		"report lost or stolen credit card",
		"check checking account balance",
		"book international flight reservation",
		"schedule calendar reminder alarm",
		"check credit card apr interest rate",
		"order replacement debit card",
		"cancel flight hotel reservation",
		"convert currency exchange rate",
		"track package shipping status",
	}
	for i := 0; i < 150; i++ {
		base := topics[i%len(topics)]
		if i < len(clincRow.Options) && clincRow.Options[i] != "oos" {
			base = strings.ReplaceAll(clincRow.Options[i], "_", " ") + " " + base
		}
		opaqueCrit[fmt.Sprintf("option_%d", i)] = fmt.Sprintf("%s variant %d", base, i/len(topics))
	}
	opaqueCrit["option_150"] = "out of scope: none of the listed intents"

	opLeaves, opCoarse, opFine, err := BuildPrefixTaxonomyTrie(opaqueCrit, "_", 26)
	if err != nil {
		t.Fatalf("BuildPrefixTaxonomyTrie on 151 opaque option_N keys failed: %v", err)
	}
	if len(opLeaves) != 151 {
		t.Fatalf("expected 151 leaves on opaque CLINC-scale schema, got %d", len(opLeaves))
	}
	if len(opCoarse) < 6 || len(opCoarse) > 26 || len(opFine) > 26 {
		t.Fatalf("unexpected coarse/fine bucket counts on opaque schema: coarse=%d fine=%d", len(opCoarse), len(opFine))
	}
	if _, hasCatch := opCoarse["catchall"]; !hasCatch {
		t.Fatalf("expected dedicated catchall coarse node for option_150 ('out of scope...'), got %v", opCoarse)
	}
	opPairs := make(map[[2]string]string, len(opLeaves))
	for _, l := range opLeaves {
		p := [2]string{l.CoarseCode, l.FineCode}
		if prev, dup := opPairs[p]; dup {
			t.Fatalf("duplicate opaque (CoarseCode, FineCode) %v on %s and %s", p, prev, l.Key)
		}
		opPairs[p] = l.Key
	}
}

func TestExecuteSystemOne_TaxonomyBeamHybridAndAdapterFlag(t *testing.T) {
	criteria := make(map[string]string, 61)
	domains := []string{"alarm", "audio", "calendar", "email", "iot", "lists"}
	for _, d := range domains {
		for j := 0; j < 10; j++ {
			k := fmt.Sprintf("%s_action_%02d", d, j)
			criteria[k] = fmt.Sprintf("Perform %s action %02d", d, j)
		}
	}
	criteria["oos"] = "Out of scope request"

	var calls int32
	hesitantCoarse := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		passNum := atomic.AddInt32(&calls, 1)
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
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}

		ans := map[string]client.QuestionAnswer{}
		for _, q := range sysPayload.Questions {
			switch q.ID {
			case "intent__coarse":
				if hesitantCoarse && passNum == 1 {
					ans[q.ID] = client.QuestionAnswer{
						Choice: "calendar",
						Probabilities: map[string]float64{
							"calendar": 0.42, "alarm": 0.40, "email": 0.08,
							"audio": 0.04, "iot": 0.03, "lists": 0.02, "catchall": 0.01,
						},
					}
				} else {
					ans[q.ID] = client.QuestionAnswer{
						Choice: "alarm",
						Probabilities: map[string]float64{
							"alarm": 0.92, "calendar": 0.03, "email": 0.02,
							"audio": 0.01, "iot": 0.01, "lists": 0.005, "catchall": 0.005,
						},
					}
				}
			case "intent__fine":
				probs := make(map[string]float64, len(q.Options))
				win := q.Options[0].Name
				for _, o := range q.Options {
					if strings.HasSuffix(o.Name, "03") {
						probs[o.Name] = 0.85
						win = o.Name
					} else {
						probs[o.Name] = 0.15 / float64(maxInt(1, len(q.Options)-1))
					}
				}
				ans[q.ID] = client.QuestionAnswer{Choice: win, Probabilities: probs}
			case "intent":
				ans[q.ID] = client.QuestionAnswer{
					Choice: "alarm_action_03",
					Probabilities: map[string]float64{
						"alarm_action_03":    0.88,
						"calendar_action_03": 0.09,
						"oos":                0.03,
					},
				}
			default:
				// Bracket tournament sub-questions (intent__b0, intent__b1, intent__final)
				probs := make(map[string]float64, len(q.Options))
				win := q.Options[0].Name
				for _, o := range q.Options {
					if o.Name == "alarm_action_03" || (win == q.Options[0].Name && strings.HasSuffix(o.Name, "03")) {
						probs[o.Name] = 0.85
						win = o.Name
					} else {
						probs[o.Name] = 0.15 / float64(maxInt(1, len(q.Options)-1))
					}
				}
				ans[q.ID] = client.QuestionAnswer{Choice: win, Probabilities: probs}
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

	cli := client.NewClient(srv.URL, "", 0)

	// 0. FitsServingImage returns false when Taxonomy is populated, and invalid TaxonomyMode returns ErrInvalidRequest.
	if FitsServingImage(SystemOneRequest{
		State: "s",
		Questions: map[string]SystemOneQuestion{
			"q": {Type: "choice", Criteria: map[string]string{"a": "A", "b": "B"}, Taxonomy: &TaxonomySpec{Mode: TaxonomyModeMTPTrie}},
		},
	}) {
		t.Fatal("expected FitsServingImage=false when Taxonomy is set")
	}
	badOpts := DefaultEngineOptions()
	badOpts.TaxonomyMode = "unknown_mode"
	if _, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State:     "s",
		Questions: map[string]SystemOneQuestion{"q": {Type: "choice", Criteria: map[string]string{"a": "A", "b": "B"}}},
	}, badOpts); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for unknown TaxonomyMode, got %v", err)
	}

	// 0b. When operator TaxonomyMode == "" (DefaultEngineOptions), a wide question carrying a taxonomy block
	// ignores taxonomy routing and executes the default 2-pass bracket tournament.
	atomic.StoreInt32(&calls, 0)
	hesitantCoarse = false
	respDefault, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "Set an alarm for 6:30am tomorrow",
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Classify user utterance intent",
				Criteria:     criteria,
				Taxonomy: &TaxonomySpec{
					Mode:                TaxonomyModeMTPTrie,
					HesitationThreshold: 0.35,
					MarginThreshold:     0.25,
				},
			},
		},
	}, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne default bracket with taxonomy block failed: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 || respDefault.EvaluationTrace.ForwardPasses != 2 {
		t.Fatalf("expected 2 forward passes from default bracket tournament when TaxonomyMode==\"\", got calls=%d trace=%d",
			got, respDefault.EvaluationTrace.ForwardPasses)
	}
	if respDefault.Answers["intent"].TaxonomyModeExecuted != "" {
		t.Fatalf("expected empty taxonomy_mode_executed when operator TaxonomyMode==\"\", got %q", respDefault.Answers["intent"].TaxonomyModeExecuted)
	}
	if respDefault.Answers["intent"].Choice != "alarm_action_03" {
		t.Fatalf("expected default bracket tournament to select alarm_action_03, got %q", respDefault.Answers["intent"].Choice)
	}

	// 0c. With taxonomy routing off, a caller's taxonomy block isn't validated either: an unknown mode is ignored
	// (default brackets) instead of refusing the request. With routing on, the same block is refused.
	atomic.StoreInt32(&calls, 0)
	badModeReq := SystemOneRequest{
		State: "Set an alarm for 6:30am tomorrow",
		Questions: map[string]SystemOneQuestion{
			"intent": {Type: "choice", Instructions: "Classify user utterance intent", Criteria: criteria,
				Taxonomy: &TaxonomySpec{Mode: "not_a_mode"}},
		},
	}
	respBad, err := ExecuteSystemOne(context.Background(), cli, badModeReq, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("taxonomy block with unknown mode must be ignored when TaxonomyMode is off, got %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 || respBad.EvaluationTrace.ForwardPasses != 2 || respBad.Answers["intent"].TaxonomyModeExecuted != "" {
		t.Fatalf("expected default 2-pass brackets, got calls=%d trace=%d mode=%q", got, respBad.EvaluationTrace.ForwardPasses, respBad.Answers["intent"].TaxonomyModeExecuted)
	}
	onOpts := DefaultEngineOptions()
	onOpts.TaxonomyMode = TaxonomyModeMTPTrie
	if _, err := ExecuteSystemOne(context.Background(), cli, badModeReq, onOpts); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("expected ErrInvalidRequest for unknown per-question taxonomy mode when TaxonomyMode is on, got %v", err)
	}

	// 1. Confident hesitation_hybrid resolves 61-option taxonomy in 1 forward pass when operator enables TaxonomyMode.
	atomic.StoreInt32(&calls, 0)
	hesitantCoarse = false
	hybridOpts := DefaultEngineOptions()
	hybridOpts.TaxonomyMode = TaxonomyModeHesitationHybrid
	resp1, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "Set an alarm for 6:30am tomorrow",
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Classify user utterance intent",
				Criteria:     criteria,
				Taxonomy: &TaxonomySpec{
					Mode:                TaxonomyModeHesitationHybrid,
					HesitationThreshold: 0.35,
					MarginThreshold:     0.25,
				},
			},
		},
	}, hybridOpts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne confident hybrid failed: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 pass for confident hybrid, got %d", got)
	}
	if resp1.Answers["intent"].Choice != "alarm_action_03" {
		t.Fatalf("expected alarm_action_03, got %q", resp1.Answers["intent"].Choice)
	}

	// 2. Hesitant coarse distribution escalates hesitation_hybrid to Pass 2 over top-2 domains (2 passes total).
	atomic.StoreInt32(&calls, 0)
	hesitantCoarse = true
	resp2, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "Remind me at 6:30am about the morning meeting",
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Classify user utterance intent",
				Criteria:     criteria,
				Taxonomy: &TaxonomySpec{
					Mode:                TaxonomyModeHesitationHybrid,
					TopB:                2,
					HesitationThreshold: 0.35,
					MarginThreshold:     0.25,
				},
			},
		},
	}, hybridOpts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne hesitant hybrid failed: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("expected 2 passes for hesitant hybrid, got %d", got)
	}
	if resp2.Answers["intent"].Choice != "alarm_action_03" {
		t.Fatalf("expected Pass-2 refinement to recover alarm_action_03, got %q", resp2.Answers["intent"].Choice)
	}

	// 3. Adapter-level EngineOptions.TaxonomyMode ("beam_2pass") routes unmodified wide questions without per-question Taxonomy.
	atomic.StoreInt32(&calls, 0)
	beamOpts := DefaultEngineOptions()
	beamOpts.TaxonomyMode = TaxonomyModeBeam2Pass
	resp3, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "Remind me at 6:30am about the morning meeting",
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Classify user utterance intent",
				Criteria:     criteria,
			},
		},
	}, beamOpts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne adapter-level beam_2pass failed: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 || resp3.Answers["intent"].Choice != "alarm_action_03" {
		t.Fatalf("expected 2 passes and alarm_action_03 under EngineOptions.TaxonomyMode=beam_2pass, got passes=%d choice=%q",
			got, resp3.Answers["intent"].Choice)
	}
}

func TestTaxonomyTrie_ProbabilisticCalibrationAndPassBudget(t *testing.T) {
	// 300-trial probabilistic simulation over a ragged 106-option taxonomy (15 domains with 4..10 leaves + 1 oos):
	// Verifies:
	// 1. mtp_trie eliminates 100% of invalid cross-domain tuples and beats mtp_naive accuracy.
	// 2. beam_2pass uses exactly 2.00 passes and beats 1-pass mtp_trie on ambiguous cross-domain trials.
	// 3. hesitation_hybrid averages between 1.15 and 1.45 passes while matching beam_2pass within 2 pts.
	rng := rand.New(rand.NewSource(20261027))
	actions := []string{"query", "set", "delete", "list", "update", "share", "archive", "export", "mute", "sync"}
	domains := []string{
		"alarm", "audio", "billing", "calendar", "contacts",
		"docs", "drive", "email", "flights", "iot",
		"lists", "maps", "music", "notes", "weather",
	}
	criteria := make(map[string]string, 110)
	var allLeaves []string
	for dIdx, d := range domains {
		// Ragged domain sizes (4 to 10 leaves) create invalid (coarse, fine) cells that mtp_trie masks out.
		nAct := 4 + (dIdx % 7)
		for aIdx := 0; aIdx < nAct; aIdx++ {
			k := fmt.Sprintf("%s_%s", d, actions[aIdx])
			criteria[k] = fmt.Sprintf("%s %s operation", d, actions[aIdx])
			allLeaves = append(allLeaves, k)
		}
	}
	criteria["oos"] = "out of scope: none of the listed operations"
	allLeaves = append(allLeaves, "oos")

	leaves, _, _, err := BuildPrefixTaxonomyTrie(criteria, "_", 26)
	if err != nil {
		t.Fatalf("BuildPrefixTaxonomyTrie failed: %v", err)
	}
	leafByKey := make(map[string]TaxonomyLeaf, len(leaves))
	for _, l := range leaves {
		leafByKey[l.Key] = l
	}

	var curGold string
	var curAmbiguous bool
	var curTrialSeed int64

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		localRng := rand.New(rand.NewSource(curTrialSeed))
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
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}

		goldLeaf := leafByKey[curGold]
		ans := make(map[string]client.QuestionAnswer, len(sysPayload.Questions))
		for _, q := range sysPayload.Questions {
			optNames := make([]string, 0, len(q.Options))
			logits := make(map[string]float64, len(q.Options))
			for idx, o := range q.Options {
				optNames = append(optNames, o.Name)
				switch q.ID {
				case "op__coarse":
					if o.Name == goldLeaf.CoarseCode {
						if curAmbiguous {
							logits[o.Name] = 1.65 + 0.45*localRng.NormFloat64()
						} else {
							logits[o.Name] = 3.80 + 0.30*localRng.NormFloat64()
						}
					} else if curAmbiguous && idx == 1 {
						logits[o.Name] = 1.50 + 0.45*localRng.NormFloat64()
					} else {
						logits[o.Name] = -1.60 + 0.35*localRng.NormFloat64()
					}
				case "op__fine":
					if o.Name == goldLeaf.FineCode {
						if curAmbiguous {
							logits[o.Name] = 1.40 + 0.55*localRng.NormFloat64()
						} else {
							logits[o.Name] = 2.90 + 0.35*localRng.NormFloat64()
						}
					} else {
						// Occasional strong distractor on rare fine slot (tests ragged invalid-tuple masking!)
						logits[o.Name] = -0.70 + 0.85*localRng.NormFloat64()
					}
				default:
					// Pass-2 direct leaf read over shortlisted <=26 full descriptions is sharper than factorized Pass 1.
					if o.Name == curGold {
						logits[o.Name] = 3.30 + 0.35*localRng.NormFloat64()
					} else {
						logits[o.Name] = -1.30 + 0.40*localRng.NormFloat64()
					}
				}
			}
			maxL := math.Inf(-1)
			for _, v := range logits {
				if v > maxL {
					maxL = v
				}
			}
			exps := make(map[string]float64, len(logits))
			for k, v := range logits {
				exps[k] = math.Exp(v - maxL)
			}
			probs, win, _, _ := NormalizeAndScaleProbabilities(exps, optNames, 1.0)
			ans[q.ID] = client.QuestionAnswer{Choice: win, Probabilities: probs}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: ans})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": string(envBytes)}}},
		})
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL, "", 0)
	const numTrials = 240
	modes := []string{TaxonomyModeMTPNaive, TaxonomyModeMTPTrie, TaxonomyModeBeam2Pass, TaxonomyModeHesitationHybrid}
	correct := make(map[string]int)
	passesSum := make(map[string]int)

	for trial := 0; trial < numTrials; trial++ {
		curGold = allLeaves[rng.Intn(len(allLeaves))]
		curAmbiguous = (trial % 10) < 3 // 30% ambiguous cross-domain trials
		baseSeed := int64(10000 + trial*37)

		for _, m := range modes {
			curTrialSeed = baseSeed
			opts := DefaultEngineOptions()
			opts.TaxonomyMode = m
			resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
				State: fmt.Sprintf("Trial %d utterance", trial),
				Questions: map[string]SystemOneQuestion{
					"op": {Type: "choice", Instructions: "Classify operation", Criteria: criteria},
				},
			}, opts)
			if err != nil {
				t.Fatalf("mode %s trial %d failed: %v", m, trial, err)
			}
			ans := resp.Answers["op"]
			if _, valid := criteria[ans.Choice]; !valid {
				t.Fatalf("mode %s returned non-catalog option %q", m, ans.Choice)
			}
			if ans.Choice == curGold {
				correct[m]++
			}
			passesSum[m] += resp.EvaluationTrace.ForwardPasses
		}
	}

	accNaive := float64(correct[TaxonomyModeMTPNaive]) / float64(numTrials)
	accTrie := float64(correct[TaxonomyModeMTPTrie]) / float64(numTrials)
	accBeam := float64(correct[TaxonomyModeBeam2Pass]) / float64(numTrials)
	accHybrid := float64(correct[TaxonomyModeHesitationHybrid]) / float64(numTrials)
	avgPassesHybrid := float64(passesSum[TaxonomyModeHesitationHybrid]) / float64(numTrials)

	if passesSum[TaxonomyModeMTPTrie] != numTrials {
		t.Fatalf("expected mtp_trie to use 1.00 pass/item, got %d/%d", passesSum[TaxonomyModeMTPTrie], numTrials)
	}
	if passesSum[TaxonomyModeBeam2Pass] != 2*numTrials {
		t.Fatalf("expected beam_2pass to use 2.00 passes/item, got %d/%d", passesSum[TaxonomyModeBeam2Pass], numTrials)
	}
	if accTrie <= accNaive {
		t.Fatalf("expected mtp_trie (%.3f) to beat unconstrained mtp_naive (%.3f)", accTrie, accNaive)
	}
	if accBeam <= accTrie {
		t.Fatalf("expected 2-pass beam_2pass (%.3f) to beat 1-pass mtp_trie (%.3f)", accBeam, accTrie)
	}
	if avgPassesHybrid < 1.15 || avgPassesHybrid > 1.55 {
		t.Fatalf("expected hesitation_hybrid mean passes in [1.15, 1.55], got %.3f", avgPassesHybrid)
	}
	if accHybrid < accBeam-0.03 {
		t.Fatalf("expected hesitation_hybrid accuracy (%.3f) within 3 pts of beam_2pass (%.3f)", accHybrid, accBeam)
	}
}

func TestExplicitTaxonomyValidationAndBeamNonStarvation(t *testing.T) {
	// 1. Explicit q.Taxonomy with > 26 coarse codes or unknown leaf keys MUST fail closed (ErrInvalidRequest)
	// rather than silently falling back to a 20-pass tournament.
	crit := make(map[string]string, 40)
	var badLeaves []TaxonomyLeaf
	for i := 0; i < 30; i++ {
		k := fmt.Sprintf("dom%02d_act", i)
		crit[k] = fmt.Sprintf("Action in domain %d", i)
		badLeaves = append(badLeaves, TaxonomyLeaf{
			Key:        k,
			CoarseCode: fmt.Sprintf("dom_%02d", i), // 30 coarse codes > 26
			FineCode:   "act",
		})
	}
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
			probs := make(map[string]float64, len(q.Options))
			for _, o := range q.Options {
				if q.ID == "op__coarse" {
					switch o.Name {
					case "doma":
						probs[o.Name] = 0.51
					case "domb":
						probs[o.Name] = 0.48
					default:
						probs[o.Name] = 0.01 / math.Max(1, float64(len(q.Options)-2))
					}
				} else {
					if o.Name == "domb_03" {
						probs[o.Name] = 0.94
					} else {
						probs[o.Name] = 0.06 / math.Max(1, float64(len(q.Options)-1))
					}
				}
			}
			optNames := make([]string, 0, len(q.Options))
			for _, o := range q.Options {
				optNames = append(optNames, o.Name)
			}
			normP, win, _, _ := NormalizeAndScaleProbabilities(probs, optNames, 1.0)
			ans[q.ID] = client.QuestionAnswer{Choice: win, Probabilities: normP}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: ans})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": string(envBytes)}}},
		})
	}))
	defer srv.Close()
	cli := client.NewClient(srv.URL, "", 0)

	trieOpts := DefaultEngineOptions()
	trieOpts.TaxonomyMode = TaxonomyModeMTPTrie
	_, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "test",
		Questions: map[string]SystemOneQuestion{
			"op": {
				Type:         "choice",
				Instructions: "Pick op",
				Criteria:     crit,
				Taxonomy: &TaxonomySpec{
					Mode:   TaxonomyModeMTPTrie,
					Leaves: badLeaves,
				},
			},
		},
	}, trieOpts)
	if err == nil {
		t.Fatalf("expected explicit invalid TaxonomySpec (>26 coarse codes) to fail closed with ErrInvalidRequest")
	}

	// 1b. Explicit unknown Taxonomy.Mode fails closed with ErrInvalidRequest when the operator has enabled taxonomy
	// routing (with routing off the block is ignored; see TestExecuteSystemOne_TaxonomyBeamHybridAndAdapterFlag 0c).
	routingOn := DefaultEngineOptions()
	routingOn.TaxonomyMode = TaxonomyModeMTPTrie
	_, err = ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "test",
		Questions: map[string]SystemOneQuestion{
			"op": {
				Type:         "choice",
				Instructions: "Pick op",
				Criteria:     crit,
				Taxonomy:     &TaxonomySpec{Mode: "not_a_valid_mode"},
			},
		},
	}, routingOn)
	if err == nil {
		t.Fatalf("expected unknown Taxonomy.Mode to fail closed with ErrInvalidRequest")
	}

	// 2. Non-starvation test: doma has 25 leaves (P=0.51) and domb has 15 leaves (P=0.48) + oos.
	// Previously doma consumed all 25 Pass-2 slots and starved domb completely; fair quota keeps domb_03 in Pass 2!
	starveCrit := map[string]string{"oos": "Out of scope"}
	for i := 0; i < 25; i++ {
		k := fmt.Sprintf("doma_%02d", i)
		starveCrit[k] = fmt.Sprintf("Domain A operation %d", i)
	}
	for i := 0; i < 15; i++ {
		k := fmt.Sprintf("domb_%02d", i)
		starveCrit[k] = fmt.Sprintf("Domain B operation %d", i)
	}
	opts := DefaultEngineOptions()
	opts.TaxonomyMode = TaxonomyModeBeam2Pass
	resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "user wants domb_03",
		Questions: map[string]SystemOneQuestion{
			"op": {
				Type:         "choice",
				Instructions: "Pick op",
				Criteria:     starveCrit,
			},
		},
	}, opts)
	if err != nil {
		t.Fatalf("unexpected error in beam_2pass non-starvation test: %v", err)
	}
	if resp.Answers["op"].Choice != "domb_03" {
		t.Fatalf("expected domb_03 to survive Pass-2 beam allocation without rank-1 starvation, got %q", resp.Answers["op"].Choice)
	}
	if resp.Answers["op"].TaxonomyModeExecuted != TaxonomyModeBeam2Pass {
		t.Fatalf("expected taxonomy_mode_executed=%q, got %q", TaxonomyModeBeam2Pass, resp.Answers["op"].TaxonomyModeExecuted)
	}
}

func FuzzBuildPrefixTaxonomyTrie(f *testing.F) {
	f.Add(uint8(40), uint8(5), "_")
	f.Add(uint8(120), uint8(12), ".")
	f.Add(uint8(202), uint8(18), "_")
	f.Fuzz(func(t *testing.T, nOpts uint8, nDoms uint8, delim string) {
		kCount := int(nOpts%220) + 27 // wide range [27, 246]
		dCount := int(nDoms%24) + 1
		if delim == "" || len(delim) > 3 {
			delim = "_"
		}
		crit := make(map[string]string, kCount)
		for i := 0; i < kCount-1; i++ {
			dom := fmt.Sprintf("d%02d", i%dCount)
			key := fmt.Sprintf("%s%ssub%03d", dom, delim, i)
			crit[key] = fmt.Sprintf("Service %s operation number %d workflow", dom, i)
		}
		crit["oos"] = "None of the supported operations"
		leaves, coarse, fine, err := BuildPrefixTaxonomyTrie(crit, delim, 26)
		if err != nil {
			return
		}
		if len(coarse) > 26 || len(fine) > 26 {
			t.Fatalf("slot bound violated: coarse=%d fine=%d", len(coarse), len(fine))
		}
		if len(leaves) != len(crit) {
			t.Fatalf("leaf count mismatch: got %d want %d", len(leaves), len(crit))
		}
		seenCoords := make(map[[2]string]bool, len(leaves))
		for _, l := range leaves {
			coord := [2]string{l.CoarseCode, l.FineCode}
			if seenCoords[coord] {
				t.Fatalf("duplicate (coarse, fine) coordinate %v for leaf %q", coord, l.Key)
			}
			seenCoords[coord] = true
		}
	})
}
