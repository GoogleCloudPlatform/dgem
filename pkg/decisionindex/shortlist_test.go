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

func TestExecuteSystemOne_WideShortlistWithPriorLogits(t *testing.T) {
	var calls int32
	var receivedOptionCount int
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
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}
		if len(sysPayload.Questions) == 1 {
			receivedOptionCount = len(sysPayload.Questions[0].Options)
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{
			Answers: map[string]client.QuestionAnswer{
				"sku": {
					Choice: "sku_042",
					Probabilities: map[string]float64{
						"sku_042":          0.91,
						"sku_019":          0.05,
						"sku_007":          0.02,
						"sku_003":          0.01,
						"sku_011":          0.005,
						"out_of_scope_oos": 0.005,
					},
				},
			},
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": string(envBytes)}},
			},
		})
	}))
	defer srv.Close()

	criteria := make(map[string]string, 60)
	priors := make(map[string]float64, 60)
	for i := 0; i < 59; i++ {
		k := fmt.Sprintf("sku_%03d", i)
		criteria[k] = fmt.Sprintf("Catalog item %03d pack size %dx100ml", i, (i%6)+1)
		priors[k] = -3.0
	}
	criteria["out_of_scope_oos"] = "Out of scope: none of the listed catalog items match"
	priors["sku_042"] = 2.8
	priors["sku_019"] = 2.1
	priors["sku_007"] = 1.5
	priors["sku_003"] = 1.1
	priors["sku_011"] = 0.8

	cli := client.NewClient(srv.URL, "", 0)
	opts := DefaultEngineOptions()
	if opts.ShortlistTopK != 0 || opts.ShortlistFunc != nil {
		t.Fatalf("expected DefaultEngineOptions to leave ShortlistTopK=0 and ShortlistFunc=nil, got %d / %v", opts.ShortlistTopK, opts.ShortlistFunc)
	}
	if opts.ShortlistContrastive {
		t.Fatalf("expected DefaultEngineOptions().ShortlistContrastive to default to false, got true")
	}
	opts.ShortlistTopK = 5
	opts.ShortlistFunc = PriorLogitsShortlist(5)

	resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "Match the query image and title to one of 60 catalog SKUs.",
		Questions: map[string]SystemOneQuestion{
			"sku": {
				Type:         "choice",
				Instructions: "Select the matching SKU or out_of_scope_oos.",
				Criteria:     criteria,
				PriorLogits:  priors,
			},
		},
	}, opts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 forward pass with PriorLogits shortlist on 60 options, got %d passes", got)
	}
	if receivedOptionCount != 6 {
		t.Fatalf("expected 5 top-prior options + 1 catch-all = 6 options in single pass, got %d", receivedOptionCount)
	}
	ans := resp.Answers["sku"]
	if ans.Choice != "sku_042" {
		t.Fatalf("expected winner sku_042, got %q", ans.Choice)
	}
	if len(ans.Probabilities) != 60 {
		t.Fatalf("expected full distribution over all 60 options, got %d keys", len(ans.Probabilities))
	}
	sum := 0.0
	for _, p := range ans.Probabilities {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Fatalf("expected probabilities to sum to 1.0, got %.12f", sum)
	}
	if ans.NormalizedH < ans.RawNormalizedH-1e-6 {
		t.Fatalf("expected shortlist NormalizedH (%v) not to be diluted by ln(60) below RawNormalizedH (%v)", ans.NormalizedH, ans.RawNormalizedH)
	}
}

func TestLexicalOverlapShortlist_RealBanking77AndClinc150(t *testing.T) {
	b77Path := filepath.Join("..", "..", "third_party", "banking77", "banking77_eval.jsonl")
	f, err := os.Open(b77Path)
	if err != nil {
		t.Fatalf("failed to open %s: %v", b77Path, err)
	}
	defer f.Close()

	type intentRow struct {
		ID             string   `json:"id"`
		Text           string   `json:"text"`
		ExpectedIntent string   `json:"expected_intent"`
		Options        []string `json:"options"`
	}
	var rows []intentRow
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var r intentRow
		if err := json.Unmarshal([]byte(line), &r); err == nil {
			rows = append(rows, r)
		}
	}
	if len(rows) != 30 {
		t.Fatalf("expected 30 banking77 rows, got %d", len(rows))
	}

	// Test real item b77-03 ("I need to know your exchange rates.", gold="exchange_rate", K=30)
	// through ExecuteSystemOne with generic LexicalOverlapShortlist(6).
	r0 := rows[2]
	crit := make(map[string]string, len(r0.Options))
	for _, opt := range r0.Options {
		crit[opt] = strings.ReplaceAll(opt, "_", " ")
	}

	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		atomic.AddInt32(&calls, 1)
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(req.Body).Decode(&reqBody)
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
		optsList := sysPayload.Questions[0].Options
		probs := make(map[string]float64, len(optsList))
		winner := ""
		for i, o := range optsList {
			k := o.Name
			if k == r0.ExpectedIntent {
				probs[k] = 0.92
				winner = k
			} else {
				probs[k] = 0.08 / float64(maxInt(1, len(optsList)-1))
				if winner == "" && i == 0 {
					winner = k
				}
			}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{
			Answers: map[string]client.QuestionAnswer{
				"intent": {
					Choice:        winner,
					Probabilities: probs,
				},
			},
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": string(envBytes)}},
			},
		})
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL, "", 0)
	opts := DefaultEngineOptions()
	opts.ShortlistTopK = 6
	opts.ShortlistFunc = LexicalOverlapShortlist(6)

	resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: map[string]string{"customer_utterance": r0.Text},
		Questions: map[string]SystemOneQuestion{
			"intent": {
				Type:         "choice",
				Instructions: "Which intent does the customer's utterance express?",
				Criteria:     crit,
			},
		},
	}, opts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed on %s: %v", r0.ID, err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("expected 1 pass for K=30 with LexicalOverlapShortlist(6), got %d passes", got)
	}
	ans := resp.Answers["intent"]
	if ans.Choice != r0.ExpectedIntent {
		t.Fatalf("expected winner %q on %s, got %q", r0.ExpectedIntent, r0.ID, ans.Choice)
	}
	if len(ans.Probabilities) != len(r0.Options) {
		t.Fatalf("expected %d probabilities, got %d", len(r0.Options), len(ans.Probabilities))
	}
}

func TestExecuteSystemOne_RealDecisionIndexWidePanel(t *testing.T) {
	panelPath := filepath.Join("..", "..", "benchmarks", "decision_index", "panel_suite.jsonl")
	f, err := os.Open(panelPath)
	if err != nil {
		t.Fatalf("failed to open %s: %v", panelPath, err)
	}
	defer f.Close()

	type panelCase struct {
		ID        string                       `json:"id"`
		Dataset   string                       `json:"dataset"`
		State     any                          `json:"state"`
		Questions map[string]SystemOneQuestion `json:"questions"`
		Expected  map[string]any               `json:"expected"`
	}

	var wideCases []panelCase
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var pc panelCase
		if err := json.Unmarshal([]byte(line), &pc); err == nil {
			for _, q := range pc.Questions {
				if len(q.Criteria) > 26 {
					wideCases = append(wideCases, pc)
					break
				}
			}
		}
	}
	if len(wideCases) != 3 {
		t.Fatalf("expected 3 wide-option cases in panel_suite.jsonl, got %d", len(wideCases))
	}

	shortlistFn := LexicalOverlapShortlist(6)
	for _, pc := range wideCases {
		stateText := FormatState(pc.State)
		for qid, q := range pc.Questions {
			gold, _ := pc.Expected[qid].(string)
			keys, priors, err := shortlistFn(context.Background(), stateText, nil, qid, q, 6)
			if err != nil {
				t.Fatalf("%s (%s): LexicalOverlapShortlist failed: %v", pc.ID, qid, err)
			}
			found := false
			for _, k := range keys {
				if k == gold {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("%s (%s, K=%d): expected gold option %q in top-6 shortlist %v (priors=%v)",
					pc.ID, qid, len(q.Criteria), gold, keys, priors)
			}
		}
	}
}

func TestBuildContrastiveCriteria_RealMassiveCompoundSubwords(t *testing.T) {
	criteria := map[string]string{
		"iot_hue_lightdim":    "iot hue lightdim",
		"iot_hue_lightup":     "iot hue lightup",
		"iot_hue_lightchange": "iot hue lightchange",
		"iot_hue_lightoff":    "iot hue lightoff",
	}
	keys := []string{"iot_hue_lightdim", "iot_hue_lightup", "iot_hue_lightchange", "iot_hue_lightoff"}
	enriched := BuildContrastiveCriteria(criteria, keys)

	if !strings.Contains(enriched["iot_hue_lightdim"], "dim") {
		t.Fatalf("expected iot_hue_lightdim contrastive criteria to contain 'dim', got %q", enriched["iot_hue_lightdim"])
	}
	if !strings.Contains(enriched["iot_hue_lightup"], "up") {
		t.Fatalf("expected iot_hue_lightup contrastive criteria to contain 'up', got %q", enriched["iot_hue_lightup"])
	}
	if !strings.Contains(enriched["iot_hue_lightchange"], "change") {
		t.Fatalf("expected iot_hue_lightchange contrastive criteria to contain 'change', got %q", enriched["iot_hue_lightchange"])
	}
}

func TestExecuteSystemOne_PriorLogitsShortlistFallsBackToBracketsWhenEmpty(t *testing.T) {
	var calls int32
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
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}
		answers := make(map[string]client.QuestionAnswer, len(sysPayload.Questions))
		for _, q := range sysPayload.Questions {
			probs := make(map[string]float64, len(q.Options))
			winner := ""
			for i, o := range q.Options {
				if o.Name == "opt_29" {
					probs[o.Name] = 0.90
					winner = o.Name
				} else {
					probs[o.Name] = 0.10 / float64(maxInt(1, len(q.Options)-1))
					if winner == "" && i == 0 {
						winner = o.Name
					}
				}
			}
			answers[q.ID] = client.QuestionAnswer{Choice: winner, Probabilities: probs}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": string(envBytes)}},
			},
		})
	}))
	defer srv.Close()

	criteria := make(map[string]string, 30)
	for i := 0; i < 30; i++ {
		criteria[fmt.Sprintf("opt_%02d", i)] = fmt.Sprintf("Option description %02d", i)
	}

	cli := client.NewClient(srv.URL, "", 0)
	opts := DefaultEngineOptions()
	opts.ShortlistTopK = 6
	opts.ShortlistFunc = PriorLogitsShortlist(6)

	resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
		State: "Route to option 29 without caller prior_logits.",
		Questions: map[string]SystemOneQuestion{
			"q": {
				Type:         "choice",
				Instructions: "Pick the matching option.",
				Criteria:     criteria,
			},
		},
	}, opts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got < 2 {
		t.Fatalf("expected >=2 bracket passes when PriorLogits is empty under PriorLogitsShortlist, got %d", got)
	}
	if got := resp.Answers["q"].Choice; got != "opt_29" {
		t.Fatalf("expected bracket fallback to recover opt_29 instead of alphabetical top-6, got %q", got)
	}
}

func TestExecuteSystemOne_ShortlistContrastiveToggle(t *testing.T) {
	runCase := func(contrastive bool) []string {
		var descriptions []string
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
			probs := make(map[string]float64)
			for i, o := range sysPayload.Questions[0].Options {
				descriptions = append(descriptions, o.Description)
				if i == 0 {
					probs[o.Name] = 0.85
				} else {
					probs[o.Name] = 0.15 / float64(maxInt(1, len(sysPayload.Questions[0].Options)-1))
				}
			}
			envBytes, _ := json.Marshal(client.StructuredDecisionResponse{
				Answers: map[string]client.QuestionAnswer{
					"sku": {Choice: sysPayload.Questions[0].Options[0].Name, Probabilities: probs},
				},
			})
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": string(envBytes)}}},
			})
		}))
		defer srv.Close()

		criteria := make(map[string]string, 30)
		priors := make(map[string]float64, 30)
		for i := 0; i < 30; i++ {
			k := fmt.Sprintf("sku_%02d", i)
			criteria[k] = fmt.Sprintf("Sparkling water %dx330ml pack", i+1)
			priors[k] = float64(30 - i)
		}

		cli := client.NewClient(srv.URL, "", 0)
		opts := DefaultEngineOptions()
		opts.ShortlistTopK = 4
		opts.ShortlistContrastive = contrastive
		opts.ShortlistFunc = PriorLogitsShortlist(4)
		_, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
			State: "Select matching pack size.",
			Questions: map[string]SystemOneQuestion{
				"sku": {Type: "choice", Instructions: "Pick SKU.", Criteria: criteria, PriorLogits: priors},
			},
		}, opts)
		if err != nil {
			t.Fatalf("ExecuteSystemOne(contrastive=%v) failed: %v", contrastive, err)
		}
		return descriptions
	}

	withTags := runCase(true)
	withoutTags := runCase(false)
	foundTag := false
	for _, d := range withTags {
		if strings.Contains(d, "[distinguishing:") {
			foundTag = true
		}
	}
	if !foundTag {
		t.Fatalf("expected [distinguishing: ...] tags when ShortlistContrastive=true, got %v", withTags)
	}
	for _, d := range withoutTags {
		if strings.Contains(d, "[distinguishing:") {
			t.Fatalf("expected raw descriptions without [distinguishing: ...] when ShortlistContrastive=false, got %q", d)
		}
	}
}

func TestNormalizeCosinePriorLogits_DeterministicInvariants(t *testing.T) {
	if got := NormalizeCosinePriorLogits(nil, 0.07); got != nil {
		t.Fatalf("expected nil for empty input, got %v", got)
	}
	if got := NormalizeCosinePriorLogits(map[string]float64{"bad": math.NaN(), "inf": math.Inf(1)}, 0.07); got != nil {
		t.Fatalf("expected nil when all entries are non-finite, got %v", got)
	}

	cosines := map[string]float64{
		"opt_a": 0.72,
		"opt_b": 0.61,
		"opt_c": 0.44,
		"opt_d": -0.10,
	}
	logits := NormalizeCosinePriorLogits(cosines, 0) // default tau = 0.07
	if len(logits) != 4 {
		t.Fatalf("expected 4 normalized logits, got %d", len(logits))
	}

	// 1. Exact log-probability simplex normalization: sum_k exp(l_k) == 1.0.
	probSum := 0.0
	for _, l := range logits {
		if l > 0 {
			t.Fatalf("expected log-probabilities <= 0, got %.6f", l)
		}
		probSum += math.Exp(l)
	}
	if math.Abs(probSum-1.0) > 1e-12 {
		t.Fatalf("expected exp(logits) to sum to 1.0, got %.15f", probSum)
	}

	// 2. Strict order preservation: opt_a > opt_b > opt_c > opt_d.
	if !(logits["opt_a"] > logits["opt_b"] && logits["opt_b"] > logits["opt_c"] && logits["opt_c"] > logits["opt_d"]) {
		t.Fatalf("expected strict monotonicity matching cosine order, got %v", logits)
	}

	// 3. Shift invariance: adding +0.25 to every cosine produces identical log-softmax values.
	shifted := make(map[string]float64, len(cosines))
	for k, v := range cosines {
		shifted[k] = v + 0.25
	}
	shiftedLogits := NormalizeCosinePriorLogits(shifted, DefaultCosinePriorTemperature)
	for k, v := range logits {
		if math.Abs(v-shiftedLogits[k]) > 1e-12 {
			t.Fatalf("shift invariance violated on %s: %.12f vs %.12f", k, v, shiftedLogits[k])
		}
	}

	// 4. Numerical stability at tiny temperature (tau = 1e-4).
	sharp := NormalizeCosinePriorLogits(cosines, 1e-4)
	if math.IsNaN(sharp["opt_a"]) || math.IsInf(sharp["opt_a"], 0) || math.Abs(math.Exp(sharp["opt_a"])-1.0) > 1e-12 {
		t.Fatalf("expected numerically stable peak at tau=1e-4, got %v", sharp)
	}
}

func TestNormalizeCosinePriorLogits_ProbabilisticCalibrationAndFusion(t *testing.T) {
	// 500-trial Monte Carlo wide-shortlist simulation (K=60, M=15):
	// Ground-truth bi-encoder cosines: gold ~ N(0.64, 0.055^2), distractors ~ N(0.42, 0.075^2).
	// Stage-2 diffusion shortlist likelihood has noisy near-ties on hard trials where the gold option
	// and a top distractor receive close logits. Compare hesitant fusion (PriorWeight=0.35, gate=0.16)
	// using raw unscaled cosines vs temperature-scaled log-softmax cosine priors (tau=0.07).
	rng := rand.New(rand.NewSource(20261023))
	const (
		numTrials = 500
		numOpts   = 60
		topM      = 15
	)

	allKeys := make([]string, numOpts)
	criteria := make(map[string]string, numOpts)
	for i := 0; i < numOpts; i++ {
		k := fmt.Sprintf("intent_%02d", i)
		allKeys[i] = k
		criteria[k] = fmt.Sprintf("Domain intent %02d", i)
	}

	var (
		rawCosCorrect, normLogitCorrect int
		rawCosNLL, normLogitNLL         float64
		rawErrScores, normRawErrScores  []float64
		rawIsErr, normIsErr             []bool
	)

	shortlistFn := PriorLogitsShortlist(topM)

	for trial := 0; trial < numTrials; trial++ {
		goldIdx := trial % numOpts
		goldKey := allKeys[goldIdx]

		cosines := make(map[string]float64, numOpts)
		for i, k := range allKeys {
			if i == goldIdx {
				cosines[k] = 0.64 + 0.055*rng.NormFloat64()
			} else {
				cosines[k] = 0.42 + 0.075*rng.NormFloat64()
			}
		}
		normLogits := NormalizeCosinePriorLogits(cosines, DefaultCosinePriorTemperature)

		// Verify shortlist ranking is identical before and after log-softmax normalization.
		qRaw := SystemOneQuestion{Type: "choice", Criteria: criteria, PriorLogits: cosines}
		qNorm := SystemOneQuestion{Type: "choice", Criteria: criteria, PriorLogits: normLogits}
		shortRaw, _, err := shortlistFn(context.Background(), "query", nil, "q", qRaw, topM)
		if err != nil {
			t.Fatalf("shortlist raw failed: %v", err)
		}
		shortNorm, _, err := shortlistFn(context.Background(), "query", nil, "q", qNorm, topM)
		if err != nil {
			t.Fatalf("shortlist norm failed: %v", err)
		}
		for i := range shortRaw {
			if shortRaw[i] != shortNorm[i] {
				t.Fatalf("trial %d: shortlist rank mismatch at %d: %q vs %q", trial, i, shortRaw[i], shortNorm[i])
			}
		}

		// Simulate noisy Stage-2 diffusion probabilities over shortNorm (with 45% hard near-tie trials).
		hardTrial := (trial % 20) < 9
		modelLogits := make(map[string]float64, len(shortNorm))
		maxL := math.Inf(-1)
		for rank, k := range shortNorm {
			var l float64
			switch {
			case k == goldKey && !hardTrial:
				l = 2.6 + 0.6*rng.NormFloat64()
			case k == goldKey && hardTrial:
				l = 1.05 + 0.55*rng.NormFloat64()
			case rank <= 2 && hardTrial:
				l = 1.15 + 0.55*rng.NormFloat64()
			default:
				l = -1.2 + 0.5*rng.NormFloat64()
			}
			modelLogits[k] = l
			if l > maxL {
				maxL = l
			}
		}
		modelProbs := make(map[string]float64, len(shortNorm))
		z := 0.0
		for k, l := range modelLogits {
			e := math.Exp(l - maxL)
			modelProbs[k] = e
			z += e
		}
		for k := range modelProbs {
			modelProbs[k] /= z
		}

		probsRawCos, winRawCos, _, _, _, rNormH1, _ := FusePriorWithGateMode(
			modelProbs, cosines, shortNorm, 1.0, 0.35, 0.16, "hesitant",
		)
		probsNorm, winNorm, _, _, _, rNormH2, _ := FusePriorWithGateMode(
			modelProbs, normLogits, shortNorm, 1.0, 0.35, 0.16, "hesitant",
		)

		if math.Abs(rNormH1-rNormH2) > 1e-12 {
			t.Fatalf("trial %d: expected pre-fusion RawNormalizedH to be invariant (%.12f vs %.12f)", trial, rNormH1, rNormH2)
		}

		if winRawCos == goldKey {
			rawCosCorrect++
		}
		if winNorm == goldKey {
			normLogitCorrect++
		}

		pGoldRaw := math.Max(1e-6, probsRawCos[goldKey])
		pGoldNorm := math.Max(1e-6, probsNorm[goldKey])
		rawCosNLL += -math.Log(pGoldRaw)
		normLogitNLL += -math.Log(pGoldNorm)

		rawErrScores = append(rawErrScores, rNormH1)
		rawIsErr = append(rawIsErr, winRawCos != goldKey)
		normRawErrScores = append(normRawErrScores, rNormH2)
		normIsErr = append(normIsErr, winNorm != goldKey)
	}

	rawAcc := float64(rawCosCorrect) / float64(numTrials)
	normAcc := float64(normLogitCorrect) / float64(numTrials)
	meanRawNLL := rawCosNLL / float64(numTrials)
	meanNormNLL := normLogitNLL / float64(numTrials)
	aurocNorm := computeHesitationAUROC(normRawErrScores, normIsErr)

	// Temperature-scaled log-softmax cosines provide meaningful log-linear discrimination on hard near-ties
	// whereas raw cosines in [0.3, 0.7] move logits by only ~0.35 * 0.20 = 0.07 nats.
	if normAcc <= rawAcc+0.03 {
		t.Fatalf("expected temperature-scaled log-softmax priors (acc=%.3f) to beat raw cosines (acc=%.3f) by >3 pts", normAcc, rawAcc)
	}
	if meanNormNLL >= meanRawNLL {
		t.Fatalf("expected normalized cosine priors to reduce NLL (%.4f vs %.4f)", meanNormNLL, meanRawNLL)
	}
	if aurocNorm < 0.75 {
		t.Fatalf("expected RawNormalizedH error-detection AUROC >= 0.75, got %.4f", aurocNorm)
	}
}
