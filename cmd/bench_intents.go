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

package cmd

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
)

var (
	intentsCorpus           string
	intentsDataset          string
	intentsFull             bool
	intentsOutput           string
	intentsOffset           int
	intentsLimit            int
	intentsWorkers          int
	intentsSamples          string
	intentsTieBreak         bool
	intentsEntropyThreshold float64
)

var benchIntentsCmd = &cobra.Command{
	Use:     "bench-intents",
	GroupID: "eval",
	Short:   "Benchmark DiffusionGemma high-cardinality intent classification, logprob calibration, and OOS detection",
	Long: `bench-intents evaluates DiffusionGemma's discrete slot readout on high-cardinality intent
datasets such as PolyAI/banking77 (77 intents, 3,080 test items) and DeepPavlov/clinc150
(150 intents across 10 domains + Out-of-Scope 'oos', 5,500 test items).

Records token-level logprobs, calibrated confidence exp(min_logprob), and Shannon entropy H.
When --tie-break is enabled (default true), high-entropy predictions (H >= --entropy-threshold)
automatically trigger a focused Pass-2 tie-breaker across the top competing finalist candidates.`,
	RunE: runBenchIntents,
}

func init() {
	benchIntentsCmd.Flags().StringVarP(&intentsCorpus, "corpus", "c", "third_party/banking77/banking77_eval.jsonl", "Path to intent evaluation JSONL file")
	benchIntentsCmd.Flags().StringVarP(&intentsDataset, "dataset", "d", "", "Preset dataset to target ('banking77' or 'clinc150')")
	benchIntentsCmd.Flags().BoolVarP(&intentsFull, "full", "F", false, "Download (if needed) and evaluate the complete official test dataset (3,080 for banking77; 5,500 for clinc150)")
	benchIntentsCmd.Flags().StringVarP(&intentsOutput, "output", "o", "", "Export JSON report to file")
	benchIntentsCmd.Flags().IntVar(&intentsOffset, "offset", 0, "Start offset index in dataset (0 = beginning)")
	benchIntentsCmd.Flags().IntVarP(&intentsLimit, "limit", "n", 0, "Limit number of cases to evaluate (0 = all)")
	benchIntentsCmd.Flags().IntVarP(&intentsWorkers, "workers", "w", 1, "Number of concurrent evaluation workers for vLLM continuous batching")
	benchIntentsCmd.Flags().StringVar(&intentsSamples, "samples", "1", "DiffusionGemma samples policy ('1', '2', '4', or 'auto')")
	benchIntentsCmd.Flags().BoolVar(&intentsTieBreak, "tie-break", true, "Enable Entropy-Gated Top-K Tie-Breaker on high-entropy predictions")
	benchIntentsCmd.Flags().Float64Var(&intentsEntropyThreshold, "entropy-threshold", 0.08, "Shannon entropy threshold (in nats) to trigger Pass-2 finalist tie-breaker")

	RootCmd.AddCommand(benchIntentsCmd)
}

type IntentCase struct {
	ID             string   `json:"id"`
	Dataset        string   `json:"dataset"`
	Text           string   `json:"text"`
	ExpectedDomain string   `json:"expected_domain,omitempty"`
	ExpectedIntent string   `json:"expected_intent"`
	IsOOS          bool     `json:"is_oos,omitempty"`
	Domains        []string `json:"domains,omitempty"`
	Options        []string `json:"options"`
}

type IntentResult struct {
	ID                string             `json:"id"`
	Dataset           string             `json:"dataset"`
	Text              string             `json:"text"`
	IsOOS             bool               `json:"is_oos,omitempty"`
	ExpectedDomain    string             `json:"expected_domain,omitempty"`
	ActualDomain      string             `json:"actual_domain,omitempty"`
	DomainAccurate    bool               `json:"domain_accurate,omitempty"`
	ExpectedIntent    string             `json:"expected_intent"`
	Pass1Intent       string             `json:"pass1_intent"`
	Pass1Accurate     bool               `json:"pass1_accurate"`
	ActualIntent      string             `json:"actual_intent"`
	IntentAccurate    bool               `json:"intent_accurate"`
	Confidence        float64            `json:"confidence"`
	Logprob           float64            `json:"logprob"`
	Entropy           float64            `json:"entropy"`
	TopProbabilities  map[string]float64 `json:"top_probabilities,omitempty"`
	TieBreakTriggered bool               `json:"tie_break_triggered"`
	FinalistOptions   []string           `json:"finalist_options,omitempty"`
	WallTimeMs        float64            `json:"wall_time_ms"`
}

// buildFinalists selects the 2..5 most plausible competing options for Pass-2 tie-breaking
// using Pass-1's top_logprobs subwords and semantic token overlap.
// maxChoiceOptions is the server's limit for one choice question (single-letter labels A-Z).
const maxChoiceOptions = 26

// intentBracketSize and intentBracketKeep: with more than 26 candidate intents, round 1 asks one question per
// group of at most intentBracketSize options and keeps each group's top intentBracketKeep, and a final round
// picks among the kept ones. `dgem systemone serve` brackets differently (top 2 per group, at most 24 finalists,
// fused probabilities over all options).
const (
	intentBracketSize = 20
	intentBracketKeep = 5
)

type intentDecideFunc func(vars map[string]interface{}) (*client.StructuredDecisionResponse, *client.RequestStats, error)

// splitBalanced splits opts into ceil(len/size) groups of near-equal size, preserving order.
func splitBalanced(opts []string, size int) [][]string {
	n := (len(opts) + size - 1) / size
	var out [][]string
	for i := 0; i < n; i++ {
		lo, hi := i*len(opts)/n, (i+1)*len(opts)/n
		out = append(out, opts[lo:hi])
	}
	return out
}

// topOptions returns up to k option names from an answer's probabilities, highest first (the chosen option first).
func topOptions(ans client.QuestionAnswer, group []string, k int) []string {
	type kv struct {
		name string
		p    float64
	}
	var ranked []kv
	for _, o := range group {
		ranked = append(ranked, kv{o, ans.Probabilities[o]})
	}
	chosen := strings.TrimSpace(ans.DisplayValue())
	for i := range ranked {
		for j := i + 1; j < len(ranked); j++ {
			if ranked[j].p > ranked[i].p || (ranked[j].name == chosen && ranked[i].name != chosen) {
				ranked[i], ranked[j] = ranked[j], ranked[i]
			}
		}
	}
	var out []string
	for _, r := range ranked {
		if len(out) == k {
			break
		}
		out = append(out, r.name)
	}
	return out
}

// decideIntentBracketed runs one decision when the options fit in one question, otherwise a 2-stage bracket (more
// rounds when the kept intents still exceed one question). The returned response is the final round's;
// stats.WallTime sums all rounds.
func decideIntentBracketed(ctx context.Context, tmplPath string, vars map[string]interface{}, options []string, decide intentDecideFunc) (*client.StructuredDecisionResponse, *client.RequestStats, error) {
	withOptions := func(opts []string) map[string]interface{} {
		v := make(map[string]interface{}, len(vars))
		for k, x := range vars {
			v[k] = x
		}
		v["options"] = opts
		return v
	}
	if len(options) <= maxChoiceOptions {
		return decide(withOptions(options))
	}
	var total time.Duration
	var finalists []string
	for _, group := range splitBalanced(options, intentBracketSize) {
		resp, stats, err := decide(withOptions(group))
		if err != nil {
			return nil, nil, fmt.Errorf("bracket round 1: %w", err)
		}
		if stats != nil {
			total += stats.WallTime
		}
		finalists = append(finalists, topOptions(intentAnswer(resp), group, intentBracketKeep)...)
	}
	// Kept intents can still exceed one question (151 CLINC150 options -> 8 groups x 5 = 40); bracket them again
	// rather than sending a final the server rejects (#119). Each round keeps at most a quarter, so it terminates.
	var resp *client.StructuredDecisionResponse
	var stats *client.RequestStats
	var err error
	if len(finalists) > maxChoiceOptions {
		resp, stats, err = decideIntentBracketed(ctx, tmplPath, vars, finalists, decide)
	} else {
		resp, stats, err = decide(withOptions(finalists))
	}
	if err != nil {
		return nil, nil, fmt.Errorf("bracket final: %w", err)
	}
	if stats != nil {
		stats.WallTime += total
	}
	return resp, stats, nil
}

// intentAnswer returns the intent answer from a response (key "intent", else "answer", else the only answer).
func intentAnswer(resp *client.StructuredDecisionResponse) client.QuestionAnswer {
	if resp == nil {
		return client.QuestionAnswer{}
	}
	if a, ok := resp.Answers["intent"]; ok {
		return a
	}
	if a, ok := resp.Answers["answer"]; ok {
		return a
	}
	for _, v := range resp.Answers {
		return v
	}
	return client.QuestionAnswer{}
}

func buildFinalists(pass1Guess string, topProbs map[string]float64, allOptions []string) []string {
	seen := make(map[string]bool)
	var finalists []string

	addCandidate := func(opt string) {
		if opt != "" && !seen[opt] && len(finalists) < 6 {
			seen[opt] = true
			finalists = append(finalists, opt)
		}
	}

	addCandidate(pass1Guess)

	// 1. Match any option starting with or containing runner-up subword tokens from top_logprobs
	for subTok, prob := range topProbs {
		subClean := strings.ToLower(strings.Trim(subTok, " _-\""))
		if len(subClean) < 3 || prob < 0.005 {
			continue
		}
		for _, opt := range allOptions {
			optLow := strings.ToLower(opt)
			if strings.HasPrefix(optLow, subClean) || strings.Contains(optLow, "_"+subClean) || strings.Contains(optLow, subClean+"_") {
				addCandidate(opt)
			}
		}
	}

	// 2. Also include semantic sibling options that share key root tokens with pass1Guess
	guessParts := strings.Split(strings.ToLower(pass1Guess), "_")
	for _, opt := range allOptions {
		if seen[opt] {
			continue
		}
		optLow := strings.ToLower(opt)
		shared := 0
		for _, gp := range guessParts {
			if len(gp) >= 3 && strings.Contains(optLow, gp) {
				shared++
			}
		}
		// Special sibling groups in Banking77 & CLINC150 where one word differs
		if shared >= 2 ||
			(strings.Contains(pass1Guess, "compromised") && strings.Contains(optLow, "not_recognised")) ||
			(strings.Contains(pass1Guess, "failed_transfer") && (strings.Contains(optLow, "beneficiary") || strings.Contains(optLow, "transfer"))) ||
			(strings.Contains(pass1Guess, "top_up") && strings.Contains(optLow, "top_up")) ||
			(strings.Contains(pass1Guess, "exchange_rate") && strings.Contains(optLow, "exchange_rate")) ||
			(strings.Contains(pass1Guess, "oil_change") && strings.Contains(optLow, "maintenance")) {
			addCandidate(opt)
		}
	}

	return finalists
}

func ensureFullDataset(dataset string) (string, error) {
	ds := strings.ToLower(strings.TrimSpace(dataset))
	if ds == "" {
		if strings.Contains(strings.ToLower(intentsCorpus), "clinc") {
			ds = "clinc150"
		} else {
			ds = "banking77"
		}
	}

	if ds == "banking77" {
		if err := os.MkdirAll("third_party/banking77", 0755); err != nil {
			return "", err
		}
		targetPath := "third_party/banking77/banking77_full_test.jsonl"
		if info, err := os.Stat(targetPath); err == nil && info.Size() > 100000 {
			return targetPath, nil
		}
		fmt.Println("==> Fetching full PolyAI/banking77 test split (3,080 utterances, 77 intents)...")
		catResp, err := http.Get("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json")
		if err != nil {
			return "", fmt.Errorf("failed to download banking77 categories: %w", err)
		}
		defer catResp.Body.Close()
		var categories []string
		if err := json.NewDecoder(catResp.Body).Decode(&categories); err != nil {
			return "", fmt.Errorf("failed to parse banking77 categories: %w", err)
		}

		csvResp, err := http.Get("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv")
		if err != nil {
			return "", fmt.Errorf("failed to download banking77 test.csv: %w", err)
		}
		defer csvResp.Body.Close()

		reader := csv.NewReader(csvResp.Body)
		rows, err := reader.ReadAll()
		if err != nil {
			return "", fmt.Errorf("failed to read banking77 CSV: %w", err)
		}

		outFile, err := os.Create(targetPath)
		if err != nil {
			return "", err
		}
		defer outFile.Close()

		count := 0
		for idx, r := range rows {
			if idx == 0 || len(r) < 2 {
				continue
			}
			count++
			ic := IntentCase{
				ID:             fmt.Sprintf("b77-%04d", count),
				Dataset:        "PolyAI/banking77 (Full 77-Intent Test Set)",
				Text:           strings.TrimSpace(r[0]),
				ExpectedIntent: strings.TrimSpace(r[1]),
				Options:        categories,
			}
			b, _ := json.Marshal(ic)
			outFile.Write(b)
			outFile.WriteString("\n")
		}
		fmt.Printf("==> Cached %d Banking77 test cases to %s\n", count, targetPath)
		return targetPath, nil
	}

	if ds == "clinc150" {
		if err := os.MkdirAll("third_party/clinc_oos", 0755); err != nil {
			return "", err
		}
		targetPath := "third_party/clinc_oos/clinc150_full_test.jsonl"
		if info, err := os.Stat(targetPath); err == nil && info.Size() > 100000 {
			return targetPath, nil
		}
		fmt.Println("==> Fetching full DeepPavlov/clinc150 test + OOS split (5,500 utterances, 151 intents)...")
		resp, err := http.Get("https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json")
		if err != nil {
			return "", fmt.Errorf("failed to download clinc150 data_full.json: %w", err)
		}
		defer resp.Body.Close()
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}

		var rawData struct {
			Test    [][]string `json:"test"`
			OOSTest [][]string `json:"oos_test"`
		}
		if err := json.Unmarshal(bodyBytes, &rawData); err != nil {
			return "", fmt.Errorf("failed to parse clinc150 json: %w", err)
		}

		seenIntents := make(map[string]bool)
		var allIntents []string
		for _, pair := range rawData.Test {
			if len(pair) >= 2 && !seenIntents[pair[1]] {
				seenIntents[pair[1]] = true
				allIntents = append(allIntents, pair[1])
			}
		}
		allIntents = append(allIntents, "oos")

		outFile, err := os.Create(targetPath)
		if err != nil {
			return "", err
		}
		defer outFile.Close()

		count := 0
		for _, pair := range rawData.Test {
			if len(pair) < 2 {
				continue
			}
			count++
			ic := IntentCase{
				ID:             fmt.Sprintf("c150-%04d", count),
				Dataset:        "DeepPavlov/clinc150 (Full 151-Intent + OOS Test Set)",
				Text:           strings.TrimSpace(pair[0]),
				ExpectedIntent: strings.TrimSpace(pair[1]),
				IsOOS:          false,
				Options:        allIntents,
			}
			b, _ := json.Marshal(ic)
			outFile.Write(b)
			outFile.WriteString("\n")
		}
		for _, pair := range rawData.OOSTest {
			if len(pair) < 2 {
				continue
			}
			count++
			ic := IntentCase{
				ID:             fmt.Sprintf("c150-%04d", count),
				Dataset:        "DeepPavlov/clinc150 (Full 151-Intent + OOS Test Set)",
				Text:           strings.TrimSpace(pair[0]),
				ExpectedIntent: "oos",
				IsOOS:          true,
				Options:        allIntents,
			}
			b, _ := json.Marshal(ic)
			outFile.Write(b)
			outFile.WriteString("\n")
		}
		fmt.Printf("==> Cached %d CLINC150 test cases (4,500 in-domain + 1,000 OOS) to %s\n", count, targetPath)
		return targetPath, nil
	}

	return "", fmt.Errorf("unsupported preset dataset %q (use 'banking77' or 'clinc150')", dataset)
}

func loadIntentCorpus(path string) ([]IntentCase, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open intent corpus %s: %w", path, err)
	}
	defer file.Close()

	var cases []IntentCase
	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var ic IntentCase
		if err := json.Unmarshal([]byte(line), &ic); err != nil {
			return nil, fmt.Errorf("invalid line %q: %w", line, err)
		}
		cases = append(cases, ic)
	}
	return cases, scanner.Err()
}

func runBenchIntents(cmd *cobra.Command, args []string) error {
	ctx := context.Background()
	c := GetClient()
	engine := template.NewEngine()

	targetCorpus := intentsCorpus
	if intentsFull {
		resolvedPath, err := ensureFullDataset(intentsDataset)
		if err != nil {
			return err
		}
		targetCorpus = resolvedPath
	} else if intentsDataset != "" {
		ds := strings.ToLower(strings.TrimSpace(intentsDataset))
		if ds == "clinc150" {
			targetCorpus = filepath.FromSlash("third_party/clinc_oos/clinc150_eval.jsonl")
		} else if ds == "banking77" {
			targetCorpus = filepath.FromSlash("third_party/banking77/banking77_eval.jsonl")
		}
	}

	cases, err := loadIntentCorpus(targetCorpus)
	if err != nil {
		return err
	}
	if intentsOffset > 0 && intentsOffset < len(cases) {
		cases = cases[intentsOffset:]
	}
	if intentsLimit > 0 && intentsLimit < len(cases) {
		cases = cases[:intentsLimit]
	}

	var samplesParam interface{} = "auto"
	if n, err := strconv.Atoi(intentsSamples); err == nil && n > 0 {
		samplesParam = n
	} else if intentsSamples != "" {
		samplesParam = intentsSamples
	}

	hasDomains := len(cases) > 0 && len(cases[0].Domains) > 0
	tmplPath := "templates/intent_banking77.json.tmpl"
	if hasDomains {
		tmplPath = "templates/intent_clinc150.json.tmpl"
	}

	workers := intentsWorkers
	if workers < 1 {
		workers = 1
	}

	fmt.Println("==========================================================================================")
	fmt.Println("  DIFFUSIONGEMMA HIGH-CARDINALITY INTENT, LOGPROB CALIBRATION & OOS BENCHMARK")
	fmt.Println("==========================================================================================")
	fmt.Printf("Corpus:         %s (%d test cases, %d candidate intents)\n", targetCorpus, len(cases), len(cases[0].Options))
	fmt.Printf("Template:       %s (Hierarchical: %v, Tie-Break: %v @ H>=%.2f nats)\n", tmplPath, hasDomains, intentsTieBreak, intentsEntropyThreshold)
	fmt.Printf("DiffusionGemma: %s (%s) [samples=%v, workers=%d, logprobs=true]\n\n", c.Model, c.BaseURL, samplesParam, workers)

	fmt.Println(strings.Repeat("-", 132))
	fmt.Printf("%-7s | %-30s | %-24s | %-24s | %-5s | %-6s | %-6s | %-4s | %-6s\n",
		"ID", "Utterance", "Expected Intent", "dgem Final Output", "Match", "Conf", "Ent(H)", "TieB", "Wall")
	fmt.Println(strings.Repeat("-", 132))

	results := make([]IntentResult, len(cases))
	var mu sync.Mutex
	var pass1Correct, intentCorrect, domainCorrect, inDomainCorrect, inDomainTotal, oosCorrect, oosTotal, tieBreaksCount int
	var sumWall, sumConf, sumEntropy float64
	var correctEntropySum, incorrectEntropySum, correctConfSum, incorrectConfSum float64
	var correctN, incorrectN int

	type jobItem struct {
		idx int
		tc  IntentCase
	}
	jobs := make(chan jobItem, len(cases))
	for idx, tc := range cases {
		jobs <- jobItem{idx: idx, tc: tc}
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				tc := j.tc
				vars := map[string]interface{}{
					"text":    tc.Text,
					"options": tc.Options,
					"domains": tc.Domains,
					"samples": samplesParam,
				}

				resp, stats, err := decideIntentBracketed(ctx, tmplPath, vars, tc.Options, func(v map[string]interface{}) (*client.StructuredDecisionResponse, *client.RequestStats, error) {
					rendered, err := engine.RenderFile(tmplPath, v)
					if err != nil {
						return nil, nil, err
					}
					schemaContent, stateContent, err := template.ParseStructuredPayload(rendered, v)
					if err != nil {
						return nil, nil, err
					}
					return c.Decide(ctx, schemaContent, stateContent)
				})
				if err != nil {
					fmt.Fprintf(os.Stderr, "Error on case %s: %v\n", tc.ID, err)
					continue
				}

				ansIntent := resp.Answers["intent"]
				if ansIntent.DisplayValue() == "" {
					if a, ok := resp.Answers["answer"]; ok && a.DisplayValue() != "" {
						ansIntent = a
					} else if len(resp.Answers) == 1 {
						for _, v := range resp.Answers {
							ansIntent = v
						}
					}
				}
				pass1Intent := strings.TrimSpace(ansIntent.DisplayValue())
				actualIntent := pass1Intent
				actualDomain := strings.TrimSpace(resp.Answers["domain"].DisplayValue())
				if actualDomain == "" && actualIntent == "oos" {
					actualDomain = "oos"
				}

				pass1Acc := strings.EqualFold(pass1Intent, tc.ExpectedIntent)
				wallMs := float64(stats.WallTime.Milliseconds())

				var tieBreakUsed bool
				var finalists []string

				// If Entropy exceeds threshold (or Confidence < 0.92), trigger focused Pass-2 Tie-Breaker
				if intentsTieBreak && (ansIntent.Entropy >= intentsEntropyThreshold || ansIntent.Confidence < 0.92) {
					finalists = buildFinalists(pass1Intent, ansIntent.Probabilities, tc.Options)
					if len(finalists) >= 2 {
						tieBreakUsed = true
						tbVars := map[string]interface{}{
							"text":        tc.Text,
							"first_guess": pass1Intent,
							"options":     finalists,
						}
						if tbRendered, err := engine.RenderFile("templates/intent_tiebreak.json.tmpl", tbVars); err == nil {
							if tbSchema, tbState, err := template.ParseStructuredPayload(tbRendered, tbVars); err == nil {
								if tbResp, tbStats, err := c.Decide(ctx, tbSchema, tbState); err == nil {
									wallMs += float64(tbStats.WallTime.Milliseconds())
									tbAns := tbResp.Answers["intent"]
									if tbAns.DisplayValue() == "" {
										for _, v := range tbResp.Answers {
											tbAns = v
										}
									}
									if strings.TrimSpace(tbAns.DisplayValue()) != "" {
										actualIntent = strings.TrimSpace(tbAns.DisplayValue())
									}
								}
							}
						}
					}
				}

				iAcc := strings.EqualFold(actualIntent, tc.ExpectedIntent)
				dAcc := !hasDomains || strings.EqualFold(actualDomain, tc.ExpectedDomain)

				ir := IntentResult{
					ID:                tc.ID,
					Dataset:           tc.Dataset,
					Text:              tc.Text,
					IsOOS:             tc.IsOOS,
					ExpectedDomain:    tc.ExpectedDomain,
					ActualDomain:      actualDomain,
					DomainAccurate:    dAcc,
					ExpectedIntent:    tc.ExpectedIntent,
					Pass1Intent:       pass1Intent,
					Pass1Accurate:     pass1Acc,
					ActualIntent:      actualIntent,
					IntentAccurate:    iAcc,
					Confidence:        ansIntent.Confidence,
					Logprob:           ansIntent.Logprob,
					Entropy:           ansIntent.Entropy,
					TopProbabilities:  ansIntent.Probabilities,
					TieBreakTriggered: tieBreakUsed,
					FinalistOptions:   finalists,
					WallTimeMs:        wallMs,
				}

				mu.Lock()
				results[j.idx] = ir
				if pass1Acc {
					pass1Correct++
					correctN++
					correctEntropySum += ansIntent.Entropy
					correctConfSum += ansIntent.Confidence
				} else {
					incorrectN++
					incorrectEntropySum += ansIntent.Entropy
					incorrectConfSum += ansIntent.Confidence
				}
				if tieBreakUsed {
					tieBreaksCount++
				}
				if iAcc {
					intentCorrect++
				}
				if dAcc {
					domainCorrect++
				}
				if tc.IsOOS {
					oosTotal++
					if iAcc {
						oosCorrect++
					}
				} else {
					inDomainTotal++
					if iAcc {
						inDomainCorrect++
					}
				}
				sumWall += wallMs
				sumConf += ansIntent.Confidence
				sumEntropy += ansIntent.Entropy

				mark := "PASS"
				if !iAcc {
					mark = "FAIL"
				}
				tbStr := " no "
				if tieBreakUsed {
					tbStr = " YES"
				}
				expLabel := tc.ExpectedIntent
				actLabel := actualIntent
				if hasDomains {
					expLabel = fmt.Sprintf("%s/%s", tc.ExpectedDomain, tc.ExpectedIntent)
					actLabel = fmt.Sprintf("%s/%s", actualDomain, actualIntent)
				}

				fmt.Printf("%-7s | %-30s | %-24s | %-24s | %-5s | %5.3f  | %5.3f  | %-4s | %4.0fms\n",
					tc.ID,
					truncateStr(tc.Text, 30),
					truncateStr(expLabel, 24),
					truncateStr(actLabel, 24),
					mark,
					ansIntent.Confidence,
					ansIntent.Entropy,
					tbStr,
					wallMs,
				)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	n := float64(len(results))
	pass1AccPct := float64(pass1Correct) / n * 100
	intentAccPct := float64(intentCorrect) / n * 100
	domainAccPct := float64(domainCorrect) / n * 100

	meanCorrectEnt := 0.0
	meanCorrectConf := 0.0
	if correctN > 0 {
		meanCorrectEnt = correctEntropySum / float64(correctN)
		meanCorrectConf = correctConfSum / float64(correctN)
	}
	meanIncorrectEnt := 0.0
	meanIncorrectConf := 0.0
	if incorrectN > 0 {
		meanIncorrectEnt = incorrectEntropySum / float64(incorrectN)
		meanIncorrectConf = incorrectConfSum / float64(incorrectN)
	}

	fmt.Println("\n" + strings.Repeat("=", 92))
	fmt.Printf("  SUMMARY: %s (%d Cases)\n", cases[0].Dataset, len(results))
	fmt.Println(strings.Repeat("=", 92))
	fmt.Printf("• Pass-1 Single-Read Accuracy:          %.1f%% (%d of %d correct)\n", pass1AccPct, pass1Correct, len(results))
	fmt.Printf("• Post-Tie-Break Final Accuracy:        %.1f%% (%d of %d correct) [Tie-Breaks Triggered: %d]\n", intentAccPct, intentCorrect, len(results), tieBreaksCount)
	if hasDomains {
		fmt.Printf("• Hierarchical Domain Accuracy:         %.1f%% (%d of %d correct)\n", domainAccPct, domainCorrect, len(results))
	}
	if inDomainTotal > 0 && oosTotal > 0 {
		fmt.Printf("• In-Domain Intent Accuracy:            %.1f%% (%d of %d correct)\n", float64(inDomainCorrect)/float64(inDomainTotal)*100, inDomainCorrect, inDomainTotal)
		fmt.Printf("• Out-of-Scope (OOS) Rejection Rate:    %.1f%% (%d of %d correct)\n", float64(oosCorrect)/float64(oosTotal)*100, oosCorrect, oosTotal)
	}
	fmt.Printf("• Calibration — Correct Predictions:    Mean Conf = %.4f | Mean Entropy H = %.4f nats (n=%d)\n", meanCorrectConf, meanCorrectEnt, correctN)
	if incorrectN > 0 {
		fmt.Printf("• Calibration — Incorrect Predictions:  Mean Conf = %.4f | Mean Entropy H = %.4f nats (n=%d)\n", meanIncorrectConf, meanIncorrectEnt, incorrectN)
	}
	fmt.Printf("• Average End-to-End Latency:           %.1f ms\n", sumWall/n)
	fmt.Println(strings.Repeat("=", 92))

	if intentsOutput != "" {
		outData := map[string]interface{}{
			"timestamp":                   time.Now().UTC().Format(time.RFC3339),
			"dataset":                     cases[0].Dataset,
			"corpus":                      targetCorpus,
			"target_url":                  c.BaseURL,
			"target_model":                c.Model,
			"samples_policy":              samplesParam,
			"workers":                     workers,
			"tie_break_enabled":           intentsTieBreak,
			"entropy_threshold":           intentsEntropyThreshold,
			"tie_breaks_triggered":        tieBreaksCount,
			"candidate_intents":           len(cases[0].Options),
			"total_cases":                 len(results),
			"pass1_accuracy_pct":          pass1AccPct,
			"pass1_correct":               pass1Correct,
			"intent_accuracy_pct":         intentAccPct,
			"intent_correct":              intentCorrect,
			"domain_accuracy_pct":         domainAccPct,
			"domain_correct":              domainCorrect,
			"in_domain_correct":           inDomainCorrect,
			"in_domain_total":             inDomainTotal,
			"oos_correct":                 oosCorrect,
			"oos_total":                   oosTotal,
			"mean_confidence":             sumConf / n,
			"mean_entropy_nats":           sumEntropy / n,
			"correct_mean_confidence":     meanCorrectConf,
			"correct_mean_entropy_nats":   meanCorrectEnt,
			"incorrect_mean_confidence":   meanIncorrectConf,
			"incorrect_mean_entropy_nats": meanIncorrectEnt,
			"avg_wall_time_ms":            sumWall / n,
			"results":                     results,
		}
		b, _ := json.MarshalIndent(outData, "", "  ")
		if err := os.WriteFile(intentsOutput, b, 0644); err != nil {
			return fmt.Errorf("failed to write output file %s: %w", intentsOutput, err)
		}
		fmt.Printf("\nStructured report written to: %s\n", intentsOutput)
	}

	return nil
}
