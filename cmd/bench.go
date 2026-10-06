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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
)

var (
	benchDataset           string
	benchMode              string
	benchOutput            string
	benchLimit             int
	benchWorkers           int
	benchIncludeGenerative bool
)

var benchCmd = &cobra.Command{
	Use:     "bench",
	GroupID: "eval",
	Short:   "Run the discrete diffusion slot readout vs autoregression benchmark",
	Long: `bench runs an empirical benchmark comparing single-pass discrete diffusion
slot readout decisions against traditional autoregressive text generation on Apple Silicon Metal or remote hosts.`,
	RunE: runBench,
}

func init() {
	benchCmd.Flags().StringVarP(&benchDataset, "dataset", "d", "", "Path to evaluation dataset in JSONL format")
	benchCmd.Flags().StringVarP(&benchMode, "mode", "M", "slot", "Benchmark mode: 'slot' (slot readout only), 'generative', or 'both'")
	benchCmd.Flags().StringVarP(&benchOutput, "output", "o", "", "Export structured JSON benchmark metrics to file")
	benchCmd.Flags().IntVarP(&benchLimit, "limit", "n", 0, "Limit number of cases to evaluate (0 = all)")
	benchCmd.Flags().IntVarP(&benchWorkers, "workers", "w", 1, "Number of concurrent evaluation workers")
	benchCmd.Flags().BoolVar(&benchIncludeGenerative, "with-generative", false, "Deprecated: use --mode both instead")

	RootCmd.AddCommand(benchCmd)
}

// EvalCase represents a single labeled benchmark item.
type EvalCase struct {
	ID        string                 `json:"id"`
	Domain    string                 `json:"domain"`
	Template  string                 `json:"template"`
	Variables map[string]interface{} `json:"variables"`
	Expected  map[string]string      `json:"expected"`
	Tier      string                 `json:"tier"`
}

// CaseResult holds measured telemetry for one evaluation item.
type CaseResult struct {
	ID             string                 `json:"id"`
	Domain         string                 `json:"domain"`
	Tier           string                 `json:"tier"`
	SlotAccurate   bool                   `json:"slot_accurate"`
	SlotDenoiseMs  float64                `json:"slot_denoise_ms"`
	SlotPrefillMs  float64                `json:"slot_prefill_ms"`
	SlotWallTimeMs float64                `json:"slot_wall_time_ms"`
	SlotSamples    int                    `json:"slot_samples"`
	SlotExtended   bool                   `json:"slot_extended"`
	SlotRetries    int                    `json:"slot_retries,omitempty"`
	Error          string                 `json:"error,omitempty"`
	HTTPStatus     int                    `json:"http_status,omitempty"`
	SlotAnswers    map[string]interface{} `json:"slot_answers,omitempty"`

	GenWallTimeMs float64 `json:"gen_wall_time_ms,omitempty"`
	GenTokens     int     `json:"gen_tokens,omitempty"`
	GenValidJSON  bool    `json:"gen_valid_json,omitempty"`
	GenResponse   string  `json:"gen_response,omitempty"`
}

// BenchmarkReport summarizes the aggregated benchmark run.
type BenchmarkReport struct {
	Timestamp       string       `json:"timestamp"`
	TargetURL       string       `json:"target_url"`
	TargetModel     string       `json:"target_model"`
	TotalCases      int          `json:"total_cases"`
	FailedCases     int          `json:"failed_cases"`
	TotalRetries    int          `json:"total_retries"`
	Workers         int          `json:"workers"`
	TotalElapsedSec float64      `json:"total_elapsed_sec"`
	Mode            string       `json:"mode"`
	SlotAvgDenoise  float64      `json:"slot_avg_denoise_ms"`
	SlotAvgWall     float64      `json:"slot_avg_wall_ms"`
	SlotAccuracy    float64      `json:"slot_accuracy_pct"`
	SlotMultiReads  int          `json:"slot_multi_reads_triggered"`
	GenAvgWall      float64      `json:"gen_avg_wall_ms,omitempty"`
	GenAvgTokens    float64      `json:"gen_avg_tokens,omitempty"`
	GenSyntaxRate   float64      `json:"gen_syntax_rate_pct,omitempty"`
	Cases           []CaseResult `json:"cases"`
}

var fallbackCases = []EvalCase{
	{
		ID:       "sup-01",
		Domain:   "support",
		Template: "templates/support_triage.json.tmpl",
		Variables: map[string]interface{}{
			"ticket": "EMERGENCY: Production API cluster returning 500 across all US-East nodes. Customer traffic is failing.",
		},
		Expected: map[string]string{"team": "engineering", "urgent": "yes"},
		Tier:     "unambiguous",
	},
	{
		ID:       "sup-02",
		Domain:   "support",
		Template: "templates/support_triage.json.tmpl",
		Variables: map[string]interface{}{
			"ticket": "I was billed twice for my annual renewal this morning ($1200 instead of $600). Please reverse the second charge.",
		},
		Expected: map[string]string{"team": "billing", "urgent": "yes"},
		Tier:     "unambiguous",
	},
	{
		ID:       "sup-03",
		Domain:   "support",
		Template: "templates/support_triage.json.tmpl",
		Variables: map[string]interface{}{
			"ticket": "Hi there, where in the dashboard can I invite my colleague as a viewer? No rush at all, thanks!",
		},
		Expected: map[string]string{"team": "support", "urgent": "no"},
		Tier:     "unambiguous",
	},
	{
		ID:       "sup-04",
		Domain:   "support",
		Template: "templates/support_triage.json.tmpl",
		Variables: map[string]interface{}{
			"ticket": "YOUR APP IS A PIECE OF GARBAGE! Deleted my database migration and corrupted files. Cancel my subscription immediately!",
		},
		Expected: map[string]string{"team": "billing", "urgent": "yes"},
		Tier:     "ambiguous",
	},
	{
		ID:       "sup-05",
		Domain:   "support",
		Template: "templates/support_triage.json.tmpl",
		Variables: map[string]interface{}{
			"ticket": "Would be nice to have dark mode or maybe some CSS customization when exporting PDFs. Not a bug, just wondering.",
		},
		Expected: map[string]string{"team": "support", "urgent": "no"},
		Tier:     "ambiguous",
	},
}

func loadDataset(path string) ([]EvalCase, error) {
	if path == "" {
		return fallbackCases, nil
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset file: %w", err)
	}
	defer file.Close()

	var cases []EvalCase
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var ec EvalCase
		if err := json.Unmarshal([]byte(line), &ec); err != nil {
			return nil, fmt.Errorf("invalid JSONL line %q: %w", line, err)
		}
		cases = append(cases, ec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading dataset: %w", err)
	}
	return cases, nil
}

func runBench(cmd *cobra.Command, args []string) error {
	ctx := context.Background()
	c := GetClient()

	mode := strings.ToLower(benchMode)
	if benchIncludeGenerative {
		mode = "both"
	}
	if mode != "slot" && mode != "generative" && mode != "both" {
		return fmt.Errorf("invalid mode %q (must be 'slot', 'generative', or 'both')", benchMode)
	}

	cases, err := loadDataset(benchDataset)
	if err != nil {
		return err
	}
	if benchLimit > 0 && benchLimit < len(cases) {
		cases = cases[:benchLimit]
	}

	fmt.Println("================================================================================")
	fmt.Println("  DIFFUSIONGEMMA: DISCRETE DIFFUSION BENCHMARK EVALUATION")
	fmt.Println("================================================================================")
	fmt.Printf("Target Server: %s\n", c.BaseURL)
	fmt.Printf("Model:         %s\n", c.Model)
	fmt.Printf("Mode:          %s\n", mode)
	fmt.Printf("Test Cases:    %d items\n\n", len(cases))

	// Warmup
	fmt.Println("Warming up server & prefilling template KV cache...")
	warmupState := `{"ticket": "Health check prefill"}`
	warmupSchema := `{"instructions":"Output valid JSON","questions":[{"id":"q","type":"boolean","instructions":"Is this active?"}]}`
	if _, _, err := c.Decide(ctx, warmupSchema, warmupState); err != nil {
		fmt.Printf("Warmup notice: %v (continuing to benchmark)\n", err)
	} else {
		fmt.Println("Warmup complete. Server is healthy.")
	}

	results := make([]CaseResult, len(cases))
	var correctCount int

	fmt.Println(strings.Repeat("-", 90))
	fmt.Printf("%-8s | %-12s | %-12s | %-6s | %-7s | %-8s | %-8s | %s\n",
		"ID", "Domain", "Tier", "Match", "Samples", "Denoise", "Wall", "Status")
	fmt.Println(strings.Repeat("-", 90))

	workers := benchWorkers
	if workers < 1 {
		workers = 1
	}
	if workers > len(cases) {
		workers = len(cases)
	}

	suiteStart := time.Now()
	jobs := make(chan int, len(cases))
	var wg sync.WaitGroup
	var printMu sync.Mutex

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localEngine := template.NewEngine()
			for idx := range jobs {
				tc := cases[idx]
				cr := CaseResult{
					ID:     tc.ID,
					Domain: tc.Domain,
					Tier:   tc.Tier,
				}

				if mode == "slot" || mode == "both" {
					rendered, err := localEngine.RenderFile(tc.Template, tc.Variables)
					if err != nil {
						cr.Error = "render: " + err.Error()
					}
					if err == nil {
						schemaContent, stateContent, err := template.ParseStructuredPayload(rendered, tc.Variables)
						if err != nil {
							cr.Error = "parse: " + err.Error()
						}
						if err == nil {
							resp, stats, err := c.Decide(ctx, schemaContent, stateContent)
							if err != nil {
								cr.Error = err.Error()
								if len(cr.Error) > 500 {
									cr.Error = cr.Error[:500]
								}
								var he *client.HTTPError
								if errors.As(err, &he) {
									cr.HTTPStatus = he.StatusCode
									cr.SlotRetries = he.Retries
								}
							}
							if err == nil {
								cr.SlotRetries = stats.Retries
								cr.SlotDenoiseMs = stats.DenoiseMs
								cr.SlotPrefillMs = stats.PrefillMs
								cr.SlotWallTimeMs = float64(stats.WallTime.Milliseconds())
								cr.SlotSamples = stats.SamplesN
								cr.SlotExtended = stats.Extended
								cr.SlotAnswers = make(map[string]interface{})

								matchAll := true
								for expKey, expVal := range tc.Expected {
									ans, ok := resp.Answers[expKey]
									if !ok {
										matchAll = false
										break
									}
									actual := strings.ToLower(ans.DisplayValue())
									expected := strings.ToLower(expVal)
									if actual != expected {
										if (actual == "true" && expected == "yes") || (actual == "false" && expected == "no") ||
											(actual == "yes" && expected == "true") || (actual == "no" && expected == "false") {
											// match
										} else {
											matchAll = false
										}
									}
									cr.SlotAnswers[expKey] = actual
								}
								cr.SlotAccurate = matchAll
							}
						}
					}
				}

				if mode == "generative" || mode == "both" {
					varParts := make([]string, 0, len(tc.Variables))
					for k, v := range tc.Variables {
						varParts = append(varParts, fmt.Sprintf("%s: %v", k, v))
					}
					prompt := fmt.Sprintf("Classify this data. Output strictly valid JSON object matching fields for this case.\n\nData: %s",
						strings.Join(varParts, "\n"))

					genModel := c.Model
					if !strings.Contains(genModel, ":think=false") {
						genModel = genModel + ":think=false"
					}

					chatResp, genStats, err := c.Complete(ctx, client.ChatCompletionRequest{
						Model: genModel,
						Messages: []client.ChatMessage{
							{Role: "user", Content: prompt},
						},
						MaxTokens: 64,
					})
					if err == nil {
						cr.GenWallTimeMs = float64(genStats.WallTime.Milliseconds())
						cr.GenTokens = genStats.OutputTokens
						if len(chatResp.Choices) > 0 {
							cr.GenResponse = strings.TrimSpace(chatResp.Choices[0].Message.RawContent())
							var testJSON map[string]interface{}
							if json.Unmarshal([]byte(cr.GenResponse), &testJSON) == nil {
								cr.GenValidJSON = true
							}
						}
					}
				}

				results[idx] = cr
				matchStr := "PASS"
				if !cr.SlotAccurate {
					matchStr = "FAIL"
				}
				status := "OK"
				if cr.Error != "" {
					matchStr = "ERR"
					status = cr.Error
					if len(status) > 80 {
						status = status[:80]
					}
				}
				printMu.Lock()
				fmt.Printf("%-8s | %-12s | %-12s | %-6s | %7d | %6.0fms | %6.0fms | %s\n",
					cr.ID, cr.Domain, cr.Tier, matchStr, cr.SlotSamples, cr.SlotDenoiseMs, cr.SlotWallTimeMs, status)
				printMu.Unlock()
			}
		}()
	}

	for i := range cases {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	totalElapsedSec := time.Since(suiteStart).Seconds()

	// Calculate aggregates
	var sumDenoise, sumSlotWall, sumGenWall float64
	var sumGenTokens int
	var multiReads, genValidCount int

	for _, r := range results {
		if r.SlotAccurate {
			correctCount++
		}
		sumDenoise += r.SlotDenoiseMs
		sumSlotWall += r.SlotWallTimeMs
		if r.SlotExtended {
			multiReads++
		}
		if mode == "generative" || mode == "both" {
			sumGenWall += r.GenWallTimeMs
			sumGenTokens += r.GenTokens
			if r.GenValidJSON {
				genValidCount++
			}
		}
	}

	n := float64(len(results))
	report := BenchmarkReport{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		TargetURL:       c.BaseURL,
		TargetModel:     c.Model,
		TotalCases:      len(results),
		Workers:         workers,
		TotalElapsedSec: totalElapsedSec,
		Mode:            mode,
		SlotAvgDenoise:  sumDenoise / n,
		SlotAvgWall:     sumSlotWall / n,
		SlotAccuracy:    float64(correctCount) / n * 100,
		SlotMultiReads:  multiReads,
		Cases:           results,
	}
	// Latency averages exclude failed requests (which have no timing); failures are counted separately.
	var okDenoise, okWall float64
	okN := 0
	for _, r := range results {
		report.TotalRetries += r.SlotRetries
		if r.Error != "" {
			report.FailedCases++
			continue
		}
		okDenoise += r.SlotDenoiseMs
		okWall += r.SlotWallTimeMs
		okN++
	}
	if okN > 0 {
		report.SlotAvgDenoise = okDenoise / float64(okN)
		report.SlotAvgWall = okWall / float64(okN)
	}

	if mode == "generative" || mode == "both" {
		report.GenAvgWall = sumGenWall / n
		report.GenAvgTokens = float64(sumGenTokens) / n
		report.GenSyntaxRate = float64(genValidCount) / n * 100
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("  AGGREGATED BENCHMARK FINDINGS")
	fmt.Println(strings.Repeat("=", 80))
	if mode == "slot" || mode == "both" {
		fmt.Printf("• Slot Readout Accuracy:        %.1f%% (%d of %d matched expected)\n", report.SlotAccuracy, correctCount, len(results))
		fmt.Printf("• Average Model GPU Denoise:    %.1f ms (pure forward compute)\n", report.SlotAvgDenoise)
		fmt.Printf("• Average End-to-End Wall Time: %.1f ms\n", report.SlotAvgWall)
		fmt.Printf("• Adaptive Multi-Reads:         %d of %d triggered multi-sampling (ambiguity flag)\n", multiReads, len(results))
	}
	if mode == "generative" || mode == "both" {
		fmt.Printf("• Generative Average Wall Time: %.1f ms\n", report.GenAvgWall)
		fmt.Printf("• Generative Completion Tokens: %.1f tokens/req\n", report.GenAvgTokens)
		fmt.Printf("• Generative Syntax Validity:   %.1f%% valid JSON\n", report.GenSyntaxRate)
		if report.SlotAvgWall > 0 {
			speedup := report.GenAvgWall / report.SlotAvgWall
			fmt.Printf("• Slot Readout Speedup:         %.1fx faster end-to-end\n", speedup)
		}
	}
	fmt.Println(strings.Repeat("=", 80))

	// Export structured report if requested
	if benchOutput != "" {
		reportBytes, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal benchmark report: %w", err)
		}
		if err := os.WriteFile(benchOutput, reportBytes, 0644); err != nil {
			return fmt.Errorf("failed to write report to %s: %w", benchOutput, err)
		}
		fmt.Printf("\nStructured benchmark report written to: %s\n", benchOutput)
	}

	return nil
}
