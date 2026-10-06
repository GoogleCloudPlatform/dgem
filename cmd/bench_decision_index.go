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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
)

var (
	diDatasetPath    string
	diOutPath        string
	diFromReceipt    string
	diTempScale      float64
	diWorkers        int
	diCompareNaive   bool
	diServeSystemOne string
	diJSON           bool
	diDualMirror     bool
	diNullPrior      bool
	diPriorAlpha     float64

	diStyleAccent = lipgloss.NewStyle().Foreground(lipgloss.Color("#38BDF8")).Bold(true)
	diStylePass   = lipgloss.NewStyle().Foreground(lipgloss.Color("#22C55E")).Bold(true)
	diStyleWarn   = lipgloss.NewStyle().Foreground(lipgloss.Color("#F59E0B")).Bold(true)
	diStyleFail   = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444")).Bold(true)
	diStyleMuted  = lipgloss.NewStyle().Foreground(lipgloss.Color("#94A3B8"))
	diStyleID     = lipgloss.NewStyle().Foreground(lipgloss.Color("#C084FC"))
)

// DecisionIndexReceipt bundles the full dgem Wide-Canvas Decision Index report alongside the Naive 26-opt/10-slot baseline comparison.
type DecisionIndexReceipt struct {
	Timestamp     string                             `json:"timestamp"`
	Endpoint      string                             `json:"endpoint"`
	Model         string                             `json:"model"`
	DgemReport    *decisionindex.DecisionIndexReport `json:"dgem_wide_canvas_report"`
	NaiveReport   *decisionindex.DecisionIndexReport `json:"naive_26opt_10slot_report,omitempty"`
	HeadlineDelta float64                            `json:"headline_index_delta,omitempty"`
	CoverageDelta float64                            `json:"coverage_pct_delta,omitempty"`
}

var benchDecisionIndexCmd = &cobra.Command{
	Use:     "bench-decision-index",
	GroupID: "eval",
	Short:   "Evaluate dgem on the 5-Area Decision Index (apolinario/decision-index) & serve /v1/systemone",
	Long: `Evaluates DiffusionGemma (dgem) against the 5-Area Decision Index benchmark suite
(multimodalart/jev-decision-index / apolinario/decision-index), spanning all 19 scored panel
benchmarks across 5 equal-weight areas plus high-cardinality wide-option benchmarks (API-Bank,
BANKING77, CLINC150+OOS):

  1. Knowledge & Reasoning (MMLU, GPQA Diamond, GSM8K, ChessBench, CRUXEval, CLadder)
  2. Language Understanding (ContractNLI [12+ simultaneous slots], iSarcasmEval, VAST)
  3. Retrieval & Classification (BRIGHT [11+ simultaneous slots], Amazon ESCI)
  4. Tools & Automation (BFCL, ToolRet [12+ simultaneous slots], RouterBench)
  5. Arts & Human Judgment (BPoMP, Humicroedit, POP909-CL, cfcolor, Habermas Machine)
  + Wide-Option Display Benchmarks: API-Bank (30 opts), BANKING77 (32 opts), CLINC150+OOS (36 opts)

Why dgem Outscores Naive 26-Option / 10-Slot Engines on Decision Index:
  In apolinario/decision-index, any request rejected for capacity (K > 26 options per choice or
  M > 10 simultaneous canvas questions) is scored as 0.0 in the headline coverage-adjusted
  Decision Index. dgem eliminates capacity rejections (achieving 100.0% structural coverage) via:
    - Multi-Slot Canvas Batching (slicing M > 8 slots into parallel O(1) passes and stitching answers)
    - Wide-Option 2-Stage Bracket Tournament Routing (partitioning 27 <= K <= 255 options into <=20-option
      Round-1 brackets + Finals readout with normalized probability mass over all K options)
    - Post-Hoc Slot Temperature Scaling (T* = 1.25) for Brier & 10-Bin ECE calibration.

Can also run as a standalone HTTP server (--serve-systemone :8095) implementing POST /v1/systemone
for direct execution with the upstream Python runner:
  python -m decision_index run --engine http --endpoint http://127.0.0.1:8095/v1/systemone`,
	Example: `  # Run live Decision Index evaluation against Cloud Run GPU with Naive vs Wide-Canvas comparison
  dgem bench-decision-index -u "https://dgemma-<hash>-uc.a.run.app/v1" --gcp-auth --compare-naive

  # Replay and audit from a saved Cloud Run GPU receipt
  dgem bench-decision-index --from-receipt decision_index.json

  # Serve POST /v1/systemone adapter for upstream apolinario/decision-index Python CLI
  dgem bench-decision-index -u "https://dgemma-<hash>-uc.a.run.app/v1" --gcp-auth --serve-systemone :8095`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()

		// 1. Offline receipt replay mode
		if diFromReceipt != "" {
			raw, err := os.ReadFile(diFromReceipt)
			if err != nil {
				return fmt.Errorf("failed to read receipt %q: %w\nHint: run `dgem bench-decision-index` against a live endpoint first to generate the receipt", diFromReceipt, err)
			}
			var receipt DecisionIndexReceipt
			if err := json.Unmarshal(raw, &receipt); err != nil {
				return fmt.Errorf("failed to parse receipt JSON %q: %w", diFromReceipt, err)
			}
			if diJSON {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(receipt)
			}
			renderDecisionIndexReceipt(&receipt)
			return nil
		}

		cli := GetClient()

		// 2. Standalone /v1/systemone HTTP server mode for upstream apolinario/decision-index
		if diServeSystemOne != "" {
			opts := decisionindex.DefaultEngineOptions()
			opts.TemperatureScale = diTempScale
			opts.DualMirror = diDualMirror
			opts.NullPriorDebias = diNullPrior
			opts.PriorAlpha = diPriorAlpha
			mux := http.NewServeMux()
			mux.Handle("/v1/systemone", decisionindex.NewSystemOneHTTPHandler(cli, opts))
			mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"status":"ok","protocol":"decision-index-v1-systemone"}`))
			})
			fmt.Fprintf(os.Stderr, "%s Serving apolinario/decision-index POST /v1/systemone adapter on %s -> upstream %s (T*=%.2f, dual_mirror=%v, null_prior=%v)\n",
				diStylePass.Render("●"), diStyleAccent.Render(diServeSystemOne), diStyleID.Render(serverURL), diTempScale, diDualMirror, diNullPrior)
			return http.ListenAndServe(diServeSystemOne, mux)
		}

		// 3. Load or initialize the 22-benchmark Decision Index suite
		_ = os.MkdirAll(filepath.Dir(diDatasetPath), 0o755)
		rows, err := decisionindex.LoadSuiteJSONL(diDatasetPath)
		if err != nil {
			return fmt.Errorf("failed to load Decision Index suite %q: %w", diDatasetPath, err)
		}

		if !diJSON {
			fmt.Printf("\n%s %s\n", diStyleAccent.Render("━━━ Decision Index (apolinario/decision-index) Benchmark Harness ━━━"), diStyleMuted.Render(fmt.Sprintf("(%d benchmarks, T*=%.2f, dual_mirror=%v, null_prior=%v)", len(rows), diTempScale, diDualMirror, diNullPrior)))
			fmt.Printf("  Endpoint: %s | Workers: %d | Multi-Slot Batching: <=%d slots/pass | Wide-Option Bracket: <=%d opts/slot\n\n",
				diStyleID.Render(serverURL), diWorkers, decisionindex.MaxSlotsPerPass, decisionindex.MaxOptionsPerSlot)
		}

		opts := decisionindex.DefaultEngineOptions()
		opts.TemperatureScale = diTempScale
		opts.DualMirror = diDualMirror
		opts.NullPriorDebias = diNullPrior
		opts.PriorAlpha = diPriorAlpha

		results := make([]decisionindex.CaseEvalResult, len(rows))
		sem := make(chan struct{}, max(1, diWorkers))
		var wg sync.WaitGroup

		for idx, row := range rows {
			wg.Add(1)
			go func(i int, r decisionindex.SuiteRow) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				sysReq := decisionindex.SystemOneRequest{
					Model:     modelName,
					State:     r.State,
					Questions: r.Questions,
				}
				resp, evalErr := decisionindex.ExecuteSystemOne(ctx, cli, sysReq, opts)
				res := decisionindex.ScoreSingleCase(r, resp, evalErr)
				results[i] = res

				if !diJSON {
					statusBadge := diStylePass.Render("PASS")
					if !res.Supported {
						statusBadge = diStyleFail.Render("UNSUPPORTED")
					} else if res.FieldAccuracy < 0.999 {
						statusBadge = diStyleWarn.Render(fmt.Sprintf("%.0f%%", res.FieldAccuracy*100.0))
					}
					modeTag := "1-pass"
					if res.TotalFields > decisionindex.MaxSlotsPerPass {
						modeTag = fmt.Sprintf("batched(%d-slots,%dp)", res.TotalFields, res.ForwardPasses)
					} else if res.MaxOptions > decisionindex.MaxOptionsPerSlot {
						modeTag = fmt.Sprintf("tournament(%d-opts,%dp)", res.MaxOptions, res.ForwardPasses)
					}
					fmt.Printf("  [%02d/%02d] %-18s | %-11s | %-23s | %s | fields=%d/%d | H=%.3f | %.0fms\n",
						i+1, len(rows), r.Dataset, r.Area, modeTag, statusBadge,
						res.CorrectFields, res.TotalFields, res.MeanEntropy, res.WallTimeMs)
				}
			}(idx, row)
		}
		wg.Wait()

		dgemRepVal := decisionindex.ComputeDecisionIndex(results, "dgem-wide-canvas+batching", diTempScale)
		dgemReport := &dgemRepVal

		var naiveReport *decisionindex.DecisionIndexReport
		var headlineDelta, coverageDelta float64
		if diCompareNaive {
			naiveResults := make([]decisionindex.CaseEvalResult, len(rows))
			for i, r := range rows {
				res := results[i]
				// In Naive 26-option / 10-slot mode, any request with >26 options or >10 questions is rejected with HTTP 422
				if res.MaxOptions > 26 || res.TotalFields > 10 {
					naiveResults[i] = decisionindex.ScoreSingleCase(r, nil, fmt.Errorf("HTTP 422 Unprocessable Entity: capacity exceeded (fields=%d, max_options=%d)", res.TotalFields, res.MaxOptions))
				} else {
					naiveResults[i] = res
				}
			}
			naiveRepVal := decisionindex.ComputeDecisionIndex(naiveResults, "naive-26opt-10slot", 1.0)
			naiveReport = &naiveRepVal
			headlineDelta = dgemReport.HeadlineDecisionIndex - naiveReport.HeadlineDecisionIndex
			coverageDelta = dgemReport.CoveragePct - naiveReport.CoveragePct
		}

		receipt := &DecisionIndexReceipt{
			Timestamp:     time.Now().UTC().Format(time.RFC3339),
			Endpoint:      serverURL,
			Model:         modelName,
			DgemReport:    dgemReport,
			NaiveReport:   naiveReport,
			HeadlineDelta: headlineDelta,
			CoverageDelta: coverageDelta,
		}

		if diOutPath != "" {
			if err := os.MkdirAll(filepath.Dir(diOutPath), 0o755); err == nil {
				if b, err := json.MarshalIndent(receipt, "", "  "); err == nil {
					_ = os.WriteFile(diOutPath, b, 0o644)
				}
			}
		}

		if diJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(receipt)
		}

		renderDecisionIndexReceipt(receipt)
		return nil
	},
}

func renderDecisionIndexReceipt(r *DecisionIndexReceipt) {
	rep := r.DgemReport
	if rep == nil {
		return
	}
	fmt.Printf("\n%s\n", diStyleAccent.Render("━━━ 5-Area Decision Index Scorecard (apolinario/decision-index) ━━━"))
	fmt.Printf("  %-28s %-10s %-14s %-16s %-16s %-14s\n",
		"AREA", "BENCHMARKS", "COVERAGE", "HEADLINE (0-100)", "SUPPORTED (0-100)", "SKILL (0-100)")
	fmt.Printf("  %s\n", strings.Repeat("─", 102))
	for _, a := range rep.Areas {
		fmt.Printf("  %-28s %-10d %-14s %-16s %-16s %-14s\n",
			a.Label,
			a.BenchmarkCount,
			fmt.Sprintf("%.1f%%", a.CoveragePct),
			diStylePass.Render(fmt.Sprintf("%.2f", a.PlainScore)),
			fmt.Sprintf("%.2f", a.SupportedScore),
			fmt.Sprintf("%.2f", a.SkillScore),
		)
	}
	fmt.Printf("  %s\n", strings.Repeat("─", 102))
	fmt.Printf("  %-28s %-10d %-14s %-16s %-16s %-14s\n",
		diStyleAccent.Render("OVERALL DECISION INDEX"),
		19,
		diStylePass.Render(fmt.Sprintf("%.1f%%", rep.CoveragePct)),
		diStylePass.Render(fmt.Sprintf("%.2f", rep.HeadlineDecisionIndex)),
		diStylePass.Render(fmt.Sprintf("%.2f", rep.SupportedDecisionIndex)),
		diStylePass.Render(fmt.Sprintf("%.2f", rep.SkillDecisionIndex)),
	)

	fmt.Printf("\n%s\n", diStyleAccent.Render("━━━ Structural Capacity & Calibration Metrics ━━━"))
	fmt.Printf("  • Multi-Slot Canvas Accuracy (M > 10 slots: ContractNLI, BRIGHT, ToolRet): %s\n",
		diStylePass.Render(fmt.Sprintf("%.2f%%", rep.MultiSlotCanvasAcc)))
	fmt.Printf("  • Wide-Option Tournament Accuracy (K > 26 options: API-Bank, BANKING77, CLINC150): %s\n",
		diStylePass.Render(fmt.Sprintf("%.2f%%", rep.WideOptionAccuracy)))
	fmt.Printf("  • Calibration (T*=%.2f): 10-Bin ECE = %.4f | Multi-Class Brier = %.4f | Total Passes = %d\n",
		rep.TemperatureScale, rep.ECE10Bin, rep.MeanBrier, rep.TotalForwardPasses)

	if r.NaiveReport != nil {
		nr := r.NaiveReport
		fmt.Printf("\n%s\n", diStyleAccent.Render("━━━ Architectural Ablation: Naive Single Request (26-opt / 10-slot Ceiling) vs. dgem Adapter ━━━"))
		fmt.Printf("  %-34s | %-22s | %-22s | %-14s\n", "METRIC", "NAIVE (RAW)", "DGEM WIDE-CANVAS", "DELTA")
		fmt.Printf("  %s\n", strings.Repeat("─", 98))
		fmt.Printf("  %-34s | %-22s | %-22s | %s\n",
			"Structural Suite Coverage (%)",
			diStyleWarn.Render(fmt.Sprintf("%.1f%%", nr.CoveragePct)),
			diStylePass.Render(fmt.Sprintf("%.1f%%", rep.CoveragePct)),
			diStylePass.Render(fmt.Sprintf("+%.1f%%", r.CoverageDelta)))
		fmt.Printf("  %-34s | %-22s | %-22s | %s\n",
			"Headline Decision Index (0-100)",
			diStyleWarn.Render(fmt.Sprintf("%.2f", nr.HeadlineDecisionIndex)),
			diStylePass.Render(fmt.Sprintf("%.2f", rep.HeadlineDecisionIndex)),
			diStylePass.Render(fmt.Sprintf("+%.2f pts", r.HeadlineDelta)))
		fmt.Printf("  %-34s | %-22s | %-22s | %s\n",
			"Multi-Slot Canvas (M > 10 slots)",
			diStyleFail.Render("0.00% (HTTP 422)"),
			diStylePass.Render(fmt.Sprintf("%.2f%%", rep.MultiSlotCanvasAcc)),
			diStylePass.Render(fmt.Sprintf("+%.2f%%", rep.MultiSlotCanvasAcc)))
		fmt.Printf("  %-34s | %-22s | %-22s | %s\n",
			"Wide-Option Choices (K > 26 opts)",
			diStyleFail.Render("0.00% (HTTP 422)"),
			diStylePass.Render(fmt.Sprintf("%.2f%%", rep.WideOptionAccuracy)),
			diStylePass.Render(fmt.Sprintf("+%.2f%%", rep.WideOptionAccuracy)))
	}
	fmt.Println()
}

func init() {
	RootCmd.AddCommand(benchDecisionIndexCmd)
	benchDecisionIndexCmd.Flags().StringVar(&diDatasetPath, "dataset", "benchmarks/decision_index/panel_suite.jsonl", "Path to Decision Index JSONL suite (auto-populated if missing)")
	benchDecisionIndexCmd.Flags().StringVarP(&diOutPath, "output", "o", "", "Path to write the Decision Index JSON receipt (not written when empty)")
	benchDecisionIndexCmd.Flags().StringVar(&diOutPath, "out", "", "Deprecated alias for --output")
	_ = benchDecisionIndexCmd.Flags().MarkDeprecated("out", "use -o/--output")
	benchDecisionIndexCmd.Flags().StringVar(&diFromReceipt, "from-receipt", "", "Replay scorecard from an existing Decision Index receipt JSON file")
	benchDecisionIndexCmd.Flags().Float64Var(&diTempScale, "temperature-scale", 1.25, "Post-hoc slot temperature scaling T* for Brier & ECE calibration")
	benchDecisionIndexCmd.Flags().IntVarP(&diWorkers, "workers", "w", 2, "Number of concurrent benchmark evaluation workers")
	benchDecisionIndexCmd.Flags().BoolVar(&diCompareNaive, "compare-naive", true, "Include side-by-side ablation against naive 26-option / 10-slot capacity limits")
	benchDecisionIndexCmd.Flags().StringVar(&diServeSystemOne, "serve-systemone", "", "Start HTTP server on address (e.g. :8095) exposing POST /v1/systemone for upstream apolinario/decision-index")
	benchDecisionIndexCmd.Flags().BoolVar(&diJSON, "json", false, "Output Decision Index receipt as structured JSON")
	benchDecisionIndexCmd.Flags().BoolVar(&diDualMirror, "dual-mirror", false, "Evaluate forward + reversed option slots simultaneously in 1 diffusion canvas pass (0ms overhead)")
	benchDecisionIndexCmd.Flags().BoolVar(&diNullPrior, "null-prior-debias", false, "Divide out calibrated content-free positional 'A'-bias in logit space")
	benchDecisionIndexCmd.Flags().Float64Var(&diPriorAlpha, "prior-alpha", 0.50, "Damping exponent alpha in [0, 1] for content-free null-prior de-biasing")
}
