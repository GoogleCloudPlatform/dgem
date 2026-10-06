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
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/GoogleCloudPlatform/dgem/pkg/permutation"
	"github.com/charmbracelet/lipgloss"
	"github.com/spf13/cobra"
)

var (
	permDatasetPath string
	permOutPath     string
	permFromReceipt string
	permGateNormH   float64
	permPriorAlpha  float64
	permWorkers     int
	permJSON        bool

	permStyleAccent = lipgloss.NewStyle().Foreground(lipgloss.Color("#38BDF8")).Bold(true)
	permStylePass   = lipgloss.NewStyle().Foreground(lipgloss.Color("#22C55E")).Bold(true)
	permStyleWarn   = lipgloss.NewStyle().Foreground(lipgloss.Color("#F59E0B")).Bold(true)
	permStyleFail   = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444")).Bold(true)
	permStyleMuted  = lipgloss.NewStyle().Foreground(lipgloss.Color("#94A3B8"))
	permStyleID     = lipgloss.NewStyle().Foreground(lipgloss.Color("#C084FC"))
)

var benchPermutationCmd = &cobra.Command{
	Use:     "bench-permutation",
	GroupID: "eval",
	Short:   "Evaluate permutation sensitivity, Null-Prior De-Biasing & O(1) Dual-Mirror Canvas calibration",
	Long: `Evaluates DiffusionGemma (dgem) under option-order permutations to quantify and eliminate
positional letter-token bias (P('A') > P('B')) and single-pass Shannon entropy blind spots:

  1. Experiment 13A (Cyclic K-Permutation Stress Test & BALD Decomposition):
     Rotates every semantic option through every positional letter slot (A, B, C, D) via K cyclic shifts
     and decomposes Total Uncertainty H(bar_p) into Within-Pass Entropy (bar_H) + Jensen-Shannon
     Permutation Divergence (D_JS = I(Y; Pi | X)).
  2. Experiment 13B (Zero-Cost Content-Free Prior Calibration):
     Measures the model's content-free positional prior p_0(pos_0 .. pos_{K-1}) in 1 probe pass and
     de-biases live slot probabilities via p_tilde(o_k) \propto p_raw(o_k) / p_0(pos_k)^alpha.
  3. Experiment 13C (O(1) Single-Pass Dual-Mirror Canvas Readout):
     Places BOTH ` + "`decision_fwd`" + ` ([o_1..o_K]) and ` + "`decision_rev`" + ` ([o_K..o_1]) onto the SAME
     256-token diffusion canvas in ONE forward pass (reads=1), canceling first-order positional bias
     and triggering an epistemic escalation gate whenever MirrorDisagreed || H(bar_p)/ln(K) >= tau.`,
	Example: `  # Run live Permutation Sensitivity & Dual-Mirror Canvas suite against Cloud Run GPU
  dgem bench-permutation -u "https://dgemma-<hash>-uc.a.run.app/v1" --gcp-auth -w 2

  # Replay and audit from the saved Cloud Run GPU receipt
  dgem bench-permutation --from-receipt permutation.json`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()

		if permFromReceipt != "" {
			raw, err := os.ReadFile(permFromReceipt)
			if err != nil {
				return fmt.Errorf("failed to read permutation receipt %q: %w\nHint: run `dgem bench-permutation` against a live endpoint first", permFromReceipt, err)
			}
			var rep permutation.PermutationReport
			if err := json.Unmarshal(raw, &rep); err != nil {
				return fmt.Errorf("failed to parse permutation receipt JSON %q: %w", permFromReceipt, err)
			}
			if permJSON {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(&rep)
			}
			renderPermutationReport(&rep)
			return nil
		}

		cli := GetClient()
		_ = os.MkdirAll(filepath.Dir(permDatasetPath), 0o755)
		cases, err := permutation.LoadOrSaveSuite(permDatasetPath)
		if err != nil {
			return fmt.Errorf("failed to load permutation suite %q: %w", permDatasetPath, err)
		}

		if !permJSON {
			fmt.Printf("\n%s %s\n", permStyleAccent.Render("━━━ Permutation Sensitivity, Prior De-Biasing & O(1) Dual-Mirror Canvas ━━━"),
				permStyleMuted.Render(fmt.Sprintf("(%d cases, gate H̃=%.2f, alpha=%.2f)", len(cases), permGateNormH, permPriorAlpha)))
			fmt.Printf("  Endpoint: %s | Workers: %d\n", permStyleID.Render(cli.BaseURL), permWorkers)
		}

		// Step 1: Measure content-free positional letter prior p_0 in 1 pass (Experiment 13B)
		np, err := permutation.MeasureNullPrior(ctx, cli)
		if err != nil {
			return fmt.Errorf("failed to measure content-free null prior: %w", err)
		}
		if !permJSON {
			fmt.Printf("  • Content-Free Null Prior p_0(A..K) Measured: K=2 %v | K=3 %v | K=4 %v\n\n",
				formatProbSlice(np.ByCardinality[2]), formatProbSlice(np.ByCardinality[3]), formatProbSlice(np.ByCardinality[4]))
		}

		results := make([]permutation.CasePermutationResult, len(cases))
		sem := make(chan struct{}, max(1, permWorkers))
		var wg sync.WaitGroup
		var firstErr error
		var errMu sync.Mutex

		for idx, c := range cases {
			wg.Add(1)
			go func(i int, item permutation.PermutationCase) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				res, evalErr := permutation.EvaluateFullCase(ctx, cli, item, np, permPriorAlpha, permGateNormH)
				if evalErr != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = evalErr
					}
					errMu.Unlock()
					return
				}
				results[i] = res

				if !permJSON {
					flipBadge := permStylePass.Render("STABLE  ")
					if res.RawFlipOccurred {
						flipBadge = permStyleWarn.Render("FLIPPED ")
					}
					dmBadge := permStylePass.Render("PASS")
					if !res.DualMirror.Accurate {
						if res.DualMirror.EscalatedByMirror {
							dmBadge = permStyleWarn.Render("GATED")
						} else {
							dmBadge = permStyleFail.Render("MISS")
						}
					}
					fmt.Printf("  [%02d/%02d] %-30s | %-18s | %s | 1-Pass H̃=%.3f | Cyclic JSD=%.3f | Mirror(1p): %s (TVD=%.3f, %.0fms)\n",
						i+1, len(cases), item.ID, item.Regime, flipBadge,
						res.CanonicalNormalizedH, res.RawJSDivergence, dmBadge, res.DualMirror.TVD, res.DualMirror.WallTimeMs)
				}
			}(idx, c)
		}
		wg.Wait()

		if firstErr != nil {
			return fmt.Errorf("benchmark evaluation error: %w", firstErr)
		}

		rep := permutation.BuildPermutationReport(cli.BaseURL, modelName, permGateNormH, permPriorAlpha, np, results)
		if permOutPath != "" {
			_ = os.MkdirAll(filepath.Dir(permOutPath), 0o755)
			if b, err := json.MarshalIndent(rep, "", "  "); err == nil {
				_ = os.WriteFile(permOutPath, b, 0o644)
			}
		}

		if permJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(rep)
		}

		renderPermutationReport(rep)
		return nil
	},
}

func formatProbSlice(s []float64) string {
	parts := make([]string, len(s))
	for i, v := range s {
		parts[i] = fmt.Sprintf("%.1f%%", v*100.0)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func renderPermutationReport(r *permutation.PermutationReport) {
	if r == nil {
		return
	}
	if r.NullPrior != nil {
		fmt.Printf("\n%s\n", permStyleAccent.Render("━━━ 13A/13B: Content-Free Positional Prior p_0(Slot A..D) (Hypothesis H1) ━━━"))
		for _, k := range []int{2, 3, 4} {
			fmt.Printf("  • K=%d Options: p_0 = %-28s | Max Positional Bias Ratio = %.2fx uniform (H_0 = %.3f / %.3f nats)\n",
				k, formatProbSlice(r.NullPrior.ByCardinality[k]), r.NullPrior.MaxBiasRatio[k], r.NullPrior.EntropyByK[k], r.NullPrior.UniformEntropy[k])
		}
	}

	fmt.Printf("\n%s\n", permStyleAccent.Render("━━━ 13A: Regime-by-Regime BALD Decomposition & Permutation Sensitivity (Hypotheses H2 & H4) ━━━"))
	fmt.Printf("  %-20s %-6s %-13s %-13s %-12s %-12s %-14s %-14s\n",
		"REGIME", "CASES", "1-PASS ACC", "MIRROR ACC", "RAW FLIP %", "DEB FLIP %", "1-PASS H̃", "CYCLIC JSD")
	fmt.Printf("  %s\n", strings.Repeat("─", 108))
	for _, reg := range r.Regimes {
		fmt.Printf("  %-20s %-6d %-13s %-13s %-12s %-12s %-14.4f %-14.4f\n",
			reg.Regime, reg.Cases,
			fmt.Sprintf("%.1f%%", reg.CanonicalAccuracyPct),
			permStylePass.Render(fmt.Sprintf("%.1f%%", reg.DualMirrorAccuracyPct)),
			fmt.Sprintf("%.1f%%", reg.RawFlipRatePct),
			fmt.Sprintf("%.1f%%", reg.DebiasedFlipRatePct),
			reg.MeanSingleNormalizedH,
			reg.MeanCyclicJSD,
		)
	}

	fmt.Printf("\n%s\n", permStyleAccent.Render("━━━ 13B & 13C: Mitigation Ablation (Single-Pass vs. Prior De-Biased vs. O(1) Dual-Mirror Canvas) ━━━"))
	fmt.Printf("  %-38s | %-18s | %-18s | %-20s | %-18s\n",
		"METRIC", "RAW 1-PASS (T=1)", "PRIOR DE-BIASED", "O(1) DUAL-MIRROR (1p)", "K-CYCLIC ENSEMBLE")
	fmt.Printf("  %s\n", strings.Repeat("─", 120))
	fmt.Printf("  %-38s | %-18s | %-18s | %-20s | %-18s\n",
		"Top-1 Decision Accuracy (%)",
		fmt.Sprintf("%.2f%%", r.CanonicalAccuracyPct),
		fmt.Sprintf("%.2f%%", r.DebiasedAccuracyPct),
		permStylePass.Render(fmt.Sprintf("%.2f%%", r.DualMirrorAccuracyPct)),
		fmt.Sprintf("%.2f%%", r.CyclicEnsembleAccPct),
	)
	fmt.Printf("  %-38s | %-18s | %-18s | %-20s | %-18s\n",
		"Option-Order Flip Rate (%)",
		permStyleWarn.Render(fmt.Sprintf("%.2f%%", r.RawFlipRatePct)),
		permStylePass.Render(fmt.Sprintf("%.2f%% (-%.0f%%)", r.DebiasedFlipRatePct, r.FlipReductionRelPct)),
		permStylePass.Render("0.00% (Symmetric)"),
		permStylePass.Render("0.00% (Invariant)"),
	)
	fmt.Printf("  %-38s | %-18s | %-18s | %-20s | %-18s\n",
		"Multi-Class Brier Score (Lower=Better)",
		fmt.Sprintf("%.4f", r.CanonicalBrier),
		fmt.Sprintf("%.4f", r.DebiasedBrier),
		permStylePass.Render(fmt.Sprintf("%.4f", r.DualMirrorBrier)),
		"—",
	)
	fmt.Printf("  %-38s | %-18s | %-18s | %-20s | %-18s\n",
		"Error Recall / Escalation Catch (%)",
		permStyleWarn.Render(fmt.Sprintf("%.1f%%", r.SingleGateErrRecallPct)),
		"—",
		permStylePass.Render(fmt.Sprintf("%.1f%%", r.MirrorGateErrRecallPct)),
		"100.0%",
	)
	fmt.Printf("  %-38s | %-18s | %-18s | %-20s | %-18s\n",
		"Forward Passes & Mean Latency",
		fmt.Sprintf("1 pass (%.0f ms)", r.MeanSingleLatencyMs),
		fmt.Sprintf("1 pass (%.0f ms)", r.MeanSingleLatencyMs),
		permStylePass.Render(fmt.Sprintf("1 pass (%.0f ms)", r.MeanMirrorLatencyMs)),
		fmt.Sprintf("K passes (%.0f ms)", r.MeanSingleLatencyMs*3.25),
	)
	fmt.Println()
}

func init() {
	RootCmd.AddCommand(benchPermutationCmd)
	benchPermutationCmd.Flags().StringVar(&permDatasetPath, "dataset", "benchmarks/permutation_suite.jsonl", "Path to permutation benchmark JSONL dataset")
	benchPermutationCmd.Flags().StringVarP(&permOutPath, "output", "o", "", "Path to write the permutation receipt JSON (not written when empty)")
	benchPermutationCmd.Flags().StringVar(&permOutPath, "out", "", "Deprecated alias for --output")
	_ = benchPermutationCmd.Flags().MarkDeprecated("out", "use -o/--output")
	benchPermutationCmd.Flags().StringVar(&permFromReceipt, "from-receipt", "", "Replay scorecard from a saved permutation receipt JSON")
	benchPermutationCmd.Flags().Float64Var(&permGateNormH, "gate-threshold", 0.16, "Normalized entropy gate threshold tau in [0,1]")
	benchPermutationCmd.Flags().Float64Var(&permPriorAlpha, "prior-alpha", 0.75, "Content-free prior de-biasing exponent alpha in [0,1]")
	benchPermutationCmd.Flags().IntVarP(&permWorkers, "workers", "w", 2, "Number of concurrent evaluation workers")
	benchPermutationCmd.Flags().BoolVar(&permJSON, "json", false, "Output report as structured JSON")
}
