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
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"text/template"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/permutation"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"google.golang.org/genai"
)

var (
	jevDataset           string
	jevLockPath          string
	jevSync              bool
	jevCheckUpstream     bool
	jevGitRef            string
	jevFromReceipt       string
	jevCascadeFrom       string
	jevCascadeThreshold  float64
	jevNormalizeEntropy  bool
	jevVertexModel       string
	jevVertexProject     string
	jevSweepThresholds   bool
	jevTierFilter        string
	jevFamilyFilter      string
	jevTopicFilter       string
	jevLimit             int
	jevOffset            int
	jevWorkers           int
	jevSamples           string
	jevTempScale         float64
	jevAutoTemp          bool
	jevAutoTempInSample  bool
	jevFlipOptions       bool
	jevFlattenScore      bool
	jevMultiSlotEvidence bool
	jevNullPriorDebias   bool
	jevDualMirror        bool
	jevSlotID            string
	jevPriorAlpha        float64
	jevScoringMode       string
	jevEndpointKind      string
	jevSealedAcc         float64
	jevOutput            string
	jevJSON              bool
)

var benchJevCmd = &cobra.Command{
	Use:     "bench-jev",
	GroupID: "eval",
	Short:   "Run JevBench v1.3.1 zero-shot classification benchmark, upstream sync, and 4-axis parity analysis",
	Long: `bench-jev evaluates DiffusionGemma (dgem) on the JevBench v1.3.1 zero-shot decision benchmark
(https://github.com/fstandhartinger/jevbench, 231 MIT-licensed public decisions across 'easy',
'standard', and 'hard' tiers, 18 reasoning families, and 7 subject topics).

Provides a Hybrid Checked-In Snapshot + Manifest-Verified Upstream Sync architecture:
  • --check-upstream: Compares local benchmarks/jevbench/manifest.lock.json against fstandhartinger/jevbench
  • --sync: Downloads, SHA-256 verifies, zero-leakage validates, and updates benchmarks/jevbench/
  • --from-receipt: Replays a saved bench-jev receipt with JevBench v1.3.1 4-Axis Geometric Mean,
    Slot Temperature Scaling (--auto-temperature), 18-Family Breakdown, and Paraphrase Consistency
  • --cascade-from: Runs Phase 2B Entropy-Gated Escalation Cascade (Stage 1 DiffusionGemma -> Stage 2 Vertex AI Gemini
    with Pass-1 Prior Forwarding) on high-entropy items (H_norm >= threshold)
  • --flip-options: Reverses option order (A..Z -> Z..A) to audit Option-Order Invariance
  • --multi-slot-evidence: Co-allocates a companion 'evidence_focus' slot on the diffusion canvas`,
	Example: `  # 1. Sync & verify the 231 MIT-licensed public JevBench tasks
  dgem bench-jev --sync

  # 2. Check whether local benchmarks/jevbench/manifest.lock.json is in sync with upstream GitHub
  dgem bench-jev --check-upstream

  # 3. Run against your endpoint, then replay the receipt with Auto-Temperature Calibration
  dgem bench-jev -u <CLOUD_RUN_URL>/v1 -o jev.json
  dgem bench-jev --from-receipt jev.json --auto-temperature

  # 4. Run Phase 2B Entropy-Gated Escalation Cascade on JevBench with Vertex AI gemini-3.8-flash
  dgem bench-jev --cascade-from jev.json \
    --vertex-model gemini-3.8-flash --cascade-threshold 0.62 --auto-temperature -w 8 \
    -o jev_cascade.json`,
	RunE: runBenchJev,
}

func init() {
	benchJevCmd.Flags().StringVarP(&jevDataset, "dataset", "d", "benchmarks/jevbench/jevbench_public.jsonl", "Path to unified JevBench public JSONL dataset")
	benchJevCmd.Flags().StringVar(&jevLockPath, "lockfile", "benchmarks/jevbench/manifest.lock.json", "Path to local JevBench manifest lockfile")
	benchJevCmd.Flags().BoolVar(&jevSync, "sync", false, "Download, SHA-256 verify, and sync JevBench public splits & upstream reference into benchmarks/jevbench/")
	benchJevCmd.Flags().BoolVar(&jevCheckUpstream, "check-upstream", false, "Check if local benchmarks/jevbench/manifest.lock.json matches upstream fstandhartinger/jevbench")
	benchJevCmd.Flags().StringVar(&jevGitRef, "ref", "", "Git branch or tag on fstandhartinger/jevbench to sync/check against (default: the git_ref pinned in the lock file; pass a ref to move the pin)")
	benchJevCmd.Flags().StringVar(&jevFromReceipt, "from-receipt", "", "Recompute JevBench v1.3.1 4-Axis parity and family/topic breakdowns from a saved JSON receipt")
	benchJevCmd.Flags().StringVar(&jevCascadeFrom, "cascade-from", "", "Path to Stage-1 JevBench JSON receipt to run Phase 2B Entropy-Gated Escalation Cascade")
	benchJevCmd.Flags().Float64Var(&jevCascadeThreshold, "cascade-threshold", 0.62, "Normalized entropy threshold H/ln(|V|) (or raw H nats) to trigger Stage-2 escalation")
	benchJevCmd.Flags().BoolVar(&jevNormalizeEntropy, "normalize-entropy", true, "Use cardinality-normalized entropy H/ln(|V|) in [0,1] for cascade threshold gating")
	benchJevCmd.Flags().StringVar(&jevVertexModel, "vertex-model", "", "Vertex AI Gemini model for Stage-2 cascade escalation (e.g. 'gemini-3.8-flash')")
	benchJevCmd.Flags().StringVar(&jevVertexProject, "vertex-project", "", "Google Cloud Project ID for Vertex AI (defaults to $GCP_PROJECT or gcloud config)")
	benchJevCmd.Flags().BoolVar(&jevSweepThresholds, "sweep-thresholds", false, "Print Entropy Gate Threshold Sweep Table across [0.40..0.75] on --cascade-from or --from-receipt")
	benchJevCmd.Flags().StringVarP(&jevTierFilter, "tier", "T", "", "Filter by JevBench tier ('easy', 'standard', or 'hard')")
	benchJevCmd.Flags().StringVar(&jevFamilyFilter, "family", "", "Filter by JevBench family (e.g., 'long_policy', 'temporal_numeric', 'multi_hop', 'trap', 'probability')")
	benchJevCmd.Flags().StringVar(&jevTopicFilter, "topic", "", "Filter by subject topic (e.g., 'math', 'coding', 'law_policy', 'finance_commerce', 'support_ops')")
	benchJevCmd.Flags().IntVarP(&jevLimit, "limit", "n", 0, "Limit number of items to evaluate (0 = all)")
	benchJevCmd.Flags().IntVar(&jevOffset, "offset", 0, "Start offset index in dataset (0 = beginning)")
	benchJevCmd.Flags().IntVarP(&jevWorkers, "workers", "w", 1, "Number of concurrent evaluation workers")
	benchJevCmd.Flags().StringVar(&jevSamples, "samples", "1", "DiffusionGemma samples policy ('1', '2', '4', or 'auto')")
	benchJevCmd.Flags().Float64Var(&jevTempScale, "temperature-scale", 1.0, "Post-hoc slot logit temperature scaling factor T > 0")
	benchJevCmd.Flags().BoolVar(&jevAutoTemp, "auto-temperature", false, "Fit temperature T* by 5-fold cross-validation: each task is scaled by the T fitted on the other folds, so reported calibration is out of sample")
	benchJevCmd.Flags().BoolVar(&jevAutoTempInSample, "auto-temperature-in-sample", false, "With --auto-temperature: fit one T on all tasks and apply it to all (optimistic; the behaviour before 2026-10-04)")
	benchJevCmd.Flags().BoolVar(&jevFlipOptions, "flip-options", false, "Reverse option order to test Option-Order Permutation Invariance")
	benchJevCmd.Flags().BoolVar(&jevFlattenScore, "flatten-score", false, "Pre-2026-10-04 behaviour: send ordinal score items as a choice over bare level indices (the rubric text was not sent). Only for comparing with old receipts")
	benchJevCmd.Flags().BoolVar(&jevMultiSlotEvidence, "multi-slot-evidence", false, "Co-allocate a companion 'evidence_focus' slot on the diffusion canvas in the same forward pass")
	benchJevCmd.Flags().BoolVar(&jevNullPriorDebias, "null-prior-debias", false, "IDC: divide out the content-free positional ('A') prior before scoring")
	benchJevCmd.Flags().StringVar(&jevSlotID, "slot-id", "decision", "Question id used for the decision slot (slot names are visible to the model)")
	benchJevCmd.Flags().StringVar(&jevScoringMode, "scoring", "both", "JevBench scoring standard to evaluate: 'v1.3.1' (geomean), 'v1.4' (harmonic + 3 gates), or 'both' (default)")
	benchJevCmd.Flags().StringVar(&jevEndpointKind, "endpoint-kind", "gpu", "Deployment kind for official latency load adjustment: 'gpu' (self-hosted x2+0.15s, default), 'api' (production API 1x), or 'cpu'")
	benchJevCmd.Flags().Float64Var(&jevSealedAcc, "sealed-acc", 0.0, "Optional accuracy on the 308 private sealed JevBench items (0.0 = public-only estimate)")
	benchJevCmd.Flags().BoolVar(&jevDualMirror, "dual-mirror", false, "IDC: add a reversed-order mirror slot on the same canvas and record Mirror TVD")
	benchJevCmd.Flags().Float64Var(&jevPriorAlpha, "prior-alpha", 0.50, "Damping exponent alpha in [0, 1] for null-prior de-biasing")
	benchJevCmd.Flags().BoolVar(&permutation.MirrorAliasNames, "mirror-alias-names", true, "Dual-mirror: rename reversed options item_1..item_K (default) instead of keeping real option names")
	benchJevCmd.Flags().StringVar(&permutation.MirrorSlotSuffix, "mirror-slot-suffix", "__rev", "Dual-mirror: suffix for the reversed slot id (legacy runs used __mirror_rev)")
	benchJevCmd.Flags().StringVar(&permutation.MirrorMode, "mirror-mode", "reversed", "Dual-mirror second slot: reversed | copy | reversed-digits | reversed-first")
	benchJevCmd.Flags().StringVarP(&jevOutput, "output", "o", "", "Export structured JSON benchmark report to file")
	benchJevCmd.Flags().BoolVar(&jevJSON, "json", false, "Output structured JSON report directly to stdout")

	RootCmd.AddCommand(benchJevCmd)
}

// JevTask represents a unified canonical task record from fstandhartinger/jevbench.
type JevTask struct {
	ID         string                 `json:"id"`
	Tier       string                 `json:"tier"`
	Family     string                 `json:"family"`
	Topic      string                 `json:"topic,omitempty"`
	State      interface{}            `json:"state"`
	Question   JevQuestionSpec        `json:"question"`
	Labels     []string               `json:"labels"`
	Expected   interface{}            `json:"expected"`
	Split      string                 `json:"split"`
	Group      *string                `json:"group,omitempty"`
	Provenance map[string]interface{} `json:"provenance,omitempty"`
}

// JevQuestionSpec holds the JevBench rubric definition.
type JevQuestionSpec struct {
	Type         string      `json:"type"`
	Instructions string      `json:"instructions"`
	Criteria     interface{} `json:"criteria,omitempty"`
}

func extractJevCriteriaMap(raw interface{}) map[string]string {
	out := make(map[string]string)
	switch v := raw.(type) {
	case map[string]interface{}:
		for k, val := range v {
			if s, ok := val.(string); ok {
				out[k] = s
			}
		}
	case []interface{}:
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				name, _ := m["name"].(string)
				if name == "" {
					name, _ = m["label"].(string)
				}
				desc, _ := m["description"].(string)
				if name != "" && desc != "" {
					out[name] = desc
				}
			}
		}
	}
	return out
}

// JevSplitLock records cryptographic hashes for one synced split.
type JevSplitLock struct {
	Name            string `json:"name"`
	Tier            string `json:"tier"`
	Items           int    `json:"items"`
	SHA256          string `json:"sha256"`
	CanonicalSHA256 string `json:"canonical_sha256,omitempty"`
	SourceURL       string `json:"source_url"`
}

// JevManifestLock is stored in benchmarks/jevbench/manifest.lock.json.
type JevManifestLock struct {
	UpstreamRepo   string         `json:"upstream_repo"`
	GitRef         string         `json:"git_ref"`
	CommitSHA      string         `json:"commit_sha"`
	Protocol       string         `json:"protocol"`
	SyncedAtUTC    string         `json:"synced_at_utc"`
	TotalPublic    int            `json:"total_public_items"`
	UnifiedSHA256  string         `json:"unified_dataset_sha256"`
	Splits         []JevSplitLock `json:"splits"`
	TopicsRevision string         `json:"topics_revision,omitempty"`
}

// JevCaseResult records per-item evaluation telemetry on JevBench.
type JevCaseResult struct {
	ID                 string                     `json:"id"`
	Tier               string                     `json:"tier"`
	Family             string                     `json:"family"`
	Topic              string                     `json:"topic,omitempty"`
	Group              string                     `json:"group,omitempty"`
	QuestionType       string                     `json:"question_type"`
	VocabCardinality   int                        `json:"vocab_cardinality"`
	ChanceBaseline     float64                    `json:"chance_baseline"`
	Expected           string                     `json:"expected"`
	Actual             string                     `json:"actual"`
	Accurate           bool                       `json:"accurate"`
	Confidence         float64                    `json:"confidence"`
	Entropy            float64                    `json:"entropy_nats"`
	NormalizedEnt      float64                    `json:"normalized_entropy"`
	BrierScore         float64                    `json:"brier_score"`
	TVDGold            float64                    `json:"tvd_gold,omitempty"`
	HasGoldProbs       bool                       `json:"has_gold_probs,omitempty"`
	EvidenceFocus      string                     `json:"evidence_focus,omitempty"`
	WallTimeMs         float64                    `json:"wall_time_ms"`
	Escalated          bool                       `json:"escalated,omitempty"`
	PriorGuided        bool                       `json:"prior_guided,omitempty"`
	Pass1Actual        string                     `json:"pass1_actual,omitempty"`
	Pass1Accurate      bool                       `json:"pass1_accurate,omitempty"`
	Pass1Entropy       float64                    `json:"pass1_entropy,omitempty"`
	Pass1NormalizedEnt float64                    `json:"pass1_normalized_entropy,omitempty"`
	Pass1LatencyMs     float64                    `json:"pass1_latency_ms,omitempty"`
	Pass2LatencyMs     float64                    `json:"pass2_latency_ms,omitempty"`
	TopProbabilities   map[string]float64         `json:"top_probabilities,omitempty"`
	TemperatureApplied float64                    `json:"temperature_applied,omitempty"` // per-task T of a k-fold --auto-temperature report
	ScoreReadout       string                     `json:"score_readout,omitempty"`       // score items: "ordinal" (levels sent as a score question) or "flattened" (--flatten-score)
	IDC                *permutation.SlotIDCDetail `json:"idc,omitempty"`
	GoldProbs          map[string]float64         `json:"gold_probs,omitempty"`
	Error              string                     `json:"error,omitempty"`
}

// JevReport represents the full exported JevBench evaluation report.
type JevReport struct {
	Timestamp             string                    `json:"timestamp"`
	IDCConfig             string                    `json:"idc_config,omitempty"`
	IDCWarning            string                    `json:"idc_warning,omitempty"`
	Source                string                    `json:"source"`
	TargetURL             string                    `json:"target_url"`
	TargetModel           string                    `json:"target_model"`
	LockCommitSHA         string                    `json:"lock_commit_sha,omitempty"`
	TotalCases            int                       `json:"total_cases"`
	TotalCorrect          int                       `json:"total_correct"`
	OverallAccuracyPct    float64                   `json:"overall_accuracy_pct"`
	ChanceBaselinePct     float64                   `json:"chance_baseline_pct"`
	ChanceCorrectedAccPct float64                   `json:"chance_corrected_acc_pct"`
	ParaphrasePairs       int                       `json:"paraphrase_pairs"`
	ParaphraseBothCorrect int                       `json:"paraphrase_both_correct"`
	ParaphraseConsistPct  float64                   `json:"paraphrase_consistency_pct"`
	TemperatureScale      float64                   `json:"temperature_scale"`
	TemperatureFit        *TemperatureFit           `json:"temperature_fit,omitempty"`
	MultiSlotEvidence     bool                      `json:"multi_slot_evidence,omitempty"`
	FlippedOptions        bool                      `json:"flipped_options,omitempty"`
	Cascade               *CascadeSummary           `json:"cascade,omitempty"`
	JevParity             *JevParitySummary         `json:"jev_parity"`
	ByTier                []CalibrationGroupSummary `json:"by_tier"`
	ByFamily              []CalibrationGroupSummary `json:"by_family"`
	ByTopic               []CalibrationGroupSummary `json:"by_topic"`
	Cases                 []JevCaseResult           `json:"cases"`
}

func runBenchJev(cmd *cobra.Command, args []string) error {
	if jevCheckUpstream {
		return checkJevBenchUpstream(jevLockPath, jevGitRef)
	}
	if jevSync {
		return syncJevBenchUpstream(jevDataset, jevLockPath, jevGitRef)
	}
	if jevCascadeFrom != "" {
		return runJevCascade(jevCascadeFrom)
	}
	if jevFromReceipt != "" {
		return replayJevBenchReceipt(jevFromReceipt)
	}

	// Ensure local dataset snapshot exists; if missing, proactively offer --sync
	if _, err := os.Stat(jevDataset); os.IsNotExist(err) {
		return fmt.Errorf("local JevBench snapshot %q not found\n  Hint: Run 'dgem bench-jev --sync' once to fetch and verify the 231 public JevBench items", jevDataset)
	}

	tasks, err := loadJevTasks(jevDataset, jevTierFilter, jevFamilyFilter, jevTopicFilter, 0)
	if err != nil {
		return fmt.Errorf("failed to load JevBench dataset %q: %w", jevDataset, err)
	}
	if len(tasks) == 0 {
		return fmt.Errorf("no JevBench tasks matched filters (tier=%q, family=%q, topic=%q)\n  Hint: Run 'dgem bench-jev' without filters to evaluate all 231 public items", jevTierFilter, jevFamilyFilter, jevTopicFilter)
	}
	if jevOffset > 0 {
		if jevOffset >= len(tasks) {
			return fmt.Errorf("--offset %d exceeds total tasks %d", jevOffset, len(tasks))
		}
		tasks = tasks[jevOffset:]
	}
	if jevLimit > 0 && len(tasks) > jevLimit {
		tasks = tasks[:jevLimit]
	}

	var lock JevManifestLock
	if rawLock, err := os.ReadFile(jevLockPath); err == nil {
		_ = json.Unmarshal(rawLock, &lock)
	}

	c := GetClient()
	ctx := context.Background()

	if !jevJSON {
		fmt.Println()
		fmt.Println(styleAccent.Render("=========================================================================================="))
		fmt.Println(styleAccent.Render("  DiffusionGemma JevBench v1.3.1 Zero-Shot Decision Evaluation (dgem bench-jev)"))
		fmt.Println(styleAccent.Render("=========================================================================================="))
		fmt.Printf("  Target Endpoint: %s\n", styleID.Render(viper.GetString("url")))
		fmt.Printf("  Target Model:    %s\n", styleID.Render(viper.GetString("model")))
		fmt.Printf("  Dataset Lock:    %s (commit=%s, items=%d, multiSlot=%v, flip=%v)\n",
			jevDataset, truncateStr(lock.CommitSHA, 10), len(tasks), jevMultiSlotEvidence, jevFlipOptions)
		fmt.Println(styleMuted.Render("------------------------------------------------------------------------------------------"))
	}

	results := make([]JevCaseResult, len(tasks))
	workers := jevWorkers
	if workers < 1 {
		workers = 1
	}
	if workers > len(tasks) {
		workers = len(tasks)
	}

	jobs := make(chan int, len(tasks))
	var wg sync.WaitGroup
	var printMu sync.Mutex

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				t := tasks[idx]
				res := evaluateJevTaskLive(ctx, c, t, jevSamples, jevFlipOptions, jevMultiSlotEvidence)
				results[idx] = res
				if !jevJSON {
					printMu.Lock()
					badge := stylePass.Render("PASS")
					if !res.Accurate {
						badge = styleFail.Render("FAIL")
					}
					if res.Error != "" {
						badge = styleWarn.Render("ERR ")
					}
					fmt.Printf("  [%03d/%03d] %-24s %-8s %-16s %s exp=%-14s act=%-14s conf=%.3f H=%.4f (%.0fms)\n",
						idx+1, len(tasks),
						styleID.Render(truncateStr(t.ID, 24)),
						t.Tier,
						truncateStr(t.Family, 16),
						badge,
						truncateStr(res.Expected, 14),
						truncateStr(res.Actual, 14),
						res.Confidence,
						res.Entropy,
						res.WallTimeMs,
					)
					printMu.Unlock()
				}
			}
		}()
	}

	for i := range tasks {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	results, activeTemp, tfit := applyJevTemperature(results)
	report := buildJevReport(results, "live-evaluation", c.BaseURL, viper.GetString("model"), lock.CommitSHA, activeTemp, jevMultiSlotEvidence, jevFlipOptions)
	report.TemperatureFit = tfit
	stampTemperatureMethod(report.JevParity, tfit)
	report.IDCConfig = idcConfigLabel(jevNullPriorDebias, jevDualMirror, jevPriorAlpha)
	if jevSlotID != "decision" {
		report.IDCConfig += "; slot_id=" + jevSlotID
	}
	if permutation.DualMirrorLetterCollisionSeen() {
		report.IDCWarning = permutation.DualMirrorLetterWarning
	}

	if jevOutput != "" {
		data, _ := json.MarshalIndent(report, "", "  ")
		_ = os.WriteFile(jevOutput, data, 0644)
	}
	if jevJSON {
		data, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(data))
		return nil
	}
	printJevReportSummary(report, jevOutput)
	return nil
}

// syncJevBenchUpstream downloads and cryptographically verifies JevBench's public datasets, topics,
// into benchmarks/jevbench/.
func syncJevBenchUpstream(datasetOut, lockOut, gitRef string) error {
	// Without --ref, download the release pinned in the lock file and refuse data that does not match it.
	var pinned *JevManifestLock
	if raw, err := os.ReadFile(lockOut); err == nil {
		var l JevManifestLock
		if json.Unmarshal(raw, &l) == nil && l.GitRef != "" {
			pinned = &l
		}
	}
	movePin := gitRef != ""
	if !movePin {
		gitRef = "main"
		if pinned != nil {
			gitRef = pinned.GitRef
		}
	}
	baseRaw := fmt.Sprintf("https://raw.githubusercontent.com/fstandhartinger/jevbench/%s", gitRef)
	httpClient := &http.Client{Timeout: 60 * time.Second}

	fetchBytes := func(url string) ([]byte, string, error) {
		resp, err := httpClient.Get(url)
		if err != nil {
			return nil, "", err
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, "", fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
		}
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, "", err
		}
		sum := sha256.Sum256(b)
		return b, hex.EncodeToString(sum[:]), nil
	}

	if !jevJSON {
		fmt.Println()
		fmt.Println(styleAccent.Render("=========================================================================================="))
		fmt.Println(styleAccent.Render("  Syncing JevBench Upstream Datasets & Reference Receipts (fstandhartinger/jevbench)"))
		fmt.Println(styleAccent.Render("=========================================================================================="))
	}

	// 1. Fetch latest commit SHA from GitHub API (best-effort)
	commitSHA := gitRef
	if req, err := http.NewRequest("GET", fmt.Sprintf("https://api.github.com/repos/fstandhartinger/jevbench/commits/%s", gitRef), nil); err == nil {
		req.Header.Set("User-Agent", "dgem-bench-jev-sync")
		if resp, err := httpClient.Do(req); err == nil && resp.StatusCode == 200 {
			var commitObj struct {
				SHA string `json:"sha"`
			}
			_ = json.NewDecoder(resp.Body).Decode(&commitObj)
			resp.Body.Close()
			if commitObj.SHA != "" {
				commitSHA = commitObj.SHA
			}
		}
	}

	// 2. Fetch datasets/manifest.json
	manifestBytes, _, err := fetchBytes(baseRaw + "/datasets/manifest.json")
	if err != nil {
		return fmt.Errorf("failed to fetch datasets/manifest.json: %w", err)
	}
	var upstreamManifest struct {
		Protocol string `json:"protocol"`
		Splits   []struct {
			Name            string `json:"name"`
			N               int    `json:"n"`
			SHA256          string `json:"sha256"`
			CanonicalSHA256 string `json:"canonical_sha256"`
		} `json:"splits"`
	}
	if err := json.Unmarshal(manifestBytes, &upstreamManifest); err != nil {
		return fmt.Errorf("invalid datasets/manifest.json: %w", err)
	}
	manifestHashes := make(map[string]string)
	canonicalHashes := make(map[string]string)
	for _, s := range upstreamManifest.Splits {
		manifestHashes[s.Name] = s.SHA256
		canonicalHashes[s.Name] = s.CanonicalSHA256
	}

	// 3. Fetch datasets/topics.json
	topicsBytes, _, err := fetchBytes(baseRaw + "/datasets/topics.json")
	if err != nil {
		return fmt.Errorf("failed to fetch datasets/topics.json: %w", err)
	}
	var topicsData struct {
		Revision string            `json:"revision"`
		Public   map[string]string `json:"public"`
	}
	_ = json.Unmarshal(topicsBytes, &topicsData)

	// 4. Fetch public splits: easy.jsonl (48), original.jsonl (72 -> standard), hard.jsonl (111)
	splitSpecs := []struct {
		Name string
		Tier string
		Path string
	}{
		{Name: "easy", Tier: "easy", Path: "/datasets/public/easy.jsonl"},
		{Name: "original", Tier: "standard", Path: "/datasets/public/original.jsonl"},
		{Name: "hard", Tier: "hard", Path: "/datasets/public/hard.jsonl"},
	}

	var allTasks []JevTask
	var splitLocks []JevSplitLock

	for _, sp := range splitSpecs {
		url := baseRaw + sp.Path
		raw, fileSHA, err := fetchBytes(url)
		if err != nil {
			return fmt.Errorf("failed to download %s: %w", url, err)
		}
		if expectedSHA, ok := manifestHashes[sp.Name]; ok && expectedSHA != "" && expectedSHA != fileSHA {
			return fmt.Errorf("SHA-256 mismatch for %s: manifest expected %s, got %s", sp.Name, expectedSHA, fileSHA)
		}

		tasks, err := parseAndValidateRawJevSplit(raw, sp.Tier, topicsData.Public)
		if err != nil {
			return fmt.Errorf("validation failed for split %s: %w", sp.Name, err)
		}
		allTasks = append(allTasks, tasks...)
		splitLocks = append(splitLocks, JevSplitLock{
			Name:            sp.Name,
			Tier:            sp.Tier,
			Items:           len(tasks),
			SHA256:          fileSHA,
			CanonicalSHA256: canonicalHashes[sp.Name],
			SourceURL:       url,
		})
		if !jevJSON {
			fmt.Printf("  %s Verified split %-10s (%3d items, tier=%-8s, sha256=%s...)\n",
				stylePass.Render("✓"), sp.Name, len(tasks), sp.Tier, fileSHA[:12])
		}
	}

	if err := os.MkdirAll(filepath.Dir(datasetOut), 0755); err != nil {
		return err
	}

	var unifiedBuf bytes.Buffer
	for _, t := range allTasks {
		line, err := json.Marshal(t)
		if err != nil {
			return err
		}
		unifiedBuf.Write(line)
		unifiedBuf.WriteByte('\n')
	}
	unifiedBytes := unifiedBuf.Bytes()
	unifiedSum := sha256.Sum256(unifiedBytes)
	unifiedSHA := hex.EncodeToString(unifiedSum[:])

	if !movePin && pinned != nil && pinned.UnifiedSHA256 != "" {
		if unifiedSHA != pinned.UnifiedSHA256 {
			return fmt.Errorf("JevBench %s does not match %s (sha256 %s, pinned %s); pass --ref <tag> to move the pin deliberately",
				gitRef, lockOut, truncateStr(unifiedSHA, 14), truncateStr(pinned.UnifiedSHA256, 14))
		}
		if err := os.WriteFile(datasetOut, unifiedBytes, 0644); err != nil {
			return err
		}
		if !jevJSON {
			fmt.Printf("  Synced %d JevBench public items (%s) to %s, verified against %s\n\n", len(allTasks), gitRef,
				styleID.Render(datasetOut), styleID.Render(lockOut))
		}
		return nil // the lock file is left unchanged
	}

	if err := os.WriteFile(datasetOut, unifiedBytes, 0644); err != nil {
		return err
	}

	lock := JevManifestLock{
		UpstreamRepo:   "https://github.com/fstandhartinger/jevbench",
		GitRef:         gitRef,
		CommitSHA:      commitSHA,
		Protocol:       upstreamManifest.Protocol + " (v1.3.1 scoring)",
		SyncedAtUTC:    time.Now().UTC().Format(time.RFC3339),
		TotalPublic:    len(allTasks),
		UnifiedSHA256:  unifiedSHA,
		Splits:         splitLocks,
		TopicsRevision: topicsData.Revision,
	}
	lockBytes, _ := json.MarshalIndent(lock, "", "  ")
	if err := os.WriteFile(lockOut, lockBytes, 0644); err != nil {
		return err
	}

	if !jevJSON {
		fmt.Println(styleMuted.Render("------------------------------------------------------------------------------------------"))
		fmt.Printf("  Synced %d JevBench public items to %s\n", len(allTasks), styleID.Render(datasetOut))
		fmt.Printf("  Pinned lockfile to %s (commit=%s)\n\n", styleID.Render(lockOut), truncateStr(commitSHA, 12))
	}
	return nil
}

// checkJevBenchUpstream checks if local benchmarks/jevbench/manifest.lock.json matches upstream GitHub.
func checkJevBenchUpstream(lockPath, gitRef string) error {
	rawLock, err := os.ReadFile(lockPath)
	if err != nil {
		return fmt.Errorf("failed to read local lockfile %q: %w\n  Hint: Run 'dgem bench-jev --sync' first", lockPath, err)
	}
	var lock JevManifestLock
	if err := json.Unmarshal(rawLock, &lock); err != nil {
		return err
	}
	if gitRef == "" { // compare upstream main with the pin to see whether a newer release exists
		gitRef = "main"
	}

	baseRaw := fmt.Sprintf("https://raw.githubusercontent.com/fstandhartinger/jevbench/%s", gitRef)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	resp, err := httpClient.Get(baseRaw + "/datasets/manifest.json")
	if err != nil {
		return fmt.Errorf("failed to reach upstream GitHub: %w", err)
	}
	defer resp.Body.Close()
	var upstream struct {
		Protocol string `json:"protocol"`
		Splits   []struct {
			Name   string `json:"name"`
			N      int    `json:"n"`
			SHA256 string `json:"sha256"`
		} `json:"splits"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&upstream); err != nil {
		return err
	}

	upMap := make(map[string]string)
	for _, s := range upstream.Splits {
		upMap[s.Name] = s.SHA256
	}

	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================="))
	fmt.Println(styleAccent.Render("  JevBench Upstream Drift & Integrity Check (--check-upstream)"))
	fmt.Println(styleAccent.Render("=========================================================================================="))
	fmt.Printf("  Local Lockfile:  %s (Synced: %s)\n", styleID.Render(lockPath), lock.SyncedAtUTC)
	fmt.Printf("  Pinned Commit:   %s\n", styleID.Render(lock.CommitSHA))
	fmt.Printf("  Protocol:        %s\n", lock.Protocol)
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))
	fmt.Printf("  %-14s %8s %-20s %-20s %-12s\n", "SPLIT", "ITEMS", "LOCAL SHA-256", "UPSTREAM SHA-256", "STATUS")
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))

	allMatch := true
	for _, sp := range lock.Splits {
		upSHA := upMap[sp.Name]
		if upSHA == "" {
			// hard.jsonl is tracked in HARD-TIER.md rather than v1.1 manifest.json
			upSHA = sp.SHA256
		}
		status := stylePass.Render("IN SYNC")
		if upSHA != sp.SHA256 {
			status = styleWarn.Render("CHANGED")
			allMatch = false
		}
		fmt.Printf("  %-14s %8d %-20s %-20s %-12s\n",
			sp.Name, sp.Items, sp.SHA256[:16]+"...", upSHA[:16]+"...", status)
	}
	fmt.Println(styleAccent.Render("=========================================================================================="))
	if allMatch {
		fmt.Printf("  %s Local JevBench snapshot (%d items) is 100%% IN SYNC with upstream.\n\n", stylePass.Render("✓"), lock.TotalPublic)
	} else {
		fmt.Printf("  %s Upstream drift detected! Run 'dgem bench-jev --sync' to update snapshot.\n\n", styleWarn.Render("!"))
	}
	return nil
}

// parseAndValidateRawJevSplit parses a JSONL split and enforces JevBench's zero-leakage and label invariants.
func parseAndValidateRawJevSplit(raw []byte, tier string, topicMap map[string]string) ([]JevTask, error) {
	var tasks []JevTask
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 4*1024*1024)

	forbiddenStateKeys := map[string]bool{
		"expected":     true,
		"label":        true,
		"ground_truth": true,
		"answer_key":   true,
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var t JevTask
		if err := json.Unmarshal([]byte(line), &t); err != nil {
			return nil, err
		}
		t.Tier = tier
		if top, ok := topicMap[t.ID]; ok {
			t.Topic = top
		} else {
			t.Topic = inferJevTopicFromFamily(t.Family)
		}

		// Zero-leakage check on structured state maps
		if stateMap, ok := t.State.(map[string]interface{}); ok {
			for k := range stateMap {
				if forbiddenStateKeys[strings.ToLower(k)] {
					return nil, fmt.Errorf("task %s state contains forbidden leakage key %q", t.ID, k)
				}
			}
		}

		// Expected label check
		expStr := formatJevExpected(t.Expected)
		if expStr != "" {
			found := false
			for _, lbl := range t.Labels {
				if lbl == expStr {
					found = true
					break
				}
			}
			if !found {
				return nil, fmt.Errorf("task %s expected %q not in labels %v", t.ID, expStr, t.Labels)
			}
		}
		tasks = append(tasks, t)
	}
	return tasks, scanner.Err()
}

func formatJevExpected(exp interface{}) string {
	switch v := exp.(type) {
	case string:
		return v
	case float64:
		return fmt.Sprintf("%.0f", v)
	case int:
		return strconv.Itoa(v)
	case bool:
		if v {
			return "yes"
		}
		return "no"
	default:
		return ""
	}
}

func inferJevTopicFromFamily(family string) string {
	switch family {
	case "temporal_numeric", "probability":
		return "math"
	case "long_policy", "policy":
		return "law_policy"
	case "adversarial":
		return "safety_security"
	case "routing", "routing_hard", "tool_selection":
		return "support_ops"
	default:
		return "everyday_language"
	}
}

func loadJevTasks(path, tierFilter, familyFilter, topicFilter string, limit int) ([]JevTask, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w\n  Hint: run 'dgem bench-jev --sync' to download the pinned public JevBench tasks", err)
		}
		return nil, err
	}
	defer f.Close()

	var out []JevTask
	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var t JevTask
		if err := json.Unmarshal([]byte(line), &t); err != nil {
			return nil, err
		}
		if tierFilter != "" && !strings.EqualFold(t.Tier, tierFilter) {
			continue
		}
		if familyFilter != "" && !strings.EqualFold(t.Family, familyFilter) {
			continue
		}
		if topicFilter != "" && !strings.EqualFold(t.Topic, topicFilter) {
			continue
		}
		out = append(out, t)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, scanner.Err()
}

// evaluateJevTaskLive compiles a JevTask through templates/jevbench_generic.json.tmpl and executes it via c.Decide.
func evaluateJevTaskLive(ctx context.Context, c *client.Client, t JevTask, samples string, flipOptions, includeEvidence bool) JevCaseResult {
	expStr := formatJevExpected(t.Expected)
	vocab := len(t.Labels)
	if vocab <= 1 {
		vocab = 2
	}
	grp := ""
	if t.Group != nil {
		grp = *t.Group
	}

	res := JevCaseResult{
		ID:               t.ID,
		Tier:             t.Tier,
		Family:           t.Family,
		Topic:            t.Topic,
		Group:            grp,
		QuestionType:     t.Question.Type,
		VocabCardinality: vocab,
		ChanceBaseline:   1.0 / float64(vocab),
		Expected:         expStr,
	}

	if gpRaw, ok := t.Provenance["gold_probs"].(map[string]interface{}); ok && len(gpRaw) > 0 {
		res.GoldProbs = make(map[string]float64, len(gpRaw))
		for k, v := range gpRaw {
			if fv, ok := v.(float64); ok {
				res.GoldProbs[k] = fv
			}
		}
		res.HasGoldProbs = true
	}

	labels := make([]string, len(t.Labels))
	copy(labels, t.Labels)
	if flipOptions {
		for i, j := 0, len(labels)-1; i < j; i, j = i+1, j-1 {
			labels[i], labels[j] = labels[j], labels[i]
		}
	}

	qType, opts, levels := buildJevQuestion(t, labels, jevFlattenScore)
	qPrompt := t.Question.Instructions
	if t.Question.Type == "score" {
		res.ScoreReadout = "ordinal"
		if levels == nil {
			res.ScoreReadout = "flattened"
		}
	}

	samplesExpr := fmt.Sprintf("%q", samples)
	if n, err := strconv.Atoi(samples); err == nil {
		samplesExpr = strconv.Itoa(n)
	}

	tmplData := map[string]interface{}{
		"Instructions":        t.Question.Instructions,
		"Samples":             samplesExpr,
		"ThinkTokens":         0,
		"IncludeEvidenceSlot": includeEvidence,
		"SlotID":              jevSlotID,
		"QuestionType":        qType,
		"QuestionPrompt":      qPrompt,
		"Options":             opts,
		"Levels":              levels,
	}

	tmplBytes, err := os.ReadFile("templates/jevbench_generic.json.tmpl")
	if err != nil {
		res.Error = err.Error()
		return res
	}
	parsedTmpl, err := template.New("jevbench").Parse(string(tmplBytes))
	if err != nil {
		res.Error = err.Error()
		return res
	}
	var renderedSchema bytes.Buffer
	if err := parsedTmpl.Execute(&renderedSchema, tmplData); err != nil {
		res.Error = err.Error()
		return res
	}

	var stateStr string
	switch sv := t.State.(type) {
	case string:
		b, _ := json.Marshal(map[string]string{"state": sv})
		stateStr = string(b)
	default:
		b, _ := json.Marshal(sv)
		stateStr = string(b)
	}

	schemaStr := renderedSchema.String()
	var slotOpts map[string][]permutation.OptionItem
	if jevDualMirror {
		schemaStr, slotOpts, _ = permutation.InjectDualMirrorSchema(schemaStr)
	} else if jevNullPriorDebias {
		slotOpts = permutation.ExtractSchemaSlotOptions(schemaStr)
	}

	resp, stats, err := c.Decide(ctx, schemaStr, stateStr)
	if err != nil {
		res.Error = err.Error()
		return res
	}
	res.WallTimeMs = float64(stats.WallTime.Milliseconds())

	if jevDualMirror || jevNullPriorDebias {
		details := permutation.PostProcessDecisionResponseDetailed(resp, slotOpts, jevDualMirror, jevNullPriorDebias, jevPriorAlpha)
		if d, ok := details[jevSlotID]; ok {
			dd := d
			res.IDC = &dd
		}
	}

	if evAns, ok := resp.Answers["evidence_focus"]; ok {
		res.EvidenceFocus = firstNonEmpty(evAns.Choice, evAns.Label)
	}
	qa, ok := resp.Answers[jevSlotID]
	if !ok {
		res.Error = "missing decision slot in response"
		return res
	}

	res.Actual = firstNonEmpty(qa.Choice, qa.Label, qa.Level)
	res.Confidence = qa.Confidence
	res.TopProbabilities = qa.Probabilities
	if levels != nil {
		// ordinal score: the server answers with level texts; map back to the task's index labels
		res.Actual, res.TopProbabilities, res.Confidence = mapJevScoreAnswer(qa, levels, labels)
	}
	res.Accurate = strings.EqualFold(res.Actual, res.Expected)
	res.Entropy = computeQuestionEntropy(qa, resp)
	if vocab > 1 {
		res.NormalizedEnt = res.Entropy / math.Log(float64(vocab))
	}
	return res
}

// jevOption is one option of a rendered JevBench question (templates/jevbench_generic.json.tmpl).
type jevOption struct {
	Name        string
	Description string
}

// buildJevQuestion renders a JevBench task's question for the dgem schema. labels is t.Labels, possibly reversed by
// --flip-options. Score items become a native "score" question whose levels are the rubric texts in label order
// (labels are level indices into criteria), unless flatten is set: then they are sent the old way, as a choice over
// bare index names whose descriptions are the indices themselves (criteria is a list of strings, which
// extractJevCriteriaMap does not read), so the rubric never reaches the model. levels is nil unless ordinal.
func buildJevQuestion(t JevTask, labels []string, flatten bool) (qType string, opts []jevOption, levels []string) {
	critMap := extractJevCriteriaMap(t.Question.Criteria)
	qType = "choice"
	switch t.Question.Type {
	case "noul":
		for _, l := range labels {
			desc := l
			if l == "yes" {
				if d, ok := critMap["true"]; ok && d != "" {
					desc = d
				} else if d, ok := critMap["yes"]; ok && d != "" {
					desc = d
				}
			} else if l == "no" {
				if d, ok := critMap["false"]; ok && d != "" {
					desc = d
				} else if d, ok := critMap["no"]; ok && d != "" {
					desc = d
				}
			}
			opts = append(opts, jevOption{Name: l, Description: desc})
		}
	case "score":
		if crit, ok := t.Question.Criteria.([]interface{}); ok && !flatten {
			texts := make([]string, 0, len(crit))
			for _, c := range crit {
				if s, ok := c.(string); ok {
					texts = append(texts, s)
				}
			}
			if len(texts) == len(crit) && len(texts) >= 2 {
				for _, l := range labels {
					i, err := strconv.Atoi(l)
					if err != nil || i < 0 || i >= len(texts) {
						levels = nil
						break
					}
					levels = append(levels, texts[i])
				}
				if levels != nil {
					return "score", nil, levels
				}
			}
		}
		fallthrough
	case "choice":
		for _, l := range labels {
			desc := l
			if d, ok := critMap[l]; ok && d != "" {
				desc = d
			}
			opts = append(opts, jevOption{Name: l, Description: desc})
		}
	}
	return qType, opts, levels
}

// mapJevScoreAnswer maps an ordinal score answer (level texts, or the server's digit labels "1".."n") back to the
// task's index labels: levels[i] is the rubric text shown for labels[i]. Returns the argmax label, probabilities keyed
// by label, and the top probability.
func mapJevScoreAnswer(qa client.QuestionAnswer, levels, labels []string) (string, map[string]float64, float64) {
	probs := make(map[string]float64, len(labels))
	for i, lvl := range levels {
		if p, ok := qa.Probabilities[lvl]; ok {
			probs[labels[i]] = p
		} else if p, ok := qa.Probabilities[strconv.Itoa(i+1)]; ok {
			probs[labels[i]] = p
		}
	}
	best, bestP := "", -1.0
	for i := range levels {
		if p, ok := probs[labels[i]]; ok && p > bestP {
			best, bestP = labels[i], p
		}
	}
	if best == "" { // no probabilities: fall back to the reported level
		for i, lvl := range levels {
			if lvl == qa.Level || strconv.Itoa(i+1) == qa.Label {
				return labels[i], nil, qa.Confidence
			}
		}
		return firstNonEmpty(qa.Level, qa.Label), nil, qa.Confidence
	}
	return best, probs, bestP
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return ""
}

func replayJevBenchReceipt(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read JevBench receipt %q: %w\n  Hint: pass a receipt written by 'dgem bench-jev -o <file>'", path, err)
	}
	var prev JevReport
	if err := json.Unmarshal(raw, &prev); err != nil {
		return fmt.Errorf("failed to parse JevBench receipt %q: %w", path, err)
	}

	// Filter cases if --tier, --family, or --topic was passed
	var filtered []JevCaseResult
	for _, c := range prev.Cases {
		if jevTierFilter != "" && !strings.EqualFold(c.Tier, jevTierFilter) {
			continue
		}
		if jevFamilyFilter != "" && !strings.EqualFold(c.Family, jevFamilyFilter) {
			continue
		}
		if jevTopicFilter != "" && !strings.EqualFold(c.Topic, jevTopicFilter) {
			continue
		}
		filtered = append(filtered, c)
		if jevLimit > 0 && len(filtered) >= jevLimit {
			break
		}
	}

	// Back to T = 1 first (per task when the receipt recorded a k-fold T, else the report-wide T), so replay is idempotent
	for i := range filtered {
		cc := unscaleTemperature(jevToCalCase(filtered[i]), filtered[i].TemperatureApplied, prev.TemperatureScale)
		filtered[i].TopProbabilities, filtered[i].Confidence, filtered[i].Entropy, filtered[i].NormalizedEnt = cc.TopProbabilities, cc.Confidence, cc.Entropy, cc.NormalizedEntropy
		filtered[i].TemperatureApplied = 0
	}

	filtered, activeTemp, tfit := applyJevTemperature(filtered)

	report := buildJevReportWithCascade(filtered, prev.Source, prev.TargetURL, prev.TargetModel, prev.LockCommitSHA, activeTemp, prev.MultiSlotEvidence, prev.FlippedOptions, prev.Cascade)
	report.TemperatureFit = tfit
	stampTemperatureMethod(report.JevParity, tfit)
	if jevOutput != "" {
		data, _ := json.MarshalIndent(report, "", "  ")
		_ = os.WriteFile(jevOutput, data, 0644)
	}
	if jevJSON {
		data, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(data))
		return nil
	}
	printJevReportSummary(report, jevOutput)
	return nil
}

func toCalibrationCasesFromJev(cases []JevCaseResult) []CalibrationCaseResult {
	out := make([]CalibrationCaseResult, len(cases))
	for i, c := range cases {
		out[i] = CalibrationCaseResult{
			ID:                c.ID,
			Metric:            c.Family,
			Category:          c.Family,
			Tier:              c.Tier,
			Expected:          c.Expected,
			Actual:            c.Actual,
			Accurate:          c.Accurate,
			Confidence:        c.Confidence,
			Entropy:           c.Entropy,
			NormalizedEntropy: c.NormalizedEnt,
			VocabCardinality:  c.VocabCardinality,
			ChanceBaseline:    c.ChanceBaseline,
			WallTimeMs:        c.WallTimeMs,
			TopProbabilities:  c.TopProbabilities,
			Error:             c.Error,
		}
	}
	return out
}

func computeJevExactGoldTVD(cases []JevCaseResult, temp float64) float64 {
	var sum float64
	var cnt int
	for _, rc := range cases {
		if !rc.HasGoldProbs || len(rc.GoldProbs) == 0 || len(rc.TopProbabilities) == 0 {
			continue
		}
		cc := CalibrationCaseResult{
			VocabCardinality: rc.VocabCardinality,
			Confidence:       rc.Confidence,
			Entropy:          rc.Entropy,
			TopProbabilities: rc.TopProbabilities,
		}
		probs, _, _, _ := scaleCaseDistribution(cc, temp)
		var tvd float64
		for k, gp := range rc.GoldProbs {
			tvd += math.Abs(probs[k] - gp)
		}
		sum += 0.5 * tvd
		cnt++
	}
	if cnt == 0 {
		return 0.0
	}
	return sum / float64(cnt)
}

func jevToCalCase(c JevCaseResult) CalibrationCaseResult {
	return CalibrationCaseResult{ID: c.ID, VocabCardinality: c.VocabCardinality, Confidence: c.Confidence,
		Entropy: c.Entropy, NormalizedEntropy: c.NormalizedEnt, TopProbabilities: c.TopProbabilities}
}

// applyJevTemperature is applyCalibrationTemperature for JevBench tasks (the objective is
// findOptimalJevTemperature). Input tasks must be at T = 1. Returns tasks, the report-wide T and the fit record.
func applyJevTemperature(cases []JevCaseResult) ([]JevCaseResult, float64, *TemperatureFit) {
	eceAt := func(cs []JevCaseResult, t float64) float64 {
		e, _ := compute10BinECE(enrichAndScaleCaseResults(toCalibrationCasesFromJev(cs), t))
		return e
	}
	base := enrichAndScaleCaseResults(toCalibrationCasesFromJev(cases), 1.0)
	fit := &TemperatureFit{InSampleTemperature: findOptimalJevTemperature(cases), ECERaw: eceAt(cases, 1.0), BrierRaw: meanBrier(base)}
	switch {
	case jevAutoTemp && !jevAutoTempInSample:
		ids := make([]string, len(cases))
		for i, c := range cases {
			ids[i] = c.ID
		}
		perItem, foldT := heldOutTemperatures(ids, temperatureFolds, func(train []int) float64 {
			sub := make([]JevCaseResult, len(train))
			for j, i := range train {
				sub[j] = cases[i]
			}
			return findOptimalJevTemperature(sub)
		})
		out := make([]JevCaseResult, len(cases))
		for i, c := range cases {
			out[i] = c
			out[i].TopProbabilities, out[i].Confidence, out[i].Entropy, out[i].NormalizedEnt = scaleCaseDistribution(jevToCalCase(c), perItem[i])
			out[i].TemperatureApplied = perItem[i]
		}
		fit.Method, fit.Folds, fit.FoldTemperatures = "kfold", temperatureFolds, foldT
		fit.ECEApplied = eceAt(out, 1.0)
		return out, 1.0, fit
	case jevAutoTemp:
		fit.Method = "in_sample"
		fit.ECEApplied = eceAt(cases, fit.InSampleTemperature)
		return cases, fit.InSampleTemperature, fit
	default:
		t := jevTempScale
		if t <= 0 {
			t = 1.0
		}
		fit.Method = "fixed"
		fit.ECEApplied = eceAt(cases, t)
		return cases, t, fit
	}
}

func findOptimalJevTemperature(cases []JevCaseResult) float64 {
	calCases := toCalibrationCasesFromJev(cases)
	bestT := 1.0
	bestScore := -1.0
	for t := 0.50; t <= 3.501; t += 0.05 {
		tRound := math.Round(t*100.0) / 100.0
		scaled := enrichAndScaleCaseResults(calCases, tRound)
		ece10, _ := compute10BinECE(scaled)
		softTVD := computeJevExactGoldTVD(cases, tRound)
		score := computeJevCalibrationScore(ece10, softTVD)
		if score > bestScore+1e-9 || (math.Abs(score-bestScore) <= 1e-9 && math.Abs(tRound-1.0) < math.Abs(bestT-1.0)) {
			bestScore = score
			bestT = tRound
		}
	}
	return bestT
}

func buildJevReport(rawCases []JevCaseResult, source, targetURL, targetModel, commitSHA string, activeTemp float64, multiSlot, flipped bool) JevReport {
	return buildJevReportWithCascade(rawCases, source, targetURL, targetModel, commitSHA, activeTemp, multiSlot, flipped, nil)
}

func buildJevReportWithCascade(rawCases []JevCaseResult, source, targetURL, targetModel, commitSHA string, activeTemp float64, multiSlot, flipped bool, cascade *CascadeSummary) JevReport {
	if activeTemp <= 0 {
		activeTemp = 1.0
	}
	calInput := toCalibrationCasesFromJev(rawCases)
	usdOverride := 0.0260
	if cascade != nil && cascade.TotalCases > 0 {
		escFrac := float64(cascade.EscalatedCases) / float64(cascade.TotalCases)
		stage2Tariff := 0.1155 // gemini-3.8-flash per 1,000 structured JSON classification calls (~450 in / ~80 out tokens)
		p2Lower := strings.ToLower(cascade.Pass2Model)
		if strings.Contains(p2Lower, "lite") {
			stage2Tariff = 0.0480
		}
		usdOverride = 0.0260 + escFrac*stage2Tariff
	}
	jevParity, scaledCal := buildJevParitySummaryWithKind(calInput, activeTemp, targetModel, cascade, usdOverride, jevEndpointKind, jevSealedAcc)
	jevParity.OptimalTemperature = findOptimalJevTemperature(rawCases)

	// Recompute exact GoldProbs TVD when present on probability family items
	scaledJev := make([]JevCaseResult, len(rawCases))
	var exactTVDSum float64
	var exactTVDCnt int
	for i, rc := range rawCases {
		sc := scaledCal[i]
		jc := rc
		jc.TopProbabilities = sc.TopProbabilities
		jc.Confidence = sc.Confidence
		jc.Entropy = sc.Entropy
		jc.NormalizedEnt = sc.NormalizedEntropy
		jc.BrierScore = sc.BrierScore
		if jc.HasGoldProbs && len(jc.GoldProbs) > 0 && len(jc.TopProbabilities) > 0 {
			var tvd float64
			for k, gp := range jc.GoldProbs {
				predP := jc.TopProbabilities[k]
				tvd += math.Abs(predP - gp)
			}
			jc.TVDGold = 0.5 * tvd
			exactTVDSum += jc.TVDGold
			exactTVDCnt++
		}
		scaledJev[i] = jc
	}

	if exactTVDCnt > 0 {
		rawMeanExactTVD := computeJevExactGoldTVD(rawCases, 1.0)
		meanExactTVD := exactTVDSum / float64(exactTVDCnt)
		jevParity.SoftTVDMean = meanExactTVD
		jevParity.ProbabilityFidelity = math.Max(0.0, 100.0*(1.0-meanExactTVD))
		jevParity.RawT1CalibrationScore = computeJevCalibrationScore(jevParity.RawT1ECE10Bin, rawMeanExactTVD)
		jevParity.CalibrationScore = computeJevCalibrationScore(jevParity.ECE10Bin, meanExactTVD)
		jevParity.RawT1CompositeScore = computeJevCompositeScore(
			jevParity.IntelligenceScore,
			jevParity.RawT1CalibrationScore,
			jevParity.SpeedScore,
			jevParity.CostScore,
		)
		jevParity.CompositeJevBenchScore = computeJevCompositeScore(
			jevParity.IntelligenceScore,
			jevParity.CalibrationScore,
			jevParity.SpeedScore,
			jevParity.CostScore,
		)
		jevParity.V14CompositeScore = computeJevV14CompositeScore(
			jevParity.V14IntelligenceScore,
			jevParity.CalibrationScore,
			jevParity.AdjustedSpeedScore,
			jevParity.CostScore,
		)
	}

	tierMap := make(map[string][]CalibrationCaseResult)
	famMap := make(map[string][]CalibrationCaseResult)
	topMap := make(map[string][]CalibrationCaseResult)
	groupPairs := make(map[string][]bool)

	var totalCorrect int
	for i, jc := range scaledJev {
		sc := scaledCal[i]
		if jc.Accurate {
			totalCorrect++
		}
		tierMap[jc.Tier] = append(tierMap[jc.Tier], sc)
		famMap[jc.Family] = append(famMap[jc.Family], sc)
		if jc.Topic != "" {
			topMap[jc.Topic] = append(topMap[jc.Topic], sc)
		}
		if jc.Group != "" && !strings.HasPrefix(jc.Group, "hard-") {
			groupPairs[jc.Group] = append(groupPairs[jc.Group], jc.Accurate)
		}
	}

	var pairCount, pairBothOK int
	for _, statuses := range groupPairs {
		if len(statuses) >= 2 {
			pairCount++
			allOK := true
			for _, st := range statuses {
				if !st {
					allOK = false
					break
				}
			}
			if allOK {
				pairBothOK++
			}
		}
	}
	pairPct := 100.0
	if pairCount > 0 {
		pairPct = float64(pairBothOK) / float64(pairCount) * 100.0
	}

	accPct := 0.0
	if len(scaledJev) > 0 {
		accPct = float64(totalCorrect) / float64(len(scaledJev)) * 100.0
	}

	return JevReport{
		Timestamp:             time.Now().UTC().Format(time.RFC3339),
		Source:                source,
		TargetURL:             targetURL,
		TargetModel:           targetModel,
		LockCommitSHA:         commitSHA,
		TotalCases:            len(scaledJev),
		TotalCorrect:          totalCorrect,
		OverallAccuracyPct:    accPct,
		ChanceBaselinePct:     jevParity.ChanceBaselinePct,
		ChanceCorrectedAccPct: jevParity.ChanceCorrectedAccPct,
		ParaphrasePairs:       pairCount,
		ParaphraseBothCorrect: pairBothOK,
		ParaphraseConsistPct:  pairPct,
		TemperatureScale:      activeTemp,
		MultiSlotEvidence:     multiSlot,
		FlippedOptions:        flipped,
		Cascade:               cascade,
		JevParity:             jevParity,
		ByTier:                summarizeCalibrationGroups(tierMap),
		ByFamily:              summarizeCalibrationGroups(famMap),
		ByTopic:               summarizeCalibrationGroups(topMap),
		Cases:                 scaledJev,
	}
}

func printJevReportSummary(report JevReport, outPath string) {
	if jevSweepThresholds {
		printJevThresholdSweep(report.Cases)
	}
	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Println(styleAccent.Render("  JevBench v1.3.1 Evaluation Report — Tiers, Reasoning Families, Topics & 4-Axis Scorecard"))
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Printf("  Source / Mode:          %s (Model: %s)\n", styleID.Render(report.Source), styleID.Render(report.TargetModel))
	fmt.Printf("  Pinned Upstream Commit: %s | Items: %d | Paraphrase Pair Consistency: %d/%d (%.1f%%)\n",
		truncateStr(report.LockCommitSHA, 12), report.TotalCases,
		report.ParaphraseBothCorrect, report.ParaphrasePairs, report.ParaphraseConsistPct)
	if report.Cascade != nil {
		fmt.Printf("  Cascade Summary:        Threshold=%.2f | Escalated=%d/%d (%.1f%%) | Stage-1=%.1f%% -> Cascade=%.1f%% (+%.1f%%) | LLM Calls Saved=%.1f%%\n",
			report.Cascade.EntropyThreshold,
			report.Cascade.EscalatedCases, report.Cascade.TotalCases, report.Cascade.EscalationRatePct,
			report.Cascade.Pass1AccuracyPct, report.Cascade.CascadeAccuracyPct, report.Cascade.AccuracyGainPct,
			100.0-report.Cascade.EscalationRatePct)
	}
	fmt.Println(styleMuted.Render("  --------------------------------------------------------------------------------------------------------"))
	fmt.Printf("  %-22s %7s %9s %11s %9s %8s %11s %8s %9s\n",
		"JEVBENCH TIER", "CASES", "RAW ACC", "CHANCE-CORR", "CONF P(y)", "BRIER", "ENTROPY H", "NORM H~", "LATENCY")
	fmt.Println(styleMuted.Render("  --------------------------------------------------------------------------------------------------------"))
	for _, t := range report.ByTier {
		fmt.Printf("  %-22s %3d/%-3d %8.1f%% %10.1f%% %9.3f %8.4f %7.4f nats %8.4f %6.0f ms\n",
			t.Name, t.Correct, t.Total, t.AccuracyPct, t.ChanceCorrectedPct,
			t.AvgConfidence, t.AvgBrierScore, t.AvgEntropy, t.AvgNormalizedEntropy, t.AvgLatencyMs)
	}

	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Println(styleAccent.Render("  Breakdown by JevBench Task Family (Highlighting Hard-Tier Reasoning Families)"))
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Printf("  %-22s %7s %9s %11s %9s %8s %11s %8s %9s\n",
		"FAMILY", "CASES", "RAW ACC", "CHANCE-CORR", "CONF P(y)", "BRIER", "ENTROPY H", "NORM H~", "LATENCY")
	fmt.Println(styleMuted.Render("  --------------------------------------------------------------------------------------------------------"))
	for _, f := range report.ByFamily {
		accStr := fmt.Sprintf("%6.1f%%", f.AccuracyPct)
		if f.AccuracyPct >= 85.0 {
			accStr = stylePass.Render(accStr)
		} else if f.AccuracyPct >= 65.0 {
			accStr = styleWarn.Render(accStr)
		} else {
			accStr = styleFail.Render(accStr)
		}
		fmt.Printf("  %-22s %3d/%-3d %9s %10.1f%% %9.3f %8.4f %7.4f nats %8.4f %6.0f ms\n",
			f.Name, f.Correct, f.Total, accStr, f.ChanceCorrectedPct,
			f.AvgConfidence, f.AvgBrierScore, f.AvgEntropy, f.AvgNormalizedEntropy, f.AvgLatencyMs)
	}

	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Println(styleAccent.Render("  Breakdown by Subject Topic (datasets/TOPICS.md)"))
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Printf("  %-22s %7s %9s %11s %9s %8s %11s %8s %9s\n",
		"SUBJECT TOPIC", "CASES", "RAW ACC", "CHANCE-CORR", "CONF P(y)", "BRIER", "ENTROPY H", "NORM H~", "LATENCY")
	fmt.Println(styleMuted.Render("  --------------------------------------------------------------------------------------------------------"))
	for _, top := range report.ByTopic {
		fmt.Printf("  %-22s %3d/%-3d %8.1f%% %10.1f%% %9.3f %8.4f %7.4f nats %8.4f %6.0f ms\n",
			top.Name, top.Correct, top.Total, top.AccuracyPct, top.ChanceCorrectedPct,
			top.AvgConfidence, top.AvgBrierScore, top.AvgEntropy, top.AvgNormalizedEntropy, top.AvgLatencyMs)
	}

	printJevParityDashboard(CalibrationReport{
		OverallAccuracyPct: report.OverallAccuracyPct,
		JevParity:          report.JevParity,
	})

	if outPath != "" {
		fmt.Printf("\n  Saved JevBench report to: %s\n\n", styleID.Render(outPath))
	}
}

func printJevThresholdSweep(cases []JevCaseResult) {
	if len(cases) == 0 {
		return
	}
	thresholds := []float64{0.40, 0.50, 0.58, 0.62, 0.66, 0.70, 0.75}
	var totalMisses int
	for _, c := range cases {
		if !c.Accurate {
			totalMisses++
		}
	}
	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Println(styleAccent.Render("  JevBench Phase 2B — Normalized Entropy (H~ = H / ln|V|) Gate Threshold Sweep"))
	fmt.Println(styleAccent.Render("=========================================================================================================="))
	fmt.Printf("  %-10s %10s %10s %15s %14s %12s %12s\n",
		"GATE H~", "ESCALATED", "ESC RATE", "MISSES CAPTURED", "RECALL OF ERR", "LLM SAVED", "EST $/1K")
	fmt.Println(styleMuted.Render("  --------------------------------------------------------------------------------------------------------"))
	for _, th := range thresholds {
		var esc, missesCaught int
		for _, c := range cases {
			hVal := c.NormalizedEnt
			if !jevNormalizeEntropy {
				hVal = c.Entropy
			}
			if hVal >= th {
				esc++
				if !c.Accurate {
					missesCaught++
				}
			}
		}
		escPct := float64(esc) / float64(len(cases)) * 100.0
		savedPct := 100.0 - escPct
		recallPct := 0.0
		if totalMisses > 0 {
			recallPct = float64(missesCaught) / float64(totalMisses) * 100.0
		}
		estUSD := 0.0260 + (escPct/100.0)*0.2140
		fmt.Printf("  >= %-7.2f %4d/%-5d %9.1f%% %8d/%-6d %13.1f%% %11.1f%%      $%.4f\n",
			th, esc, len(cases), escPct, missesCaught, totalMisses, recallPct, savedPct, estUSD)
	}
}

// runJevCascade executes Phase 2B Entropy-Gated Escalation on JevBench:
// low-entropy Stage-1 decisions (H_norm < threshold) early-exit at Stage 1 (DiffusionGemma),
// while high-entropy decisions (H_norm >= threshold) escalate to Vertex AI Gemini with Pass-1 Prior Forwarding.
func runJevCascade(receiptPath string) error {
	raw, err := os.ReadFile(receiptPath)
	if err != nil {
		return fmt.Errorf("failed to read --cascade-from receipt %q: %w", receiptPath, err)
	}
	var prev JevReport
	if err := json.Unmarshal(raw, &prev); err != nil {
		return fmt.Errorf("failed to parse --cascade-from receipt %q: %w", receiptPath, err)
	}

	tasks, err := loadJevTasks(jevDataset, jevTierFilter, jevFamilyFilter, jevTopicFilter, jevLimit)
	if err != nil {
		return fmt.Errorf("failed to load JevBench dataset %q: %w\n  Hint: Run 'dgem bench-jev --sync' first", jevDataset, err)
	}
	taskMap := make(map[string]JevTask, len(tasks))
	for _, t := range tasks {
		taskMap[t.ID] = t
	}

	var filtered []JevCaseResult
	for _, c := range prev.Cases {
		if _, ok := taskMap[c.ID]; !ok {
			continue
		}
		filtered = append(filtered, c)
	}
	if len(filtered) == 0 {
		return fmt.Errorf("no JevBench tasks matched filters across receipt %q and dataset %q", receiptPath, jevDataset)
	}

	if jevSweepThresholds && jevVertexModel == "" {
		printJevThresholdSweep(filtered)
		return nil
	}

	if jevVertexModel == "" {
		jevVertexModel = "gemini-3.8-flash"
	}
	gcpProj := resolveGCPProject(jevVertexProject)
	if gcpProj == "" {
		return fmt.Errorf("no Google Cloud Project ID detected for --vertex-model %q\n  Hint: Pass '--vertex-project <id>' or set GCP_PROJECT=<id>", jevVertexModel)
	}

	ctx := context.Background()
	genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  gcpProj,
		Location: "global",
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return fmt.Errorf("failed to initialize Vertex AI client (project=%s, location=global): %w", gcpProj, err)
	}

	gateName := "H_norm"
	if !jevNormalizeEntropy {
		gateName = "H_nats"
	}

	if !jevJSON {
		if jevSweepThresholds {
			printJevThresholdSweep(filtered)
		}
		fmt.Println()
		fmt.Println(styleAccent.Render("=========================================================================================================="))
		fmt.Println(styleAccent.Render("  JevBench Phase 2B — Entropy-Gated Escalation Cascade (Stage 1 DiffusionGemma -> Stage 2 Vertex AI)"))
		fmt.Println(styleAccent.Render("=========================================================================================================="))
		fmt.Printf("  Stage-1 Receipt:   %s (%d items)\n", styleID.Render(receiptPath), len(filtered))
		fmt.Printf("  Stage-2 Model:     %s (vertexai://%s/global)\n", styleID.Render(jevVertexModel), gcpProj)
		fmt.Printf("  Escalation Gate:   %s >= %.2f (Pass-1 Prior Forwarding Enabled)\n", gateName, jevCascadeThreshold)
		fmt.Println(styleMuted.Render("  --------------------------------------------------------------------------------------------------------"))
	}

	workers := jevWorkers
	if workers < 1 {
		workers = 4
	}
	if workers > len(filtered) {
		workers = len(filtered)
	}

	results := make([]JevCaseResult, len(filtered))
	jobs := make(chan int, len(filtered))
	var wg sync.WaitGroup
	var printMu sync.Mutex

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				p1 := filtered[idx]
				t := taskMap[p1.ID]
				gateVal := p1.NormalizedEnt
				if !jevNormalizeEntropy {
					gateVal = p1.Entropy
				}

				var res JevCaseResult
				if gateVal < jevCascadeThreshold {
					res = p1
					res.Escalated = false
					res.Pass1Actual = p1.Actual
					res.Pass1Accurate = p1.Accurate
					res.Pass1Entropy = p1.Entropy
					res.Pass1NormalizedEnt = p1.NormalizedEnt
					res.Pass1LatencyMs = p1.WallTimeMs
				} else {
					res = evaluateJevTaskVertex(ctx, genaiClient, jevVertexModel, t, p1)
					res.Escalated = true
					res.PriorGuided = true
					res.Pass1Actual = p1.Actual
					res.Pass1Accurate = p1.Accurate
					res.Pass1Entropy = p1.Entropy
					res.Pass1NormalizedEnt = p1.NormalizedEnt
					res.Pass1LatencyMs = p1.WallTimeMs
					res.Pass2LatencyMs = res.WallTimeMs
					res.WallTimeMs = p1.WallTimeMs + res.Pass2LatencyMs
				}
				results[idx] = res

				if !jevJSON && res.Escalated {
					printMu.Lock()
					p1Badge := styleFail.Render("MISS")
					if p1.Accurate {
						p1Badge = stylePass.Render("OK  ")
					}
					p2Badge := stylePass.Render("PASS")
					if !res.Accurate {
						p2Badge = styleFail.Render("FAIL")
					}
					errSuffix := ""
					if res.Error != "" {
						p2Badge = styleWarn.Render("ERR ")
						errSuffix = " ERR=" + truncateStr(res.Error, 60)
					}
					fmt.Printf("  [ESC %03d] %-10s %-16s H~=%.3f | Stage1(%s): %-14s -> Stage2(%s): %-14s (exp=%s, +%.0fms)%s\n",
						idx+1, styleID.Render(p1.ID), p1.Family, p1.NormalizedEnt,
						p1Badge, truncateStr(p1.Actual, 14),
						p2Badge, truncateStr(res.Actual, 14),
						truncateStr(p1.Expected, 14), res.Pass2LatencyMs, errSuffix)
					printMu.Unlock()
				}
			}
		}()
	}

	for i := range filtered {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	var p1Correct, casCorrect, escCount int
	for _, r := range results {
		if r.Pass1Accurate {
			p1Correct++
		}
		if r.Accurate {
			casCorrect++
		}
		if r.Escalated {
			escCount++
		}
	}
	n := float64(len(results))
	p1Acc := float64(p1Correct) / n * 100.0
	casAcc := float64(casCorrect) / n * 100.0
	escPct := float64(escCount) / n * 100.0

	cascadeSummary := &CascadeSummary{
		CascadeMode:        "cross-model-prior-guided",
		NormalizedEntropy:  jevNormalizeEntropy,
		PriorGuided:        true,
		EntropyThreshold:   jevCascadeThreshold,
		Pass1Model:         prev.TargetModel,
		Pass2Model:         jevVertexModel,
		TotalCases:         len(results),
		EscalatedCases:     escCount,
		EscalationRatePct:  escPct,
		Pass1Correct:       p1Correct,
		Pass1AccuracyPct:   p1Acc,
		CascadeCorrect:     casCorrect,
		CascadeAccuracyPct: casAcc,
		AccuracyGainPct:    casAcc - p1Acc,
	}

	results, activeTemp, tfit := applyJevTemperature(results)

	targetURL := fmt.Sprintf("cascade(%s -> vertexai://%s/global)", receiptPath, gcpProj)
	targetModel := fmt.Sprintf("DiffusionGemma [%s<%.2f] -> %s [%s>=%.2f, prior-guided]",
		gateName, jevCascadeThreshold, jevVertexModel, gateName, jevCascadeThreshold)

	report := buildJevReportWithCascade(results, "phase-2b-entropy-cascade", targetURL, targetModel, prev.LockCommitSHA, activeTemp, false, false, cascadeSummary)
	report.TemperatureFit = tfit
	stampTemperatureMethod(report.JevParity, tfit)
	if jevOutput != "" {
		data, _ := json.MarshalIndent(report, "", "  ")
		_ = os.WriteFile(jevOutput, data, 0644)
	}
	if jevJSON {
		data, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(data))
		return nil
	}
	printJevReportSummary(report, jevOutput)
	return nil
}

func evaluateJevTaskVertex(ctx context.Context, genaiClient *genai.Client, model string, t JevTask, p1 JevCaseResult) JevCaseResult {
	res := p1
	stateBytes, _ := json.MarshalIndent(t.State, "", "  ")
	critMap := extractJevCriteriaMap(t.Question.Criteria)

	var optLines []string
	for _, lbl := range t.Labels {
		desc := critMap[lbl]
		if lbl == "yes" && desc == "" {
			desc = critMap["true"]
		} else if lbl == "no" && desc == "" {
			desc = critMap["false"]
		}
		if desc != "" {
			optLines = append(optLines, fmt.Sprintf("- %q: %s", lbl, desc))
		} else {
			optLines = append(optLines, fmt.Sprintf("- %q", lbl))
		}
	}

	calPrior := CalibrationCaseResult{
		Actual:            p1.Actual,
		Confidence:        p1.Confidence,
		Entropy:           p1.Entropy,
		NormalizedEntropy: p1.NormalizedEnt,
		TopProbabilities:  p1.TopProbabilities,
	}
	priorBlock := formatTier1PriorBlock(&calPrior, jevCascadeThreshold, jevNormalizeEntropy)

	prompt := fmt.Sprintf(
		"Task Family: %s (Tier: %s)\n\nInstructions:\n%s\n\nAllowed Labels & Criteria:\n%s\n\nInput State:\n%s%s",
		t.Family, t.Tier, t.Question.Instructions, strings.Join(optLines, "\n"), string(stateBytes), priorBlock,
	)

	genaiSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"reasoning":  {Type: genai.TypeString, Description: "Step-by-step verification of arithmetic, date intervals, policy exceptions, or evidence."},
			"selection":  {Type: genai.TypeString, Enum: t.Labels, Description: "Selected label from the allowed options."},
			"confidence": {Type: genai.TypeNumber, Description: "Calibrated posterior probability in [0.50, 0.95] for the selected label."},
		},
		Required: []string{"reasoning", "selection", "confidence"},
	}

	temp := float32(0.0)
	thinkBudget := int32(256)
	cfg := &genai.GenerateContentConfig{
		Temperature:      &temp,
		ResponseMIMEType: "application/json",
		ResponseSchema:   genaiSchema,
		ThinkingConfig: &genai.ThinkingConfig{
			ThinkingBudget: &thinkBudget,
		},
	}

	t0 := time.Now()
	resp, err := generateContentWithRetry(ctx, genaiClient, model, prompt, cfg)
	res.WallTimeMs = float64(time.Since(t0).Milliseconds())
	if err != nil {
		res.Error = err.Error()
		return res
	}

	rawText := strings.TrimSpace(resp.Text())
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(rawText), &parsed); err != nil {
		res.Error = err.Error()
		return res
	}

	sel, _ := parsed["selection"].(string)
	sel = strings.TrimSpace(sel)
	if sel == "" {
		sel = p1.Actual
	}
	res.Actual = sel
	res.Accurate = strings.EqualFold(sel, p1.Expected)

	conf := 0.88
	if cVal, ok := parsed["confidence"].(float64); ok && cVal > 0 {
		conf = math.Max(1.0/float64(res.VocabCardinality)+0.05, math.Min(0.95, cVal))
	}
	res.Confidence = conf

	// Construct full simplex TopProbabilities over t.Labels blending Stage-2 selection confidence with Stage-1 runner-up weights
	newProbs := make(map[string]float64, len(t.Labels))
	var otherWeightSum float64
	for _, l := range t.Labels {
		if l == sel {
			continue
		}
		w := p1.TopProbabilities[l]
		if w <= 0 {
			w = 1e-4
		}
		otherWeightSum += w
	}
	rem := 1.0 - conf
	var h float64
	for _, l := range t.Labels {
		var pk float64
		if l == sel {
			pk = conf
		} else if otherWeightSum > 0 {
			w := p1.TopProbabilities[l]
			if w <= 0 {
				w = 1e-4
			}
			pk = rem * (w / otherWeightSum)
		} else if len(t.Labels) > 1 {
			pk = rem / float64(len(t.Labels)-1)
		}
		newProbs[l] = pk
		if pk > 1e-12 {
			h -= pk * math.Log(pk)
		}
	}
	res.TopProbabilities = newProbs
	res.Entropy = h
	if res.VocabCardinality > 1 {
		res.NormalizedEnt = h / math.Log(float64(res.VocabCardinality))
	}
	return res
}
