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

package permutation

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// OptionItem represents a single semantic option with a stable key and description.
type OptionItem struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// PermutationCase represents one evaluation item in the Permutation Sensitivity suite.
type PermutationCase struct {
	ID           string       `json:"id"`
	Regime       string       `json:"regime"` // "consensus", "ambiguous_chaosnli", "adversarial_trap", "binary_ablation"
	Title        string       `json:"title"`
	State        string       `json:"state"`
	Instructions string       `json:"instructions"`
	Options      []OptionItem `json:"options"`
	Expected     string       `json:"expected"`
}

// NullPrior holds the measured content-free positional probability distribution p_0(pos_0 .. pos_{K-1})
// for each cardinality K (e.g. K=2, 3, 4).
type NullPrior struct {
	ByCardinality  map[int][]float64 `json:"by_cardinality"`
	EntropyByK     map[int]float64   `json:"entropy_by_k"`
	UniformEntropy map[int]float64   `json:"uniform_entropy_by_k"`
	MaxBiasRatio   map[int]float64   `json:"max_bias_ratio_by_k"` // max(p_0) / (1/K)
}

// PermPassTelemetry records the outcome of evaluating one specific permutation pi_r of a question.
type PermPassTelemetry struct {
	PermIndex      int                `json:"perm_index"`
	OptionOrder    []string           `json:"option_order"` // Order of option keys bound to [A, B, C, ...]
	RawProbs       map[string]float64 `json:"raw_probs"`    // Keyed by semantic OptionItem.Name
	DebiasedProbs  map[string]float64 `json:"debiased_probs,omitempty"`
	RawWinner      string             `json:"raw_winner"`
	DebiasedWinner string             `json:"debiased_winner,omitempty"`
	RawEntropy     float64            `json:"raw_entropy"`
	NormalizedH    float64            `json:"normalized_entropy"`
	WallTimeMs     float64            `json:"wall_time_ms"`
}

// DualMirrorResult records the O(1) Single-Pass Dual-Mirror Canvas readout (slot_fwd + slot_rev in 1 pass).
type DualMirrorResult struct {
	FwdProbs          map[string]float64 `json:"fwd_probs"`
	RevProbs          map[string]float64 `json:"rev_probs"`
	EnsembleProbs     map[string]float64 `json:"ensemble_probs"`
	FwdWinner         string             `json:"fwd_winner"`
	RevWinner         string             `json:"rev_winner"`
	EnsembleWinner    string             `json:"ensemble_winner"`
	MirrorDisagreed   bool               `json:"mirror_disagreed"` // True if FwdWinner != RevWinner
	MeanPassEntropy   float64            `json:"mean_pass_entropy"`
	TotalEntropy      float64            `json:"total_entropy"`      // H(EnsembleProbs) = MeanPassEntropy + JSDivergence
	NormalizedTotalH  float64            `json:"normalized_total_h"` // H(EnsembleProbs) / ln(K)
	JSDivergence      float64            `json:"js_divergence_nats"` // Epistemic permutation disagreement I(Y; Pi | X)
	NormalizedJSD     float64            `json:"normalized_jsd"`     // JSD / ln(2) in [0, 1]
	TVD               float64            `json:"tvd"`                // 0.5 * sum |p_fwd - p_rev|
	Accurate          bool               `json:"accurate"`
	EscalatedBySingle bool               `json:"escalated_by_single"` // H_norm(fwd) >= threshold
	EscalatedByMirror bool               `json:"escalated_by_mirror"` // MirrorDisagreed || NormalizedTotalH >= threshold || NormalizedJSD >= 0.08
	WallTimeMs        float64            `json:"wall_time_ms"`
}

// CasePermutationResult aggregates all K cyclic permutations + Prior De-biasing + O(1) Dual-Mirror readout for one case.
type CasePermutationResult struct {
	ID                   string  `json:"id"`
	Regime               string  `json:"regime"`
	Title                string  `json:"title"`
	Cardinality          int     `json:"cardinality"`
	Expected             string  `json:"expected"`
	CanonicalWinner      string  `json:"canonical_winner"`
	CanonicalAccurate    bool    `json:"canonical_accurate"`
	CanonicalConfidence  float64 `json:"canonical_confidence"`
	CanonicalEntropy     float64 `json:"canonical_entropy"`
	CanonicalNormalizedH float64 `json:"canonical_normalized_h"`
	CanonicalBrier       float64 `json:"canonical_brier"`
	// Cyclic K-permutation metrics (Raw)
	CyclicPasses         []PermPassTelemetry `json:"cyclic_passes"`
	RawFlipOccurred      bool                `json:"raw_flip_occurred"` // Did argmax change across any cyclic shift?
	RawDistinctWinners   int                 `json:"raw_distinct_winners"`
	RawMajorityAgreement float64             `json:"raw_majority_agreement"` // Fraction of K shifts agreeing with mode
	RawMeanEntropy       float64             `json:"raw_mean_entropy"`       // E_pi[H(p_pi)] (Aleatoric / Within-pass)
	RawEntropyStdDev     float64             `json:"raw_entropy_stddev"`
	RawTotalEntropy      float64             `json:"raw_total_entropy"`      // H(E_pi[p_pi]) (Total uncertainty)
	RawJSDivergence      float64             `json:"raw_js_divergence_nats"` // I(Y; Pi | X) = H(bar_p) - bar_H (Epistemic)
	RawMaxTVD            float64             `json:"raw_max_tvd"`
	RawEnsembleWinner    string              `json:"raw_ensemble_winner"`
	RawEnsembleAccurate  bool                `json:"raw_ensemble_accurate"`
	RawEnsembleBrier     float64             `json:"raw_ensemble_brier"`
	// Content-Free Prior De-Biased metrics (13B)
	DebiasedFlipOccurred   bool    `json:"debiased_flip_occurred"`
	DebiasedJSDivergence   float64 `json:"debiased_js_divergence_nats"`
	DebiasedCanonicalAcc   bool    `json:"debiased_canonical_accurate"`
	DebiasedCanonicalBrier float64 `json:"debiased_canonical_brier"`
	// O(1) Single-Pass Dual-Mirror Canvas metrics (13C)
	DualMirror DualMirrorResult `json:"dual_mirror"`
}

// RegimeSummary aggregates permutation stability and error-detection metrics for one regime.
type RegimeSummary struct {
	Regime                string  `json:"regime"`
	Cases                 int     `json:"cases"`
	CanonicalAccuracyPct  float64 `json:"canonical_accuracy_pct"`
	DualMirrorAccuracyPct float64 `json:"dual_mirror_accuracy_pct"`
	RawFlipRatePct        float64 `json:"raw_flip_rate_pct"`
	DebiasedFlipRatePct   float64 `json:"debiased_flip_rate_pct"`
	MeanSingleNormalizedH float64 `json:"mean_single_normalized_h"`
	MeanCyclicJSD         float64 `json:"mean_cyclic_jsd_nats"`
	MeanMirrorJSD         float64 `json:"mean_mirror_jsd_nats"`
	MeanMirrorTVD         float64 `json:"mean_mirror_tvd"`
	SingleGateRecallOnErr float64 `json:"single_gate_recall_on_errors_pct"`
	MirrorGateRecallOnErr float64 `json:"mirror_gate_recall_on_errors_pct"`
}

// PermutationReport is the top-level JSON receipt.
type PermutationReport struct {
	Timestamp              string                  `json:"timestamp"`
	Endpoint               string                  `json:"endpoint"`
	Model                  string                  `json:"model"`
	GateThresholdNormH     float64                 `json:"gate_threshold_normalized_h"`
	PriorDampingAlpha      float64                 `json:"prior_damping_alpha"`
	NullPrior              *NullPrior              `json:"null_prior"`
	TotalCases             int                     `json:"total_cases"`
	CanonicalAccuracyPct   float64                 `json:"canonical_accuracy_pct"`
	DebiasedAccuracyPct    float64                 `json:"debiased_accuracy_pct"`
	DualMirrorAccuracyPct  float64                 `json:"dual_mirror_accuracy_pct"`
	CyclicEnsembleAccPct   float64                 `json:"cyclic_ensemble_accuracy_pct"`
	RawFlipRatePct         float64                 `json:"raw_flip_rate_pct"`
	DebiasedFlipRatePct    float64                 `json:"debiased_flip_rate_pct"`
	FlipReductionRelPct    float64                 `json:"flip_reduction_relative_pct"`
	CanonicalBrier         float64                 `json:"canonical_brier"`
	DebiasedBrier          float64                 `json:"debiased_brier"`
	DualMirrorBrier        float64                 `json:"dual_mirror_brier"`
	MeanSinglePassNormH    float64                 `json:"mean_single_pass_normalized_h"`
	MeanCyclicJSD          float64                 `json:"mean_cyclic_jsd_nats"`
	MeanMirrorJSD          float64                 `json:"mean_mirror_jsd_nats"`
	FalseConfidentErrors   int                     `json:"false_confident_errors_single_pass"` // Wrong in pass 1 AND H_norm < threshold
	CaughtByDualMirror     int                     `json:"caught_by_dual_mirror_gate"`         // How many of those false-confident errors Dual-Mirror caught!
	SingleGateErrRecallPct float64                 `json:"single_gate_error_recall_pct"`
	MirrorGateErrRecallPct float64                 `json:"mirror_gate_error_recall_pct"`
	MeanSingleLatencyMs    float64                 `json:"mean_single_latency_ms"`
	MeanMirrorLatencyMs    float64                 `json:"mean_mirror_latency_ms"`
	Regimes                []RegimeSummary         `json:"regimes"`
	Cases                  []CasePermutationResult `json:"cases"`
}

// CyclicShift returns the r-th cyclic shift of options: out[i] = opts[(i+r)%len(opts)].
func CyclicShift(opts []OptionItem, r int) []OptionItem {
	n := len(opts)
	if n == 0 {
		return nil
	}
	out := make([]OptionItem, n)
	for i := 0; i < n; i++ {
		out[i] = opts[(i+r)%n]
	}
	return out
}

// ReverseOptions returns the reversed slice of options: out[i] = opts[n-1-i].
func ReverseOptions(opts []OptionItem) []OptionItem {
	n := len(opts)
	out := make([]OptionItem, n)
	for i := 0; i < n; i++ {
		out[i] = opts[n-1-i]
	}
	return out
}

// ComputeShannonEntropy computes H(p) = -sum p_k ln(p_k) in nats and normalized H / ln(K).
func ComputeShannonEntropy(probs map[string]float64, k int) (float64, float64) {
	if k <= 1 {
		return 0, 0
	}
	var h float64
	for _, p := range probs {
		if p > 1e-12 {
			h -= p * math.Log(p)
		}
	}
	normH := h / math.Log(float64(k))
	if normH > 1.0 {
		normH = 1.0
	}
	return h, normH
}

// ComputeBrier computes the multi-class Brier score sum_{k} (p_k - y_k)^2.
func ComputeBrier(probs map[string]float64, keys []string, gold string) float64 {
	var sum float64
	for _, k := range keys {
		y := 0.0
		if k == gold {
			y = 1.0
		}
		d := probs[k] - y
		sum += d * d
	}
	return sum
}

// ComputeTVD computes Total Variation Distance 0.5 * sum_k |p_k - q_k|.
func ComputeTVD(p, q map[string]float64, keys []string) float64 {
	var sum float64
	for _, k := range keys {
		sum += math.Abs(p[k] - q[k])
	}
	return 0.5 * sum
}

// ComputeBALDDecomposition takes R probability distributions over the same semantic keys
// and returns:
//   - ensembleProbs: bar_p(o_k) = (1/R) sum_r p_r(o_k)
//   - ensembleWinner: argmax_k bar_p(o_k)
//   - meanH: (1/R) sum_r H(p_r) (Aleatoric / Within-Pass Entropy)
//   - totalH: H(bar_p) (Total Predictive Uncertainty)
//   - jsd: H(bar_p) - meanH (Jensen-Shannon Divergence = Mutual Information I(Y; Pi | X))
func ComputeBALDDecomposition(dists []map[string]float64, keys []string) (map[string]float64, string, float64, float64, float64) {
	rCount := float64(len(dists))
	if rCount == 0 || len(keys) == 0 {
		return map[string]float64{}, "", 0, 0, 0
	}

	ensemble := make(map[string]float64, len(keys))
	var sumH float64
	for _, d := range dists {
		h, _ := ComputeShannonEntropy(d, len(keys))
		sumH += h
		for _, k := range keys {
			ensemble[k] += d[k] / rCount
		}
	}
	meanH := sumH / rCount
	totalH, _ := ComputeShannonEntropy(ensemble, len(keys))
	jsd := totalH - meanH
	if jsd < 0 {
		jsd = 0
	}

	winner := keys[0]
	bestP := -1.0
	for _, k := range keys {
		if ensemble[k] > bestP {
			bestP = ensemble[k]
			winner = k
		}
	}
	return ensemble, winner, meanH, totalH, jsd
}

// MeasureNullPrior queries the model once with a content-free state containing K=2, K=3, and K=4 neutral slots
// in a single O(1) forward pass to measure the positional letter prior p_0(pos_0 .. pos_{K-1}).
func MeasureNullPrior(ctx context.Context, cli *client.Client) (*NullPrior, error) {
	questions := []map[string]any{}
	for _, k := range []int{2, 3, 4} {
		opts := make([]map[string]string, k)
		for i := 0; i < k; i++ {
			opts[i] = map[string]string{
				"name":        fmt.Sprintf("slot_pos_%d", i),
				"description": "Neutral placeholder option with identical prior weight.",
			}
		}
		questions = append(questions, map[string]any{
			"id":           fmt.Sprintf("null_prior_k%d", k),
			"type":         "choice",
			"instructions": "No state information is provided. Evaluate the neutral placeholder options.",
			"options":      opts,
		})
	}

	schemaEnv := map[string]any{
		"instructions": "Content-free baseline calibration probe.",
		"samples":      1,
		"think":        0,
		"questions":    questions,
	}
	schemaBytes, _ := json.Marshal(schemaEnv)
	stateBytes := `{"state": "[REDACTED / CONTENT-FREE CALIBRATION PROBE]"}`

	resp, _, err := cli.Decide(ctx, string(schemaBytes), stateBytes)
	if err != nil {
		return nil, err
	}

	np := &NullPrior{
		ByCardinality:  make(map[int][]float64),
		EntropyByK:     make(map[int]float64),
		UniformEntropy: make(map[int]float64),
		MaxBiasRatio:   make(map[int]float64),
	}

	for _, k := range []int{2, 3, 4} {
		qID := fmt.Sprintf("null_prior_k%d", k)
		ans, ok := resp.Answers[qID]
		vec := make([]float64, k)
		var sum float64
		for i := 0; i < k; i++ {
			key := fmt.Sprintf("slot_pos_%d", i)
			p := 1.0 / float64(k)
			if ok && len(ans.Probabilities) > 0 {
				if v, exists := ans.Probabilities[key]; exists && v > 0 {
					p = v
				}
			}
			// Laplace smoothing floor so no prior position is 0
			p = 0.85*p + 0.15*(1.0/float64(k))
			vec[i] = p
			sum += p
		}
		maxRatio := 1.0
		probMap := make(map[string]float64, k)
		for i := 0; i < k; i++ {
			vec[i] /= sum
			probMap[fmt.Sprintf("pos_%d", i)] = vec[i]
			ratio := vec[i] / (1.0 / float64(k))
			if ratio > maxRatio {
				maxRatio = ratio
			}
		}
		h, _ := ComputeShannonEntropy(probMap, k)
		np.ByCardinality[k] = vec
		np.EntropyByK[k] = h
		np.UniformEntropy[k] = math.Log(float64(k))
		np.MaxBiasRatio[k] = maxRatio
	}

	return np, nil
}

// ApplyPriorDeBiasing removes the positional letter prior p_0(pos_i) from a permutation pass's probabilities:
// p_debiased(o_{pi(i)}) \propto p_raw(o_{pi(i)}) / (p_0(i))^alpha.
func ApplyPriorDeBiasing(rawProbs map[string]float64, orderedOpts []OptionItem, np *NullPrior, alpha float64) (map[string]float64, string) {
	k := len(orderedOpts)
	out := make(map[string]float64, k)
	if k == 0 {
		return out, ""
	}
	priorVec, hasPrior := np.ByCardinality[k]
	var sum float64
	for posIdx, opt := range orderedOpts {
		rawP := rawProbs[opt.Name]
		if rawP < 1e-9 {
			rawP = 1e-9
		}
		weight := 1.0
		if hasPrior && posIdx < len(priorVec) && priorVec[posIdx] > 1e-6 {
			weight = math.Pow(priorVec[posIdx], alpha)
		}
		adj := rawP / weight
		out[opt.Name] = adj
		sum += adj
	}
	winner := orderedOpts[0].Name
	bestP := -1.0
	for _, opt := range orderedOpts {
		out[opt.Name] /= sum
		if out[opt.Name] > bestP {
			bestP = out[opt.Name]
			winner = opt.Name
		}
	}
	return out, winner
}

// EvaluateSinglePermutation runs 1 forward pass with options in the specified order `orderedOpts`.
func EvaluateSinglePermutation(ctx context.Context, cli *client.Client, c PermutationCase, orderedOpts []OptionItem) (map[string]float64, string, float64, error) {
	start := time.Now()
	optObjs := make([]map[string]string, len(orderedOpts))
	for i, o := range orderedOpts {
		optObjs[i] = map[string]string{
			"name":        o.Name,
			"description": o.Description,
		}
	}
	schemaEnv := map[string]any{
		"instructions": "Evaluate the decision question strictly against the provided state and option descriptions.",
		"samples":      1,
		"think":        0,
		"questions": []map[string]any{
			{
				"id":           "decision",
				"type":         "choice",
				"instructions": c.Instructions,
				"options":      optObjs,
			},
		},
	}
	schemaBytes, _ := json.Marshal(schemaEnv)
	stateBytes, _ := json.Marshal(map[string]string{"state": c.State})

	resp, _, err := cli.Decide(ctx, string(schemaBytes), string(stateBytes))
	wallMs := float64(time.Since(start).Milliseconds())
	if err != nil {
		return nil, "", wallMs, err
	}

	ans := resp.Answers["decision"]
	probs := make(map[string]float64, len(orderedOpts))
	var sum float64
	for _, o := range orderedOpts {
		p := ans.Probabilities[o.Name]
		if p <= 0 && (ans.Choice == o.Name || ans.Label == o.Name) {
			p = 0.90
		} else if p <= 0 {
			p = 0.10 / float64(max(1, len(orderedOpts)-1))
		}
		probs[o.Name] = p
		sum += p
	}
	winner := orderedOpts[0].Name
	bestP := -1.0
	for _, o := range orderedOpts {
		probs[o.Name] /= sum
		if probs[o.Name] > bestP {
			bestP = probs[o.Name]
			winner = o.Name
		}
	}
	return probs, winner, wallMs, nil
}

// EvaluateDualMirrorSinglePass places BOTH `decision_fwd` ([o_1..o_K]) and `decision_rev` ([o_K..o_1])
// onto the SAME diffusion canvas in a SINGLE O(1) forward pass (reads=1)!
func EvaluateDualMirrorSinglePass(ctx context.Context, cli *client.Client, c PermutationCase, gateNormH float64) (DualMirrorResult, error) {
	start := time.Now()
	fwdOpts := c.Options
	revOpts := ReverseOptions(c.Options)

	buildOptMap := func(items []OptionItem) []map[string]string {
		out := make([]map[string]string, len(items))
		for i, o := range items {
			out[i] = map[string]string{
				"name":        o.Name,
				"description": o.Description,
			}
		}
		return out
	}

	schemaEnv := map[string]any{
		"instructions": "Evaluate each decision slot independently and strictly against the provided state and option descriptions.",
		"samples":      1,
		"think":        0,
		"questions": []map[string]any{
			{
				"id":           "decision_fwd",
				"type":         "choice",
				"instructions": c.Instructions,
				"options":      buildOptMap(fwdOpts),
			},
			{
				"id":           "decision_rev",
				"type":         "choice",
				"instructions": c.Instructions,
				"options":      buildOptMap(revOpts),
			},
		},
	}
	schemaBytes, _ := json.Marshal(schemaEnv)
	stateBytes, _ := json.Marshal(map[string]string{"state": c.State})

	resp, _, err := cli.Decide(ctx, string(schemaBytes), string(stateBytes))
	wallMs := float64(time.Since(start).Milliseconds())
	if err != nil {
		return DualMirrorResult{}, err
	}

	keys := make([]string, len(c.Options))
	for i, o := range c.Options {
		keys[i] = o.Name
	}

	extractDist := func(slotID string) (map[string]float64, string) {
		ans := resp.Answers[slotID]
		probs := make(map[string]float64, len(keys))
		var sum float64
		for _, k := range keys {
			p := ans.Probabilities[k]
			if p <= 0 && (ans.Choice == k || ans.Label == k) {
				p = 0.90
			} else if p <= 0 {
				p = 0.10 / float64(max(1, len(keys)-1))
			}
			probs[k] = p
			sum += p
		}
		winner := keys[0]
		bestP := -1.0
		for _, k := range keys {
			probs[k] /= sum
			if probs[k] > bestP {
				bestP = probs[k]
				winner = k
			}
		}
		return probs, winner
	}

	fwdProbs, fwdWinner := extractDist("decision_fwd")
	revProbs, revWinner := extractDist("decision_rev")

	ensProbs, ensWinner, meanH, totalH, jsd := ComputeBALDDecomposition([]map[string]float64{fwdProbs, revProbs}, keys)
	_, fwdNormH := ComputeShannonEntropy(fwdProbs, len(keys))
	normTotalH := 0.0
	if len(keys) > 1 {
		normTotalH = totalH / math.Log(float64(len(keys)))
	}
	normJSD := jsd / math.Log(2.0)
	tvd := ComputeTVD(fwdProbs, revProbs, keys)
	disagreed := (fwdWinner != revWinner)

	escSingle := fwdNormH >= gateNormH
	// Dual-Mirror Gate triggers if:
	// 1) Forward and Reversed slots disagree on argmax (MirrorDisagreed), OR
	// 2) Normalized Total Uncertainty H(bar_p)/ln(K) >= gateNormH, OR
	// 3) Normalized JSD I(Y; Pi | X)/ln(2) >= 0.06 (significant logit shift under reversal)
	escMirror := disagreed || (normTotalH >= gateNormH) || (normJSD >= 0.06)

	return DualMirrorResult{
		FwdProbs:          fwdProbs,
		RevProbs:          revProbs,
		EnsembleProbs:     ensProbs,
		FwdWinner:         fwdWinner,
		RevWinner:         revWinner,
		EnsembleWinner:    ensWinner,
		MirrorDisagreed:   disagreed,
		MeanPassEntropy:   meanH,
		TotalEntropy:      totalH,
		NormalizedTotalH:  normTotalH,
		JSDivergence:      jsd,
		NormalizedJSD:     normJSD,
		TVD:               tvd,
		Accurate:          strings.EqualFold(ensWinner, c.Expected),
		EscalatedBySingle: escSingle,
		EscalatedByMirror: escMirror,
		WallTimeMs:        wallMs,
	}, nil
}

// EvaluateFullCase runs all K cyclic shifts + Prior De-biasing + O(1) Dual-Mirror Canvas for one PermutationCase.
func EvaluateFullCase(ctx context.Context, cli *client.Client, c PermutationCase, np *NullPrior, alpha, gateNormH float64) (CasePermutationResult, error) {
	k := len(c.Options)
	keys := make([]string, k)
	for i, o := range c.Options {
		keys[i] = o.Name
	}

	passes := make([]PermPassTelemetry, k)
	rawDists := make([]map[string]float64, k)
	debDists := make([]map[string]float64, k)
	rawWinnerCounts := make(map[string]int)
	debWinnerCounts := make(map[string]int)

	for r := 0; r < k; r++ {
		shifted := CyclicShift(c.Options, r)
		orderNames := make([]string, k)
		for i, o := range shifted {
			orderNames[i] = o.Name
		}
		probs, winner, wallMs, err := EvaluateSinglePermutation(ctx, cli, c, shifted)
		if err != nil {
			return CasePermutationResult{}, err
		}
		h, normH := ComputeShannonEntropy(probs, k)
		debProbs, debWinner := ApplyPriorDeBiasing(probs, shifted, np, alpha)

		passes[r] = PermPassTelemetry{
			PermIndex:      r,
			OptionOrder:    orderNames,
			RawProbs:       probs,
			DebiasedProbs:  debProbs,
			RawWinner:      winner,
			DebiasedWinner: debWinner,
			RawEntropy:     h,
			NormalizedH:    normH,
			WallTimeMs:     wallMs,
		}
		rawDists[r] = probs
		debDists[r] = debProbs
		rawWinnerCounts[winner]++
		debWinnerCounts[debWinner]++
	}

	rawEns, rawEnsWinner, rawMeanH, rawTotalH, rawJSD := ComputeBALDDecomposition(rawDists, keys)
	_, _, _, _, debJSD := ComputeBALDDecomposition(debDists, keys)

	// Compute std dev of single-pass entropy across the K cyclic shifts
	var varH float64
	var maxTVD float64
	for r := 0; r < k; r++ {
		diff := passes[r].RawEntropy - rawMeanH
		varH += diff * diff
		tvd := ComputeTVD(passes[0].RawProbs, passes[r].RawProbs, keys)
		if tvd > maxTVD {
			maxTVD = tvd
		}
	}
	stdH := math.Sqrt(varH / float64(k))

	maxCount := 0
	for _, cnt := range rawWinnerCounts {
		if cnt > maxCount {
			maxCount = cnt
		}
	}

	dm, err := EvaluateDualMirrorSinglePass(ctx, cli, c, gateNormH)
	if err != nil {
		return CasePermutationResult{}, err
	}

	p0 := passes[0]
	return CasePermutationResult{
		ID:                     c.ID,
		Regime:                 c.Regime,
		Title:                  c.Title,
		Cardinality:            k,
		Expected:               c.Expected,
		CanonicalWinner:        p0.RawWinner,
		CanonicalAccurate:      strings.EqualFold(p0.RawWinner, c.Expected),
		CanonicalConfidence:    p0.RawProbs[p0.RawWinner],
		CanonicalEntropy:       p0.RawEntropy,
		CanonicalNormalizedH:   p0.NormalizedH,
		CanonicalBrier:         ComputeBrier(p0.RawProbs, keys, c.Expected),
		CyclicPasses:           passes,
		RawFlipOccurred:        len(rawWinnerCounts) > 1,
		RawDistinctWinners:     len(rawWinnerCounts),
		RawMajorityAgreement:   float64(maxCount) / float64(k),
		RawMeanEntropy:         rawMeanH,
		RawEntropyStdDev:       stdH,
		RawTotalEntropy:        rawTotalH,
		RawJSDivergence:        rawJSD,
		RawMaxTVD:              maxTVD,
		RawEnsembleWinner:      rawEnsWinner,
		RawEnsembleAccurate:    strings.EqualFold(rawEnsWinner, c.Expected),
		RawEnsembleBrier:       ComputeBrier(rawEns, keys, c.Expected),
		DebiasedFlipOccurred:   len(debWinnerCounts) > 1,
		DebiasedJSDivergence:   debJSD,
		DebiasedCanonicalAcc:   strings.EqualFold(p0.DebiasedWinner, c.Expected),
		DebiasedCanonicalBrier: ComputeBrier(p0.DebiasedProbs, keys, c.Expected),
		DualMirror:             dm,
	}, nil
}

// BuildPermutationReport aggregates CasePermutationResults into the complete report.
func BuildPermutationReport(endpoint, model string, gateNormH, alpha float64, np *NullPrior, results []CasePermutationResult) *PermutationReport {
	n := len(results)
	if n == 0 {
		return &PermutationReport{}
	}

	var canAcc, debAcc, dmAcc, ensAcc int
	var rawFlips, debFlips int
	var sumCanBrier, sumDebBrier, sumDmBrier float64
	var sumSingleNormH, sumCyclicJSD, sumMirrorJSD float64
	var sumSingleLat, sumMirrorLat float64
	var totalErrors, singleCaughtErrors, mirrorCaughtErrors int
	var falseConfidentErrors, caughtByMirror int

	byRegime := make(map[string][]CasePermutationResult)

	for _, r := range results {
		byRegime[r.Regime] = append(byRegime[r.Regime], r)
		if r.CanonicalAccurate {
			canAcc++
		} else {
			totalErrors++
			if r.DualMirror.EscalatedBySingle {
				singleCaughtErrors++
			} else {
				// False-confident error in single pass!
				falseConfidentErrors++
				if r.DualMirror.EscalatedByMirror || r.DualMirror.Accurate {
					caughtByMirror++
				}
			}
			if r.DualMirror.EscalatedByMirror || r.DualMirror.Accurate {
				mirrorCaughtErrors++
			}
		}
		if r.DebiasedCanonicalAcc {
			debAcc++
		}
		if r.DualMirror.Accurate {
			dmAcc++
		}
		if r.RawEnsembleAccurate {
			ensAcc++
		}
		if r.RawFlipOccurred {
			rawFlips++
		}
		if r.DebiasedFlipOccurred {
			debFlips++
		}

		keys := make([]string, len(r.CyclicPasses[0].OptionOrder))
		copy(keys, r.CyclicPasses[0].OptionOrder)

		sumCanBrier += r.CanonicalBrier
		sumDebBrier += r.DebiasedCanonicalBrier
		sumDmBrier += ComputeBrier(r.DualMirror.EnsembleProbs, keys, r.Expected)
		sumSingleNormH += r.CanonicalNormalizedH
		sumCyclicJSD += r.RawJSDivergence
		sumMirrorJSD += r.DualMirror.JSDivergence
		sumSingleLat += r.CyclicPasses[0].WallTimeMs
		sumMirrorLat += r.DualMirror.WallTimeMs
	}

	flipRed := 0.0
	if rawFlips > 0 {
		flipRed = float64(rawFlips-debFlips) / float64(rawFlips) * 100.0
	}
	singleErrRec := 100.0
	mirrorErrRec := 100.0
	if totalErrors > 0 {
		singleErrRec = float64(singleCaughtErrors) / float64(totalErrors) * 100.0
		mirrorErrRec = float64(mirrorCaughtErrors) / float64(totalErrors) * 100.0
	}

	regimeOrder := []string{"consensus", "ambiguous_chaosnli", "adversarial_trap", "binary_ablation"}
	var regimes []RegimeSummary
	for _, regKey := range regimeOrder {
		sub := byRegime[regKey]
		if len(sub) == 0 {
			continue
		}
		m := float64(len(sub))
		var cAcc, dAcc, rFlip, dFlip, sErr, sCaught, mCaught int
		var sNormH, cJSD, mJSD, mTVD float64
		for _, item := range sub {
			if item.CanonicalAccurate {
				cAcc++
			} else {
				sErr++
				if item.DualMirror.EscalatedBySingle {
					sCaught++
				}
				if item.DualMirror.EscalatedByMirror || item.DualMirror.Accurate {
					mCaught++
				}
			}
			if item.DualMirror.Accurate {
				dAcc++
			}
			if item.RawFlipOccurred {
				rFlip++
			}
			if item.DebiasedFlipOccurred {
				dFlip++
			}
			sNormH += item.CanonicalNormalizedH
			cJSD += item.RawJSDivergence
			mJSD += item.DualMirror.JSDivergence
			mTVD += item.DualMirror.TVD
		}
		sRec, mRec := 100.0, 100.0
		if sErr > 0 {
			sRec = float64(sCaught) / float64(sErr) * 100.0
			mRec = float64(mCaught) / float64(sErr) * 100.0
		}
		regimes = append(regimes, RegimeSummary{
			Regime:                regKey,
			Cases:                 len(sub),
			CanonicalAccuracyPct:  float64(cAcc) / m * 100.0,
			DualMirrorAccuracyPct: float64(dAcc) / m * 100.0,
			RawFlipRatePct:        float64(rFlip) / m * 100.0,
			DebiasedFlipRatePct:   float64(dFlip) / m * 100.0,
			MeanSingleNormalizedH: sNormH / m,
			MeanCyclicJSD:         cJSD / m,
			MeanMirrorJSD:         mJSD / m,
			MeanMirrorTVD:         mTVD / m,
			SingleGateRecallOnErr: sRec,
			MirrorGateRecallOnErr: mRec,
		})
	}

	sort.Slice(results, func(i, j int) bool { return results[i].ID < results[j].ID })

	fn := float64(n)
	return &PermutationReport{
		Timestamp:              time.Now().UTC().Format(time.RFC3339),
		Endpoint:               endpoint,
		Model:                  model,
		GateThresholdNormH:     gateNormH,
		PriorDampingAlpha:      alpha,
		NullPrior:              np,
		TotalCases:             n,
		CanonicalAccuracyPct:   float64(canAcc) / fn * 100.0,
		DebiasedAccuracyPct:    float64(debAcc) / fn * 100.0,
		DualMirrorAccuracyPct:  float64(dmAcc) / fn * 100.0,
		CyclicEnsembleAccPct:   float64(ensAcc) / fn * 100.0,
		RawFlipRatePct:         float64(rawFlips) / fn * 100.0,
		DebiasedFlipRatePct:    float64(debFlips) / fn * 100.0,
		FlipReductionRelPct:    flipRed,
		CanonicalBrier:         sumCanBrier / fn,
		DebiasedBrier:          sumDebBrier / fn,
		DualMirrorBrier:        sumDmBrier / fn,
		MeanSinglePassNormH:    sumSingleNormH / fn,
		MeanCyclicJSD:          sumCyclicJSD / fn,
		MeanMirrorJSD:          sumMirrorJSD / fn,
		FalseConfidentErrors:   falseConfidentErrors,
		CaughtByDualMirror:     caughtByMirror,
		SingleGateErrRecallPct: singleErrRec,
		MirrorGateErrRecallPct: mirrorErrRec,
		MeanSingleLatencyMs:    sumSingleLat / fn,
		MeanMirrorLatencyMs:    sumMirrorLat / fn,
		Regimes:                regimes,
		Cases:                  results,
	}
}

// DefaultCalibratedNullPriors returns the empirically measured content-free positional null priors
// p_0(k) from on Cloud Run DiffusionGemma (plus analytical primacy-recency interpolation for K >= 5).
func DefaultCalibratedNullPriors() NullPrior {
	byCard := map[int][]float64{
		2: {0.8832, 0.1168},
		3: {0.7834, 0.0859, 0.1307},
		4: {0.4932, 0.0484, 0.1630, 0.2954},
	}
	for k := 5; k <= 26; k++ {
		vec := make([]float64, k)
		var sum float64
		for i := 0; i < k; i++ {
			// Primacy on slot 0 + slight recency on slot K-1
			w := 1.0
			if i == 0 {
				w = 2.4
			} else if i == k-1 {
				w = 1.35
			}
			vec[i] = w
			sum += w
		}
		for i := range vec {
			vec[i] /= sum
		}
		byCard[k] = vec
	}
	return NullPrior{ByCardinality: byCard}
}

// ExtractSchemaSlotOptions parses a dgem JSON schema and returns the ordered option items for each choice/boolean slot.
func ExtractSchemaSlotOptions(schemaJSON string) map[string][]OptionItem {
	out := make(map[string][]OptionItem)
	var schemaMap map[string]any
	if err := json.Unmarshal([]byte(schemaJSON), &schemaMap); err != nil {
		return out
	}
	qRaw, ok := schemaMap["questions"].([]any)
	if !ok {
		return out
	}
	for _, item := range qRaw {
		qMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		qID, _ := qMap["id"].(string)
		qType, _ := qMap["type"].(string)
		if qID == "" {
			continue
		}
		switch qType {
		case "boolean":
			out[qID] = []OptionItem{
				{Name: "yes", Description: "Affirmative / True"},
				{Name: "no", Description: "Negative / False"},
			}
		case "choice":
			optsRaw, ok := qMap["options"].([]any)
			if !ok {
				continue
			}
			var opts []OptionItem
			for _, oItem := range optsRaw {
				if oMap, ok := oItem.(map[string]any); ok {
					name, _ := oMap["name"].(string)
					desc, _ := oMap["description"].(string)
					if name != "" {
						opts = append(opts, OptionItem{Name: name, Description: desc})
					}
				} else if oStr, ok := oItem.(string); ok && oStr != "" {
					opts = append(opts, OptionItem{Name: oStr})
				}
			}
			if len(opts) >= 2 {
				out[qID] = opts
			}
		}
	}
	return out
}

// InjectDualMirrorSchema transforms a dgem JSON schema by adding a companion reversed-option slot
// (`<id>` + MirrorSlotSuffix, default `<id>__rev`) for every choice/boolean slot so both forward and reversed option orderings
// are evaluated simultaneously on the SAME bidirectional diffusion canvas (reads=1).
// MirrorAliasNames controls how the reversed mirror slot names its options. When true (the historical
// default), options are renamed item_1..item_K with "name: description" text. When false, the reversed
// slot keeps the real option names and descriptions, matching the bench-permutation setup.
var MirrorAliasNames = true

// MirrorSlotSuffix is appended to a question id to name its reversed mirror slot. Slot ids are visible to the
// model: the historical suffix "__mirror_rev" measurably degraded both the mirror and the forward readings on
// the Vertex endpoint, while neutral suffixes such as "__rev" did not. Kept configurable so older
// receipts can be reproduced with --mirror-slot-suffix=__mirror_rev.
var MirrorSlotSuffix = "__rev"

// MirrorMode selects how the second slot is built:
//
//	reversed        reversed options, lettered choice (historical default; letters collide)
//	copy            identical copy of the question in the same order (letters mean the same options)
//	reversed-digits reversed options sent as a score question, so the server labels them 1, 2, 3...
//	reversed-first  reversed lettered choice placed before the forward question
var MirrorMode = "reversed"

// mirrorSlotSpec returns the mirror slot's options (in presentation order), the answer keys the server
// will report for each, and the slot type.
func mirrorSlotSpec(opts []OptionItem) ([]OptionItem, []string, string) {
	ordered := ReverseOptions(opts)
	if MirrorMode == "copy" {
		ordered = append([]OptionItem(nil), opts...)
	}
	keys := make([]string, len(ordered))
	if MirrorMode == "reversed-digits" && len(ordered) <= 9 {
		for i, o := range ordered {
			if o.Description != "" && o.Description != o.Name {
				keys[i] = o.Name + ": " + o.Description
			} else {
				keys[i] = o.Name
			}
		}
		return ordered, keys, "score"
	}
	for i, o := range ordered {
		if MirrorMode == "copy" || !MirrorAliasNames {
			keys[i] = o.Name
		} else {
			keys[i] = fmt.Sprintf("item_%d", i+1)
		}
	}
	return ordered, keys, "choice"
}

func mirrorDescription(o OptionItem, key string) string {
	desc := o.Description
	if desc == "" {
		return o.Name
	}
	if key != o.Name && len(o.Name) > 1 && !strings.HasPrefix(strings.ToLower(o.Name), "opt_") {
		return o.Name + ": " + desc
	}
	return desc
}

// DualMirrorLetterWarning is the message emitted when --dual-mirror mirrors a lettered choice question.
const DualMirrorLetterWarning = "--dual-mirror on lettered choice questions: the forward and reversed slots share letters (A, B, C...) " +
	"that mean different options, and the model can copy the letter across slots, changing the forward answer. " +
	"Research use only."

var (
	letterCollisionSeen atomic.Bool
	letterWarnOnce      sync.Once
	// WarnWriter receives the one-time dual-mirror warning (stderr by default; set to io.Discard to silence).
	WarnWriter io.Writer = os.Stderr
)

// DualMirrorLetterCollisionSeen reports whether any schema in this process mirrored a lettered choice question.
func DualMirrorLetterCollisionSeen() bool { return letterCollisionSeen.Load() }

func noteLetterCollision() {
	letterCollisionSeen.Store(true)
	letterWarnOnce.Do(func() { fmt.Fprintln(WarnWriter, "WARNING: "+DualMirrorLetterWarning) })
}

func InjectDualMirrorSchema(schemaJSON string) (string, map[string][]OptionItem, bool) {
	slotOpts := ExtractSchemaSlotOptions(schemaJSON)
	if len(slotOpts) == 0 || len(slotOpts) > 5 {
		return schemaJSON, slotOpts, false
	}
	var schemaMap map[string]any
	if err := json.Unmarshal([]byte(schemaJSON), &schemaMap); err != nil {
		return schemaJSON, slotOpts, false
	}
	qRaw, ok := schemaMap["questions"].([]any)
	if !ok {
		return schemaJSON, slotOpts, false
	}
	var newQuestions []any
	injected := false
	for _, item := range qRaw {
		newQuestions = append(newQuestions, item)
		qMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		qID, _ := qMap["id"].(string)
		qType, _ := qMap["type"].(string)
		opts, hasOpts := slotOpts[qID]
		if !hasOpts || len(opts) < 2 {
			continue
		}
		// Skip if question has depends_on / ask_if to preserve single-pass level-0 scheduling
		if _, hasDep := qMap["depends_on"]; hasDep {
			continue
		}
		if qType == "boolean" || qType == "choice" {
			mOpts, mKeys, mType := mirrorSlotSpec(opts)
			instr, _ := qMap["instructions"].(string)
			mirrorQ := map[string]any{
				"id":           qID + MirrorSlotSuffix,
				"type":         mType,
				"instructions": instr,
			}
			if mType == "score" {
				mirrorQ["levels"] = mKeys
			} else {
				objs := make([]map[string]string, len(mOpts))
				for i, mo := range mOpts {
					objs[i] = map[string]string{"name": mKeys[i], "description": mirrorDescription(mo, mKeys[i])}
				}
				mirrorQ["options"] = objs
			}
			if MirrorMode == "reversed-first" {
				newQuestions = append(newQuestions[:len(newQuestions)-1], mirrorQ, item)
			} else {
				newQuestions = append(newQuestions, mirrorQ)
			}
			injected = true
			// Boolean slots are labelled yes/no and digit/copy modes do not reuse letters with other meanings.
			if qType == "choice" && (MirrorMode == "reversed" || MirrorMode == "reversed-first") {
				noteLetterCollision()
			}
		}
	}
	if !injected {
		return schemaJSON, slotOpts, false
	}
	schemaMap["questions"] = newQuestions
	updatedBytes, err := json.Marshal(schemaMap)
	if err != nil {
		return schemaJSON, slotOpts, false
	}
	return string(updatedBytes), slotOpts, true
}

// SlotIDCDetail records the raw and post-processed distributions for one slot so that receipts can be
// re-scored offline (merge rules, prior strength, mirror gates) without re-running the model.
type SlotIDCDetail struct {
	RawFwdProbs map[string]float64 `json:"raw_fwd_probs,omitempty"`
	RawRevProbs map[string]float64 `json:"raw_rev_probs,omitempty"`
	FwdProbs    map[string]float64 `json:"fwd_probs,omitempty"`
	RevProbs    map[string]float64 `json:"rev_probs,omitempty"`
	MirrorTVD   float64            `json:"mirror_tvd"`
	MirrorJSD   float64            `json:"mirror_jsd"`
	HasMirror   bool               `json:"has_mirror"`
}

// PostProcessDecisionResponse applies Null-Prior De-Biasing and/or O(1) Dual-Mirror Canvas
// distribution merging in-place on a StructuredDecisionResponse, returning per-slot Mirror TVD and Mirror JSD.
func PostProcessDecisionResponse(
	resp *client.StructuredDecisionResponse,
	slotOpts map[string][]OptionItem,
	enableDualMirror bool,
	enableNullPrior bool,
	alpha float64,
) (map[string]float64, map[string]float64) {
	details := PostProcessDecisionResponseDetailed(resp, slotOpts, enableDualMirror, enableNullPrior, alpha)
	mirrorTVD := make(map[string]float64)
	mirrorJSD := make(map[string]float64)
	for q, d := range details {
		if d.HasMirror {
			mirrorTVD[q] = d.MirrorTVD
			mirrorJSD[q] = d.MirrorJSD
		}
	}
	return mirrorTVD, mirrorJSD
}

// PostProcessDecisionResponseDetailed is PostProcessDecisionResponse plus per-slot raw/processed distributions.
func PostProcessDecisionResponseDetailed(
	resp *client.StructuredDecisionResponse,
	slotOpts map[string][]OptionItem,
	enableDualMirror bool,
	enableNullPrior bool,
	alpha float64,
) map[string]SlotIDCDetail {
	details := make(map[string]SlotIDCDetail)
	if resp == nil || len(resp.Answers) == 0 || len(slotOpts) == 0 {
		return details
	}
	if alpha <= 0 {
		alpha = 0.50 // Conservative default damping factor so weak priors don't over-invert
	}
	np := DefaultCalibratedNullPriors()

	for qID, opts := range slotOpts {
		qaFwd, hasFwd := resp.Answers[qID]
		if !hasFwd || len(opts) < 2 {
			continue
		}
		optKeys := make([]string, len(opts))
		for i, o := range opts {
			optKeys[i] = o.Name
		}

		probsFwd := normalizeAnswerProbs(qaFwd, opts)
		det := SlotIDCDetail{RawFwdProbs: copyProbs(probsFwd)}
		if enableNullPrior {
			probsFwd, _ = ApplyPriorDeBiasing(probsFwd, opts, &np, alpha)
		}
		det.FwdProbs = copyProbs(probsFwd)

		finalProbs := probsFwd
		revID := qID + MirrorSlotSuffix
		if enableDualMirror {
			if qaRev, hasRev := resp.Answers[revID]; hasRev {
				revOpts, mKeys, _ := mirrorSlotSpec(opts)
				aliasOpts := make([]OptionItem, len(revOpts))
				for i, ro := range revOpts {
					aliasOpts[i] = OptionItem{Name: mKeys[i], Description: ro.Description}
				}
				probsAlias := normalizeAnswerProbs(qaRev, aliasOpts)
				probsRev := make(map[string]float64, len(revOpts))
				for i, ro := range revOpts {
					probsRev[ro.Name] = probsAlias[aliasOpts[i].Name]
				}
				det.RawRevProbs = copyProbs(probsRev)
				if enableNullPrior {
					probsRev, _ = ApplyPriorDeBiasing(probsRev, revOpts, &np, alpha)
				}
				det.RevProbs = copyProbs(probsRev)
				_, _, _, _, mJSD := ComputeBALDDecomposition([]map[string]float64{probsFwd, probsRev}, optKeys)
				mTVD := ComputeTVD(probsFwd, probsRev, optKeys)

				fwdWinner := optKeys[0]
				for _, k := range optKeys {
					if probsFwd[k] > probsFwd[fwdWinner] {
						fwdWinner = k
					}
				}
				revWinner := optKeys[0]
				for _, k := range optKeys {
					if probsRev[k] > probsRev[revWinner] {
						revWinner = k
					}
				}

				// Determine winner:
				// 1. If forward slot picked a non-first option (k > 0), it already overcame Slot-0 ('A') primacy bias -> keep fwdWinner.
				// 2. Only allow reverse slot to flip fwdWinner if fwdWinner was Slot 0 ('A') AND revWinner is an interior
				//    option (not reverse slot's own Slot 0, which is optKeys[len-1]) with strong confidence.
				chosenWinner := fwdWinner
				lastKey := optKeys[len(optKeys)-1]
				if fwdWinner == optKeys[0] && revWinner != optKeys[0] && revWinner != lastKey && probsRev[revWinner] > 0.75 && probsRev[optKeys[0]] < 0.05 {
					chosenWinner = revWinner
				}

				// Blend distributions (70% forward + 30% mirror) so MirrorTVD/MirrorJSD softens overconfidence
				// and raises Shannon entropy on permutation-sensitive items while keeping chosenWinner as argmax.
				comb := make(map[string]float64, len(optKeys))
				var sum float64
				for _, k := range optKeys {
					comb[k] = 0.70*probsFwd[k] + 0.30*probsRev[k]
					sum += comb[k]
				}
				for _, k := range optKeys {
					comb[k] /= sum
				}
				// Ensure chosenWinner remains strict argmax
				maxOther := 0.0
				for _, k := range optKeys {
					if k != chosenWinner && comb[k] > maxOther {
						maxOther = comb[k]
					}
				}
				if comb[chosenWinner] <= maxOther {
					comb[chosenWinner] = maxOther + 0.02
					sum = 0
					for _, k := range optKeys {
						sum += comb[k]
					}
					for _, k := range optKeys {
						comb[k] /= sum
					}
				}

				finalProbs = comb
				det.MirrorTVD = mTVD
				det.MirrorJSD = mJSD
				det.HasMirror = true
				delete(resp.Answers, revID)
			}
		}

		// Update qaFwd in-place with the calibrated distribution
		winner := optKeys[0]
		bestP := -1.0
		for _, k := range optKeys {
			if finalProbs[k] > bestP {
				bestP = finalProbs[k]
				winner = k
			}
		}
		h, _ := ComputeShannonEntropy(finalProbs, len(optKeys))
		qaFwd.Probabilities = finalProbs
		qaFwd.Confidence = bestP
		qaFwd.Entropy = h
		if qaFwd.Choice != "" || qaFwd.Label == "" {
			qaFwd.Choice = winner
		}
		if qaFwd.Label != "" {
			qaFwd.Label = winner
		}
		if len(optKeys) == 2 && (optKeys[0] == "yes" || optKeys[0] == "true") {
			qaFwd.Noul = finalProbs[optKeys[0]]
		}
		resp.Answers[qID] = qaFwd
		details[qID] = det
	}
	return details
}

func copyProbs(in map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func normalizeAnswerProbs(qa client.QuestionAnswer, orderedOpts []OptionItem) map[string]float64 {
	out := make(map[string]float64, len(orderedOpts))
	var sum float64
	chosen := strings.TrimSpace(qa.Choice)
	if chosen == "" {
		chosen = strings.TrimSpace(qa.Label)
	}
	for _, o := range orderedOpts {
		p := qa.Probabilities[o.Name]
		if p <= 0 && o.Name == "yes" && qa.Noul > 0 {
			p = qa.Noul
		} else if p <= 0 && o.Name == "no" && qa.Noul > 0 {
			p = 1.0 - qa.Noul
		}
		if p <= 0 && strings.EqualFold(chosen, o.Name) {
			p = 0.90
		} else if p <= 0 {
			p = 0.10 / float64(max(1, len(orderedOpts)-1))
		}
		out[o.Name] = p
		sum += p
	}
	if sum > 0 {
		for k := range out {
			out[k] /= sum
		}
	}
	return out
}
