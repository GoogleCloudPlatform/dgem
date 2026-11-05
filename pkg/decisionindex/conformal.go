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
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"
)

// ConformalMethod identifies the nonconformity score family used for split conformal prediction.
type ConformalMethod string

const (
	// ConformalLAC uses Least Ambiguous Set Classifiers nonconformity s(x, y) = 1 - p(y|x).
	ConformalLAC ConformalMethod = "lac"
	// ConformalAPS uses Adaptive Prediction Sets cumulative mass of strictly higher-ranked tie groups.
	ConformalAPS ConformalMethod = "aps"
	// ConformalRAPS adds rank-thresholded size regularization above KReg to APS.
	ConformalRAPS ConformalMethod = "raps"
)

const (
	// ContextActionFull retains the complete trajectory block verbatim.
	ContextActionFull = "full"
	// ContextActionStub replaces the block body with its compact summary or signature stub.
	ContextActionStub = "stub"
	// ContextActionDrop omits the block from the active context window.
	ContextActionDrop = "drop"
)

const (
	conformalScoreTolerance   = 1e-12
	maxSeededConformalStreams = 256
	// DefaultConformalAlpha is the standard split-conformal target miscoverage rate (10%).
	DefaultConformalAlpha = 0.10
	// DefaultConformalKReg is the standard RAPS unpenalized top-rank cutoff.
	DefaultConformalKReg = 1
	// DefaultConformalLambdaReg is the standard RAPS per-rank size penalty weight.
	DefaultConformalLambdaReg = 0.01
)

var (
	conformalRNGMu      sync.Mutex
	defaultConformalRNG = rand.New(rand.NewSource(time.Now().UnixNano()))
	seededConformalRNGs = make(map[int64]*rand.Rand)
	seededConformalKeys []int64
)

var defaultAbstainLabels = []string{
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

var (
	// ErrEmptyCalibration is returned when an empty calibration slice is supplied.
	ErrEmptyCalibration = errors.New("decisionindex: calibration set must be non-empty")
	// ErrCalibrationLenMismatch is returned when calibration probability maps and gold labels differ in length.
	ErrCalibrationLenMismatch = errors.New("decisionindex: calibration probabilities and labels length mismatch")
	// ErrInvalidAlpha is returned when alpha lies outside the open interval (0, 1).
	ErrInvalidAlpha = errors.New("decisionindex: alpha must be strictly inside (0, 1)")
	// ErrUnknownCalibrationKey is returned when a gold calibration label is missing from its probability map.
	ErrUnknownCalibrationKey = errors.New("decisionindex: gold calibration label missing from probability map")
	// ErrInvalidConformalMethod is returned when ConformalConfig.Method is not recognized.
	ErrInvalidConformalMethod = errors.New("decisionindex: unknown conformal method")
	// ErrTooManyContextBlocks is returned when more than MaxSlotsPerPass (8) blocks are passed for single-pass pruning.
	ErrTooManyContextBlocks = errors.New("decisionindex: context block count exceeds MaxSlotsPerPass (8)")
	// ErrNegativeTokenBudget is returned when ContextPruneConfig.TokenBudget is negative.
	ErrNegativeTokenBudget = errors.New("decisionindex: token budget must be non-negative")
	// ErrInvalidProbabilities is returned when a probability map is empty, contains NaN/Inf/negative values, or sums to <= 0.
	ErrInvalidProbabilities = errors.New("decisionindex: probability map must be non-empty with finite non-negative values summing to > 0")
	// ErrInvalidConformalQHat is returned when qHat is negative, NaN, or -Inf.
	ErrInvalidConformalQHat = errors.New("decisionindex: qHat must be non-negative and non-NaN")
	// ErrInvalidConformalParam is returned when RAPS KReg or LambdaReg is invalid.
	ErrInvalidConformalParam = errors.New("decisionindex: invalid conformal regularization parameter")
	// ErrInvalidContextBlock is returned when a ContextBlockSpec has negative/inverted token counts or duplicate IDs.
	ErrInvalidContextBlock = errors.New("decisionindex: invalid context block specification")
	// ErrInfeasiblePinnedBudget is returned when TokenBudget is smaller than the minimum stub token sum of pinned blocks.
	ErrInfeasiblePinnedBudget = errors.New("decisionindex: token budget below minimum pinned block cost")
)

// ConformalConfig configures split conformal score computation and prediction-set construction.
// Use DefaultConformalConfig() (or zero-value ConformalConfig{}) for standard RAPS defaults
// (Method: ConformalRAPS, Alpha: 0.10, KReg: 1, LambdaReg: 0.01). When Method is explicitly set
// to ConformalRAPS, KReg == 0 and LambdaReg == 0.0 are honored literally without override.
type ConformalConfig struct {
	// Method selects "lac", "aps", or "raps" (defaults to "raps" when empty).
	Method ConformalMethod `json:"method,omitempty"`
	// Alpha stores the target miscoverage rate in (0, 1) (populated as 0.10 by DefaultConformalConfig).
	Alpha float64 `json:"alpha,omitempty"`
	// KReg is the unpenalized top-rank cutoff k_reg >= 0 for RAPS (0 penalizes starting at rank 1).
	KReg int `json:"k_reg,omitempty"`
	// LambdaReg is the per-rank penalty weight lambda_reg >= 0 for RAPS (0.0 reduces RAPS to APS).
	LambdaReg float64 `json:"lambda_reg,omitempty"`
	// Randomized enables Uniform(0, 1) boundary smoothing and tie-group Fisher-Yates shuffling for APS and RAPS.
	Randomized bool `json:"randomized,omitempty"`
	// Seed selects a stateful seeded RNG stream across CalibrateConformalQuantile and ConformalPredictionSet when Randomized is true and Seed != 0.
	Seed int64 `json:"seed,omitempty"`
	// RNG supplies an explicit caller-managed *rand.Rand stream when Randomized is true.
	RNG *rand.Rand `json:"-"`
	// AbstainLabels lists option names treated as abstention triggers when included in the prediction set.
	// Defaults to {"0", "abstain", "insufficient_evidence", "none", "none_of_the_above",
	// "none_visible", "oos", "uncertain", "unknown"} when nil.
	AbstainLabels []string `json:"abstain_labels,omitempty"`
}

// DefaultConformalConfig returns the standard RAPS configuration
// (Method: ConformalRAPS, Alpha: 0.10, KReg: 1, LambdaReg: 0.01).
func DefaultConformalConfig() ConformalConfig {
	return ConformalConfig{
		Method:    ConformalRAPS,
		Alpha:     DefaultConformalAlpha,
		KReg:      DefaultConformalKReg,
		LambdaReg: DefaultConformalLambdaReg,
	}
}

// ConformalCalibrationResult records the calibrated split-conformal quantile and calibration summary.
type ConformalCalibrationResult struct {
	Method         ConformalMethod `json:"method"`
	Alpha          float64         `json:"alpha"`
	NumCalibration int             `json:"num_calibration"`
	OrderIndex     int             `json:"order_index"`
	QHat           float64         `json:"q_hat"`
	EmpiricalCover float64         `json:"empirical_coverage"`
	SortedScores   []float64       `json:"sorted_scores,omitempty"`
}

// ConformalSetResult holds the calibrated prediction set and cascade escalation decision for one slot.
type ConformalSetResult struct {
	PredictionSet   []string           `json:"prediction_set"`
	Scores          map[string]float64 `json:"scores"`
	QHat            float64            `json:"q_hat"`
	TopOption       string             `json:"top_option"`
	TopProbability  float64            `json:"top_probability"`
	SetSize         int                `json:"set_size"`
	Singleton       bool               `json:"singleton"`
	ContainsAbstain bool               `json:"contains_abstain"`
	// ShouldEscalate is true iff SetSize != 1 || ContainsAbstain.
	ShouldEscalate bool `json:"should_escalate"`
}

// QuestionResult carries post-fusion and optional pre-prior-fusion option probabilities
// so conformal escalation gates can inspect raw likelihoods before caller prior fusion.
type QuestionResult struct {
	Choice           string             `json:"choice,omitempty"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RawProbabilities map[string]float64 `json:"raw_probabilities,omitempty"`
	PriorFused       bool               `json:"prior_fused,omitempty"`
}

type optionProb struct {
	name string
	prob float64
}

func parseConformalMethod(m ConformalMethod) (ConformalMethod, error) {
	norm := strings.ToLower(strings.TrimSpace(string(m)))
	switch norm {
	case "", string(ConformalRAPS):
		return ConformalRAPS, nil
	case string(ConformalLAC):
		return ConformalLAC, nil
	case string(ConformalAPS):
		return ConformalAPS, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrInvalidConformalMethod, m)
	}
}

func validateConformalConfig(cfg ConformalConfig) (ConformalMethod, int, float64, error) {
	rawTrimmed := strings.TrimSpace(string(cfg.Method))
	method, err := parseConformalMethod(cfg.Method)
	if err != nil {
		return "", 0, 0, err
	}
	if cfg.KReg < 0 {
		return "", 0, 0, fmt.Errorf("%w: KReg=%d must be >= 0", ErrInvalidConformalParam, cfg.KReg)
	}
	if math.IsNaN(cfg.LambdaReg) || math.IsInf(cfg.LambdaReg, 0) || cfg.LambdaReg < 0 {
		return "", 0, 0, fmt.Errorf("%w: LambdaReg=%v must be finite and >= 0", ErrInvalidConformalParam, cfg.LambdaReg)
	}
	kReg := cfg.KReg
	lamReg := cfg.LambdaReg
	if rawTrimmed == "" && kReg == 0 && lamReg == 0 {
		kReg = DefaultConformalKReg
		lamReg = DefaultConformalLambdaReg
	}
	return method, kReg, lamReg, nil
}

func normalizeOrderedProbs(probs map[string]float64) ([]optionProb, error) {
	if len(probs) == 0 {
		return nil, fmt.Errorf("%w: empty map", ErrInvalidProbabilities)
	}
	keys := make([]string, 0, len(probs))
	for k := range probs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	cleaned := make([]optionProb, len(keys))
	sum := 0.0
	for i, k := range keys {
		p := probs[k]
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 {
			return nil, fmt.Errorf("%w: key %q has value %v", ErrInvalidProbabilities, k, p)
		}
		cleaned[i] = optionProb{name: k, prob: p}
		sum += p
	}
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return nil, fmt.Errorf("%w: total probability mass=%v", ErrInvalidProbabilities, sum)
	}
	for i := range cleaned {
		cleaned[i].prob /= sum
	}

	sort.Slice(cleaned, func(i, j int) bool {
		if cleaned[i].prob != cleaned[j].prob {
			return cleaned[i].prob > cleaned[j].prob
		}
		return cleaned[i].name < cleaned[j].name
	})
	return cleaned, nil
}

func partitionOptionTieGroups(ordered []optionProb) [][]optionProb {
	var groups [][]optionProb
	for idx := 0; idx < len(ordered); {
		groupProb := ordered[idx].prob
		end := idx + 1
		for end < len(ordered) && math.Abs(ordered[end].prob-groupProb) <= conformalScoreTolerance {
			end++
		}
		groups = append(groups, ordered[idx:end])
		idx = end
	}
	return groups
}

func touchSeededKeyLocked(seed int64) {
	for i, k := range seededConformalKeys {
		if k == seed {
			seededConformalKeys = append(seededConformalKeys[:i], seededConformalKeys[i+1:]...)
			break
		}
	}
	seededConformalKeys = append(seededConformalKeys, seed)
	for len(seededConformalKeys) > maxSeededConformalStreams {
		evict := seededConformalKeys[0]
		seededConformalKeys = seededConformalKeys[1:]
		delete(seededConformalRNGs, evict)
	}
}

func storeSeededConformalRNG(seed int64, r *rand.Rand) {
	conformalRNGMu.Lock()
	defer conformalRNGMu.Unlock()
	seededConformalRNGs[seed] = r
	touchSeededKeyLocked(seed)
}

func drawWithConformalRNG(r *rand.Rand, groups [][]optionProb, total int) (float64, []optionProb) {
	u := r.Float64()
	flat := make([]optionProb, 0, total)
	for _, grp := range groups {
		if len(grp) > 1 {
			cp := make([]optionProb, len(grp))
			copy(cp, grp)
			r.Shuffle(len(cp), func(i, j int) { cp[i], cp[j] = cp[j], cp[i] })
			flat = append(flat, cp...)
		} else {
			flat = append(flat, grp[0])
		}
	}
	return u, flat
}

func drawInferenceOrderAndU(groups [][]optionProb, total int, cfg ConformalConfig) (float64, []optionProb) {
	if cfg.RNG != nil {
		return drawWithConformalRNG(cfg.RNG, groups, total)
	}
	conformalRNGMu.Lock()
	defer conformalRNGMu.Unlock()
	if cfg.Seed == 0 {
		return drawWithConformalRNG(defaultConformalRNG, groups, total)
	}
	r, ok := seededConformalRNGs[cfg.Seed]
	if !ok {
		r = rand.New(rand.NewSource(cfg.Seed))
		seededConformalRNGs[cfg.Seed] = r
	}
	touchSeededKeyLocked(cfg.Seed)
	return drawWithConformalRNG(r, groups, total)
}

func computeDeterministicScores(
	ordered []optionProb,
	method ConformalMethod,
	kReg int,
	lamReg float64,
) map[string]float64 {
	scores := make(map[string]float64, len(ordered))
	if len(ordered) == 0 {
		return scores
	}
	if method == ConformalLAC {
		for _, opt := range ordered {
			scores[opt.name] = 1.0 - opt.prob
		}
		return scores
	}

	groups := partitionOptionTieGroups(ordered)
	cumStrictlyGreater := 0.0
	strictCount := 0
	for _, grp := range groups {
		groupStartRank1 := strictCount + 1
		penalty := 0.0
		if method == ConformalRAPS && groupStartRank1 > kReg {
			penalty = lamReg * float64(groupStartRank1-kReg)
		}
		grpScore := cumStrictlyGreater + penalty
		groupMass := 0.0
		for _, opt := range grp {
			scores[opt.name] = grpScore
			groupMass += opt.prob
		}
		cumStrictlyGreater += groupMass
		strictCount += len(grp)
	}
	return scores
}

func computeRandomizedScoresFromDraw(
	flat []optionProb,
	u float64,
	method ConformalMethod,
	kReg int,
	lamReg float64,
) map[string]float64 {
	scores := make(map[string]float64, len(flat))
	cumBefore := 0.0
	for idx, opt := range flat {
		rank1 := idx + 1
		switch method {
		case ConformalLAC:
			scores[opt.name] = 1.0 - opt.prob
		case ConformalAPS:
			scores[opt.name] = cumBefore + u*opt.prob
		default:
			pen := 0.0
			if rank1 > kReg {
				pen = lamReg * float64(rank1-kReg)
			}
			scores[opt.name] = cumBefore + u*opt.prob + pen
		}
		cumBefore += opt.prob
	}
	return scores
}

func computeInferenceScores(
	ordered []optionProb,
	method ConformalMethod,
	kReg int,
	lamReg float64,
	cfg ConformalConfig,
) map[string]float64 {
	if !cfg.Randomized {
		return computeDeterministicScores(ordered, method, kReg, lamReg)
	}
	groups := partitionOptionTieGroups(ordered)
	u, flat := drawInferenceOrderAndU(groups, len(ordered), cfg)
	return computeRandomizedScoresFromDraw(flat, u, method, kReg, lamReg)
}

// ConformalScore computes the nonconformity score s(x, label) for one probability map and candidate label.
func ConformalScore(probs map[string]float64, label string, cfg ConformalConfig) (float64, error) {
	method, kReg, lamReg, err := validateConformalConfig(cfg)
	if err != nil {
		return 0, err
	}
	ordered, err := normalizeOrderedProbs(probs)
	if err != nil {
		return 0, err
	}
	if _, ok := probs[label]; !ok {
		return 0, fmt.Errorf("%w: %q", ErrUnknownCalibrationKey, label)
	}
	scores := computeInferenceScores(ordered, method, kReg, lamReg, cfg)
	return scores[label], nil
}

// CalibrateConformalQuantile computes the finite-sample split conformal quantile
// at order index k = ceil((n + 1) * (1 - alpha) - 1e-12) over held-out calibration examples.
func CalibrateConformalQuantile(
	calibProbs []map[string]float64,
	calibLabels []string,
	alpha float64,
	cfg ConformalConfig,
) (ConformalCalibrationResult, error) {
	if len(calibProbs) == 0 {
		return ConformalCalibrationResult{}, ErrEmptyCalibration
	}
	if len(calibProbs) != len(calibLabels) {
		return ConformalCalibrationResult{}, fmt.Errorf(
			"%w: got %d probability maps and %d labels",
			ErrCalibrationLenMismatch, len(calibProbs), len(calibLabels),
		)
	}
	if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha <= 0 || alpha >= 1 {
		return ConformalCalibrationResult{}, fmt.Errorf("%w: got %v", ErrInvalidAlpha, alpha)
	}
	method, kReg, lamReg, err := validateConformalConfig(cfg)
	if err != nil {
		return ConformalCalibrationResult{}, err
	}

	n := len(calibProbs)
	orderedItems := make([][]optionProb, n)
	for i := 0; i < n; i++ {
		ordered, nErr := normalizeOrderedProbs(calibProbs[i])
		if nErr != nil {
			return ConformalCalibrationResult{}, fmt.Errorf("calibration item %d: %w", i, nErr)
		}
		lbl := calibLabels[i]
		if _, ok := calibProbs[i][lbl]; !ok {
			return ConformalCalibrationResult{}, fmt.Errorf("%w at index %d: %q", ErrUnknownCalibrationKey, i, lbl)
		}
		orderedItems[i] = ordered
	}

	var calRNG *rand.Rand
	useGlobalDefaultRNG := false
	if cfg.Randomized {
		if cfg.RNG != nil {
			calRNG = cfg.RNG
		} else if cfg.Seed != 0 {
			calRNG = rand.New(rand.NewSource(cfg.Seed))
		} else {
			useGlobalDefaultRNG = true
		}
	}

	rawScores := make([]float64, n)
	for i := 0; i < n; i++ {
		ordered := orderedItems[i]
		var sMap map[string]float64
		if !cfg.Randomized {
			sMap = computeDeterministicScores(ordered, method, kReg, lamReg)
		} else {
			groups := partitionOptionTieGroups(ordered)
			var u float64
			var flat []optionProb
			if useGlobalDefaultRNG {
				conformalRNGMu.Lock()
				u, flat = drawWithConformalRNG(defaultConformalRNG, groups, len(ordered))
				conformalRNGMu.Unlock()
			} else {
				u, flat = drawWithConformalRNG(calRNG, groups, len(ordered))
			}
			sMap = computeRandomizedScoresFromDraw(flat, u, method, kReg, lamReg)
		}
		rawScores[i] = sMap[calibLabels[i]]
	}

	if cfg.Randomized && cfg.RNG == nil && cfg.Seed != 0 && calRNG != nil {
		storeSeededConformalRNG(cfg.Seed, calRNG)
	}

	sorted := append([]float64(nil), rawScores...)
	sort.Float64s(sorted)

	k := int(math.Ceil(float64(n+1)*(1.0-alpha) - conformalScoreTolerance))
	if k < 1 {
		k = 1
	}

	var qHat float64
	orderIdx := k
	if k > n {
		orderIdx = n + 1
		qHat = math.Inf(1)
	} else {
		qHat = sorted[k-1]
	}

	covered := 0
	for _, s := range rawScores {
		if math.IsInf(qHat, 1) || s <= qHat+conformalScoreTolerance {
			covered++
		}
	}

	return ConformalCalibrationResult{
		Method:         method,
		Alpha:          alpha,
		NumCalibration: n,
		OrderIndex:     orderIdx,
		QHat:           qHat,
		EmpiricalCover: float64(covered) / float64(n),
		SortedScores:   sorted,
	}, nil
}

// ConformalPredictionSet constructs the calibrated prediction set C_alpha(x) = {y : s(x, y) <= qHat}
// ordered by descending probability, and sets ShouldEscalate when the set is not a non-abstain singleton.
func ConformalPredictionSet(probs map[string]float64, qHat float64, cfg ConformalConfig) (ConformalSetResult, error) {
	method, kReg, lamReg, err := validateConformalConfig(cfg)
	if err != nil {
		return ConformalSetResult{}, err
	}
	if math.IsNaN(qHat) || math.IsInf(qHat, -1) || qHat < 0 {
		return ConformalSetResult{}, fmt.Errorf("%w: got %v", ErrInvalidConformalQHat, qHat)
	}
	ordered, err := normalizeOrderedProbs(probs)
	if err != nil {
		return ConformalSetResult{}, err
	}
	scores := computeInferenceScores(ordered, method, kReg, lamReg, cfg)

	abstainList := cfg.AbstainLabels
	if abstainList == nil {
		abstainList = defaultAbstainLabels
	}
	abstainSet := make(map[string]bool, len(abstainList))
	for _, a := range abstainList {
		trimmed := strings.ToLower(strings.TrimSpace(a))
		if trimmed != "" {
			abstainSet[trimmed] = true
		}
	}

	predSet := make([]string, 0, len(ordered))
	containsAbstain := false
	for _, opt := range ordered {
		if math.IsInf(qHat, 1) || scores[opt.name] <= qHat+conformalScoreTolerance {
			predSet = append(predSet, opt.name)
			if abstainSet[strings.ToLower(strings.TrimSpace(opt.name))] {
				containsAbstain = true
			}
		}
	}

	topOpt := ""
	topProb := 0.0
	if len(ordered) > 0 {
		topOpt = ordered[0].name
		topProb = ordered[0].prob
	}

	singleton := len(predSet) == 1
	shouldEscalate := !singleton || containsAbstain

	return ConformalSetResult{
		PredictionSet:   predSet,
		Scores:          scores,
		QHat:            qHat,
		TopOption:       topOpt,
		TopProbability:  topProb,
		SetSize:         len(predSet),
		Singleton:       singleton,
		ContainsAbstain: containsAbstain,
		ShouldEscalate:  shouldEscalate,
	}, nil
}

// EvaluatePreFusionConformalGate evaluates ConformalPredictionSet on qr.RawProbabilities when populated,
// falling back to qr.Probabilities when no prior fusion occurred, so caller priors cannot mask ambiguity.
func EvaluatePreFusionConformalGate(qr QuestionResult, qHat float64, cfg ConformalConfig) (ConformalSetResult, error) {
	target := qr.Probabilities
	if len(qr.RawProbabilities) > 0 {
		target = qr.RawProbabilities
	}
	return ConformalPredictionSet(target, qHat, cfg)
}

// RecoverPreFusionProbabilities inverts log-linear prior fusion when ans.PriorFused is true,
// returning the pre-fusion option probability distribution for downstream conformal gating.
func RecoverPreFusionProbabilities(ans SystemOneAnswer, priorLogits map[string]float64, priorWeight float64) map[string]float64 {
	if !ans.PriorFused || len(priorLogits) == 0 || priorWeight <= 0 || len(ans.Probabilities) == 0 {
		out := make(map[string]float64, len(ans.Probabilities))
		for k, v := range ans.Probabilities {
			out[k] = v
		}
		return out
	}

	for _, v := range ans.Probabilities {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			out := make(map[string]float64, len(ans.Probabilities))
			for k, pv := range ans.Probabilities {
				out[k] = pv
			}
			return out
		}
	}

	keys := make([]string, 0, len(ans.Probabilities))
	for k := range ans.Probabilities {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	logits := make(map[string]float64, len(keys))
	maxLogit := math.Inf(-1)
	for _, k := range keys {
		p := math.Max(1e-12, ans.Probabilities[k])
		l := math.Log(p) - priorWeight*priorLogits[k]
		logits[k] = l
		if l > maxLogit {
			maxLogit = l
		}
	}

	raw := make(map[string]float64, len(keys))
	sum := 0.0
	for _, k := range keys {
		v := math.Exp(logits[k] - maxLogit)
		raw[k] = v
		sum += v
	}
	if sum > 0 {
		for _, k := range keys {
			raw[k] /= sum
		}
	}
	return raw
}

// EvaluateSystemOnePreFusionGate evaluates ConformalPredictionSet on a SystemOneAnswer after
// recovering pre-prior-fusion probabilities when PriorFused is true.
func EvaluateSystemOnePreFusionGate(
	ans SystemOneAnswer,
	priorLogits map[string]float64,
	priorWeight float64,
	qHat float64,
	cfg ConformalConfig,
) (ConformalSetResult, error) {
	raw := RecoverPreFusionProbabilities(ans, priorLogits, priorWeight)
	return ConformalPredictionSet(raw, qHat, cfg)
}

// NormalizedShannonEntropy computes H(p) / ln(K) in [0, 1] for a non-negative slice of K probabilities.
func NormalizedShannonEntropy(probs []float64) float64 {
	k := len(probs)
	if k <= 1 {
		return 0
	}
	sum := 0.0
	for _, p := range probs {
		if !math.IsNaN(p) && !math.IsInf(p, 0) && p > 0 {
			sum += p
		}
	}
	if sum <= 0 {
		return 0
	}
	h := 0.0
	for _, p := range probs {
		if !math.IsNaN(p) && !math.IsInf(p, 0) && p > 0 {
			pn := p / sum
			h -= pn * math.Log(pn)
		}
	}
	normH := h / math.Log(float64(k))
	if normH < 0 {
		return 0
	}
	if normH > 1 {
		return 1
	}
	return normH
}

// ContextBlockSpec describes one candidate trajectory block (up to MaxSlotsPerPass = 8 blocks per pass).
type ContextBlockSpec struct {
	// ID is the caller's block identifier (mapped in source order to neutral slot IDs b0..b7).
	ID string `json:"id"`
	// Kind categorizes the block (for example "tool_output", "file_excerpt", "test_trace", "turn_summary").
	Kind string `json:"kind,omitempty"`
	// Summary is a concise description rendered into the slot instructions prompt.
	Summary string `json:"summary,omitempty"`
	// FullText is the unpruned block content.
	FullText string `json:"full_text,omitempty"`
	// StubText is the compact header, signature, or digest retained when Action == "stub".
	StubText string `json:"stub_text,omitempty"`
	// FullTokens is the token cost of keeping FullText (must satisfy FullTokens >= StubTokens >= 0).
	FullTokens int `json:"full_tokens"`
	// StubTokens is the token cost of keeping StubText (must satisfy StubTokens >= 0).
	StubTokens int `json:"stub_tokens"`
	// Pinned restricts allowed actions to {"full", "stub"} (forbidding "drop").
	Pinned bool `json:"pinned,omitempty"`
	// ImportanceWeight scales this block's expected utility in knapsack selection (defaults to 1.0 when == 0; must be finite and >= 0).
	ImportanceWeight float64 `json:"importance_weight,omitempty"`
}

// ContextPruneConfig controls expected-utility Multiple-Choice Knapsack (MCKP) context pruning.
type ContextPruneConfig struct {
	// TokenBudget is the maximum total tokens allowed across retained full and stub blocks.
	TokenBudget int `json:"token_budget"`
	// StubUtilityRatio overrides the partial utility credit when > 0 (must lie in [0, 1]).
	// When both StubUtilityRatio == 0 and UncertaintyStubBonus == 0, canonical weights
	// u("full") = w*(1.0*pFull + 0.5*pStub), u("stub") = w*(0.6*pFull + 1.0*pStub), u("drop") = 0 are used.
	StubUtilityRatio float64 `json:"stub_utility_ratio,omitempty"`
	// UncertaintyStubBonus adds normalized-entropy weight toward "stub" when > 0 (set <= 0 for none).
	UncertaintyStubBonus float64 `json:"uncertainty_stub_bonus,omitempty"`
}

// ContextBlockDecision records the selected action, token cost, and posterior distribution for one block.
type ContextBlockDecision struct {
	BlockID           string             `json:"block_id"`
	SlotID            string             `json:"slot_id"`
	Action            string             `json:"action"`
	RetainedText      string             `json:"retained_text,omitempty"`
	RetainedTokens    int                `json:"retained_tokens"`
	SavedTokens       int                `json:"saved_tokens"`
	NormalizedEntropy float64            `json:"normalized_entropy"`
	Probabilities     map[string]float64 `json:"probabilities"`
}

// ContextPruneResult aggregates per-block pruning decisions and token accounting.
type ContextPruneResult struct {
	Decisions       []ContextBlockDecision `json:"decisions"`
	OriginalTokens  int                    `json:"original_tokens"`
	RetainedTokens  int                    `json:"retained_tokens"`
	SavedTokens     int                    `json:"saved_tokens"`
	TokenBudget     int                    `json:"token_budget"`
	BudgetSatisfied bool                   `json:"budget_satisfied"`
	ExpectedUtility float64                `json:"expected_utility"`
}

func validateContextBlockSpec(b ContextBlockSpec, idx int, seenIDs map[string]bool) (slotID, blockID string, err error) {
	slotID = fmt.Sprintf("b%d", idx)
	trimmedID := strings.TrimSpace(b.ID)
	if trimmedID == "" {
		blockID = slotID
	} else {
		blockID = trimmedID
	}
	if seenIDs[blockID] {
		return "", "", fmt.Errorf("%w: duplicate block ID %q", ErrInvalidContextBlock, blockID)
	}
	seenIDs[blockID] = true
	if b.StubTokens < 0 || b.FullTokens < 0 || b.FullTokens < b.StubTokens {
		return "", "", fmt.Errorf(
			"%w: block %q requires FullTokens (%d) >= StubTokens (%d) >= 0",
			ErrInvalidContextBlock, blockID, b.FullTokens, b.StubTokens,
		)
	}
	if math.IsNaN(b.ImportanceWeight) || math.IsInf(b.ImportanceWeight, 0) || b.ImportanceWeight < 0 {
		return "", "", fmt.Errorf(
			"%w: block %q has invalid ImportanceWeight %v",
			ErrInvalidContextBlock, blockID, b.ImportanceWeight,
		)
	}
	return slotID, blockID, nil
}

// BuildContextPruningQuestions constructs up to MaxSlotsPerPass (8) independent choice questions
// keyed by neutral slot IDs b0..b7 with options {"full", "stub", "drop"}.
func BuildContextPruningQuestions(blocks []ContextBlockSpec) (map[string]SystemOneQuestion, []string, error) {
	if len(blocks) > MaxSlotsPerPass {
		return nil, nil, fmt.Errorf("%w: got %d blocks", ErrTooManyContextBlocks, len(blocks))
	}
	questions := make(map[string]SystemOneQuestion, len(blocks))
	slotIDs := make([]string, 0, len(blocks))
	seenIDs := make(map[string]bool, len(blocks))
	for i, b := range blocks {
		slotID, blockID, err := validateContextBlockSpec(b, i, seenIDs)
		if err != nil {
			return nil, nil, err
		}
		slotIDs = append(slotIDs, slotID)

		kind := strings.TrimSpace(b.Kind)
		if kind == "" {
			kind = "context_block"
		}

		inst := fmt.Sprintf("Select retention level for block %s (%s, full=%d tok, stub=%d tok).", blockID, kind, b.FullTokens, b.StubTokens)
		if summary := strings.TrimSpace(b.Summary); summary != "" {
			inst = fmt.Sprintf("Select retention level for block %s (%s, full=%d tok, stub=%d tok): %s", blockID, kind, b.FullTokens, b.StubTokens, summary)
		}

		questions[slotID] = SystemOneQuestion{
			Type:         "choice",
			Instructions: inst,
			Criteria: map[string]string{
				ContextActionFull: "Retain complete block text verbatim.",
				ContextActionStub: "Replace full body with compact summary or signature stub.",
				ContextActionDrop: "Omit block from active context window.",
			},
			Order: []string{ContextActionFull, ContextActionStub, ContextActionDrop},
		}
	}
	return questions, slotIDs, nil
}

func validateActionTriplet(blockID string, raw map[string]float64) (pFull, pStub, pDrop float64, err error) {
	if len(raw) == 0 {
		return 0, 0, 0, fmt.Errorf("%w: missing action probabilities for block %q", ErrInvalidProbabilities, blockID)
	}
	for k, v := range raw {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return 0, 0, 0, fmt.Errorf("%w: block %q action %q has invalid probability %v", ErrInvalidProbabilities, blockID, k, v)
		}
	}
	vFull := raw[ContextActionFull]
	vStub := raw[ContextActionStub]
	vDrop := raw[ContextActionDrop]
	sum := vFull + vStub + vDrop
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return 0, 0, 0, fmt.Errorf("%w: block %q action probabilities sum to %v", ErrInvalidProbabilities, blockID, sum)
	}
	return vFull / sum, vStub / sum, vDrop / sum, nil
}

// SelectContextPruningActions solves the exact Multiple-Choice Knapsack Problem (MCKP)
// across up to MaxSlotsPerPass (8) blocks (at most 3^8 = 6,561 joint states) under cfg.TokenBudget,
// restricting Pinned blocks to {"full", "stub"} and returning ErrInfeasiblePinnedBudget if cfg.TokenBudget
// cannot cover the minimum stub token sum of pinned blocks.
func SelectContextPruningActions(
	blocks []ContextBlockSpec,
	slotProbs map[string]map[string]float64,
	cfg ContextPruneConfig,
) (ContextPruneResult, error) {
	if len(blocks) > MaxSlotsPerPass {
		return ContextPruneResult{}, fmt.Errorf("%w: got %d blocks", ErrTooManyContextBlocks, len(blocks))
	}
	if cfg.TokenBudget < 0 {
		return ContextPruneResult{}, fmt.Errorf("%w: got %d", ErrNegativeTokenBudget, cfg.TokenBudget)
	}
	if math.IsNaN(cfg.StubUtilityRatio) || math.IsInf(cfg.StubUtilityRatio, 0) || cfg.StubUtilityRatio < 0 || cfg.StubUtilityRatio > 1 {
		return ContextPruneResult{}, fmt.Errorf("%w: StubUtilityRatio=%v must lie in [0, 1]", ErrInvalidContextBlock, cfg.StubUtilityRatio)
	}
	if math.IsNaN(cfg.UncertaintyStubBonus) || math.IsInf(cfg.UncertaintyStubBonus, 0) {
		return ContextPruneResult{}, fmt.Errorf("%w: UncertaintyStubBonus=%v must be finite", ErrInvalidContextBlock, cfg.UncertaintyStubBonus)
	}
	if len(blocks) == 0 {
		return ContextPruneResult{
			Decisions:       []ContextBlockDecision{},
			TokenBudget:     cfg.TokenBudget,
			BudgetSatisfied: true,
		}, nil
	}

	m := len(blocks)
	prep := make([]preparedBlock, m)
	seenIDs := make(map[string]bool, m)
	origTokens := 0
	minPinnedStubTokens := 0

	for i, b := range blocks {
		slotID, blockID, err := validateContextBlockSpec(b, i, seenIDs)
		if err != nil {
			return ContextPruneResult{}, err
		}
		origTokens += b.FullTokens
		if b.Pinned {
			minPinnedStubTokens += b.StubTokens
		}

		var rawMap map[string]float64
		hasMap := false
		if trimmed := strings.TrimSpace(b.ID); trimmed != "" {
			if pm, ok := slotProbs[trimmed]; ok {
				rawMap = pm
				hasMap = true
			}
		}
		if !hasMap && b.ID != "" {
			if pm, ok := slotProbs[b.ID]; ok {
				rawMap = pm
				hasMap = true
			}
		}
		if !hasMap {
			if pm, ok := slotProbs[slotID]; ok {
				rawMap = pm
				hasMap = true
			}
		}
		if !hasMap {
			return ContextPruneResult{}, fmt.Errorf("%w: missing slotProbs entry for block %q (%s)", ErrInvalidProbabilities, blockID, slotID)
		}

		pFull, pStub, pDrop, pErr := validateActionTriplet(blockID, rawMap)
		if pErr != nil {
			return ContextPruneResult{}, pErr
		}
		hNorm := NormalizedShannonEntropy([]float64{pFull, pStub, pDrop})

		w := b.ImportanceWeight
		if w == 0 {
			w = 1.0
		}

		var uFull, uStub, uDrop float64
		if cfg.StubUtilityRatio == 0 && cfg.UncertaintyStubBonus == 0 {
			// Default expected-utility matrix: u("full") = w*(1.0*pFull + 0.5*pStub), u("stub") = w*(0.6*pFull + 1.0*pStub), u("drop") = 0.
			uFull = w * (1.0*pFull + 0.5*pStub)
			uStub = w * (0.6*pFull + 1.0*pStub)
			uDrop = 0.0
		} else {
			rho := cfg.StubUtilityRatio
			if rho == 0 {
				rho = 0.6
			}
			beta := cfg.UncertaintyStubBonus
			if beta < 0 {
				beta = 0
			}
			uFull = w * (1.0*pFull + rho*pStub)
			uStub = w * (rho*pFull + 1.0*pStub + beta*hNorm*(1.0-pDrop))
			uDrop = 0.0
		}

		prep[i] = preparedBlock{
			slotID:  slotID,
			blockID: blockID,
			costs:   [3]int{b.FullTokens, b.StubTokens, 0},
			utils:   [3]float64{uFull, uStub, uDrop},
			probs:   [3]float64{pFull, pStub, pDrop},
			hNorm:   hNorm,
			pinned:  b.Pinned,
		}
	}

	if minPinnedStubTokens > cfg.TokenBudget {
		return ContextPruneResult{}, fmt.Errorf(
			"%w: budget %d < minimum pinned stub tokens %d",
			ErrInfeasiblePinnedBudget, cfg.TokenBudget, minPinnedStubTokens,
		)
	}

	frontier := []mckpState{{tokens: 0, utility: 0}}
	for i := 0; i < m; i++ {
		maxAct := 3
		if prep[i].pinned {
			maxAct = 2
		}
		expanded := make([]mckpState, 0, len(frontier)*maxAct)
		for _, st := range frontier {
			for act := 0; act < maxAct; act++ {
				nt := st.tokens + prep[i].costs[act]
				if nt <= cfg.TokenBudget {
					next := mckpState{
						tokens:  nt,
						utility: st.utility + prep[i].utils[act],
						choices: st.choices,
						depth:   i + 1,
					}
					next.choices[i] = act
					expanded = append(expanded, next)
				}
			}
		}
		if len(expanded) == 0 {
			return ContextPruneResult{}, fmt.Errorf(
				"%w: no feasible assignment within budget %d",
				ErrInfeasiblePinnedBudget, cfg.TokenBudget,
			)
		}
		frontier = paretoFilterMCKPStates(expanded, 4096)
	}

	best := frontier[0]
	for _, st := range frontier[1:] {
		uDiff := math.Round(st.utility*1e12) - math.Round(best.utility*1e12)
		if uDiff > 0 ||
			(uDiff == 0 && st.tokens < best.tokens) ||
			(uDiff == 0 && st.tokens == best.tokens && compareMCKPChoices(st.choices, best.choices, m) > 0) {
			best = st
		}
	}
	return buildContextPruneResult(blocks, prep, best.choices[:m], origTokens, best.tokens, cfg.TokenBudget, true, best.utility), nil
}

type mckpState struct {
	tokens  int
	utility float64
	choices [MaxSlotsPerPass]int
	depth   int
}

func compareMCKPChoices(a, b [MaxSlotsPerPass]int, depth int) int {
	// Lexicographical ASCII order on action names: "drop" (2) < "full" (0) < "stub" (1).
	asciiOrder := [3]int{1, 2, 0}
	for i := 0; i < depth; i++ {
		ra, rb := asciiOrder[a[i]], asciiOrder[b[i]]
		if ra < rb {
			return -1
		}
		if ra > rb {
			return 1
		}
	}
	return 0
}

func paretoFilterMCKPStates(states []mckpState, maxStates int) []mckpState {
	sort.Slice(states, func(i, j int) bool {
		if states[i].tokens != states[j].tokens {
			return states[i].tokens < states[j].tokens
		}
		if states[i].utility != states[j].utility {
			return states[i].utility > states[j].utility
		}
		return compareMCKPChoices(states[i].choices, states[j].choices, states[i].depth) < 0
	})

	pareto := make([]mckpState, 0, len(states))
	bestU := -1.0
	for _, st := range states {
		if st.utility > bestU+conformalScoreTolerance {
			pareto = append(pareto, st)
			bestU = st.utility
		}
	}
	if len(pareto) <= maxStates {
		return pareto
	}

	minT := pareto[0].tokens
	maxT := pareto[len(pareto)-1].tokens
	if maxT <= minT {
		return pareto[:1]
	}
	span := float64(maxT - minT)
	buckets := make([]mckpState, maxStates)
	hasBucket := make([]bool, maxStates)
	for _, st := range pareto {
		bIdx := int((float64(st.tokens-minT) / span) * float64(maxStates-1))
		if bIdx >= maxStates {
			bIdx = maxStates - 1
		}
		if !hasBucket[bIdx] || st.utility > buckets[bIdx].utility || (st.utility == buckets[bIdx].utility && st.tokens < buckets[bIdx].tokens) {
			buckets[bIdx] = st
			hasBucket[bIdx] = true
		}
	}
	compacted := make([]mckpState, 0, maxStates)
	for i := 0; i < maxStates; i++ {
		if hasBucket[i] {
			compacted = append(compacted, buckets[i])
		}
	}
	return compacted
}

// PruneAgentContextBlocks is an alias for SelectContextPruningActions for callers using the verb form.
func PruneAgentContextBlocks(
	blocks []ContextBlockSpec,
	slotProbs map[string]map[string]float64,
	cfg ContextPruneConfig,
) (ContextPruneResult, error) {
	return SelectContextPruningActions(blocks, slotProbs, cfg)
}

type preparedBlock struct {
	slotID  string
	blockID string
	costs   [3]int
	utils   [3]float64
	probs   [3]float64
	hNorm   float64
	pinned  bool
}

func buildContextPruneResult(
	blocks []ContextBlockSpec,
	meta []preparedBlock,
	actions []int,
	origTokens int,
	retainedTokens int,
	budget int,
	satisfied bool,
	expectedUtil float64,
) ContextPruneResult {
	actionNames := [3]string{ContextActionFull, ContextActionStub, ContextActionDrop}
	decisions := make([]ContextBlockDecision, len(blocks))
	for i, b := range blocks {
		actIdx := actions[i]
		actName := actionNames[actIdx]
		retTok := meta[i].costs[actIdx]
		var retText string
		switch actIdx {
		case 0:
			retText = b.FullText
		case 1:
			retText = b.StubText
		default:
			retText = ""
		}
		decisions[i] = ContextBlockDecision{
			BlockID:           meta[i].blockID,
			SlotID:            meta[i].slotID,
			Action:            actName,
			RetainedText:      retText,
			RetainedTokens:    retTok,
			SavedTokens:       meta[i].costs[0] - retTok,
			NormalizedEntropy: meta[i].hNorm,
			Probabilities: map[string]float64{
				ContextActionFull: meta[i].probs[0],
				ContextActionStub: meta[i].probs[1],
				ContextActionDrop: meta[i].probs[2],
			},
		}
	}
	return ContextPruneResult{
		Decisions:       decisions,
		OriginalTokens:  origTokens,
		RetainedTokens:  retainedTokens,
		SavedTokens:     origTokens - retainedTokens,
		TokenBudget:     budget,
		BudgetSatisfied: satisfied,
		ExpectedUtility: expectedUtil,
	}
}
