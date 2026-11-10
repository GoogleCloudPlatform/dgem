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

package client

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
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
	conformalScoreTolerance = 1e-12
	defaultConformalSeedKey = "dgem-conformal-v1"
	// DefaultConformalAlpha is the standard split-conformal target miscoverage rate (10%).
	DefaultConformalAlpha = 0.10
	// DefaultConformalKReg is the standard RAPS unpenalized top-rank cutoff.
	DefaultConformalKReg = 1
	// DefaultConformalLambdaReg is the standard RAPS per-rank size penalty weight.
	DefaultConformalLambdaReg = 0.01
)

var defaultAbstainLabels = []string{
	"0",
	"abstain",
	"insufficient_evidence",
	"none",
	"none_of_the_above",
	"none_visible",
	"oos",
	"out_of_scope",
	"uncertain",
	"unknown",
}

var (
	// ErrEmptyCalibration reports that an empty calibration slice was supplied.
	ErrEmptyCalibration = errors.New("client: calibration set must be non-empty")
	// ErrCalibrationLenMismatch reports that calibration probability maps and gold labels differ in length.
	ErrCalibrationLenMismatch = errors.New("client: calibration probabilities and labels length mismatch")
	// ErrInvalidAlpha reports that alpha lies outside the open interval (0, 1).
	ErrInvalidAlpha = errors.New("client: alpha must lie strictly inside (0, 1)")
	// ErrUnknownCalibrationKey reports that a gold calibration label is missing from its probability map.
	ErrUnknownCalibrationKey = errors.New("client: gold calibration label missing from probability map")
	// ErrInvalidConformalMethod reports an unrecognized ConformalConfig.Method.
	ErrInvalidConformalMethod = errors.New("client: unknown conformal method")
	// ErrInvalidProbabilities reports an empty, negative, or non-finite probability map.
	ErrInvalidProbabilities = errors.New("client: probability map must be non-empty with finite non-negative values summing above zero")
	// ErrInvalidConformalQHat reports a negative, NaN, or -Inf qHat threshold.
	ErrInvalidConformalQHat = errors.New("client: qHat must be non-negative and non-NaN")
	// ErrInvalidConformalParam reports an invalid RAPS KReg or LambdaReg parameter.
	ErrInvalidConformalParam = errors.New("client: invalid conformal regularization parameter")
)

// ConformalConfig configures stateless split conformal score computation and prediction-set construction.
type ConformalConfig struct {
	// Method selects "lac", "aps", or "raps" (defaults to "raps" when empty).
	Method ConformalMethod `json:"method,omitempty"`
	// Alpha stores the target miscoverage rate in (0, 1).
	Alpha float64 `json:"alpha,omitempty"`
	// KReg is the unpenalized top-rank cutoff k_reg >= 0 for RAPS (0 penalizes starting at rank 1).
	KReg int `json:"k_reg,omitempty"`
	// LambdaReg is the per-rank penalty weight lambda_reg >= 0 for RAPS (0.0 reduces RAPS to APS).
	LambdaReg float64 `json:"lambda_reg,omitempty"`
	// Randomized enables stateless HMAC-SHA256 Uniform[0, 1) boundary smoothing and tie-group permutation.
	Randomized bool `json:"randomized,omitempty"`
	// SeedKey keys stateless HMAC-SHA256 pseudorandom draws when Randomized is true (defaults to "dgem-conformal-v1").
	SeedKey string `json:"seed_key,omitempty"`
	// ItemKey scopes stateless HMAC-SHA256 draws to a specific request/question identifier (for example "req-42:intent").
	ItemKey string `json:"item_key,omitempty"`
	// AbstainLabels lists option names treated as abstention triggers when included in the prediction set.
	AbstainLabels []string `json:"abstain_labels,omitempty"`
}

// DefaultConformalConfig returns the standard RAPS configuration (Method: "raps", Alpha: 0.10, KReg: 1, LambdaReg: 0.01).
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

type optionProb struct {
	name string
	prob float64
}

// ParseConformalMethod validates and normalizes a conformal method string ("lac", "aps", or "raps").
func ParseConformalMethod(raw string) (ConformalMethod, error) {
	norm := strings.ToLower(strings.TrimSpace(raw))
	switch norm {
	case "", string(ConformalRAPS):
		return ConformalRAPS, nil
	case string(ConformalLAC):
		return ConformalLAC, nil
	case string(ConformalAPS):
		return ConformalAPS, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrInvalidConformalMethod, raw)
	}
}

func validateConformalConfig(cfg ConformalConfig) (ConformalMethod, int, float64, error) {
	rawTrimmed := strings.TrimSpace(string(cfg.Method))
	method, err := ParseConformalMethod(string(cfg.Method))
	if err != nil {
		return "", 0, 0, err
	}
	if cfg.KReg < 0 {
		return "", 0, 0, fmt.Errorf("%w: k_reg %d must be >= 0", ErrInvalidConformalParam, cfg.KReg)
	}
	if math.IsNaN(cfg.LambdaReg) || math.IsInf(cfg.LambdaReg, 0) || cfg.LambdaReg < 0 {
		return "", 0, 0, fmt.Errorf("%w: lambda_reg %v must be finite and >= 0", ErrInvalidConformalParam, cfg.LambdaReg)
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
		return nil, fmt.Errorf("%w: total probability mass %v", ErrInvalidProbabilities, sum)
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

func hmacUint64(seedKey, itemKey, tag string) uint64 {
	key := seedKey
	if key == "" {
		key = defaultConformalSeedKey
	}
	mac := hmac.New(sha256.New, []byte(key))
	_, _ = mac.Write([]byte(itemKey))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(tag))
	sum := mac.Sum(nil)
	return binary.BigEndian.Uint64(sum[:8])
}

func hmacUniformFloat64(seedKey, itemKey, tag string) float64 {
	top53 := hmacUint64(seedKey, itemKey, tag) >> 11
	return float64(top53) * (1.0 / (1 << 53))
}

func statelessRandomizedDraw(groups [][]optionProb, total int, seedKey, itemKey string) (float64, []optionProb) {
	u := hmacUniformFloat64(seedKey, itemKey, "u")
	flat := make([]optionProb, 0, total)
	for _, grp := range groups {
		if len(grp) <= 1 {
			flat = append(flat, grp...)
			continue
		}
		cp := make([]optionProb, len(grp))
		copy(cp, grp)
		sort.Slice(cp, func(i, j int) bool {
			hi := hmacUint64(seedKey, itemKey, "tie:"+cp[i].name)
			hj := hmacUint64(seedKey, itemKey, "tie:"+cp[j].name)
			if hi != hj {
				return hi < hj
			}
			return cp[i].name < cp[j].name
		})
		flat = append(flat, cp...)
	}
	return u, flat
}

func computeDeterministicScores(ordered []optionProb, method ConformalMethod, kReg int, lamReg float64) map[string]float64 {
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

func computeRandomizedScoresFromDraw(flat []optionProb, u float64, method ConformalMethod, kReg int, lamReg float64) map[string]float64 {
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

func computeScoresForItem(ordered []optionProb, method ConformalMethod, kReg int, lamReg float64, cfg ConformalConfig, itemKey string) map[string]float64 {
	if !cfg.Randomized {
		return computeDeterministicScores(ordered, method, kReg, lamReg)
	}
	groups := partitionOptionTieGroups(ordered)
	u, flat := statelessRandomizedDraw(groups, len(ordered), cfg.SeedKey, itemKey)
	return computeRandomizedScoresFromDraw(flat, u, method, kReg, lamReg)
}

// ResolveGoldProbabilityKey resolves an expected label against a probability map, matching exact keys, case-insensitive keys, boolean yes/true aliases, and numeric score strings.
func ResolveGoldProbabilityKey(probs map[string]float64, expected string) (string, bool) {
	if len(probs) == 0 {
		return "", false
	}
	if _, ok := probs[expected]; ok {
		return expected, true
	}
	trimmed := strings.TrimSpace(expected)
	if _, ok := probs[trimmed]; ok {
		return trimmed, true
	}
	for k := range probs {
		if strings.EqualFold(strings.TrimSpace(k), trimmed) {
			return k, true
		}
	}
	switch strings.ToLower(trimmed) {
	case "true", "yes", "1":
		for _, alias := range []string{"yes", "true", "1"} {
			for k := range probs {
				if strings.EqualFold(strings.TrimSpace(k), alias) {
					return k, true
				}
			}
		}
	case "false", "no", "0":
		for _, alias := range []string{"no", "false", "0"} {
			for k := range probs {
				if strings.EqualFold(strings.TrimSpace(k), alias) {
					return k, true
				}
			}
		}
	}
	if expFloat, err := strconv.ParseFloat(trimmed, 64); err == nil {
		for k := range probs {
			if kFloat, kErr := strconv.ParseFloat(strings.TrimSpace(k), 64); kErr == nil && math.Abs(expFloat-kFloat) <= 1e-9 {
				return k, true
			}
		}
	}
	return "", false
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
	resolved, ok := ResolveGoldProbabilityKey(probs, label)
	if !ok {
		return 0, fmt.Errorf("%w: %q", ErrUnknownCalibrationKey, label)
	}
	scores := computeScoresForItem(ordered, method, kReg, lamReg, cfg, cfg.ItemKey)
	return scores[resolved], nil
}

// CalibrateConformalQuantile computes the finite-sample split conformal quantile at order index k = ceil((n + 1) * (1 - alpha) - 1e-12).
func CalibrateConformalQuantile(calibProbs []map[string]float64, calibLabels []string, alpha float64, cfg ConformalConfig) (ConformalCalibrationResult, error) {
	return CalibrateConformalQuantileWithKeys(calibProbs, calibLabels, nil, alpha, cfg)
}

// CalibrateConformalQuantileWithKeys computes the split conformal quantile using optional per-item keys for stateless randomized tie-breaking.
func CalibrateConformalQuantileWithKeys(calibProbs []map[string]float64, calibLabels []string, itemKeys []string, alpha float64, cfg ConformalConfig) (ConformalCalibrationResult, error) {
	if len(calibProbs) == 0 {
		return ConformalCalibrationResult{}, ErrEmptyCalibration
	}
	if len(calibProbs) != len(calibLabels) {
		return ConformalCalibrationResult{}, fmt.Errorf("%w: got %d probability maps and %d labels", ErrCalibrationLenMismatch, len(calibProbs), len(calibLabels))
	}
	if len(itemKeys) > 0 && len(itemKeys) != len(calibProbs) {
		return ConformalCalibrationResult{}, fmt.Errorf("%w: got %d item keys and %d probability maps", ErrCalibrationLenMismatch, len(itemKeys), len(calibProbs))
	}
	if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha <= 0 || alpha >= 1 {
		return ConformalCalibrationResult{}, fmt.Errorf("%w: got %v", ErrInvalidAlpha, alpha)
	}
	method, kReg, lamReg, err := validateConformalConfig(cfg)
	if err != nil {
		return ConformalCalibrationResult{}, err
	}

	n := len(calibProbs)
	rawScores := make([]float64, n)
	for i := 0; i < n; i++ {
		ordered, nErr := normalizeOrderedProbs(calibProbs[i])
		if nErr != nil {
			return ConformalCalibrationResult{}, fmt.Errorf("calibration item %d: %w", i, nErr)
		}
		resolved, ok := ResolveGoldProbabilityKey(calibProbs[i], calibLabels[i])
		if !ok {
			return ConformalCalibrationResult{}, fmt.Errorf("%w at index %d: %q", ErrUnknownCalibrationKey, i, calibLabels[i])
		}
		key := fmt.Sprintf("cal:%d", i)
		if len(itemKeys) > 0 && itemKeys[i] != "" {
			key = itemKeys[i]
		}
		sMap := computeScoresForItem(ordered, method, kReg, lamReg, cfg, key)
		rawScores[i] = sMap[resolved]
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

// ConformalPredictionSet constructs the calibrated prediction set C_alpha(x) = {y : s(x, y) <= qHat} ordered by descending probability.
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
	scores := computeScoresForItem(ordered, method, kReg, lamReg, cfg, cfg.ItemKey)

	abstainList := cfg.AbstainLabels
	if abstainList == nil {
		abstainList = defaultAbstainLabels
	}
	abstainSet := make(map[string]bool, len(abstainList))
	for _, a := range abstainList {
		if trimmed := strings.ToLower(strings.TrimSpace(a)); trimmed != "" {
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

// EvaluateQuestionConformalSet evaluates ConformalPredictionSet on rawProbs when non-empty (preserving pre-prior-fusion marginal coverage), falling back to qa.Probabilities.
func EvaluateQuestionConformalSet(qa QuestionAnswer, rawProbs map[string]float64, qHat float64, cfg ConformalConfig) (ConformalSetResult, error) {
	target := qa.Probabilities
	if len(rawProbs) > 0 {
		target = rawProbs
	}
	return ConformalPredictionSet(target, qHat, cfg)
}
