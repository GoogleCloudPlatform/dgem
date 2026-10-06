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
	"fmt"
	"math"
	"sort"
	"strings"
)

// ReliabilityBin represents one of the 10 equal-width confidence bins [0.0..1.0] for ECE and Reliability Diagrams.
type ReliabilityBin struct {
	BinIndex     int     `json:"bin_index"`
	RangeLabel   string  `json:"range_label"`
	Lower        float64 `json:"lower"`
	Upper        float64 `json:"upper"`
	Count        int     `json:"count"`
	AvgConf      float64 `json:"avg_confidence"`
	EmpiricalAcc float64 `json:"empirical_accuracy"`
	AbsGap       float64 `json:"abs_gap"`
}

// JevParitySummary holds the JevBench v1.3.1 & v1.4 4-Axis parity metrics and temperature-scaling comparison.
type JevParitySummary struct {
	ProtocolVersion        string             `json:"protocol_version"`
	TemperatureApplied     float64            `json:"temperature_applied"`
	TemperatureMethod      string             `json:"temperature_method,omitempty"` // kfold: cases carry their own held-out T; the T=1 columns are then held-out values
	OptimalTemperature     float64            `json:"optimal_temperature"`
	ChanceBaselinePct      float64            `json:"chance_baseline_pct"`
	ChanceCorrectedAccPct  float64            `json:"chance_corrected_acc_pct"`
	IntelligenceScore      float64            `json:"intelligence_score"`
	CalibrationScore       float64            `json:"calibration_score"`
	SpeedScore             float64            `json:"speed_score"`
	CostScore              float64            `json:"cost_score"`
	CompositeJevBenchScore float64            `json:"composite_jevbench_score"`
	ECE10Bin               float64            `json:"ece_10bin"`
	BrierMean              float64            `json:"brier_mean"`
	SoftTVDMean            float64            `json:"soft_tvd_mean"`
	ProbabilityFidelity    float64            `json:"probability_fidelity"`
	P50LatencySec          float64            `json:"p50_latency_sec"`
	P95LatencySec          float64            `json:"p95_latency_sec"`
	USDPer1kDecisions      float64            `json:"usd_per_1k_decisions"`
	Presets                map[string]float64 `json:"presets"`
	RawT1ECE10Bin          float64            `json:"raw_t1_ece_10bin"`
	RawT1BrierMean         float64            `json:"raw_t1_brier_mean"`
	RawT1CalibrationScore  float64            `json:"raw_t1_calibration_score"`
	RawT1CompositeScore    float64            `json:"raw_t1_composite_score"`
	ReliabilityBins        []ReliabilityBin   `json:"reliability_bins"`

	// JevBench v1.4 Protocol Metrics (composite_v14.py)
	EndpointKind          string  `json:"endpoint_kind,omitempty"`
	RawSpeedScore         float64 `json:"raw_speed_score,omitempty"`
	AdjustedSpeedScore    float64 `json:"adjusted_speed_score,omitempty"`
	V14IntelligenceScore  float64 `json:"v14_intelligence_score,omitempty"`
	V14CalibrationScore   float64 `json:"v14_calibration_score,omitempty"`
	V14CompositeScore     float64 `json:"v14_composite_score,omitempty"`
	V14PublicOnlyEstimate bool    `json:"v14_public_only_estimate,omitempty"`
	SealedAccuracy        float64 `json:"sealed_accuracy,omitempty"`
	GapPenaltyPct         float64 `json:"gap_penalty_pct,omitempty"`
}

// enrichAndScaleCaseResults populates VocabCardinality, ChanceBaseline, BrierScore, TVDGold,
// and applies post-hoc logit temperature scaling p_k(T) = p_k^(1/T) / sum_j p_j^(1/T) when temp != 1.0.
// Argmax predictions (Actual, Accurate) are strictly invariant to any monotonic T > 0.
func enrichAndScaleCaseResults(cases []CalibrationCaseResult, temp float64) []CalibrationCaseResult {
	if temp <= 0 {
		temp = 1.0
	}
	out := make([]CalibrationCaseResult, len(cases))
	for i, c := range cases {
		rc := c
		vocab := rc.VocabCardinality
		if vocab <= 1 {
			vocab = calibrationMetricVocabSize(rc.Metric, len(rc.TopProbabilities))
		}
		rc.VocabCardinality = vocab
		rc.ChanceBaseline = 1.0 / float64(vocab)

		// Apply temperature scaling to TopProbabilities / Confidence
		rc.TopProbabilities, rc.Confidence, rc.Entropy, rc.NormalizedEntropy = scaleCaseDistribution(rc, temp)
		rc.BrierScore = computeCaseBrierScore(rc)
		rc.TVDGold, _ = computeCaseSoftTVD(rc)
		out[i] = rc
	}
	return out
}

// scaleCaseDistribution applies post-hoc temperature scaling T > 0 to a case's probability map.
func scaleCaseDistribution(rc CalibrationCaseResult, temp float64) (map[string]float64, float64, float64, float64) {
	vocab := rc.VocabCardinality
	if vocab <= 1 {
		vocab = 2
	}
	invT := 1.0 / temp

	if len(rc.TopProbabilities) > 0 {
		var sumTop float64
		for _, p := range rc.TopProbabilities {
			if p > 0 {
				sumTop += p
			}
		}
		tailCount := vocab - len(rc.TopProbabilities)
		var pTail float64
		if tailCount > 0 && sumTop < 1.0 {
			pTail = math.Max(1e-12, (1.0-sumTop)/float64(tailCount))
		} else {
			tailCount = 0
		}

		// Compute partition function Z(T) = sum_{k in Top} p_k^(1/T) + tailCount * pTail^(1/T)
		var z float64
		scaledTop := make(map[string]float64, len(rc.TopProbabilities))
		for k, p := range rc.TopProbabilities {
			pClamped := math.Max(1e-12, math.Min(1.0, p))
			w := math.Pow(pClamped, invT)
			scaledTop[k] = w
			z += w
		}
		var wTail float64
		if tailCount > 0 {
			wTail = math.Pow(pTail, invT)
			z += float64(tailCount) * wTail
		}
		if z <= 0 {
			z = 1.0
		}

		var maxP, entropy float64
		for k, w := range scaledTop {
			pk := w / z
			scaledTop[k] = pk
			if pk > maxP {
				maxP = pk
			}
			if pk > 1e-12 {
				entropy -= pk * math.Log(pk)
			}
		}
		if tailCount > 0 {
			pkTail := wTail / z
			if pkTail > 1e-12 {
				entropy -= float64(tailCount) * pkTail * math.Log(pkTail)
			}
		}
		normEnt := 0.0
		if vocab > 1 {
			normEnt = entropy / math.Log(float64(vocab))
		}
		return scaledTop, maxP, entropy, normEnt
	}

	// Fallback when only scalar Confidence is present (e.g. Vertex AI receipts)
	pMax := math.Max(1e-6, math.Min(0.999999, rc.Confidence))
	tailCount := vocab - 1
	pTail := (1.0 - pMax) / float64(tailCount)
	wMax := math.Pow(pMax, invT)
	wTail := math.Pow(math.Max(1e-12, pTail), invT)
	z := wMax + float64(tailCount)*wTail
	newConf := wMax / z
	newTail := wTail / z
	entropy := 0.0
	if newConf > 1e-12 {
		entropy -= newConf * math.Log(newConf)
	}
	if newTail > 1e-12 {
		entropy -= float64(tailCount) * newTail * math.Log(newTail)
	}
	normEnt := 0.0
	if vocab > 1 {
		normEnt = entropy / math.Log(float64(vocab))
	}
	return rc.TopProbabilities, newConf, entropy, normEnt
}

// computeCaseBrierScore computes JevBench's exact multi-class Brier score:
// sum_{k in V} (p_k - y_k)^2 = (1 - p_gold)^2 + sum_{k != gold} p_k^2.
// For 2-class binary tasks (V=2), this equals 2 * (1 - p_gold)^2, matching jevbench/metrics.py verbatim.
func computeCaseBrierScore(rc CalibrationCaseResult) float64 {
	vocab := rc.VocabCardinality
	if vocab <= 1 {
		vocab = 2
	}
	var pGold float64
	if rc.Accurate {
		pGold = rc.Confidence
	} else {
		found := false
		if len(rc.TopProbabilities) > 0 {
			for k, p := range rc.TopProbabilities {
				if labelsMatch(k, rc.Expected) {
					pGold = p
					found = true
					break
				}
			}
		}
		if !found {
			pGold = math.Max(0.0, (1.0-rc.Confidence)/float64(vocab-1))
		}
	}
	pGold = math.Max(0.0, math.Min(1.0, pGold))

	if vocab == 2 {
		return 2.0 * (1.0 - pGold) * (1.0 - pGold)
	}

	if len(rc.TopProbabilities) > 0 && !rc.Accurate {
		var sumOtherSq float64
		var sumKnown float64
		for k, p := range rc.TopProbabilities {
			sumKnown += p
			if !labelsMatch(k, rc.Expected) {
				sumOtherSq += p * p
			}
		}
		return (1.0-pGold)*(1.0-pGold) + sumOtherSq
	}

	// Symmetric tail approximation for remaining V-1 classes when Accurate or scalar confidence
	if len(rc.TopProbabilities) > 1 {
		var sumOtherSq float64
		maxKey := ""
		maxVal := -1.0
		for k, p := range rc.TopProbabilities {
			if p > maxVal {
				maxVal = p
				maxKey = k
			}
		}
		for k, p := range rc.TopProbabilities {
			if k != maxKey {
				sumOtherSq += p * p
			}
		}
		return (1.0-pGold)*(1.0-pGold) + sumOtherSq
	}

	rem := 1.0 - pGold
	return (1.0-pGold)*(1.0-pGold) + (rem*rem)/float64(vocab-1)
}

func labelsMatch(a, b string) bool {
	a = strings.ToLower(strings.TrimSpace(a))
	b = strings.ToLower(strings.TrimSpace(b))
	if a == b {
		return true
	}
	if (a == "yes" && b == "true") || (a == "true" && b == "yes") {
		return true
	}
	if (a == "no" && b == "false") || (a == "false" && b == "no") {
		return true
	}
	return false
}

// computeCaseSoftTVD computes Total Variation Distance (TVD = 0.5 * sum_k |p_k - q_k|)
// on items with human crowd annotator soft distributions (ChaosNLI, Ambiguous rater splits, and Civil Comments rates).
func computeCaseSoftTVD(rc CalibrationCaseResult) (float64, bool) {
	vocab := rc.VocabCardinality
	if vocab <= 1 {
		vocab = 3
	}
	var goldTop float64
	isSoft := false

	switch {
	case rc.Category == "nli-soft" && rc.Tier == "low-entropy":
		// ChaosNLI high-consensus human rater distribution (~92% primary, 8% split)
		goldTop = 0.92
		isSoft = true
	case rc.Category == "nli-soft" && rc.Tier == "high-entropy":
		// ChaosNLI 3-way crowd split (~40% primary, 34% secondary, 26% tertiary)
		goldTop = 0.40
		isSoft = true
	case rc.Tier == "ambiguous":
		// Borderline rater split (~75% primary consensus, 25% secondary)
		goldTop = 0.75
		isSoft = true
	}
	if !isSoft {
		return 0, false
	}

	// Predicted probability assigned to the top consensus class
	var predGold float64
	if rc.Accurate {
		predGold = rc.Confidence
	} else {
		predGold = math.Max(0.0, 1.0-rc.Confidence)
	}
	// For a 2-partition (gold vs rest), TVD = |predGold - goldTop|
	tvd := math.Abs(predGold - goldTop)
	if tvd > 1.0 {
		tvd = 1.0
	}
	return tvd, true
}

// compute10BinECE computes the 10-bin equal-width Expected Calibration Error (ECE) on [0.0, 1.0]
// matching jevbench/metrics.py ("ECE: top-label confidence, 10 equal-width bins").
func compute10BinECE(cases []CalibrationCaseResult) (float64, []ReliabilityBin) {
	bins := make([]ReliabilityBin, 10)
	for b := 0; b < 10; b++ {
		lo := float64(b) * 0.1
		hi := float64(b+1) * 0.1
		bins[b] = ReliabilityBin{
			BinIndex:   b,
			RangeLabel: fmt.Sprintf("[%.1f, %.1f)", lo, hi),
			Lower:      lo,
			Upper:      hi,
		}
		if b == 9 {
			bins[b].RangeLabel = "[0.9, 1.0]"
		}
	}

	if len(cases) == 0 {
		return 0, bins
	}

	var sumConf [10]float64
	var sumAcc [10]float64

	for _, c := range cases {
		conf := math.Max(0.0, math.Min(1.0, c.Confidence))
		idx := int(math.Floor(conf * 10.0))
		if idx >= 10 {
			idx = 9
		}
		if idx < 0 {
			idx = 0
		}
		bins[idx].Count++
		sumConf[idx] += conf
		if c.Accurate {
			sumAcc[idx] += 1.0
		}
	}

	nTotal := float64(len(cases))
	var ece float64
	for b := 0; b < 10; b++ {
		if bins[b].Count > 0 {
			nb := float64(bins[b].Count)
			bins[b].AvgConf = sumConf[b] / nb
			bins[b].EmpiricalAcc = sumAcc[b] / nb
			bins[b].AbsGap = math.Abs(bins[b].EmpiricalAcc - bins[b].AvgConf)
			ece += (nb / nTotal) * bins[b].AbsGap
		}
	}
	return ece, bins
}

// computeLatencyPercentilesSec computes p50 and p95 latencies in seconds.
func computeLatencyPercentilesSec(cases []CalibrationCaseResult) (float64, float64) {
	if len(cases) == 0 {
		return 0, 0
	}
	lats := make([]float64, len(cases))
	for i, c := range cases {
		lats[i] = c.WallTimeMs / 1000.0
	}
	sort.Float64s(lats)
	p50 := percentileSorted(lats, 0.50)
	p95 := percentileSorted(lats, 0.95)
	return p50, p95
}

func percentileSorted(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	pos := p * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi || hi >= len(sorted) {
		return sorted[lo]
	}
	frac := pos - float64(lo)
	return sorted[lo]*(1.0-frac) + sorted[hi]*frac
}

// computeJevSpeedScore implements jevbench/composite_v13.py Speed axis:
// score(s) = clamp(100 - 20 * log10(s / 0.1), 0, 100), mean of score(p50) and score(p95).
func computeJevSpeedScore(p50Sec, p95Sec float64) float64 {
	scoreOne := func(s float64) float64 {
		if s <= 0.1 {
			return 100.0
		}
		v := 100.0 - 20.0*math.Log10(s/0.1)
		return math.Max(0.0, math.Min(100.0, v))
	}
	return 0.5 * (scoreOne(p50Sec) + scoreOne(p95Sec))
}

// computeJevSpeedScoreWithKind computes Speed axis with endpoint kind adjustment:
// If endpointKind is "gpu" or "cpu" (self-hosted / evaluator pod), latency is adjusted
// by 2.0x (+0.15s) following JevBench official method (composite_v13.py / composite_v14.py).
// If endpointKind is "api", latency is unadjusted (1.0x).
// Returns (adjustedSpeedScore, rawSpeedScore).
func computeJevSpeedScoreWithKind(p50Sec, p95Sec float64, endpointKind string) (float64, float64) {
	raw := computeJevSpeedScore(p50Sec, p95Sec)
	ek := strings.ToLower(strings.TrimSpace(endpointKind))
	if ek == "api" {
		return raw, raw
	}
	// Self-hosted GPU or CPU endpoint
	adjP50 := p50Sec*2.0 + 0.15
	adjP95 := p95Sec*2.0 + 0.15
	adj := computeJevSpeedScore(adjP50, adjP95)
	return adj, raw
}

// computeJevV14CompositeScore computes the official JevBench v1.4 equal-weight harmonic mean
// with the three <50 threshold penalties (Intelligence, Speed, Cost) from composite_v14.py:
//
//	score = 4 / (1/I + 1/C + 1/S + 1/Cost)
//	if I < 50:    score *= (I / 50)^2
//	if S < 50:    score *= (S / 50)^2
//	if Cost < 50: score *= (Cost / 50)^2
func computeJevV14CompositeScore(intel, cal, speed, cost float64) float64 {
	axes := []float64{intel, cal, speed, cost}
	for _, v := range axes {
		if v <= 0.0 {
			return 0.0
		}
	}
	harmonic := 4.0 / (1.0/intel + 1.0/cal + 1.0/speed + 1.0/cost)

	const gateThreshold = 50.0
	if intel < gateThreshold {
		r := math.Max(0.0, intel) / gateThreshold
		harmonic *= r * r
	}
	if speed < gateThreshold {
		r := math.Max(0.0, speed) / gateThreshold
		harmonic *= r * r
	}
	if cost < gateThreshold {
		r := math.Max(0.0, cost) / gateThreshold
		harmonic *= r * r
	}
	return harmonic
}

// computeJevV14Intelligence computes the official JevBench v1.4 Intelligence axis from composite_v14.py:
//   - If sealedAcc <= 0 (unknown/public-only), returns (publicIntel, 0.0, true).
//   - If sealedAcc > 0:
//     I_sealed = clamp(100 * max(0, (sealedAcc - 0.293) / (1 - 0.293)))
//     I_base   = 0.8 * publicIntel + 0.2 * I_sealed
//     gap      = 100 * (publicAcc - sealedAcc)
//     penalty  = max(0, gap - 25) / 100
//     I        = I_base * (1 - penalty)
func computeJevV14Intelligence(publicIntel, publicAcc, sealedAcc float64) (float64, float64, bool) {
	if sealedAcc <= 0.0 {
		return publicIntel, 0.0, true
	}
	const sealedWeight = 0.20
	const sealedChance = 0.293
	const gapAllowancePoints = 25.0
	const gapPenaltyK = 1.0

	sealedChanceCorr := 0.0
	if sealedAcc > sealedChance {
		sealedChanceCorr = 100.0 * (sealedAcc - sealedChance) / (1.0 - sealedChance)
		if sealedChanceCorr > 100.0 {
			sealedChanceCorr = 100.0
		}
	}

	combined := (1.0-sealedWeight)*publicIntel + sealedWeight*sealedChanceCorr
	gapPoints := 100.0 * (publicAcc - sealedAcc)
	gapPenalty := math.Max(0.0, gapPoints-gapAllowancePoints) * gapPenaltyK / 100.0
	finalIntel := combined * math.Max(0.0, 1.0-gapPenalty)

	return finalIntel, gapPenalty * 100.0, false
}

// computeJevCostScore implements jevbench/composite_v13.py Cost axis:
// clamp(100 - 30 * log10(usd_per_1000 / 0.001), 0, 100).
func computeJevCostScore(usdPer1k float64) float64 {
	if usdPer1k <= 0.001 {
		return 100.0
	}
	v := 100.0 - 30.0*math.Log10(usdPer1k/0.001)
	return math.Max(0.0, math.Min(100.0, v))
}

// estimateUSDPer1kDecisions estimates the cost in USD per 1,000 decisions following JevBench v1.2.3:
// - DiffusionGemma (dgem, 26B-A4B NVFP4): $0.0260 per 1,000 decisions (~742 input tokens @ $0.035/1M input, 0 output)
// - Gemini 3.8 Flash / Vertex AI: $0.2638 per 1,000 decisions
// - Entropy-Gated Cascade: $0.0260 + (escalation_rate * $0.2638)
func estimateUSDPer1kDecisions(targetModel string, cascade *CascadeSummary, overrideUSD float64) float64 {
	if overrideUSD > 0 {
		return overrideUSD
	}
	if cascade != nil {
		escRate := cascade.EscalationRatePct / 100.0
		if strings.Contains(strings.ToLower(cascade.CascadeMode), "self") {
			return 0.0260 * (1.0 + escRate)
		}
		return 0.0260 + escRate*0.2638
	}
	m := strings.ToLower(targetModel)
	if strings.Contains(m, "gemini") {
		return 0.2638
	}
	return 0.0260
}

// computeJevCalibrationScore implements jevbench/composite_v12.py Calibration axis:
// mean of 100 * max(0, 1 - ECE / 0.5) and 100 * (1 - mean_soft_TVD).
func computeJevCalibrationScore(ece10Bin, softTVDMean float64) float64 {
	eceComponent := math.Max(0.0, 100.0*(1.0-ece10Bin/0.5))
	tvdComponent := math.Max(0.0, 100.0*(1.0-softTVDMean))
	return 0.5 * (eceComponent + tvdComponent)
}

// computeJevWeightedIntelligence computes JevBench v1.3.1's weighted chance-corrected Intelligence:
// easy (14%), standard/held-out (28%), judge/guardrail/localization (28%), hard/adversarial/ambiguous/high-entropy (30%).
func computeJevWeightedIntelligence(cases []CalibrationCaseResult) (float64, float64, float64) {
	if len(cases) == 0 {
		return 0, 0, 0
	}
	var sumChance, sumAcc float64
	type bucket struct {
		accSum    float64
		chanceSum float64
		count     int
	}
	buckets := map[string]*bucket{
		"easy":     {},
		"standard": {},
		"judge":    {},
		"hard":     {},
	}

	for _, c := range cases {
		ch := c.ChanceBaseline
		if ch <= 0 {
			ch = 1.0 / float64(calibrationMetricVocabSize(c.Metric, len(c.TopProbabilities)))
		}
		sumChance += ch
		acc := 0.0
		if c.Accurate {
			acc = 1.0
		}
		sumAcc += acc

		var bKey string
		switch c.Tier {
		case "easy", "low-entropy":
			bKey = "easy"
		case "held-out", "out-of-scope":
			bKey = "standard"
		case "localization":
			bKey = "judge"
		default:
			// adversarial, ambiguous, high-entropy
			bKey = "hard"
		}
		b := buckets[bKey]
		b.accSum += acc
		b.chanceSum += ch
		b.count++
	}

	n := float64(len(cases))
	meanChance := sumChance / n
	meanAcc := sumAcc / n
	overallChanceCorr := math.Max(0.0, (meanAcc-meanChance)/(1.0-meanChance)) * 100.0

	weights := map[string]float64{
		"easy":     0.14,
		"standard": 0.28,
		"judge":    0.28,
		"hard":     0.30,
	}
	var weightedInt, weightSum float64
	for k, w := range weights {
		b := buckets[k]
		if b.count > 0 {
			cnt := float64(b.count)
			bAcc := b.accSum / cnt
			bCh := b.chanceSum / cnt
			score := math.Max(0.0, (bAcc-bCh)/(1.0-bCh)) * 100.0
			weightedInt += w * score
			weightSum += w
		}
	}
	if weightSum > 0 {
		weightedInt /= weightSum
	} else {
		weightedInt = overallChanceCorr
	}
	return meanChance * 100.0, overallChanceCorr, weightedInt
}

// computeJevCompositeScore computes the 4-axis geometric mean from jevbench/composite_v13.py,
// including the <50 Intelligence penalty: score * (Intelligence / 50)^2.
func computeJevCompositeScore(intel, cal, speed, cost float64) float64 {
	clamp1 := func(x float64) float64 {
		return math.Max(1.0, math.Min(100.0, x))
	}
	geo := math.Exp(0.25 * (math.Log(clamp1(intel)) + math.Log(clamp1(cal)) + math.Log(clamp1(speed)) + math.Log(clamp1(cost))))
	if intel < 50.0 {
		penalty := (intel / 50.0) * (intel / 50.0)
		geo *= penalty
	}
	return geo
}

// findOptimalTemperature grid-searches T in [0.50, 3.50] to maximize JevBench Calibration Score (minimizing ECE + Brier).
func findOptimalTemperature(rawCases []CalibrationCaseResult) float64 {
	bestT := 1.0
	bestScore := -1e9
	for step := 50; step <= 350; step += 5 {
		t := float64(step) / 100.0
		scaled := enrichAndScaleCaseResults(rawCases, t)
		ece, _ := compute10BinECE(scaled)
		var sumBrier, sumTVD float64
		var cntTVD int
		for _, c := range scaled {
			sumBrier += c.BrierScore
			if tvd, ok := computeCaseSoftTVD(c); ok {
				sumTVD += tvd
				cntTVD++
			}
		}
		meanBrier := sumBrier / float64(len(scaled))
		meanTVD := 0.0
		if cntTVD > 0 {
			meanTVD = sumTVD / float64(cntTVD)
		}
		calScore := computeJevCalibrationScore(ece, meanTVD)
		// Objective: maximize JevBench CalibrationScore while breaking ties with lower Brier score
		obj := calScore - 10.0*meanBrier
		if obj > bestScore {
			bestScore = obj
			bestT = t
		}
	}
	return bestT
}

// buildJevParitySummary computes the complete JevBench v1.3.1 & v1.4 4-Axis telemetry and comparison against T=1.0.
func buildJevParitySummary(rawCases []CalibrationCaseResult, activeTemp float64, targetModel string, cascade *CascadeSummary, overrideUSD float64) (*JevParitySummary, []CalibrationCaseResult) {
	return buildJevParitySummaryWithKind(rawCases, activeTemp, targetModel, cascade, overrideUSD, "gpu", 0.0)
}

// buildJevParitySummaryWithKind computes both v1.3.1 and v1.4 metrics with optional endpoint kind adjustment and sealed accuracy.
func buildJevParitySummaryWithKind(rawCases []CalibrationCaseResult, activeTemp float64, targetModel string, cascade *CascadeSummary, overrideUSD float64, endpointKind string, sealedAcc float64) (*JevParitySummary, []CalibrationCaseResult) {
	if activeTemp <= 0 {
		activeTemp = 1.0
	}
	if endpointKind == "" {
		endpointKind = "gpu"
	}
	optT := findOptimalTemperature(rawCases)

	// Compute T=1.0 baseline first
	t1Cases := enrichAndScaleCaseResults(rawCases, 1.0)
	t1ECE, _ := compute10BinECE(t1Cases)
	var t1BrierSum, t1TVDSum float64
	var t1TVDCnt int
	for _, c := range t1Cases {
		t1BrierSum += c.BrierScore
		if tvd, ok := computeCaseSoftTVD(c); ok {
			t1TVDSum += tvd
			t1TVDCnt++
		}
	}
	t1BrierMean := 0.0
	if len(t1Cases) > 0 {
		t1BrierMean = t1BrierSum / float64(len(t1Cases))
	}
	t1TVDMean := 0.0
	if t1TVDCnt > 0 {
		t1TVDMean = t1TVDSum / float64(t1TVDCnt)
	}
	t1CalScore := computeJevCalibrationScore(t1ECE, t1TVDMean)

	// Now compute at activeTemp
	scaledCases := enrichAndScaleCaseResults(rawCases, activeTemp)
	ece, bins := compute10BinECE(scaledCases)
	var brierSum, tvdSum float64
	var tvdCnt int
	for _, c := range scaledCases {
		brierSum += c.BrierScore
		if tvd, ok := computeCaseSoftTVD(c); ok {
			tvdSum += tvd
			tvdCnt++
		}
	}
	brierMean := 0.0
	if len(scaledCases) > 0 {
		brierMean = brierSum / float64(len(scaledCases))
	}
	tvdMean := 0.0
	if tvdCnt > 0 {
		tvdMean = tvdSum / float64(tvdCnt)
	}
	calScore := computeJevCalibrationScore(ece, tvdMean)
	chanceBasePct, chanceCorrPct, intelScore := computeJevWeightedIntelligence(scaledCases)
	p50Sec, p95Sec := computeLatencyPercentilesSec(scaledCases)
	adjSpeed, rawSpeed := computeJevSpeedScoreWithKind(p50Sec, p95Sec, endpointKind)
	usdPer1k := estimateUSDPer1kDecisions(targetModel, cascade, overrideUSD)
	costScore := computeJevCostScore(usdPer1k)

	composite := computeJevCompositeScore(intelScore, calScore, adjSpeed, costScore)
	t1Composite := computeJevCompositeScore(intelScore, t1CalScore, adjSpeed, costScore)

	v14Intel, gapPenalty, publicOnly := computeJevV14Intelligence(intelScore, chanceCorrPct/100.0, sealedAcc)
	v14Composite := computeJevV14CompositeScore(v14Intel, calScore, adjSpeed, costScore)

	// Weighted presets from jevbench/composite_v13.py
	weightedGeo := func(wI, wCal, wS, wC float64) float64 {
		c1 := func(x float64) float64 { return math.Max(1.0, math.Min(100.0, x)) }
		return math.Exp(wI*math.Log(c1(intelScore)) + wCal*math.Log(c1(calScore)) + wS*math.Log(c1(adjSpeed)) + wC*math.Log(c1(costScore)))
	}
	presets := map[string]float64{
		"JevBench Score (25:25:25:25)":       composite,
		"Balanced 33:33:33 (no calibration)": weightedGeo(1.0/3.0, 0.0, 1.0/3.0, 1.0/3.0),
		"Emphasis on Accuracy 60:20:20":      weightedGeo(0.60, 0.0, 0.20, 0.20),
		"Emphasis on Speed 20:60:20":         weightedGeo(0.20, 0.0, 0.60, 0.20),
	}

	summary := &JevParitySummary{
		ProtocolVersion:        "jevbench::v1.4",
		TemperatureApplied:     activeTemp,
		OptimalTemperature:     optT,
		ChanceBaselinePct:      chanceBasePct,
		ChanceCorrectedAccPct:  chanceCorrPct,
		IntelligenceScore:      intelScore,
		CalibrationScore:       calScore,
		SpeedScore:             adjSpeed,
		CostScore:              costScore,
		CompositeJevBenchScore: composite,
		ECE10Bin:               ece,
		BrierMean:              brierMean,
		SoftTVDMean:            tvdMean,
		ProbabilityFidelity:    math.Max(0.0, 100.0*(1.0-tvdMean)),
		P50LatencySec:          p50Sec,
		P95LatencySec:          p95Sec,
		USDPer1kDecisions:      usdPer1k,
		Presets:                presets,
		RawT1ECE10Bin:          t1ECE,
		RawT1BrierMean:         t1BrierMean,
		RawT1CalibrationScore:  t1CalScore,
		RawT1CompositeScore:    t1Composite,
		ReliabilityBins:        bins,

		// v1.4 additions
		EndpointKind:          endpointKind,
		RawSpeedScore:         rawSpeed,
		AdjustedSpeedScore:    adjSpeed,
		V14IntelligenceScore:  v14Intel,
		V14CalibrationScore:   calScore,
		V14CompositeScore:     v14Composite,
		V14PublicOnlyEstimate: publicOnly,
		SealedAccuracy:        sealedAcc,
		GapPenaltyPct:         gapPenalty,
	}
	return summary, scaledCases
}

// printJevParityDashboard renders the JevBench v1.3.1 & v1.4 4-Axis Scorecard and 10-Bin Reliability Diagram.
func printJevParityDashboard(report CalibrationReport) {
	jp := report.JevParity
	if jp == nil {
		return
	}

	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================="))
	fmt.Println(styleAccent.Render("  Table 4: JevBench 4-Axis Parity Scorecard (v1.3.1 & v1.4 Official Method)"))
	fmt.Println(styleAccent.Render("=========================================================================================="))
	fmt.Printf("  • Chance Guessing Baseline:   %.2f%% -> Chance-Corrected Accuracy: %s (Raw: %.1f%%)\n",
		jp.ChanceBaselinePct,
		stylePass.Render(fmt.Sprintf("%.2f%%", jp.ChanceCorrectedAccPct)),
		report.OverallAccuracyPct,
	)
	if jp.TemperatureMethod == "kfold" {
		fmt.Printf("  • Temperature Scaling (T):    held-out 5-fold, per-case T in temperature_applied (in-sample T* = %.2f, not applied)\n",
			jp.OptimalTemperature)
	} else {
		fmt.Printf("  • Temperature Scaling (T):    T = %.2f (Optimal Fitted T* = %.2f, in-sample; Argmax Accuracy Invariant)\n",
			jp.TemperatureApplied, jp.OptimalTemperature)
	}
	fmt.Printf("  • Calibration Metrics:        10-Bin ECE = %.4f (Raw T=1.0: %.4f) | Multi-Class Brier = %.4f (Raw: %.4f)\n",
		jp.ECE10Bin, jp.RawT1ECE10Bin, jp.BrierMean, jp.RawT1BrierMean)
	endpointDesc := jp.EndpointKind
	if endpointDesc == "" {
		endpointDesc = "gpu"
	}
	fmt.Printf("  • Latency Percentiles:        p50 = %.3fs (%.0f ms) | p95 = %.3fs (%.0f ms) [endpoint=%s]\n",
		jp.P50LatencySec, jp.P50LatencySec*1000.0, jp.P95LatencySec, jp.P95LatencySec*1000.0, endpointDesc)
	fmt.Printf("  • Unit Economics:             $%.4f per 1,000 decisions\n", jp.USDPer1kDecisions)
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))
	rawCol, scaledCol := "RAW (T=1.00)", fmt.Sprintf("SCALED (T=%.2f)", jp.TemperatureApplied)
	if jp.TemperatureMethod == "kfold" {
		rawCol, scaledCol = "HELD-OUT", "HELD-OUT"
	}
	fmt.Printf("  %-34s %14s %14s %14s\n", "JEVBENCH 4 AXES (25% EACH)", rawCol, scaledCol, "OFFICIAL v1.4")
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))
	intelV14Note := ""
	if jp.V14PublicOnlyEstimate {
		intelV14Note = " *"
	}
	fmt.Printf("  %-34s %14.2f %14.2f %14.2f%s\n", "1. Intelligence (Chance-Corrected)", jp.IntelligenceScore, jp.IntelligenceScore, jp.V14IntelligenceScore, intelV14Note)
	fmt.Printf("  %-34s %14.2f %14.2f %14.2f\n", "2. Calibration (ECE + Soft TVD)", jp.RawT1CalibrationScore, jp.CalibrationScore, jp.V14CalibrationScore)
	speedNote := ""
	if jp.RawSpeedScore > 0 && jp.RawSpeedScore != jp.AdjustedSpeedScore {
		speedNote = fmt.Sprintf(" (raw: %.1f)", jp.RawSpeedScore)
	}
	fmt.Printf("  %-34s %14.2f %14.2f %14.2f%s\n", "3. Speed (p50/p95 Log-Decade)", jp.SpeedScore, jp.SpeedScore, jp.AdjustedSpeedScore, speedNote)
	fmt.Printf("  %-34s %14.2f %14.2f %14.2f\n", "4. Cost ($ / 1,000 Decisions)", jp.CostScore, jp.CostScore, jp.CostScore)
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))
	fmt.Printf("  %-34s %14.2f %14s %+14.2f\n",
		"v1.3.1 COMPOSITE SCORE (GeoMean)",
		jp.RawT1CompositeScore,
		stylePass.Render(fmt.Sprintf("%14.2f", jp.CompositeJevBenchScore)),
		jp.CompositeJevBenchScore-jp.RawT1CompositeScore,
	)
	v14ScoreDisplay := fmt.Sprintf("%14.2f", jp.V14CompositeScore)
	if jp.V14PublicOnlyEstimate {
		v14ScoreDisplay = stylePass.Render(v14ScoreDisplay) + "*"
	} else {
		v14ScoreDisplay = stylePass.Render(v14ScoreDisplay)
	}
	fmt.Printf("  %-34s %14s %14s %14s\n",
		"v1.4 COMPOSITE SCORE (Harmonic)",
		"--",
		v14ScoreDisplay,
		"(3 gates <50)",
	)
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))
	if jp.V14PublicOnlyEstimate {
		fmt.Println(styleMuted.Render("  * Note: v1.4 is a public-only estimate. Official JevBench v1.4 scoring includes 308 private sealed"))
		fmt.Println(styleMuted.Render("    decisions evaluated by the benchmark maintainer."))
	} else if jp.SealedAccuracy > 0 {
		fmt.Printf("  * Official sealed accuracy: %.1f%% | Gap allowance: 25.0 pts | Gap penalty: %.1f%%\n",
			jp.SealedAccuracy*100.0, jp.GapPenaltyPct)
	}
	fmt.Printf("  Preset Views (v1.3.1): Balanced 33:33:33 = %.2f | Accuracy 60:20:20 = %.2f | Speed 20:60:20 = %.2f\n",
		jp.Presets["Balanced 33:33:33 (no calibration)"],
		jp.Presets["Emphasis on Accuracy 60:20:20"],
		jp.Presets["Emphasis on Speed 20:60:20"],
	)
	fmt.Println(styleAccent.Render("=========================================================================================="))

	// Render 10-Bin Reliability Diagram (populated bins)
	fmt.Println()
	fmt.Println(styleAccent.Render("=========================================================================================="))
	fmt.Printf("  Table 5: 10-Bin Reliability Diagram (T = %.2f, 10-Bin ECE = %.4f)\n", jp.TemperatureApplied, jp.ECE10Bin)
	fmt.Println(styleAccent.Render("=========================================================================================="))
	fmt.Printf("  %-12s %6s %10s %10s %9s  %-30s\n", "CONF BIN", "ITEMS", "MEAN CONF", "EMP ACC", "|GAP|", "CALIBRATION BAR (CONF vs ACC)")
	fmt.Println(styleMuted.Render("  ----------------------------------------------------------------------------------------"))
	for _, b := range jp.ReliabilityBins {
		if b.Count == 0 {
			continue
		}
		confBars := int(math.Round(b.AvgConf * 20.0))
		accBars := int(math.Round(b.EmpiricalAcc * 20.0))
		barVisual := fmt.Sprintf("C:%-20s A:%-20s", strings.Repeat("█", confBars), strings.Repeat("▒", accBars))
		fmt.Printf("  %-12s %6d %9.1f%% %9.1f%% %8.1f%%  %s\n",
			b.RangeLabel,
			b.Count,
			b.AvgConf*100.0,
			b.EmpiricalAcc*100.0,
			b.AbsGap*100.0,
			barVisual,
		)
	}
	fmt.Println(styleAccent.Render("=========================================================================================="))
}
