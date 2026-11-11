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
	"math"
	"regexp"
	"strconv"
	"strings"
)

const maxReasonablePackCount = 500

var (
	// Compound multiplier before explicit unit volume/weight/count (for example, "2 x 500ml", "6x330 ml", "12 x 16 fl oz").
	reCompoundUnitPrefix = regexp.MustCompile(
		`(?i)\b(\d+)\s*[x×]\s*\d+(?:\.\d+)?\s*(?:ml|l|ltr|oz|fl\.?\s*oz|g|gm|kg|lb|lbs|mg|cl|ct|count|pk|pack|packs|pcs|caps|capsules|tabs|tablets|rolls|sheets|wipes|bars|cans|bottles|pouches|pods|sachets)\b`,
	)
	// Reverse compound unit before multiplier (for example, "500ml x 2", "330 ml x 6").
	reCompoundUnitSuffix = regexp.MustCompile(
		`(?i)\b\d+(?:\.\d+)?\s*(?:ml|l|ltr|oz|fl\.?\s*oz|g|gm|kg|lb|lbs|mg|cl)\s*[x×]\s*(\d+)\b`,
	)
	// Explicit "pack of N", "case of N", "box of N", "bundle of N", or "carton of N".
	rePackOfCount = regexp.MustCompile(
		`(?i)\b(?:pack|case|box|bundle|carton)\s+of\s+(\d+)\b`,
	)
	// Hyphenated or spaced count before multipack container nouns (for example, "6-pack", "24 ct", "12 cans").
	reHyphenatedPack = regexp.MustCompile(
		`(?i)\b(\d+)\s*(?:-|\s)?\s*(?:pack|packs|pk|count|ct|cans|bottles|bars|pods|pouches|sachets)\b`,
	)
	// Bare compound "N x M" without unit suffix when N is a modest pack count (<= 96) and M >= 10 (excluding "4x4" or "1920x1080").
	reBareCompound = regexp.MustCompile(
		`(?i)\b(\d{1,2})\s*[x×]\s*(\d{2,4})\b`,
	)
	reTwinPack   = regexp.MustCompile(`(?i)\b(?:twin|double|duo)\s*(?:-|\s)?\s*pack\b`)
	reTriplePack = regexp.MustCompile(`(?i)\btriple\s*(?:-|\s)?\s*pack\b`)
	reQuadPack   = regexp.MustCompile(`(?i)\bquad\s*(?:-|\s)?\s*pack\b`)
)

// NormalizedOrdinalVariance computes Var(S) / ((L-1)^2 / 4) in [0, 1] for L >= 2 ordered rubric levels.
// Dividing by Popoviciu's maximum variance bound ((L-1)^2 / 4 on {0, ..., L-1}) makes thresholds scale-invariant
// across 3-point, 5-point, and 10-point ordered scales while distinguishing bimodal opposition ([0.5, 0, 0, 0, 0.5] -> 1.0)
// from benign adjacent rounding ([0, 0.5, 0.5, 0, 0] -> 0.0625).
// Returns 0.0 if len(probs) < 2, if any entry is NaN/Inf/negative, or if total probability mass is <= 0.
func NormalizedOrdinalVariance(probs []float64) float64 {
	numBuckets := len(probs)
	if numBuckets < 2 {
		return 0.0
	}

	totalMass := 0.0
	for _, prob := range probs {
		if math.IsNaN(prob) || math.IsInf(prob, 0) || prob < 0.0 {
			return 0.0
		}
		totalMass += prob
	}
	if totalMass <= 0.0 || math.IsNaN(totalMass) || math.IsInf(totalMass, 0) {
		return 0.0
	}

	meanPos := 0.0
	for idx, prob := range probs {
		normalizedProb := prob / totalMass
		meanPos += float64(idx) * normalizedProb
	}

	variance := 0.0
	for idx, prob := range probs {
		normalizedProb := prob / totalMass
		diff := float64(idx) - meanPos
		variance += normalizedProb * diff * diff
	}

	span := float64(numBuckets - 1)
	maxVariance := (span * span) / 4.0
	if maxVariance <= 0.0 {
		return 0.0
	}

	normalizedVariance := variance / maxVariance
	if normalizedVariance < 0.0 {
		return 0.0
	}
	if normalizedVariance > 1.0 {
		return 1.0
	}
	return normalizedVariance
}

// ParsePackMultiplier extracts an integer multipack count from a product title.
// Compound expressions ("2 x 500ml", "500ml x 4", "Pack of 12", "6-pack") are matched ahead of bare unit
// volumes or weights ("500ml", "1.5L", "250g") so single containers with metric/imperial sizes return 1.
func ParsePackMultiplier(title string) int {
	trimmed := strings.TrimSpace(title)
	if trimmed == "" {
		return 1
	}

	for _, pattern := range []*regexp.Regexp{
		reCompoundUnitPrefix,
		reCompoundUnitSuffix,
		rePackOfCount,
		reHyphenatedPack,
	} {
		matches := pattern.FindStringSubmatch(trimmed)
		if len(matches) == 2 {
			parsedCount, err := strconv.Atoi(matches[1])
			if err == nil && parsedCount >= 1 && parsedCount <= maxReasonablePackCount {
				return parsedCount
			}
		}
	}

	if reTwinPack.MatchString(trimmed) {
		return 2
	}
	if reTriplePack.MatchString(trimmed) {
		return 3
	}
	if reQuadPack.MatchString(trimmed) {
		return 4
	}

	if bareMatches := reBareCompound.FindStringSubmatch(trimmed); len(bareMatches) == 3 {
		packCount, errCount := strconv.Atoi(bareMatches[1])
		unitSize, errSize := strconv.Atoi(bareMatches[2])
		if errCount == nil && errSize == nil && packCount >= 1 && packCount <= 96 && unitSize >= 10 && unitSize <= 5000 {
			return packCount
		}
	}

	return 1
}
