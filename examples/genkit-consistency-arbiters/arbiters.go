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

package main

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/firebase/genkit/go/genkit"
)

// PairwiseEdge represents an entity-resolution equivalence judgment between NodeA and NodeB
// with estimated equivalence probability EquivProb in [0, 1].
type PairwiseEdge struct {
	NodeA     string  `json:"node_a"`
	NodeB     string  `json:"node_b"`
	EquivProb float64 `json:"equiv_prob"`
}

// TriangleContradiction reports a 3-cycle (A, B, C) that violates transitive equivalence:
// two edges have P(equiv) >= highThresh while the third edge has P(equiv) <= lowThresh.
type TriangleContradiction struct {
	Nodes     [3]string  `json:"nodes"`
	EdgeProbs [3]float64 `json:"edge_probs"` // [P(A,B), P(B,C), P(A,C)]
	WeakEdge  [2]string  `json:"weak_edge"`
	WeakProb  float64    `json:"weak_prob"`
	Severity  float64    `json:"severity"` // min(strong1, strong2) - weakProb
}

// ProbabilisticTriangleContradiction scores a 3-cycle (A, B, C) using continuous transitivity-violation mass
// across all three 2-positive-1-negative rotations:
//
//	m_ab = P(B,C) * P(A,C) * (1 - P(A,B))
//	m_bc = P(A,B) * P(A,C) * (1 - P(B,C))
//	m_ac = P(A,B) * P(B,C) * (1 - P(A,C))
//
// Continuous violation mass captures near-threshold contradictions (for example, 0.79, 0.85, 0.22)
// and provides a smooth ranking signal for audit escalation.
type ProbabilisticTriangleContradiction struct {
	Nodes              [3]string  `json:"nodes"`
	EdgeProbs          [3]float64 `json:"edge_probs"` // [P(A,B), P(B,C), P(A,C)]
	WeakEdge           [2]string  `json:"weak_edge"`
	WeakProb           float64    `json:"weak_prob"`
	MaxViolationMass   float64    `json:"max_violation_mass"`
	TotalViolationMass float64    `json:"total_violation_mass"`
}

type undirectedEdgeKey struct{ u, v string }

func buildUndirectedProbGraph(edges []PairwiseEdge) ([]string, map[string][]string, map[undirectedEdgeKey]float64) {
	type accum struct {
		sum   float64
		count int
	}
	pairMap := make(map[undirectedEdgeKey]accum)
	nodeSet := make(map[string]struct{})

	for _, e := range edges {
		if math.IsNaN(e.EquivProb) {
			continue
		}
		u := strings.TrimSpace(e.NodeA)
		v := strings.TrimSpace(e.NodeB)
		if u == "" || v == "" || u == v {
			continue
		}
		if u > v {
			u, v = v, u
		}
		p := math.Max(0.0, math.Min(1.0, e.EquivProb))
		nodeSet[u] = struct{}{}
		nodeSet[v] = struct{}{}
		k := undirectedEdgeKey{u, v}
		cur := pairMap[k]
		cur.sum += p
		cur.count++
		pairMap[k] = cur
	}

	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	adj := make(map[string][]string, len(nodes))
	probOf := make(map[undirectedEdgeKey]float64, len(pairMap))
	for k, acc := range pairMap {
		probOf[k] = acc.sum / float64(acc.count)
		adj[k.u] = append(adj[k.u], k.v)
	}
	for u := range adj {
		sort.Strings(adj[u])
	}
	return nodes, adj, probOf
}

// CheckTransitiveTriangles inspects a set of pairwise entity-resolution edges and returns all
// 3-cycles (A, B, C) where two edges have EquivProb >= highThresh and the third has EquivProb <= lowThresh.
// If highThresh <= 0, it defaults to 0.80; if lowThresh <= 0, it defaults to 0.25.
func CheckTransitiveTriangles(edges []PairwiseEdge, highThresh, lowThresh float64) []TriangleContradiction {
	if highThresh <= 0 {
		highThresh = 0.80
	}
	if lowThresh <= 0 {
		lowThresh = 0.25
	}
	nodes, adj, probOf := buildUndirectedProbGraph(edges)

	var out []TriangleContradiction
	for _, a := range nodes {
		neighborsA := adj[a]
		for i := 0; i < len(neighborsA); i++ {
			b := neighborsA[i]
			for j := i + 1; j < len(neighborsA); j++ {
				c := neighborsA[j]
				pbc, ok := probOf[undirectedEdgeKey{b, c}]
				if !ok {
					continue
				}
				pab := probOf[undirectedEdgeKey{a, b}]
				pac := probOf[undirectedEdgeKey{a, c}]

				probs := [3]float64{pab, pbc, pac}
				pairs := [3][2]string{{a, b}, {b, c}, {a, c}}

				highCount := 0
				lowIdx := -1
				for idx, p := range probs {
					if p >= highThresh {
						highCount++
					} else if p <= lowThresh {
						lowIdx = idx
					}
				}
				if highCount == 2 && lowIdx >= 0 {
					minStrong := 1.0
					for idx, p := range probs {
						if idx != lowIdx && p < minStrong {
							minStrong = p
						}
					}
					out = append(out, TriangleContradiction{
						Nodes:     [3]string{a, b, c},
						EdgeProbs: probs,
						WeakEdge:  pairs[lowIdx],
						WeakProb:  probs[lowIdx],
						Severity:  minStrong - probs[lowIdx],
					})
				}
			}
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if math.Abs(out[i].Severity-out[j].Severity) > 1e-9 {
			return out[i].Severity > out[j].Severity
		}
		if out[i].Nodes[0] != out[j].Nodes[0] {
			return out[i].Nodes[0] < out[j].Nodes[0]
		}
		if out[i].Nodes[1] != out[j].Nodes[1] {
			return out[i].Nodes[1] < out[j].Nodes[1]
		}
		return out[i].Nodes[2] < out[j].Nodes[2]
	})
	return out
}

// ScoreProbabilisticTriangles evaluates all 3-cycles in edges and returns those with MaxViolationMass >= minMass
// (default 0.35 when minMass <= 0), sorted descending by MaxViolationMass.
func ScoreProbabilisticTriangles(edges []PairwiseEdge, minMass float64) []ProbabilisticTriangleContradiction {
	if minMass <= 0 {
		minMass = 0.35
	}
	nodes, adj, probOf := buildUndirectedProbGraph(edges)

	var out []ProbabilisticTriangleContradiction
	for _, a := range nodes {
		neighborsA := adj[a]
		for i := 0; i < len(neighborsA); i++ {
			b := neighborsA[i]
			for j := i + 1; j < len(neighborsA); j++ {
				c := neighborsA[j]
				pbc, ok := probOf[undirectedEdgeKey{b, c}]
				if !ok {
					continue
				}
				pab := probOf[undirectedEdgeKey{a, b}]
				pac := probOf[undirectedEdgeKey{a, c}]

				probs := [3]float64{pab, pbc, pac}
				pairs := [3][2]string{{a, b}, {b, c}, {a, c}}
				masses := [3]float64{
					pbc * pac * (1.0 - pab), // weak edge (a,b)
					pab * pac * (1.0 - pbc), // weak edge (b,c)
					pab * pbc * (1.0 - pac), // weak edge (a,c)
				}

				bestIdx := 0
				maxM := masses[0]
				totalM := masses[0] + masses[1] + masses[2]
				for idx := 1; idx < 3; idx++ {
					if masses[idx] > maxM {
						maxM = masses[idx]
						bestIdx = idx
					}
				}
				if maxM >= minMass {
					out = append(out, ProbabilisticTriangleContradiction{
						Nodes:              [3]string{a, b, c},
						EdgeProbs:          probs,
						WeakEdge:           pairs[bestIdx],
						WeakProb:           probs[bestIdx],
						MaxViolationMass:   maxM,
						TotalViolationMass: totalM,
					})
				}
			}
		}
	}

	sort.Slice(out, func(i, j int) bool {
		if math.Abs(out[i].MaxViolationMass-out[j].MaxViolationMass) > 1e-9 {
			return out[i].MaxViolationMass > out[j].MaxViolationMass
		}
		if out[i].Nodes[0] != out[j].Nodes[0] {
			return out[i].Nodes[0] < out[j].Nodes[0]
		}
		if out[i].Nodes[1] != out[j].Nodes[1] {
			return out[i].Nodes[1] < out[j].Nodes[1]
		}
		return out[i].Nodes[2] < out[j].Nodes[2]
	})
	return out
}

const maxReasonablePackCount = 500

type packTokenKind uint8

const (
	tokSep packTokenKind = iota
	tokInt
	tokDecimal
	tokWord
	tokTimes
	tokHyphen
)

type packToken struct {
	kind        packTokenKind
	intVal      int
	word        string
	spaceBefore bool
}

var (
	compoundUnitWords = map[string]bool{
		"ml": true, "l": true, "ltr": true, "oz": true, "g": true, "gm": true,
		"kg": true, "lb": true, "lbs": true, "mg": true, "cl": true, "ct": true,
		"count": true, "pk": true, "pack": true, "packs": true, "pcs": true,
		"caps": true, "capsules": true, "tabs": true, "tablets": true,
		"rolls": true, "sheets": true, "wipes": true, "bars": true,
		"cans": true, "bottles": true, "pouches": true, "pods": true, "sachets": true,
	}
	volumeOrWeightWords = map[string]bool{
		"ml": true, "l": true, "ltr": true, "oz": true, "g": true, "gm": true,
		"kg": true, "lb": true, "lbs": true, "mg": true, "cl": true,
	}
	containerMultWords = map[string]bool{
		"pack": true, "packs": true, "cans": true, "bottles": true, "bars": true,
		"boxes": true, "cases": true, "cartons": true, "bundles": true, "jars": true, "tins": true,
	}
	packOfContainerWords = map[string]bool{
		"pack": true, "case": true, "box": true, "bundle": true, "carton": true,
	}
	hyphenatedPackWords = map[string]bool{
		"pack": true, "packs": true, "pk": true, "count": true, "ct": true,
		"cans": true, "bottles": true, "bars": true,
	}
)

func tokenizePackTitle(title string) []packToken {
	trimmed := strings.TrimSpace(title)
	if trimmed == "" {
		return nil
	}
	tokens := make([]packToken, 0, 16)
	spaceRun := 0
	n := len(trimmed)
	for i := 0; i < n; {
		ch := trimmed[i]
		if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
			start := i
			for i < n && (trimmed[i] == ' ' || trimmed[i] == '\t' || trimmed[i] == '\r' || trimmed[i] == '\n') {
				i++
			}
			spaceRun += i - start
			if spaceRun > 3 && (len(tokens) == 0 || tokens[len(tokens)-1].kind != tokSep) {
				tokens = append(tokens, packToken{kind: tokSep})
			}
			continue
		}
		hasSpace := spaceRun > 0
		spaceRun = 0

		// UTF-8 multiplication sign '×' (0xC3 0x97).
		if ch == 0xC3 && i+1 < n && trimmed[i+1] == 0x97 {
			tokens = append(tokens, packToken{kind: tokTimes, spaceBefore: hasSpace})
			i += 2
			continue
		}
		if ch == '-' {
			start := i
			for i < n && trimmed[i] == '-' {
				i++
			}
			if i-start == 1 {
				tokens = append(tokens, packToken{kind: tokHyphen, spaceBefore: hasSpace})
			} else if len(tokens) == 0 || tokens[len(tokens)-1].kind != tokSep {
				tokens = append(tokens, packToken{kind: tokSep})
			}
			continue
		}
		if ch >= '0' && ch <= '9' {
			start := i
			for i < n && trimmed[i] >= '0' && trimmed[i] <= '9' {
				i++
			}
			digitLen := i - start
			if i+1 < n && trimmed[i] == '.' && trimmed[i+1] >= '0' && trimmed[i+1] <= '9' {
				i++
				for i < n && trimmed[i] >= '0' && trimmed[i] <= '9' {
					i++
				}
				tokens = append(tokens, packToken{kind: tokDecimal, intVal: -1, spaceBefore: hasSpace})
				continue
			}
			val := -1
			if digitLen <= 4 {
				parsed := 0
				for k := start; k < i; k++ {
					parsed = parsed*10 + int(trimmed[k]-'0')
				}
				val = parsed
			}
			tokens = append(tokens, packToken{kind: tokInt, intVal: val, spaceBefore: hasSpace})
			continue
		}
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') {
			start := i
			for i < n && ((trimmed[i] >= 'a' && trimmed[i] <= 'z') || (trimmed[i] >= 'A' && trimmed[i] <= 'Z')) {
				i++
			}
			wordLen := i - start
			word := "_long_"
			if wordLen <= 16 {
				word = strings.ToLower(trimmed[start:i])
			}
			if word == "x" {
				tokens = append(tokens, packToken{kind: tokTimes, word: "x", spaceBefore: hasSpace})
			} else {
				tokens = append(tokens, packToken{kind: tokWord, word: word, spaceBefore: hasSpace})
			}
			continue
		}
		if ch == '.' {
			spaceRun = 1
			i++
			continue
		}
		if len(tokens) == 0 || tokens[len(tokens)-1].kind != tokSep {
			tokens = append(tokens, packToken{kind: tokSep})
		}
		i++
	}
	return tokens
}

func matchesUnitAt(tokens []packToken, idx int, unitSet map[string]bool) bool {
	if idx >= len(tokens) || tokens[idx].kind != tokWord {
		return false
	}
	if unitSet[tokens[idx].word] {
		return true
	}
	if tokens[idx].word == "fl" && idx+1 < len(tokens) && tokens[idx+1].kind == tokWord && tokens[idx+1].word == "oz" {
		return true
	}
	return false
}

// ParsePackMultiplier extracts an integer multipack count from a product title in single-pass O(N) time
// without regular-expression backtracking (for example, "2 x 500ml" -> 2, "6x330ml" -> 6,
// "6x Brand Product 330ml" -> 6, "Pack of 12" -> 12, "24-Pack" -> 24, "2 x 20 pods" -> 2).
// Bare single-container unit sizes ("500ml", "1.5L", "250g") and standalone unit-content counts ("20 pods") return 1.
func ParsePackMultiplier(title string) int {
	tokens := tokenizePackTitle(title)
	numToks := len(tokens)
	if numToks < 2 {
		return 1
	}

	// 1. Compound multiplier before explicit unit ("2 x 500ml", "6x330ml", "12×16 fl oz", "2 x 20 pods").
	for i := 0; i+3 < numToks; i++ {
		if tokens[i].kind == tokInt && tokens[i].intVal >= 1 && tokens[i].intVal <= maxReasonablePackCount &&
			tokens[i+1].kind == tokTimes &&
			(tokens[i+2].kind == tokInt || tokens[i+2].kind == tokDecimal) &&
			matchesUnitAt(tokens, i+3, compoundUnitWords) {
			return tokens[i].intVal
		}
	}

	// 2. Reverse compound unit before multiplier ("500ml x 4", "16 fl oz x 12").
	for i := 0; i+3 < numToks; i++ {
		if tokens[i].kind != tokInt && tokens[i].kind != tokDecimal {
			continue
		}
		unitSpan := 0
		if i+1 < numToks && tokens[i+1].kind == tokWord && volumeOrWeightWords[tokens[i+1].word] {
			unitSpan = 1
		} else if i+2 < numToks && tokens[i+1].kind == tokWord && tokens[i+1].word == "fl" &&
			tokens[i+2].kind == tokWord && tokens[i+2].word == "oz" {
			unitSpan = 2
		}
		multIdx := i + 1 + unitSpan + 1
		if unitSpan > 0 && multIdx < numToks &&
			tokens[i+1+unitSpan].kind == tokTimes &&
			tokens[multIdx].kind == tokInt &&
			tokens[multIdx].intVal >= 1 && tokens[multIdx].intVal <= maxReasonablePackCount {
			return tokens[multIdx].intVal
		}
	}

	// 3. Leading multipack prefix ("6x Brand Product 330ml", "12 x Sparkling Water").
	if numToks >= 3 && tokens[0].kind == tokInt &&
		tokens[0].intVal >= 1 && tokens[0].intVal <= maxReasonablePackCount &&
		tokens[1].kind == tokTimes &&
		tokens[2].kind == tokWord && tokens[2].spaceBefore {
		return tokens[0].intVal
	}

	// 4. Multiplier before container noun ("6x Cans", "4 x Bottles").
	for i := 0; i+2 < numToks; i++ {
		if tokens[i].kind == tokInt && tokens[i].intVal >= 1 && tokens[i].intVal <= maxReasonablePackCount &&
			tokens[i+1].kind == tokTimes &&
			tokens[i+2].kind == tokWord && containerMultWords[tokens[i+2].word] {
			return tokens[i].intVal
		}
	}

	// 5. Explicit "<container> of N" ("Pack of 12", "Case of 24", "Box of 6").
	for i := 0; i+2 < numToks; i++ {
		if tokens[i].kind == tokWord && packOfContainerWords[tokens[i].word] &&
			tokens[i+1].kind == tokWord && tokens[i+1].word == "of" &&
			tokens[i+2].kind == tokInt && tokens[i+2].intVal >= 1 && tokens[i+2].intVal <= maxReasonablePackCount {
			return tokens[i+2].intVal
		}
	}

	// 6. Hyphenated or spaced count before multipack noun ("24-Pack", "6-pack", "10 ct").
	for i := 0; i+1 < numToks; i++ {
		if tokens[i].kind != tokInt || tokens[i].intVal < 1 || tokens[i].intVal > maxReasonablePackCount {
			continue
		}
		if tokens[i+1].kind == tokWord && hyphenatedPackWords[tokens[i+1].word] {
			return tokens[i].intVal
		}
		if i+2 < numToks && tokens[i+1].kind == tokHyphen &&
			tokens[i+2].kind == tokWord && hyphenatedPackWords[tokens[i+2].word] {
			return tokens[i].intVal
		}
	}

	// 7. Named multipacks ("Twin-Pack", "Triple Pack", "Quad Pack").
	for i := 0; i+1 < numToks; i++ {
		if tokens[i].kind != tokWord {
			continue
		}
		nextIdx := i + 1
		if nextIdx < numToks && tokens[nextIdx].kind == tokHyphen {
			nextIdx++
		}
		if nextIdx >= numToks || tokens[nextIdx].kind != tokWord || tokens[nextIdx].word != "pack" {
			continue
		}
		switch tokens[i].word {
		case "twin", "double", "duo":
			return 2
		case "triple":
			return 3
		case "quad":
			return 4
		}
	}

	// 8. Bare compound "N x M" when 1 <= N <= 96 and 10 <= M <= 5000 (excluding "4x4" and "1920x1080").
	for i := 0; i+2 < numToks; i++ {
		if tokens[i].kind == tokInt && tokens[i].intVal >= 1 && tokens[i].intVal <= 96 &&
			tokens[i+1].kind == tokTimes &&
			tokens[i+2].kind == tokInt && tokens[i+2].intVal >= 10 && tokens[i+2].intVal <= 5000 {
			return tokens[i].intVal
		}
	}

	return 1
}

// NormalizedOrdinalVariance computes Var(S) / ((L-1)^2 / 4) in [0, 1] for L >= 2 ordered rubric levels.
// Dividing variance on {0, ..., L-1} by Popoviciu's maximum variance bound ((L-1)^2 / 4) makes cutoffs
// scale-invariant across ordered rubrics while separating bimodal opposition ([0.5, 0, 0, 0, 0.5] -> 1.0)
// from benign adjacent splits ([0, 0.5, 0.5, 0, 0] -> 0.0625).
// Returns 0.0 when len(probs) < 2, when any entry is negative/NaN/Inf, or when total mass <= 0.
func NormalizedOrdinalVariance(probs []float64) float64 {
	numLevels := len(probs)
	if numLevels < 2 {
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
		meanPos += float64(idx) * (prob / totalMass)
	}

	variance := 0.0
	for idx, prob := range probs {
		diff := float64(idx) - meanPos
		variance += (prob / totalMass) * diff * diff
	}

	span := float64(numLevels - 1)
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

// SidakSlotAlpha returns the per-slot miscoverage level 1 - (1 - bundleAlpha)^(1/numSlots)
// for family-wise error control across numSlots concurrent slots in a multi-slot bundle.
func SidakSlotAlpha(bundleAlpha float64, numSlots int) float64 {
	if numSlots <= 0 {
		return 0.0
	}
	if math.IsNaN(bundleAlpha) || math.IsInf(bundleAlpha, 0) || bundleAlpha <= 0.0 || bundleAlpha >= 1.0 {
		bundleAlpha = 0.10
	}
	return 1.0 - math.Pow(1.0-bundleAlpha, 1.0/float64(numSlots))
}

// OrderedProbabilities extracts probability mass in the exact order of levels,
// supporting both named level keys and systemonex "0".."L-1" index keys.
func OrderedProbabilities(probMap map[string]float64, levels []string) []float64 {
	out := make([]float64, len(levels))
	for idx, level := range levels {
		if val, ok := probMap[level]; ok {
			out[idx] = val
		} else if valIdx, okIdx := probMap[strconv.Itoa(idx)]; okIdx {
			out[idx] = valIdx
		}
	}
	return out
}

func resolveOrderedChoiceLabel(choice string, legend map[string]string, probs map[string]float64, levels []string) string {
	raw := topProbabilityKey(probs, choice)
	if mapped, ok := legend[raw]; ok && mapped != "" {
		return mapped
	}
	if idx, err := strconv.Atoi(raw); err == nil && idx >= 0 && idx < len(levels) {
		return levels[idx]
	}
	return raw
}

// ShouldEscalateOrdinalSlot evaluates an ordered rubric slot using Popoviciu normalized variance
// and optional Sidak support-span gating rather than Shannon entropy.
// Invalid or empty probability vectors fail closed (true).
func ShouldEscalateOrdinalSlot(probs []float64, maxOrdinalVar, minSupportProb float64) bool {
	numLevels := len(probs)
	if numLevels < 2 {
		return true
	}
	normProbs := make([]float64, numLevels)
	totalMass := 0.0
	for _, prob := range probs {
		if math.IsNaN(prob) || math.IsInf(prob, 0) || prob < 0.0 {
			return true
		}
		totalMass += prob
	}
	if totalMass <= 0.0 || math.IsNaN(totalMass) || math.IsInf(totalMass, 0) {
		return true
	}
	for idx, prob := range probs {
		normProbs[idx] = prob / totalMass
	}

	if maxOrdinalVar <= 0.0 || math.IsNaN(maxOrdinalVar) {
		// Permit benign 50/50 adjacent rounding (1 / (L-1)^2) while catching >= 2-step or polar splits.
		adjacentBound := 1.0 / float64((numLevels-1)*(numLevels-1))
		maxOrdinalVar = math.Max(0.10, adjacentBound+0.01)
	}
	if NormalizedOrdinalVariance(normProbs) > maxOrdinalVar {
		return true
	}

	if minSupportProb > 0.0 && !math.IsNaN(minSupportProb) {
		minIdx := -1
		maxIdx := -1
		for idx, prob := range normProbs {
			if prob >= minSupportProb {
				if minIdx == -1 {
					minIdx = idx
				}
				maxIdx = idx
			}
		}
		if minIdx == -1 || (maxIdx-minIdx) > 1 {
			return true
		}
	}
	return false
}

// BundleSlotGateInput describes one slot inside a multi-slot decision bundle for selective escalation.
type BundleSlotGateInput struct {
	SlotID        string
	OrderedLevels []string
	Probabilities map[string]float64
	Hesitation    float64
}

// SelectEscalatedBundleSlots returns the subset of slot IDs that require Stage-2 escalation,
// avoiding bundle-wide escalation blow-up on multi-slot policies:
//   - Ordered rubric slots (len(OrderedLevels) >= 2) escalate via ShouldEscalateOrdinalSlot.
//   - Nominal and boolean slots escalate when Hesitation >= maxNominalHesitation (default 0.16).
func SelectEscalatedBundleSlots(slots []BundleSlotGateInput, maxNominalHesitation, maxOrdinalVar, minSupportProb float64) []string {
	if maxNominalHesitation <= 0.0 || math.IsNaN(maxNominalHesitation) {
		maxNominalHesitation = 0.16
	}
	var escalated []string
	for _, slot := range slots {
		if len(slot.OrderedLevels) >= 2 {
			probs := OrderedProbabilities(slot.Probabilities, slot.OrderedLevels)
			if ShouldEscalateOrdinalSlot(probs, maxOrdinalVar, minSupportProb) {
				escalated = append(escalated, slot.SlotID)
			}
			continue
		}
		if math.IsNaN(slot.Hesitation) || math.IsInf(slot.Hesitation, 0) || slot.Hesitation >= maxNominalHesitation {
			escalated = append(escalated, slot.SlotID)
		}
	}
	return escalated
}

// ValidatePackQuantityConsistency reports whether two product titles' parsed pack multipliers
// are consistent with the model's entity_relation and pack_quantity_match slot predictions.
func ValidatePackQuantityConsistency(titleA, titleB, predictedRelation, predictedPackMatch string) bool {
	multA := ParsePackMultiplier(titleA)
	multB := ParsePackMultiplier(titleB)
	if multA != multB {
		if predictedRelation == "exact_duplicate" || predictedPackMatch == "identical_pack_and_volume" {
			return false
		}
	}
	return true
}

// SwarmAgentProposal describes a single worker agent's proposed file/resource lock acquisition
// along with the model's slot confidence and normalized hesitation.
type SwarmAgentProposal struct {
	AgentID     string   `json:"agent_id"`
	Proposal    string   `json:"proposal,omitempty"`
	TargetPaths []string `json:"target_paths"`
	Confidence  float64  `json:"confidence"`
	Hesitation  float64  `json:"hesitation"`
}

// SwarmLockDecision records the deterministic lock arbitration result across concurrent worker proposals.
type SwarmLockDecision struct {
	AgentActions       map[string]string `json:"agent_actions"` // grant_lock, serialize_after_peer, escalate_conflict
	WriteCollision     bool              `json:"write_collision"`
	WinningAgent       string            `json:"winning_agent"`
	EscalateToHuman    bool              `json:"escalate_to_human"`
	RollbackSeverity   string            `json:"rollback_severity,omitempty"`
	RollbackOrdinalVar float64           `json:"rollback_ordinal_var,omitempty"`
	EscalatedSlots     []string          `json:"escalated_slots,omitempty"`
}

// ArbitrateSwarmLocks resolves file/resource lock collisions across concurrent worker agents.
// Non-colliding agents receive "grant_lock" in parallel. When two or more agents request overlapping
// TargetPaths, the agent with highest Confidence (and Hesitation < maxHesitation) wins "grant_lock"
// while colliding peers receive "serialize_after_peer" (or "escalate_conflict" if any colliding agent
// has Hesitation >= maxHesitation).
func ArbitrateSwarmLocks(proposals []SwarmAgentProposal, maxHesitation float64) SwarmLockDecision {
	if maxHesitation <= 0 {
		maxHesitation = 0.16
	}
	decision := SwarmLockDecision{
		AgentActions: make(map[string]string, len(proposals)),
	}
	if len(proposals) == 0 {
		return decision
	}

	pathOwners := make(map[string][]int)
	for idx, p := range proposals {
		seen := make(map[string]struct{})
		for _, rawPath := range p.TargetPaths {
			norm := strings.TrimSpace(rawPath)
			if norm == "" {
				continue
			}
			if _, ok := seen[norm]; ok {
				continue
			}
			seen[norm] = struct{}{}
			pathOwners[norm] = append(pathOwners[norm], idx)
		}
	}

	collidingIdx := make(map[int]bool)
	for _, idxs := range pathOwners {
		if len(idxs) > 1 {
			decision.WriteCollision = true
			for _, idx := range idxs {
				collidingIdx[idx] = true
			}
		}
	}

	order := make([]int, len(proposals))
	for i := range proposals {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := proposals[order[i]], proposals[order[j]]
		if math.Abs(a.Confidence-b.Confidence) > 1e-9 {
			return a.Confidence > b.Confidence
		}
		if math.Abs(a.Hesitation-b.Hesitation) > 1e-9 {
			return a.Hesitation < b.Hesitation
		}
		return a.AgentID < b.AgentID
	})

	decision.WinningAgent = proposals[order[0]].AgentID

	claimedPaths := make(map[string]string)
	for _, idx := range order {
		p := proposals[idx]
		if !collidingIdx[idx] {
			if p.Hesitation >= maxHesitation {
				decision.AgentActions[p.AgentID] = "escalate_conflict"
				decision.EscalateToHuman = true
			} else {
				decision.AgentActions[p.AgentID] = "grant_lock"
			}
			for _, path := range p.TargetPaths {
				if norm := strings.TrimSpace(path); norm != "" {
					claimedPaths[norm] = p.AgentID
				}
			}
			continue
		}

		if p.Hesitation >= maxHesitation {
			decision.AgentActions[p.AgentID] = "escalate_conflict"
			decision.EscalateToHuman = true
			continue
		}
		overlapAlreadyClaimed := false
		for _, path := range p.TargetPaths {
			norm := strings.TrimSpace(path)
			if norm == "" {
				continue
			}
			if _, taken := claimedPaths[norm]; taken {
				overlapAlreadyClaimed = true
				break
			}
		}
		if !overlapAlreadyClaimed {
			decision.AgentActions[p.AgentID] = "grant_lock"
			for _, path := range p.TargetPaths {
				if norm := strings.TrimSpace(path); norm != "" {
					claimedPaths[norm] = p.AgentID
				}
			}
		} else {
			decision.AgentActions[p.AgentID] = "serialize_after_peer"
		}
	}

	return decision
}

// EntityPairEvaluation combines the dgem entity_resolution_dedup policy readout with
// deterministic pack-multiplier post-verification.
type EntityPairEvaluation struct {
	NodeA          string
	NodeB          string
	Relation       string
	PackMatch      string
	MergeAction    string
	EquivProb      float64
	PackConsistent bool
	WallTimeMs     int64
}

// EvaluateEntityPairWithGuardrail executes templates/multimodal/entity_resolution_dedup.json.tmpl
// via Genkit and applies ValidatePackQuantityConsistency to quarantine multipack-vs-single false merges.
func EvaluateEntityPairWithGuardrail(ctx context.Context, g *genkit.Genkit, modelName, templatesDir, nodeA, titleA, nodeB, titleB string) (EntityPairEvaluation, error) {
	tmplPath := filepath.Join(templatesDir, "multimodal", "entity_resolution_dedup.json.tmpl")
	dec, err := dgemgenkit.DecideWithTemplate(ctx, g, modelName, tmplPath, map[string]any{
		"title_a": titleA,
		"title_b": titleB,
		"samples": 1,
	})
	if err != nil {
		return EntityPairEvaluation{}, err
	}
	relAns := dec.Answers["entity_relation"]
	packAns := dec.Answers["pack_quantity_match"]
	mergeAns := dec.Answers["merge_action"]
	equivProb := relAns.Probabilities["exact_duplicate"]
	consistent := ValidatePackQuantityConsistency(titleA, titleB, relAns.Choice, packAns.Choice)
	action := mergeAns.Choice
	if !consistent {
		action = "quarantine_for_human_review"
	}
	return EntityPairEvaluation{
		NodeA:          nodeA,
		NodeB:          nodeB,
		Relation:       relAns.Choice,
		PackMatch:      packAns.Choice,
		MergeAction:    action,
		EquivProb:      equivProb,
		PackConsistent: consistent,
		WallTimeMs:     dec.WallTimeMs,
	}, nil
}

var (
	rollbackSeverityLevels = []string{
		"clean_fast_forward",
		"minor_textual_rebase",
		"build_or_test_breakage",
		"corrupted_shared_state",
	}
	interruptUrgencyLevels = []string{
		"silent_no_alert",
		"passive_sidebar_update",
		"soft_nudge_before_closing",
		"immediate_compliance_interruption",
	}
)

// EvaluateSwarmLockProposals executes templates/swarm_lock_arbiter.json.tmpl for 4 worker proposals,
// feeds dgem's per-worker confidence and normalized hesitation into ArbitrateSwarmLocks, and applies
// Popoviciu ordinal variance gating on rollback_severity via SelectEscalatedBundleSlots.
func EvaluateSwarmLockProposals(ctx context.Context, g *genkit.Genkit, modelName, templatesDir string, workers [4]SwarmAgentProposal, maxHesitation float64) (SwarmLockDecision, int64, error) {
	tmplPath := filepath.Join(templatesDir, "swarm_lock_arbiter.json.tmpl")
	dec, err := dgemgenkit.DecideWithTemplate(ctx, g, modelName, tmplPath, map[string]any{
		"agent_1": workers[0].Proposal,
		"agent_2": workers[1].Proposal,
		"agent_3": workers[2].Proposal,
		"agent_4": workers[3].Proposal,
		"samples": 1,
	})
	if err != nil {
		return SwarmLockDecision{}, 0, err
	}
	enriched := make([]SwarmAgentProposal, 4)
	bundleInputs := make([]BundleSlotGateInput, 0, 8)
	for i := range workers {
		slotID := fmt.Sprintf("agent_%d_lock_action", i+1)
		ans := dec.Answers[slotID]
		hes := dec.Hesitation(slotID)
		enriched[i] = workers[i]
		enriched[i].Confidence = ans.Confidence
		enriched[i].Hesitation = hes
		bundleInputs = append(bundleInputs, BundleSlotGateInput{
			SlotID:        slotID,
			Probabilities: ans.Probabilities,
			Hesitation:    hes,
		})
	}
	for _, nominalSlot := range []string{"file_write_collision", "shared_state_hazard", "winning_agent"} {
		ans := dec.Answers[nominalSlot]
		bundleInputs = append(bundleInputs, BundleSlotGateInput{
			SlotID:        nominalSlot,
			Probabilities: ans.Probabilities,
			Hesitation:    dec.Hesitation(nominalSlot),
		})
	}
	rollbackAns := dec.Answers["rollback_severity"]
	rollbackProbs := OrderedProbabilities(rollbackAns.Probabilities, rollbackSeverityLevels)
	bundleInputs = append(bundleInputs, BundleSlotGateInput{
		SlotID:        "rollback_severity",
		OrderedLevels: rollbackSeverityLevels,
		Probabilities: rollbackAns.Probabilities,
		Hesitation:    dec.Hesitation("rollback_severity"),
	})

	lockDec := ArbitrateSwarmLocks(enriched, maxHesitation)
	lockDec.RollbackSeverity = resolveOrderedChoiceLabel(rollbackAns.Choice, rollbackAns.Legend, rollbackAns.Probabilities, rollbackSeverityLevels)
	lockDec.RollbackOrdinalVar = NormalizedOrdinalVariance(rollbackProbs)
	lockDec.EscalatedSlots = SelectEscalatedBundleSlots(bundleInputs, maxHesitation, 0.0, 0.0)
	return lockDec, dec.WallTimeMs, nil
}

// ChecklistWindowSummary records the live_checklist_verify policy evaluation for a transcript window,
// including Popoviciu normalized ordinal variance on interrupt_urgency and per-slot selective escalation.
type ChecklistWindowSummary struct {
	WindowID           string
	ContradictedItems  []string
	CompletedItems     []string
	InterruptUrgency   string
	UrgencyOrdinalVar  float64
	SidakSlotAlpha     float64
	EscalatedSlots     []string
	CommitmentRecorded bool
	MaxHesitation      float64
	WallTimeMs         int64
}

// EvaluateChecklistWindow executes templates/live_checklist_verify.json.tmpl on a sliding transcript window
// and applies per-slot selective escalation with Popoviciu normalized ordinal variance on interrupt_urgency.
func EvaluateChecklistWindow(ctx context.Context, g *genkit.Genkit, modelName, templatesDir, windowID, transcript string) (ChecklistWindowSummary, error) {
	tmplPath := filepath.Join(templatesDir, "live_checklist_verify.json.tmpl")
	dec, err := dgemgenkit.DecideWithTemplate(ctx, g, modelName, tmplPath, map[string]any{
		"window_id":  windowID,
		"transcript": transcript,
		"samples":    1,
	})
	if err != nil {
		return ChecklistWindowSummary{}, err
	}
	itemSlots := []string{
		"item_01_identity_verified",
		"item_02_consent_recorded",
		"item_03_scope_confirmed",
		"item_04_risk_precheck",
		"item_05_action_authorized",
		"item_06_next_steps_logged",
	}
	urgAns := dec.Answers["interrupt_urgency"]
	urgProbs := OrderedProbabilities(urgAns.Probabilities, interruptUrgencyLevels)
	commitAns := dec.Answers["speaker_commitment_present"]

	bundleInputs := make([]BundleSlotGateInput, 0, 8)
	out := ChecklistWindowSummary{
		WindowID:           windowID,
		InterruptUrgency:   resolveOrderedChoiceLabel(urgAns.Choice, urgAns.Legend, urgAns.Probabilities, interruptUrgencyLevels),
		UrgencyOrdinalVar:  NormalizedOrdinalVariance(urgProbs),
		SidakSlotAlpha:     SidakSlotAlpha(0.10, 8),
		CommitmentRecorded: commitAns.Probability >= 0.5 || commitAns.Probabilities["yes"] >= 0.5 || commitAns.Choice == "yes",
		WallTimeMs:         dec.WallTimeMs,
	}
	for _, slot := range itemSlots {
		ans := dec.Answers[slot]
		choice := ans.Choice
		if choice == "contradicted" {
			out.ContradictedItems = append(out.ContradictedItems, slot)
		} else if choice == "completed_verified" {
			out.CompletedItems = append(out.CompletedItems, slot)
		}
		hes := dec.Hesitation(slot)
		if hes > out.MaxHesitation {
			out.MaxHesitation = hes
		}
		bundleInputs = append(bundleInputs, BundleSlotGateInput{
			SlotID:        slot,
			Probabilities: ans.Probabilities,
			Hesitation:    hes,
		})
	}
	bundleInputs = append(bundleInputs,
		BundleSlotGateInput{
			SlotID:        "speaker_commitment_present",
			Probabilities: commitAns.Probabilities,
			Hesitation:    dec.Hesitation("speaker_commitment_present"),
		},
		BundleSlotGateInput{
			SlotID:        "interrupt_urgency",
			OrderedLevels: interruptUrgencyLevels,
			Probabilities: urgAns.Probabilities,
			Hesitation:    dec.Hesitation("interrupt_urgency"),
		},
	)
	out.EscalatedSlots = SelectEscalatedBundleSlots(bundleInputs, 0.16, 0.0, 0.0)
	return out, nil
}

func topProbabilityKey(probs map[string]float64, fallback string) string {
	if fallback != "" {
		return fallback
	}
	bestKey := ""
	bestProb := -1.0
	for k, p := range probs {
		if p > bestProb || (p == bestProb && (bestKey == "" || k < bestKey)) {
			bestProb = p
			bestKey = k
		}
	}
	return bestKey
}
