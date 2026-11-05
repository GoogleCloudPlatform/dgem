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
	"sort"
	"strconv"
	"strings"
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

	type edgeKey struct{ u, v string }
	type accum struct {
		sum   float64
		count int
	}
	pairMap := make(map[edgeKey]accum)
	nodeSet := make(map[string]struct{})

	for _, e := range edges {
		u := strings.TrimSpace(e.NodeA)
		v := strings.TrimSpace(e.NodeB)
		if u == "" || v == "" || u == v {
			continue
		}
		if u > v {
			u, v = v, u
		}
		nodeSet[u] = struct{}{}
		nodeSet[v] = struct{}{}
		k := edgeKey{u, v}
		cur := pairMap[k]
		cur.sum += e.EquivProb
		cur.count++
		pairMap[k] = cur
	}

	nodes := make([]string, 0, len(nodeSet))
	for n := range nodeSet {
		nodes = append(nodes, n)
	}
	sort.Strings(nodes)

	adj := make(map[string][]string, len(nodes))
	probOf := make(map[edgeKey]float64, len(pairMap))
	for k, acc := range pairMap {
		probOf[k] = acc.sum / float64(acc.count)
		adj[k.u] = append(adj[k.u], k.v)
	}
	for u := range adj {
		sort.Strings(adj[u])
	}

	var out []TriangleContradiction
	for _, a := range nodes {
		neighborsA := adj[a]
		for i := 0; i < len(neighborsA); i++ {
			b := neighborsA[i]
			for j := i + 1; j < len(neighborsA); j++ {
				c := neighborsA[j]
				pbc, ok := probOf[edgeKey{b, c}]
				if !ok {
					continue
				}
				pab := probOf[edgeKey{a, b}]
				pac := probOf[edgeKey{a, c}]

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

// ProbabilisticTriangleContradiction scores a 3-cycle (A, B, C) using continuous transitivity-violation mass
// across all three 2-positive-1-negative rotations:
//
//	m_ac = P(A,B) * P(B,C) * (1 - P(A,C))
//	m_bc = P(A,B) * P(A,C) * (1 - P(B,C))
//	m_ab = P(B,C) * P(A,C) * (1 - P(A,B))
//
// Unlike hard step thresholds, continuous violation mass captures near-threshold contradictions (e.g. 0.79, 0.85, 0.20)
// and provides a smooth ranking signal for audit escalation.
type ProbabilisticTriangleContradiction struct {
	Nodes              [3]string  `json:"nodes"`
	EdgeProbs          [3]float64 `json:"edge_probs"` // [P(A,B), P(B,C), P(A,C)]
	WeakEdge           [2]string  `json:"weak_edge"`
	WeakProb           float64    `json:"weak_prob"`
	MaxViolationMass   float64    `json:"max_violation_mass"`
	TotalViolationMass float64    `json:"total_violation_mass"`
}

// ScoreProbabilisticTriangles evaluates all 3-cycles in edges and returns those with MaxViolationMass >= minMass
// (default 0.35 when minMass <= 0), sorted descending by MaxViolationMass.
func ScoreProbabilisticTriangles(edges []PairwiseEdge, minMass float64) []ProbabilisticTriangleContradiction {
	if minMass <= 0 {
		minMass = 0.35
	}

	type edgeKey struct{ u, v string }
	type accum struct {
		sum   float64
		count int
	}
	pairMap := make(map[edgeKey]accum)
	nodeSet := make(map[string]struct{})

	for _, e := range edges {
		u := strings.TrimSpace(e.NodeA)
		v := strings.TrimSpace(e.NodeB)
		if u == "" || v == "" || u == v {
			continue
		}
		if u > v {
			u, v = v, u
		}
		p := math.Max(0.0, math.Min(1.0, e.EquivProb))
		if math.IsNaN(e.EquivProb) {
			continue
		}
		nodeSet[u] = struct{}{}
		nodeSet[v] = struct{}{}
		k := edgeKey{u, v}
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
	probOf := make(map[edgeKey]float64, len(pairMap))
	for k, acc := range pairMap {
		probOf[k] = acc.sum / float64(acc.count)
		adj[k.u] = append(adj[k.u], k.v)
	}
	for u := range adj {
		sort.Strings(adj[u])
	}

	var out []ProbabilisticTriangleContradiction
	for _, a := range nodes {
		neighborsA := adj[a]
		for i := 0; i < len(neighborsA); i++ {
			b := neighborsA[i]
			for j := i + 1; j < len(neighborsA); j++ {
				c := neighborsA[j]
				pbc, ok := probOf[edgeKey{b, c}]
				if !ok {
					continue
				}
				pab := probOf[edgeKey{a, b}]
				pac := probOf[edgeKey{a, c}]

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

var (
	reMultiPackPrefix = regexp.MustCompile(`(?i)\b(\d+)\s*[x×]\s*\d+`)
	rePackOf          = regexp.MustCompile(`(?i)\b(?:pack\s+of|case\s+of|box\s+of)\s+(\d+)\b`)
	reHyphenPack      = regexp.MustCompile(`(?i)\b(\d+)\s*(?:-|\s)?\s*(?:pack|pk|count|ct|cans|bottles|bars|pods|pouches)\b`)
)

// ParsePackMultiplier extracts an explicit integer pack count from a product title
// (for example, "6x330ml" -> 6, "Pack of 12" -> 12, "24-Pack" -> 24).
// Returns 1 when no explicit multipack multiplier is present.
func ParsePackMultiplier(title string) int {
	for _, re := range []*regexp.Regexp{reMultiPackPrefix, rePackOf, reHyphenPack} {
		if m := re.FindStringSubmatch(title); len(m) == 2 {
			if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
				return n
			}
		}
	}
	return 1
}

// ValidatePackQuantityConsistency checks whether two product titles' parsed pack multipliers
// are consistent with the model's entity_relation and pack_quantity_match slot predictions.
// It returns false (indicating a contradiction requiring escalation or downgrade to variant)
// if the parsed multipliers differ while the model predicts an exact duplicate or identical pack.
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
// along with the model's slot confidence and normalized hesitation H_raw.
type SwarmAgentProposal struct {
	AgentID     string   `json:"agent_id"`
	TargetPaths []string `json:"target_paths"`
	Confidence  float64  `json:"confidence"`
	Hesitation  float64  `json:"hesitation"`
}

// SwarmLockDecision records the deterministic lock arbitration result across concurrent worker proposals.
type SwarmLockDecision struct {
	AgentActions    map[string]string `json:"agent_actions"` // grant_lock, serialize_after_peer, escalate_conflict
	WriteCollision  bool              `json:"write_collision"`
	WinningAgent    string            `json:"winning_agent"`
	EscalateToHuman bool              `json:"escalate_to_human"`
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

	// Rank proposals by descending Confidence, breaking ties by ascending Hesitation then AgentID.
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

	if len(order) > 0 {
		decision.WinningAgent = proposals[order[0]].AgentID
	}

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
				norm := strings.TrimSpace(path)
				if norm != "" {
					claimedPaths[norm] = p.AgentID
				}
			}
			continue
		}

		// Colliding proposal.
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
				norm := strings.TrimSpace(path)
				if norm != "" {
					claimedPaths[norm] = p.AgentID
				}
			}
		} else {
			decision.AgentActions[p.AgentID] = "serialize_after_peer"
		}
	}

	return decision
}
