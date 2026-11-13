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

// Command genkit-consistency-arbiters demonstrates combining single-pass dgem
// .json.tmpl policies (entity_resolution_dedup, swarm_lock_arbiter, and
// live_checklist_verify) with deterministic post-decision graph and lock arbiters.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/firebase/genkit/go/genkit"
)

func resolveTemplatesDir(explicit string) string {
	if explicit != "" {
		return explicit
	}
	for _, cand := range []string{"../../templates", "../templates", "templates"} {
		if info, err := os.Stat(cand); err == nil && info.IsDir() {
			return cand
		}
	}
	return "../../templates"
}

func main() {
	cfg := dgemgenkit.EndpointConfigFromEnv()
	fs := flag.NewFlagSet("genkit-consistency-arbiters", flag.ExitOnError)
	dgemgenkit.BindEndpointFlags(fs, &cfg)
	var templatesDir string
	fs.StringVar(&templatesDir, "templates-dir", "", "Path to dgem templates/ directory")
	_ = fs.Parse(os.Args[1:])
	templatesDir = resolveTemplatesDir(templatesDir)

	ctx := context.Background()
	plugin, cleanup := dgemgenkit.NewSystemOnePlugin(cfg)
	defer cleanup()

	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	modelName := cfg.GenkitModelName()
	fmt.Printf("Target: %s (model=%s, templates=%s)\n\n", cfg.DescribeTarget(), modelName, filepath.Clean(templatesDir))

	// 1. Entity Resolution Deduplication + Pack-Multiplier & Triangle Transitivity Guardrail.
	pairAB, err := EvaluateEntityPairWithGuardrail(ctx, g, modelName, templatesDir,
		"sku_a", "San Pellegrino Sparkling Natural Mineral Water 6x330ml Cans",
		"sku_b", "San Pellegrino Sparkling Mineral Water 6-Pack 330ml Cans",
	)
	if err != nil {
		log.Fatalf("entity pair (A, B): %v", err)
	}
	pairBC, err := EvaluateEntityPairWithGuardrail(ctx, g, modelName, templatesDir,
		"sku_b", "San Pellegrino Sparkling Mineral Water 6-Pack 330ml Cans",
		"sku_c", "San Pellegrino Sparkling Mineral Water 330ml Single Can",
	)
	if err != nil {
		log.Fatalf("entity pair (B, C): %v", err)
	}
	fmt.Printf("[1] Entity Dedup (A, B): relation=%s pack=%s consistent=%v action=%s (%dms)\n",
		pairAB.Relation, pairAB.PackMatch, pairAB.PackConsistent, pairAB.MergeAction, pairAB.WallTimeMs)
	fmt.Printf("[1] Entity Dedup (B, C): relation=%s pack=%s consistent=%v action=%s (%dms)\n",
		pairBC.Relation, pairBC.PackMatch, pairBC.PackConsistent, pairBC.MergeAction, pairBC.WallTimeMs)

	triangles := ScoreProbabilisticTriangles([]PairwiseEdge{
		{NodeA: pairAB.NodeA, NodeB: pairAB.NodeB, EquivProb: pairAB.EquivProb},
		{NodeA: pairBC.NodeA, NodeB: pairBC.NodeB, EquivProb: pairBC.EquivProb},
		{NodeA: "sku_a", NodeB: "sku_c", EquivProb: 0.05},
	}, 0.35)
	fmt.Printf("[1] Triangle Contradictions (minMass=0.35): %d found\n\n", len(triangles))

	// 2. Multi-Agent Swarm Lock Arbitration.
	workers := [4]SwarmAgentProposal{
		{AgentID: "agent_1", Proposal: "Edit pkg/decisionindex/engine.go to add PriorLogits validation", TargetPaths: []string{"pkg/decisionindex/engine.go"}},
		{AgentID: "agent_2", Proposal: "Edit pkg/decisionindex/engine.go to refactor ScoreSinglePass", TargetPaths: []string{"pkg/decisionindex/engine.go"}},
		{AgentID: "agent_3", Proposal: "Edit docs/overview.md to update architecture summary", TargetPaths: []string{"docs/overview.md"}},
		{AgentID: "agent_4", Proposal: "Edit Makefile to add lint target", TargetPaths: []string{"Makefile"}},
	}
	lockDec, lockMs, err := EvaluateSwarmLockProposals(ctx, g, modelName, templatesDir, workers, 0.16)
	if err != nil {
		log.Fatalf("swarm lock arbiter: %v", err)
	}
	fmt.Printf("[2] Swarm Lock Arbiter (%dms): collision=%v winner=%s escalate=%v rollback=%s (ord_var=%.3f) escalated_slots=%v actions=%v\n\n",
		lockMs, lockDec.WriteCollision, lockDec.WinningAgent, lockDec.EscalateToHuman,
		lockDec.RollbackSeverity, lockDec.RollbackOrdinalVar, lockDec.EscalatedSlots, lockDec.AgentActions)

	// 3. Sliding-Window Operational Checklist Verification.
	chk, err := EvaluateChecklistWindow(ctx, g, modelName, templatesDir, "win-101",
		"Operator: Badge EMP-4491 verified; session recorded with consent. Caller: I consent, asset is prod-db-04, prechecks green, rotation authorized.")
	if err != nil {
		log.Fatalf("live checklist verify: %v", err)
	}
	fmt.Printf("[3] Live Checklist (%dms): completed=%d contradicted=%d urgency=%s (ord_var=%.3f) commitment=%v max_hesitation=%.1f%% escalated_slots=%v\n",
		chk.WallTimeMs, len(chk.CompletedItems), len(chk.ContradictedItems), chk.InterruptUrgency,
		chk.UrgencyOrdinalVar, chk.CommitmentRecorded, chk.MaxHesitation*100, chk.EscalatedSlots)
}
