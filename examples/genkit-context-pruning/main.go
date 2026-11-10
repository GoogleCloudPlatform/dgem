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
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/firebase/genkit/go/genkit"
)

func main() {
	cfg := dgemgenkit.EndpointConfigFromEnv()
	dgemgenkit.BindEndpointFlags(flag.CommandLine, &cfg)
	templatesDir := flag.String("templates-dir", filepath.Join("..", "..", "templates"), "Path to dgem templates directory")
	tokenBudget := flag.Int("token-budget", 1400, "Maximum active context token budget")
	uncertaintyBonus := flag.Float64("uncertainty-stub-bonus", 0.20, "Normalized-entropy bonus toward retaining compact stubs under uncertainty")
	jsonOut := flag.Bool("json", false, "Emit structured JSON output")
	flag.Parse()

	plugin, cleanup := dgemgenkit.NewSystemOnePlugin(cfg)
	defer cleanup()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	modelName := cfg.GenkitModelName()

	blocks := sampleTrajectoryBlocks()
	res, err := EvaluateAndPruneContextBlocks(
		ctx,
		g,
		modelName,
		*templatesDir,
		"Fix data race in worker_pool.go Shutdown() reported by TestWorkerPoolDrain.",
		6,
		blocks,
		ContextPruneConfig{
			TokenBudget:          *tokenBudget,
			UncertaintyStubBonus: *uncertaintyBonus,
		},
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "context pruning failed: %v\n", err)
		os.Exit(1)
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return
	}

	fmt.Printf("8-Slot Context Pruning (budget=%d tok, retained=%d/%d tok, saved=%d tok, wall=%dms)\n",
		res.TokenBudget, res.RetainedTokens, res.OriginalTokens, res.SavedTokens, res.WallTimeMs)
	for _, d := range res.Decisions {
		fmt.Printf("  %-4s %-16s action=%-4s tok=%4d (saved=%4d) H_norm=%.3f P={full:%.2f stub:%.2f drop:%.2f}\n",
			d.SlotID, d.BlockID, d.Action, d.RetainedTokens, d.SavedTokens, d.NormalizedEntropy,
			d.Probabilities[ContextActionFull], d.Probabilities[ContextActionStub], d.Probabilities[ContextActionDrop])
	}
}

func sampleTrajectoryBlocks() []ContextBlockSpec {
	return []ContextBlockSpec{
		{
			ID:         "worker_pool.go",
			Kind:       "file_excerpt",
			Summary:    "Shutdown() implementation containing unsynchronized close(ch) write.",
			FullText:   "func (p *Pool) Shutdown() { close(p.jobs); p.wg.Wait() }",
			StubText:   "worker_pool.go: func (p *Pool) Shutdown()",
			FullTokens: 520,
			StubTokens: 90,
			Pinned:     true,
		},
		{
			ID:         "race_trace",
			Kind:       "test_trace",
			Summary:    "Failing race stack trace from TestWorkerPoolDrain at worker_pool.go:88 and :114.",
			FullText:   "WARNING: DATA RACE Write at 0x00c00012a0 by goroutine 19 in Pool.Shutdown()",
			StubText:   "TestWorkerPoolDrain: DATA RACE in Pool.Shutdown() vs Pool.Submit()",
			FullTokens: 410,
			StubTokens: 80,
			Pinned:     true,
		},
		{
			ID:         "dispatcher.go",
			Kind:       "file_excerpt",
			Summary:    "TaskDispatcher interface referenced by worker_pool.go.",
			FullText:   "type TaskDispatcher interface { Submit(ctx context.Context, fn func()) error; Shutdown() }",
			StubText:   "type TaskDispatcher interface { Submit(...); Shutdown() }",
			FullTokens: 640,
			StubTokens: 110,
		},
		{
			ID:         "turn_4_notes",
			Kind:       "turn_summary",
			Summary:    "Prior lock ordering invariant notes for package pool.",
			FullText:   "Acquire mu before transitioning closed=true so Submit observes closed under RLock.",
			StubText:   "Invariant: hold mu across closed=true transition.",
			FullTokens: 480,
			StubTokens: 75,
		},
		{
			ID:         "marketing_ls",
			Kind:       "tool_output",
			Summary:    "Stale directory listing of docs/marketing/ from turn 1.",
			FullTokens: 590,
			StubTokens: 60,
		},
		{
			ID:         "css_lint",
			Kind:       "tool_output",
			Summary:    "Completed CSS lint output from an unrelated check in turn 2.",
			FullTokens: 510,
			StubTokens: 55,
		},
		{
			ID:         "readme_badges",
			Kind:       "file_excerpt",
			Summary:    "Unrelated README badge header inspected during onboarding.",
			FullTokens: 700,
			StubTokens: 95,
		},
		{
			ID:         "old_git_status",
			Kind:       "tool_output",
			Summary:    "Superseded git status output before checking out the current branch.",
			FullTokens: 430,
			StubTokens: 50,
		},
	}
}
