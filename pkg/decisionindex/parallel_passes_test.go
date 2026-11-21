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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func delayedMockServer(t *testing.T, delay time.Duration, failCall int32) (*httptest.Server, *atomic.Int32, *atomic.Int32) {
	t.Helper()
	var inFlight atomic.Int32
	var maxInFlight atomic.Int32
	var totalCalls atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := totalCalls.Add(1)
		cur := inFlight.Add(1)
		for {
			prev := maxInFlight.Load()
			if cur <= prev || maxInFlight.CompareAndSwap(prev, cur) {
				break
			}
		}
		defer inFlight.Add(-1)

		if failCall > 0 && n == failCall {
			http.Error(w, "injected upstream failure", http.StatusInternalServerError)
			return
		}

		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}

		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		var schema map[string]any
		for _, m := range body.Messages {
			if m.Role == "system" {
				if s, ok := m.Content.(string); ok {
					_ = json.Unmarshal([]byte(s), &schema)
				}
			}
		}
		qs, _ := schema["questions"].([]any)
		answers := make(map[string]any, len(qs))
		for _, rawQ := range qs {
			qm, _ := rawQ.(map[string]any)
			id, _ := qm["id"].(string)
			switch qm["type"] {
			case "boolean", "noul":
				answers[id] = map[string]any{
					"answer": "yes", "confidence": 0.9,
					"probabilities": map[string]float64{"yes": 0.9, "no": 0.1},
				}
			default:
				opts, _ := qm["options"].([]any)
				first := "opt_000"
				probs := make(map[string]float64, len(opts))
				for i, o := range opts {
					name := o.(map[string]any)["name"].(string)
					if i == 0 {
						first = name
						probs[name] = 0.7
					} else {
						probs[name] = 0.3 / float64(len(opts)-1)
					}
				}
				answers[id] = map[string]any{
					"answer": first, "confidence": 0.7,
					"probabilities": probs,
				}
			}
		}
		payload, _ := json.Marshal(map[string]any{
			"answers":     answers,
			"diagnostics": map[string]any{"timing": map[string]any{"total_ms": 20.0}},
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": string(payload)}}},
		})
	}))
	return srv, &maxInFlight, &totalCalls
}

func TestExecuteSystemOne_ParallelMultiBatchAndSerialMode(t *testing.T) {
	// 24 questions = 3 batches of MaxSlotsPerPass (8).
	qs := make(map[string]SystemOneQuestion, 24)
	for i := 0; i < 24; i++ {
		qs[fmt.Sprintf("q%02d", i)] = SystemOneQuestion{Type: "noul", Instructions: "check"}
	}
	req := SystemOneRequest{State: "multi-batch state", Questions: qs}

	t.Run("parallel batches overlap up to MaxConcurrency", func(t *testing.T) {
		srv, maxInFlight, totalCalls := delayedMockServer(t, 50*time.Millisecond, 0)
		defer srv.Close()

		cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
		opts := DefaultEngineOptions() // MaxConcurrency = 4
		start := time.Now()
		resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("ExecuteSystemOne: %v", err)
		}
		if len(resp.Answers) != 24 {
			t.Fatalf("answers = %d, want 24", len(resp.Answers))
		}
		if totalCalls.Load() != 3 || resp.EvaluationTrace.ForwardPasses != 3 {
			t.Fatalf("passes = %d / trace %d, want 3", totalCalls.Load(), resp.EvaluationTrace.ForwardPasses)
		}
		if maxInFlight.Load() < 2 {
			t.Errorf("maxInFlight = %d, want >= 2 concurrent batches", maxInFlight.Load())
		}
		if elapsed >= 135*time.Millisecond {
			t.Errorf("elapsed = %v, expected parallel batches faster than 3x50ms serial", elapsed)
		}
		if resp.EvaluationTrace.ServerMs != 60.0 {
			t.Errorf("server_ms = %v, want 60.0 (sum of 3 passes)", resp.EvaluationTrace.ServerMs)
		}
	})

	t.Run("MaxConcurrency=1 executes strictly serially", func(t *testing.T) {
		srv, maxInFlight, _ := delayedMockServer(t, 15*time.Millisecond, 0)
		defer srv.Close()

		cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
		opts := DefaultEngineOptions()
		opts.MaxConcurrency = 1
		resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
		if err != nil {
			t.Fatalf("ExecuteSystemOne: %v", err)
		}
		if len(resp.Answers) != 24 {
			t.Fatalf("answers = %d, want 24", len(resp.Answers))
		}
		if maxInFlight.Load() != 1 {
			t.Errorf("maxInFlight = %d, want 1 in serial mode", maxInFlight.Load())
		}
	})
}

func TestExecuteSystemOne_WideTournamentRound1Parallel(t *testing.T) {
	// 180 options -> 9 brackets of 20 -> 2 Round-1 batches (8 + 1) that run in parallel, then 1 Finals pass = 3 passes.
	crit := make(map[string]string, 180)
	for i := 0; i < 180; i++ {
		crit[fmt.Sprintf("opt_%03d", i)] = fmt.Sprintf("option %d", i)
	}
	req := SystemOneRequest{
		State: "wide choice state",
		Questions: map[string]SystemOneQuestion{
			"tool": {Type: "choice", Instructions: "pick tool", Criteria: crit},
		},
	}

	srv, maxInFlight, totalCalls := delayedMockServer(t, 45*time.Millisecond, 0)
	defer srv.Close()

	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
	resp, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne: %v", err)
	}
	if totalCalls.Load() != 3 || resp.EvaluationTrace.ForwardPasses != 3 {
		t.Fatalf("passes = %d / trace %d, want 3", totalCalls.Load(), resp.EvaluationTrace.ForwardPasses)
	}
	if maxInFlight.Load() < 2 {
		t.Errorf("Round-1 maxInFlight = %d, want >= 2 parallel batches", maxInFlight.Load())
	}
	if resp.Answers["tool"].Choice == "" {
		t.Errorf("missing winner for wide choice: %+v", resp.Answers["tool"])
	}
}

func TestExecuteSystemOne_ParallelErrorCancelsSiblings(t *testing.T) {
	qs := make(map[string]SystemOneQuestion, 24)
	for i := 0; i < 24; i++ {
		qs[fmt.Sprintf("q%02d", i)] = SystemOneQuestion{Type: "noul", Instructions: "check"}
	}
	req := SystemOneRequest{State: "error cancel state", Questions: qs}

	// First call fails immediately; siblings sleep 500ms unless canceled.
	srv, _, _ := delayedMockServer(t, 500*time.Millisecond, 1)
	defer srv.Close()

	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
	start := time.Now()
	_, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions())
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected error from failing batch, got nil")
	}
	if elapsed >= 400*time.Millisecond {
		t.Errorf("elapsed = %v, expected sibling batches to cancel promptly", elapsed)
	}
}
