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
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// localDiffgemmaContent is the structured content returned by local diffgemma
// (Apple Silicon Metal) for a 3-slot urgent/team/sentiment decision, captured
// from a real run. Entropy is reported only under diagnostics.questions, never
// on answers, which is what exposed issue #14.
const localDiffgemmaContent = `{
  "answers": {
    "urgent": {"type": "bool", "label": "yes", "confidence": 0.9999927878379822, "noul": 0.9999927878379822,
      "agreement": 1, "stderr": 0.000004818155048269546,
      "probabilities": {"yes": 0.9999927878379822, "no": 0.000007202716460597003}},
    "team": {"type": "choice", "label": "A", "choice": "billing", "confidence": 0.9999858140945435,
      "agreement": 1, "stderr": 0.000007922635631985031,
      "probabilities": {"billing": 0.9999858140945435, "technical": 0.000014139069207885768, "sales": 5.1909125886595575e-8}},
    "sentiment": {"type": "score", "label": "1", "level": "1", "score": 1.1650534868240356, "confidence": 0.8373942971229553,
      "agreement": 1, "stderr": 0.050104521214962006,
      "probabilities": {"1": 0.8373942971229553, "2": 0.16156376898288727, "3": 0.00022755855752620846, "4": 0.00022313944646157324, "5": 0.000591284828260541}}
  },
  "diagnostics": {
    "hole": "noise",
    "steps": 1,
    "questions": {
      "urgent": {"argmax_is_label": true, "argmax_token": "▁yes", "entropy": 0.0006862637237645686, "label_mass": 0.9999611377716064},
      "team": {"argmax_is_label": true, "argmax_token": "▁A", "entropy": 0.0034505254589021206, "label_mass": 0.9996464848518372},
      "sentiment": {"argmax_is_label": true, "argmax_token": "1", "entropy": 0.501485288143158, "label_mass": 0.9997917413711548}
    },
    "samples": {"n": 4, "policy": {"extended": true, "first_read_max_entropy": 0.501485288143158, "max": 4, "mode": "auto", "pinned_slots": 0, "threshold": 0.1}},
    "timing": {"denoise_ms": 3821.401124, "prefill_ms": 12161.772292, "prompt_tokens": 203, "reads": 0, "reused_tokens": 0, "rounds": 4, "samples": 4, "steps_run": 4, "total_ms": 0}
  }
}`

const localSentimentEntropy = 0.501485288143158

// fakeLocalForwardDelay makes the fake upstream's round trip measurable in
// whole milliseconds so gpu_forward_ms must be > 0.
const fakeLocalForwardDelay = 20 * time.Millisecond

// startFakeLocalDiffgemma serves an OpenAI-compatible /v1/chat/completions that
// always answers with localDiffgemmaContent, and points the "local" backend at it.
func startFakeLocalDiffgemma(t *testing.T) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			time.Sleep(fakeLocalForwardDelay)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":    "chatcmpl-test",
				"model": "diffgemma-26b-a4b-it-q4",
				"choices": []map[string]any{{
					"index":         0,
					"finish_reason": "stop",
					"message":       map[string]any{"role": "assistant", "content": localDiffgemmaContent},
				}},
			})
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "diffgemma-26b-a4b-it-q4"}}})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)

	backendConfigMu.Lock()
	origDefB, origLocURL := serveDefaultBackend, serveLocalURL
	serveDefaultBackend = "local"
	serveLocalURL = srv.URL + "/v1"
	backendConfigMu.Unlock()

	origTemplatesDir := serveTemplatesDir
	serveTemplatesDir = "../templates"

	t.Cleanup(func() {
		backendConfigMu.Lock()
		serveDefaultBackend, serveLocalURL = origDefB, origLocURL
		backendConfigMu.Unlock()
		serveTemplatesDir = origTemplatesDir
	})
}

func connectTestMCPClient(t *testing.T) *mcp.ClientSession {
	t.Helper()
	ctx := context.Background()
	serverT, clientT := mcp.NewInMemoryTransports()
	if _, err := buildMCPServer().Connect(ctx, serverT, nil); err != nil {
		t.Fatalf("server connect: %v", err)
	}
	cs, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "0"}, nil).Connect(ctx, clientT, nil)
	if err != nil {
		t.Fatalf("client connect: %v", err)
	}
	t.Cleanup(func() { _ = cs.Close() })
	return cs
}

func callDecideTool(t *testing.T, cs *mcp.ClientSession, name string, args map[string]any) GatewayDecideResponse {
	t.Helper()
	res, err := cs.CallTool(context.Background(), &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		t.Fatalf("%s: CallTool: %v", name, err)
	}
	if res.IsError {
		raw, _ := json.Marshal(res.Content)
		t.Fatalf("%s: tool returned error: %s", name, raw)
	}
	raw, err := json.Marshal(res.StructuredContent)
	if err != nil {
		t.Fatalf("%s: marshal structured content: %v", name, err)
	}
	var out GatewayDecideResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%s: unmarshal structured content: %v\n%s", name, err, raw)
	}
	return out
}

// TestMCPDecideToolsSummaryFields is the regression test for issue #14: both
// MCP decide tools must report max_entropy from diagnostics.questions (local
// diffgemma puts entropy there, not on answers) and must populate
// gpu_forward_ms, matching POST /api/decide.
func TestMCPDecideToolsSummaryFields(t *testing.T) {
	startFakeLocalDiffgemma(t)
	cs := connectTestMCPClient(t)

	ticket := "I was double charged on invoice #9481. Please refund ASAP, I'm really frustrated."
	cases := []struct {
		tool string
		args map[string]any
	}{
		{
			tool: "decide_custom_questions",
			args: map[string]any{
				"context": ticket,
				"backend": "local",
				"questions": []map[string]any{
					{"id": "urgent", "type": "boolean", "question": "Does this ticket require urgent handling?"},
					{"id": "team", "type": "choice", "question": "Which team should handle it?", "options": []string{"billing", "technical", "sales"}},
					{"id": "sentiment", "type": "score", "question": "Customer sentiment from 1 (very negative) to 5 (very positive)."},
				},
			},
		},
		{
			tool: "decide_policy",
			args: map[string]any{
				"template":  "support_triage",
				"backend":   "local",
				"variables": map[string]any{"ticket": ticket},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.tool, func(t *testing.T) {
			out := callDecideTool(t, cs, tc.tool, tc.args)

			if out.BackendTarget != "local" {
				t.Errorf("backend_target = %q, want local", out.BackendTarget)
			}
			if got := out.Answers["team"].Choice; got != "billing" {
				t.Errorf("answers.team.choice = %q, want billing", got)
			}
			if got := out.Answers["urgent"].Label; got != "yes" {
				t.Errorf("answers.urgent.label = %q, want yes", got)
			}
			if math.Abs(out.MaxEntropy-localSentimentEntropy) > 1e-9 {
				t.Errorf("max_entropy = %v, want %v (max of diagnostics.questions[*].entropy)", out.MaxEntropy, localSentimentEntropy)
			}
			if out.GpuForwardMs <= 0 {
				t.Errorf("gpu_forward_ms = %d, want > 0", out.GpuForwardMs)
			}
			if out.GpuForwardMs > out.WallTimeMs {
				t.Errorf("gpu_forward_ms (%d) exceeds wall_time_ms (%d)", out.GpuForwardMs, out.WallTimeMs)
			}
			if out.ColdStartWaitMs < 0 {
				t.Errorf("cold_start_wait_ms = %d, want >= 0", out.ColdStartWaitMs)
			}
		})
	}
}
