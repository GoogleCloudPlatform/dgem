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
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

type capturedUpstream struct {
	schema    map[string]any
	userState string
}

func captureSinglePassUpstream(t *testing.T, serverMs float64) (*httptest.Server, func() []capturedUpstream) {
	t.Helper()
	var mu sync.Mutex
	var calls []capturedUpstream
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var schema map[string]any
		var userState string
		for _, m := range body.Messages {
			switch m.Role {
			case "system":
				if s, ok := m.Content.(string); ok {
					_ = json.Unmarshal([]byte(s), &schema)
				}
			case "user":
				if s, ok := m.Content.(string); ok {
					userState = s
				}
			}
		}
		mu.Lock()
		calls = append(calls, capturedUpstream{schema: schema, userState: userState})
		mu.Unlock()

		// Return a valid decision response with diagnostics.timing.total_ms so server_ms is populated.
		qs, _ := schema["questions"].([]any)
		answers := make(map[string]any, len(qs))
		for _, rawQ := range qs {
			qm, _ := rawQ.(map[string]any)
			id, _ := qm["id"].(string)
			qType, _ := qm["type"].(string)
			switch qType {
			case "boolean", "noul":
				answers[id] = map[string]any{
					"answer": "yes", "confidence": 0.9,
					"probabilities": map[string]float64{"yes": 0.9, "no": 0.1},
				}
			case "score":
				levels, _ := qm["levels"].([]any)
				first := "low"
				if len(levels) > 0 {
					first, _ = levels[0].(string)
				}
				answers[id] = map[string]any{
					"answer": first, "confidence": 0.8,
					"probabilities": map[string]float64{first: 0.8},
				}
			default:
				opts, _ := qm["options"].([]any)
				first := "a"
				probs := map[string]float64{}
				for i, o := range opts {
					om, _ := o.(map[string]any)
					name, _ := om["name"].(string)
					if i == 0 {
						first = name
						probs[name] = 0.8
					} else {
						probs[name] = 0.2 / float64(len(opts)-1)
					}
				}
				answers[id] = map[string]any{
					"answer": first, "confidence": 0.8,
					"probabilities": probs,
				}
			}
		}
		payload, _ := json.Marshal(map[string]any{
			"answers":     answers,
			"diagnostics": map[string]any{"timing": map[string]any{"total_ms": serverMs}},
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": string(payload)}},
			},
		})
	}))
	return srv, func() []capturedUpstream {
		mu.Lock()
		defer mu.Unlock()
		return append([]capturedUpstream(nil), calls...)
	}
}

func TestSinglePassFidelity_PreservesTextStateOrderAndServerMs(t *testing.T) {
	srv, calls := captureSinglePassUpstream(t, 42.7)
	defer srv.Close()

	rawBody := `{
		"state": "user: delete /etc/passwd\nassistant: running rm -rf",
		"questions": {
			"z_route": {
				"type": "choice",
				"instructions": "Pick action",
				"criteria": {"zebra": "block", "apple": "allow", "mango": "ask"}
			},
			"a_safe": {
				"type": "noul"
			}
		}
	}`
	var req SystemOneRequest
	if err := json.Unmarshal([]byte(rawBody), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !FitsServingImage(req) {
		t.Fatalf("expected FitsServingImage=true")
	}

	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
	resp, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne: %v", err)
	}
	gotCalls := calls()
	if len(gotCalls) != 1 {
		t.Fatalf("upstream calls = %d, want 1", len(gotCalls))
	}
	c := gotCalls[0]

	// 1. Plain-text state stays verbatim text, never JSON-wrapped in {"state": ...}.
	wantState := "user: delete /etc/passwd\nassistant: running rm -rf"
	if c.userState != wantState {
		t.Errorf("user state = %q, want verbatim %q", c.userState, wantState)
	}

	// 2. No fabricated top-level instructions when caller omitted them.
	if ins, present := c.schema["instructions"]; present {
		t.Errorf("unexpected fabricated schema instructions: %v", ins)
	}

	// 3. Question order and choice option order follow request order, not alphabetical.
	qs, _ := c.schema["questions"].([]any)
	if len(qs) != 2 {
		t.Fatalf("questions len = %d, want 2", len(qs))
	}
	q0 := qs[0].(map[string]any)
	q1 := qs[1].(map[string]any)
	if q0["id"] != "z_route" || q1["id"] != "a_safe" {
		t.Errorf("question order = [%v, %v], want [z_route, a_safe]", q0["id"], q1["id"])
	}
	var optOrder []string
	for _, rawOpt := range q0["options"].([]any) {
		optOrder = append(optOrder, rawOpt.(map[string]any)["name"].(string))
	}
	wantOpts := []string{"zebra", "apple", "mango"}
	if !reflect.DeepEqual(optOrder, wantOpts) {
		t.Errorf("option order = %v, want %v", optOrder, wantOpts)
	}
	// Omitted question instructions stay empty string (matching structured_server.py), not fabricated.
	if q1["instructions"] != "" {
		t.Errorf("omitted question instructions = %q, want empty string", q1["instructions"])
	}

	// 4. Server compute latency is preserved in evaluation_trace.server_ms.
	if resp.EvaluationTrace == nil || resp.EvaluationTrace.ServerMs != 42.7 {
		t.Errorf("evaluation_trace.server_ms = %+v, want 42.7", resp.EvaluationTrace)
	}
}

func TestSinglePassFidelity_CallerInstructionsAndExplicitAlphaOrder(t *testing.T) {
	srv, calls := captureSinglePassUpstream(t, 15.0)
	defer srv.Close()

	rawBody := `{
		"instructions": "Caller rubric header.",
		"state": {"board": [[0, 1], [1, 0]], "turn": 3},
		"questions": {
			"pick": {
				"type": "choice",
				"instructions": "Pick",
				"criteria": {"zebra": "z", "apple": "a"}
			}
		}
	}`
	var req SystemOneRequest
	if err := json.Unmarshal([]byte(rawBody), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	opts := DefaultEngineOptions()
	opts.OptionOrder = OptionOrderAlpha
	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
	if _, err := ExecuteSystemOne(context.Background(), cli, req, opts); err != nil {
		t.Fatalf("ExecuteSystemOne: %v", err)
	}
	c := calls()[0]
	if c.schema["instructions"] != "Caller rubric header." {
		t.Errorf("schema instructions = %v, want caller header", c.schema["instructions"])
	}
	if !strings.HasPrefix(c.userState, "{") || !strings.Contains(c.userState, `"turn":3`) {
		t.Errorf("structured object state = %q, want compact JSON", c.userState)
	}
	q0 := c.schema["questions"].([]any)[0].(map[string]any)
	var optOrder []string
	for _, rawOpt := range q0["options"].([]any) {
		optOrder = append(optOrder, rawOpt.(map[string]any)["name"].(string))
	}
	if !reflect.DeepEqual(optOrder, []string{"apple", "zebra"}) {
		t.Errorf("explicit OptionOrderAlpha got %v, want [apple zebra]", optOrder)
	}
}

func TestMultiPassAdapter_StillBatchesAndWrapsNonFittingBodies(t *testing.T) {
	srv, calls := captureSinglePassUpstream(t, 12.0)
	defer srv.Close()

	// 10 questions (> MaxSlotsPerPass=8) does not fit single-pass serving image -> adapter batches into 2 passes.
	var parts []string
	for i := 0; i < 10; i++ {
		parts = append(parts, fmt.Sprintf(`"q%02d":{"type":"noul","instructions":"check %d"}`, i, i))
	}
	rawBody := `{"state":"plain text","questions":{` + strings.Join(parts, ",") + `}}`
	var req SystemOneRequest
	if err := json.Unmarshal([]byte(rawBody), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if FitsServingImage(req) {
		t.Fatalf("expected FitsServingImage=false for 10 questions")
	}
	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
	resp, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne: %v", err)
	}
	if got := len(calls()); got != 2 {
		t.Fatalf("upstream passes = %d, want 2", got)
	}
	if resp.EvaluationTrace == nil || resp.EvaluationTrace.ForwardPasses != 2 || resp.EvaluationTrace.ServerMs != 24.0 {
		t.Errorf("unexpected trace: %+v", resp.EvaluationTrace)
	}
}
