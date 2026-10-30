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
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestExecuteSystemOne_BatchingAndWideTournament(t *testing.T) {
	// Mock vLLM/structured_server that inspects the questions in the system message
	// and always assigns highest probability to the option containing "TARGET" or the 1st option.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Extract structured schema from the system message
		var sysPayload struct {
			Questions []struct {
				ID      string `json:"id"`
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}

		if len(sysPayload.Questions) > MaxSlotsPerPass {
			http.Error(w, "the canvas holds at most 10 slots", http.StatusUnprocessableEntity)
			return
		}

		answers := make(map[string]client.QuestionAnswer)
		for _, q := range sysPayload.Questions {
			if len(q.Options) > MaxOptionsPerSlot {
				http.Error(w, "at most 26 options per choice", http.StatusUnprocessableEntity)
				return
			}
			chosen := q.Options[0].Name
			for _, o := range q.Options {
				if strings.Contains(o.Name, "TARGET") {
					chosen = o.Name
					break
				}
			}
			probs := make(map[string]float64)
			rem := 0.10 / float64(maxInt(1, len(q.Options)-1))
			for _, o := range q.Options {
				if o.Name == chosen {
					probs[o.Name] = 0.90
				} else {
					probs[o.Name] = rem
				}
			}
			answers[q.ID] = client.QuestionAnswer{
				Choice:        chosen,
				Probabilities: probs,
			}
		}

		env := client.StructuredDecisionResponse{
			Answers: answers,
		}
		envBytes, _ := json.Marshal(env)

		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": string(envBytes),
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)

	// 1. Test Multi-Slot Canvas Batching (18 slots, similar to ContractNLI's 17 slots)
	questions := make(map[string]SystemOneQuestion)
	for i := 0; i < 18; i++ {
		questions[fmt.Sprintf("hypothesis_%02d", i)] = SystemOneQuestion{
			Type:         "choice",
			Instructions: fmt.Sprintf("Does the NDA permit clause %d?", i),
			Criteria: map[string]string{
				"contradiction":     "Clause is contradicted",
				"entailment_TARGET": "Clause is entailed",
				"neutral":           "Clause is not mentioned",
			},
		}
	}

	// 2. Also add a 77-option question (similar to BANKING77) in the same request!
	wideCriteria := make(map[string]string)
	for i := 0; i < 77; i++ {
		if i == 54 {
			wideCriteria["card_payment_wrong_exchange_rate_TARGET"] = "Wrong FX rate applied to card transaction"
		} else {
			wideCriteria[fmt.Sprintf("banking_intent_%02d", i)] = fmt.Sprintf("Banking intent description %d", i)
		}
	}
	questions["banking77_intent"] = SystemOneQuestion{
		Type:         "choice",
		Instructions: "Customer says exchange rate on card payment was wrong.",
		Criteria:     wideCriteria,
	}

	sysReq := SystemOneRequest{
		State:     "Sample NDA and customer banking inquiry.",
		Questions: questions,
	}

	opts := DefaultEngineOptions()
	opts.TemperatureScale = 1.25

	sysResp, err := ExecuteSystemOne(context.Background(), cli, sysReq, opts)
	if err != nil {
		t.Fatalf("ExecuteSystemOne failed: %v", err)
	}

	if sysResp.EvaluationTrace == nil {
		t.Fatalf("expected EvaluationTrace to be non-nil")
	}
	if sysResp.EvaluationTrace.MultiSlotBatches < 3 {
		t.Errorf("expected MultiSlotBatches >= 3 for 18 normal slots, got %d", sysResp.EvaluationTrace.MultiSlotBatches)
	}
	if sysResp.EvaluationTrace.WideBracketedQs != 1 {
		t.Errorf("expected WideBracketedQs = 1 for 77-option question, got %d", sysResp.EvaluationTrace.WideBracketedQs)
	}
	if len(sysResp.Answers) != 19 {
		t.Fatalf("expected 19 answers, got %d", len(sysResp.Answers))
	}

	// Verify 77-option question picked TARGET and probabilities sum to 1.0 over all 77 keys
	b77 := sysResp.Answers["banking77_intent"]
	if b77.Choice != "card_payment_wrong_exchange_rate_TARGET" {
		t.Errorf("expected banking77_intent choice to be card_payment_wrong_exchange_rate_TARGET, got %q", b77.Choice)
	}
	if len(b77.Probabilities) != 77 {
		t.Errorf("expected 77 probability keys in banking77_intent, got %d", len(b77.Probabilities))
	}
	var sumP float64
	for _, p := range b77.Probabilities {
		sumP += p
	}
	if math.Abs(sumP-1.0) > 1e-4 {
		t.Errorf("expected probabilities to sum to 1.0, got %.6f", sumP)
	}

	// Verify Naive limit mode rejects >10 slots or >26 options
	naiveOpts := DefaultEngineOptions()
	naiveOpts.NaiveLimits = true
	_, err = ExecuteSystemOne(context.Background(), cli, sysReq, naiveOpts)
	if err == nil {
		t.Errorf("expected NaiveLimits to reject 19-slot/77-option request")
	}
}

func TestExecuteSystemOne_NoulQuestion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		answers := map[string]client.QuestionAnswer{
			"is_hallucinated": {
				Label: "yes",
				Noul:  0.88,
				Probabilities: map[string]float64{
					"yes": 0.88,
					"no":  0.12,
				},
				Entropy: 0.36,
			},
		}
		env := client.StructuredDecisionResponse{Answers: answers}
		envBytes, _ := json.Marshal(env)
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": string(envBytes),
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)
	req := SystemOneRequest{
		State: "The patient was prescribed amoxicillin.",
		Questions: map[string]SystemOneQuestion{
			"is_hallucinated": {
				Type:         "noul",
				Instructions: "Is the claim hallucinated?",
			},
		},
	}

	opts := DefaultEngineOptions()
	resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ans, ok := resp.Answers["is_hallucinated"]
	if !ok {
		t.Fatalf("missing answer for is_hallucinated")
	}
	if ans.Type != "noul" {
		t.Errorf("expected type 'noul', got %q", ans.Type)
	}
	if ans.Noul == nil || *ans.Noul < 0.80 || *ans.Noul > 0.95 {
		t.Errorf("expected noul ~0.88, got %v", ans.Noul)
	}
	if ans.Choice != "" {
		t.Errorf("expected choice to be empty for noul, got %q", ans.Choice)
	}
	if len(ans.Probabilities) != 0 {
		t.Errorf("expected probabilities to be empty for noul, got %v", ans.Probabilities)
	}
}

func TestNewSystemOneHTTPHandler_CapacityMarkers(t *testing.T) {
	// Dummy client with invalid URL to trigger error
	cli := client.NewClient("http://127.0.0.1:9999/v1", "dgemma", 1*time.Second)
	opts := DefaultEngineOptions()
	opts.NaiveLimits = true // will reject >26 options with "options per choice"

	wideCriteria := make(map[string]string)
	for i := 0; i < 30; i++ {
		wideCriteria[fmt.Sprintf("opt_%d", i)] = fmt.Sprintf("desc %d", i)
	}

	reqBody, _ := json.Marshal(SystemOneRequest{
		State: "Sample state",
		Questions: map[string]SystemOneQuestion{
			"q1": {Type: "choice", Criteria: wideCriteria},
		},
	})

	handler := NewSystemOneHTTPHandler(cli, opts)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(string(reqBody)))
	handler(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Errorf("expected HTTP 422 for capacity refusal, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "options per choice") {
		t.Errorf("expected body to contain capacity marker 'options per choice', got %q", rec.Body.String())
	}
}

// TestNewSystemOneHTTPHandler_ServerSchemaError: a server HTTP 400 schema error ("at most 26 alternatives") becomes a
// 422 with the kit's capacity marker, not a 500 (#120). A score question with more levels than one read can label is
// refused by the adapter itself with 422, before any upstream call.
func TestNewSystemOneHTTPHandler_ServerSchemaError(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "question 'q1': at most 26 alternatives", "type": "invalid_request_error"}}`))
	}))
	defer srv.Close()
	cli := client.NewClient(srv.URL+"/v1", "dgemma", 2*time.Second)
	cli.MaxRetries = 0
	h := NewSystemOneHTTPHandler(cli, DefaultEngineOptions())

	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(`{"state": "s", "questions": {"q1": {"type": "choice", "criteria": {"a": "", "b": ""}}}}`)))
	if rec.Code != http.StatusUnprocessableEntity || !strings.Contains(rec.Body.String(), "options per choice") {
		t.Fatalf("server schema error: got %d %q, want 422 with the kit marker", rec.Code, rec.Body.String())
	}

	levels := make([]string, 30)
	for i := range levels {
		levels[i] = fmt.Sprintf("level %02d", i)
	}
	lb, _ := json.Marshal(levels)
	calls = 0
	rec = httptest.NewRecorder()
	h(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(`{"state": "s", "questions": {"q1": {"type": "score", "criteria": `+string(lb)+`}}}`)))
	if rec.Code != http.StatusUnprocessableEntity || calls != 0 {
		t.Fatalf("30-level score: got %d %q with %d upstream calls, want 422 and none", rec.Code, rec.Body.String(), calls)
	}
}
