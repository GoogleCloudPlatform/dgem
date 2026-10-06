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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
)

func TestSystemOneServe_AuthAndHealth(t *testing.T) {
	// Mock upstream vLLM returning a simple answer
	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		answers := map[string]client.QuestionAnswer{
			"q1": {
				Label:         "A",
				Choice:        "A",
				Probabilities: map[string]float64{"A": 0.95, "B": 0.05},
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
	defer mockUpstream.Close()

	cli := client.NewClient(mockUpstream.URL, "dgemma", 5*time.Second)
	opts := decisionindex.DefaultEngineOptions()

	// Handler with API key required
	apiKey := "secret-test-token"
	authWrap := func(handler http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if apiKey != "" {
				authHeader := r.Header.Get("Authorization")
				expected := "Bearer " + apiKey
				if authHeader != expected {
					w.Header().Set("WWW-Authenticate", `Bearer realm="dgem-systemone"`)
					http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
					return
				}
			}
			handler(w, r)
		}
	}

	soHandler := decisionindex.NewSystemOneHTTPHandler(cli, opts)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/systemone", authWrap(soHandler))
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "auth_enabled": true})
	})

	testServer := httptest.NewServer(mux)
	defer testServer.Close()

	// 1. Health check returns 200
	hResp, err := http.Get(testServer.URL + "/health")
	if err != nil || hResp.StatusCode != http.StatusOK {
		t.Fatalf("expected health status 200, got %v (err: %v)", hResp.StatusCode, err)
	}

	// 2. Unauthenticated POST to /v1/systemone returns 401
	payload := `{"state":"test","questions":{"q1":{"type":"choice","criteria":{"A":"opt A","B":"opt B"}}}}`
	uResp, err := http.Post(testServer.URL+"/v1/systemone", "application/json", strings.NewReader(payload))
	if err != nil || uResp.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected 401 Unauthorized for missing token, got %v", uResp.StatusCode)
	}

	// 3. Authenticated POST with Bearer token succeeds with 200
	req, _ := http.NewRequest("POST", testServer.URL+"/v1/systemone", strings.NewReader(payload))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	cResp, err := http.DefaultClient.Do(req)
	if err != nil || cResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK with valid bearer token, got %v (err: %v)", cResp.StatusCode, err)
	}

	var soResp decisionindex.SystemOneResponse
	if err := json.NewDecoder(cResp.Body).Decode(&soResp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if soResp.Answers["q1"].Choice != "A" {
		t.Errorf("expected choice A, got %q", soResp.Answers["q1"].Choice)
	}
}
