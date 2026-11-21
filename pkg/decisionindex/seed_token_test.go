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
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestExecuteSystemOne_SeedTokenForwarding(t *testing.T) {
	var mu sync.Mutex
	var schemas []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		for _, m := range body.Messages {
			if m.Role == "system" {
				if s, ok := m.Content.(string); ok {
					var sc map[string]any
					_ = json.Unmarshal([]byte(s), &sc)
					mu.Lock()
					schemas = append(schemas, sc)
					mu.Unlock()
				}
			}
		}
		content := `{"answers":{"q":{"answer":"yes","confidence":0.9,"probabilities":{"yes":0.9,"no":0.1}}}}`
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}}},
		})
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)

	tests := []struct {
		name     string
		reqST    string
		optST    string
		wantST   any // nil means key absent
		wantFail bool
	}{
		{name: "unset omits seed_token", reqST: "", optST: "", wantST: nil},
		{name: "option default fixed applied when request unset", reqST: "", optST: "fixed", wantST: "fixed"},
		{name: "explicit request random overrides option fixed", reqST: "random", optST: "fixed", wantST: "random"},
		{name: "explicit request fixed applied without option", reqST: "fixed", optST: "", wantST: "fixed"},
		{name: "invalid seed_token rejected", reqST: "bogus", optST: "", wantFail: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mu.Lock()
			schemas = nil
			mu.Unlock()
			req := SystemOneRequest{
				SeedToken: tc.reqST,
				State:     "state text",
				Questions: map[string]SystemOneQuestion{"q": {Type: "noul", Instructions: "check"}},
			}
			opts := DefaultEngineOptions()
			opts.SeedToken = tc.optST
			_, err := ExecuteSystemOne(context.Background(), cli, req, opts)
			if tc.wantFail {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("ExecuteSystemOne: %v", err)
			}
			mu.Lock()
			got := schemas[0]["seed_token"]
			mu.Unlock()
			if got != tc.wantST {
				t.Errorf("schema seed_token = %v, want %v", got, tc.wantST)
			}
		})
	}
}
