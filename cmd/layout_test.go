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
	"sync"
	"testing"
)

func TestApplyPromptLayout(t *testing.T) {
	s := `{"questions":[{"id":"a","type":"boolean"}]}`
	if got, err := ApplyPromptLayout(s, ""); err != nil || got != s {
		t.Fatalf("empty layout must leave the schema untouched: %q %v", got, err)
	}
	got, err := ApplyPromptLayout(s, " Schema_First ")
	if err != nil || !strings.Contains(got, `"layout":"schema_first"`) {
		t.Fatalf("schema_first: %q %v", got, err)
	}
	if _, err := ApplyPromptLayout(s, "sideways"); err == nil {
		t.Fatal("invalid layout must be rejected")
	}
}

// TestMCPCustomQuestionsLayout checks that the MCP layout field reaches the upstream schema, and that no layout key is
// sent when the field is empty (older servers and the local backend never see it).
func TestMCPCustomQuestionsLayout(t *testing.T) {
	startFakeLocalDiffgemma(t)
	var mu sync.Mutex
	var schemas []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct{ Role, Content string } `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		for _, m := range body.Messages {
			if m.Role == "system" {
				var s map[string]any
				_ = json.Unmarshal([]byte(m.Content), &s)
				mu.Lock()
				schemas = append(schemas, s)
				mu.Unlock()
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"index": 0, "finish_reason": "stop",
			"message": map[string]any{"role": "assistant", "content": localDiffgemmaContent}}}})
	}))
	defer srv.Close()
	backendConfigMu.Lock()
	serveLocalURL = srv.URL + "/v1"
	backendConfigMu.Unlock()
	cs := connectTestMCPClient(t)
	args := map[string]any{"context": "My card was charged twice.", "backend": "local",
		"questions": []map[string]any{{"id": "urgent", "type": "boolean", "question": "Is it urgent?"}}}
	for _, want := range []string{"", "schema_first"} {
		schemas = nil
		if want != "" {
			args["layout"] = want
		}
		callDecideTool(t, cs, "decide_custom_questions", args)
		if len(schemas) == 0 {
			t.Fatalf("%q: no upstream request captured", want)
		}
		got, has := schemas[len(schemas)-1]["layout"]
		if want == "" && has {
			t.Fatalf("empty layout must not send a layout key, sent %v", got)
		}
		if want != "" && got != want {
			t.Fatalf("layout %q: upstream schema had %v", want, got)
		}
	}
}
