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
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// TestPromptLayout checks that --prompt-layout reaches every upstream request, including the bracket rounds of a
// wide-option tournament, and that the default sends no "layout" key (byte-identical to older servers).
func TestPromptLayout(t *testing.T) {
	var mu sync.Mutex
	var layouts []string
	var seen []string
	mock := catchAllServer(t, &seen)
	defer mock.Close()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct{ Role, Content string } `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		for _, m := range body.Messages {
			if m.Role == "system" {
				var s map[string]any
				_ = json.Unmarshal([]byte(m.Content), &s)
				l, _ := s["layout"].(string)
				mu.Lock()
				layouts = append(layouts, l)
				mu.Unlock()
			}
		}
		b, _ := json.Marshal(body)
		req, _ := http.NewRequest(http.MethodPost, mock.URL, bytes.NewReader(b))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Error(err)
			return
		}
		defer resp.Body.Close()
		var out any
		_ = json.NewDecoder(resp.Body).Decode(&out)
		_ = json.NewEncoder(w).Encode(out)
	}))
	defer srv.Close()
	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)
	req := SystemOneRequest{State: "x", Questions: map[string]SystemOneQuestion{
		"intent": {Type: "choice", Instructions: "Which intent?", Criteria: wideWithCatchAll()},
		"h":      {Type: "noul", Instructions: "Hallucination?", Criteria: map[string]string{"true": "a", "false": "b"}}}}
	if DefaultEngineOptions().PromptLayout != "document_first" {
		t.Fatalf("default layout should be document_first")
	}
	for _, want := range []string{"", "document_first"} {
		layouts = nil
		opts := DefaultEngineOptions()
		opts.PromptLayout = want
		if _, err := ExecuteSystemOne(context.Background(), cli, req, opts); err != nil {
			t.Fatalf("%q: %v", want, err)
		}
		if len(layouts) < 3 {
			t.Fatalf("%q: expected bracket rounds plus a batch, saw %d requests", want, len(layouts))
		}
		for _, l := range layouts {
			if l != want {
				t.Fatalf("layout %q: a request carried %q (%v)", want, l, layouts)
			}
		}
	}
}
