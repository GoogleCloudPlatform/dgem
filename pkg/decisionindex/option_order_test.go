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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// recordingUpstream answers every question with its first option and records each question's option names in the
// order the adapter sent them.
func recordingUpstream(t *testing.T) (*httptest.Server, func() [][]string) {
	t.Helper()
	var mu sync.Mutex
	var seen [][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		var schema struct {
			Questions []struct {
				ID      string `json:"id"`
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range body.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &schema)
			}
		}
		answers := map[string]client.QuestionAnswer{}
		mu.Lock()
		for _, q := range schema.Questions {
			var names []string
			probs := map[string]float64{}
			for i, o := range q.Options {
				names = append(names, o.Name)
				probs[o.Name] = 0.01
				if i == 0 {
					probs[o.Name] = 0.9
				}
			}
			seen = append(seen, names)
			answers[q.ID] = client.QuestionAnswer{Type: "choice", Choice: names[0], Label: "A", Probabilities: probs}
		}
		mu.Unlock()
		content, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": string(content)}}}})
	}))
	return srv, func() [][]string { mu.Lock(); defer mu.Unlock(); return append([][]string(nil), seen...) }
}

func postSystemOne(t *testing.T, srvURL, body string, order string) SystemOneResponse {
	t.Helper()
	cli := client.NewClient(srvURL+"/v1", "dgemma", 5*time.Second)
	opts := DefaultEngineOptions()
	opts.OptionOrder = order
	rec := httptest.NewRecorder()
	NewSystemOneHTTPHandler(cli, opts)(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("%s: HTTP %d %s", order, rec.Code, rec.Body.String())
	}
	var resp SystemOneResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	return resp
}

// TestOptionOrderSourceEndToEnd: with "source" the upstream sees options in the request's order; with the default
// "alpha" it sees them sorted (unchanged behaviour).
func TestOptionOrderSourceEndToEnd(t *testing.T) {
	body := `{"state": "x", "questions": {"q": {"type": "choice", "instructions": "pick",
		"criteria": {"zebra": "z", "apple": "a", "mango": "m", "kiwi": "k"}}}}`
	for _, tc := range []struct {
		order string
		want  []string
		pick  string
	}{
		{OptionOrderAlpha, []string{"apple", "kiwi", "mango", "zebra"}, "apple"},
		{OptionOrderSource, []string{"zebra", "apple", "mango", "kiwi"}, "zebra"},
		{"", []string{"apple", "kiwi", "mango", "zebra"}, "apple"},
	} {
		srv, seen := recordingUpstream(t)
		resp := postSystemOne(t, srv.URL, body, tc.order)
		srv.Close()
		got := seen()
		if len(got) != 1 || !reflect.DeepEqual(got[0], tc.want) {
			t.Fatalf("order %q: upstream saw %v, want %v", tc.order, got, tc.want)
		}
		if c := resp.Answers["q"].Choice; c != tc.pick {
			t.Fatalf("order %q: answer %q, want the first-shown %q", tc.order, c, tc.pick)
		}
	}
}

// TestOptionOrderSourceBrackets: with 30 options, Round-1 brackets are contiguous slices of the request order and
// the final lists finalists in request order.
func TestOptionOrderSourceBrackets(t *testing.T) {
	var keys []string
	var parts []string
	for i := 0; i < 30; i++ {
		k := fmt.Sprintf("opt_%02d", (i*7)%30) // a fixed non-alphabetical order
		keys = append(keys, k)
		parts = append(parts, fmt.Sprintf("%q: %q", k, "desc "+k))
	}
	body := `{"state": "x", "questions": {"q": {"type": "choice", "criteria": {` + strings.Join(parts, ", ") + `}}}}`
	srv, seen := recordingUpstream(t)
	defer srv.Close()
	postSystemOne(t, srv.URL, body, OptionOrderSource)
	got := seen()
	if len(got) != 3 { // two brackets of 15, then the final
		t.Fatalf("upstream questions %d: %v", len(got), got)
	}
	var round1 [][]string
	for _, g := range got[:2] {
		round1 = append(round1, g)
	}
	sort.Slice(round1, func(i, j int) bool { return round1[i][0] == keys[0] })
	if !reflect.DeepEqual(round1[0], keys[:15]) || !reflect.DeepEqual(round1[1], keys[15:]) {
		t.Fatalf("brackets %v, want request-order halves", round1)
	}
	pos := map[string]int{}
	for i, k := range keys {
		pos[k] = i
	}
	final := got[2]
	if !sort.SliceIsSorted(final, func(i, j int) bool { return pos[final[i]] < pos[final[j]] }) {
		t.Fatalf("final %v not in request order", final)
	}
}

func TestCriteriaKeyOrder(t *testing.T) {
	var q SystemOneQuestion
	if err := json.Unmarshal([]byte(`{"instructions": {"a": 1}, "criteria": {"b": {"x": [1]}, "a": "s", "b": "dup"}, "type": "choice"}`), &q); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(q.Order, []string{"b", "a"}) || q.Criteria["b"] != "dup" {
		t.Fatalf("order %v criteria %v", q.Order, q.Criteria)
	}
	// Unknown order (built in code) falls back to sorted keys.
	if got := optionKeys(SystemOneQuestion{Criteria: map[string]string{"c": "", "a": ""}}, OptionOrderSource); !reflect.DeepEqual(got, []string{"a", "c"}) {
		t.Fatalf("fallback %v", got)
	}
}
