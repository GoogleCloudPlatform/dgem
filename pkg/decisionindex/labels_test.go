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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

type sentQuestion struct {
	labels string
	n      int
}

// labelUpstream records, per upstream request, the schema's "labels" and each question's option count; it refuses
// more options than the label set allows, like the server, and picks the option named "TARGET" when present.
func labelUpstream(t *testing.T) (*httptest.Server, func() []sentQuestion) {
	t.Helper()
	var mu sync.Mutex
	var seen []sentQuestion
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		var schema struct {
			Labels    string `json:"labels"`
			Questions []struct {
				ID      string                  `json:"id"`
				Options []struct{ Name string } `json:"options"`
			} `json:"questions"`
		}
		for _, m := range body.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &schema)
			}
		}
		limit := optionLimit(schema.Labels)
		answers := map[string]client.QuestionAnswer{}
		for _, q := range schema.Questions {
			mu.Lock()
			seen = append(seen, sentQuestion{schema.Labels, len(q.Options)})
			mu.Unlock()
			if len(q.Options) > limit {
				http.Error(w, fmt.Sprintf(`{"error":{"message":"question %q: at most %d alternatives"}}`, q.ID, limit), http.StatusBadRequest)
				return
			}
			pick := q.Options[0].Name
			probs := map[string]float64{}
			for _, o := range q.Options {
				probs[o.Name] = 0.01
				if strings.Contains(o.Name, "TARGET") {
					pick = o.Name
				}
			}
			probs[pick] = 0.9
			answers[q.ID] = client.QuestionAnswer{Type: "choice", Choice: pick, Probabilities: probs}
		}
		content, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": string(content)}}}})
	}))
	return srv, func() []sentQuestion { mu.Lock(); defer mu.Unlock(); return append([]sentQuestion(nil), seen...) }
}

func wideRequest(n int) SystemOneRequest {
	crit := map[string]string{}
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("opt_%02d", i)
		if i == n-3 {
			name = "opt_TARGET"
		}
		crit[name] = "d"
	}
	return SystemOneRequest{State: "x", Questions: map[string]SystemOneQuestion{"q": {Type: "choice", Criteria: crit}}}
}

func TestLabelsWideNativeAndBrackets(t *testing.T) {
	for _, tc := range []struct {
		labels string
		n      int
		want   []sentQuestion // upstream questions in any order: Round-1 brackets then final, or one native read
	}{
		{LabelsAZ, 30, []sentQuestion{{"", 15}, {"", 15}, {"", 4}}},                  // default: bracket (unchanged)
		{LabelsAZAA, 30, []sentQuestion{{"az_aa", 30}}},                              // one native read
		{LabelsAZAA, 52, []sentQuestion{{"az_aa", 52}}},                              // at the limit
		{LabelsAZLower, 40, []sentQuestion{{"az_lower", 40}}},                        //
		{LabelsAZAA, 60, []sentQuestion{{"az_aa", 30}, {"az_aa", 30}, {"az_aa", 4}}}, // 52-wide brackets: 2 of 30
	} {
		srv, seen := labelUpstream(t)
		cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
		cli.MaxRetries = 0
		opts := DefaultEngineOptions()
		opts.Labels = tc.labels
		resp, err := ExecuteSystemOne(t.Context(), cli, wideRequest(tc.n), opts)
		srv.Close()
		if err != nil {
			t.Fatalf("%s/%d: %v", tc.labels, tc.n, err)
		}
		got := seen()
		if len(got) != len(tc.want) {
			t.Fatalf("%s/%d: upstream questions %v, want %v", tc.labels, tc.n, got, tc.want)
		}
		count := map[sentQuestion]int{}
		for _, g := range got {
			count[g]++
		}
		for _, w := range tc.want {
			count[w]--
		}
		for k, v := range count {
			if v != 0 {
				t.Fatalf("%s/%d: upstream questions %v, want %v (mismatch at %v)", tc.labels, tc.n, got, tc.want, k)
			}
		}
		if c := resp.Answers["q"].Choice; c != "opt_TARGET" {
			t.Fatalf("%s/%d: answer %q", tc.labels, tc.n, c)
		}
		if p := resp.Answers["q"].Probabilities; len(p) != tc.n {
			t.Fatalf("%s/%d: %d probabilities", tc.labels, tc.n, len(p))
		}
	}
}
