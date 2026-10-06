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

// catchAllServer mimics the failure mode: the option named *TARGET* wins when present; otherwise an option whose
// description says "none of the listed" wins (the catch-all is the right answer in a bracket without the target).
// For yes/no questions rendered as choices it reports the "yes" option at 0.8; for plain booleans, noul = 0.2.
func catchAllServer(t *testing.T, seen *[]string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct{ Role, Content string } `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		var sys struct {
			Questions []struct {
				ID      string                               `json:"id"`
				Type    string                               `json:"type"`
				Options []struct{ Name, Description string } `json:"options"`
			} `json:"questions"`
		}
		for _, m := range body.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sys)
			}
		}
		answers := map[string]client.QuestionAnswer{}
		for _, q := range sys.Questions {
			*seen = append(*seen, q.Type)
			if q.Type == "boolean" {
				answers[q.ID] = client.QuestionAnswer{Noul: 0.2}
				continue
			}
			win := ""
			for _, o := range q.Options {
				if strings.Contains(o.Name, "TARGET") || o.Name == "yes" {
					win = o.Name
				}
			}
			if win == "" {
				for _, o := range q.Options {
					if strings.Contains(o.Description, "none of the listed") {
						win = o.Name
					}
				}
			}
			if win == "" {
				win = q.Options[0].Name
			}
			p := map[string]float64{}
			rest := 0.2 / float64(maxInt(1, len(q.Options)-1))
			for _, o := range q.Options {
				p[o.Name] = rest
			}
			p[win] = 0.8
			answers[q.ID] = client.QuestionAnswer{Choice: win, Probabilities: p}
		}
		env, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": string(env)}}}})
	}))
}

func wideWithCatchAll() map[string]string {
	c := map[string]string{"out_of_scope": "out of scope: none of the listed intents"}
	for i := 0; i < 150; i++ {
		c[fmt.Sprintf("intent_%03d", i)] = fmt.Sprintf("intent number %d", i)
	}
	delete(c, "intent_077")
	c["intent_077_TARGET"] = "the right intent"
	return c
}

func TestCatchAllModes(t *testing.T) {
	var seen []string
	srv := catchAllServer(t, &seen)
	defer srv.Close()
	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)
	req := SystemOneRequest{State: "x", Questions: map[string]SystemOneQuestion{
		"intent": {Type: "choice", Instructions: "Which intent?", Criteria: wideWithCatchAll()}}}
	for mode, want := range map[string]string{"final": "intent_077_TARGET", "verify": "intent_077_TARGET", "both": "intent_077_TARGET"} {
		opts := DefaultEngineOptions()
		opts.CatchAll = mode
		resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		a := resp.Answers["intent"]
		sum := 0.0
		for _, v := range a.Probabilities {
			sum += v
		}
		if a.Choice != want || len(a.Probabilities) != 151 || math.Abs(sum-1) > 1e-6 {
			t.Fatalf("%s: choice %s (want %s), %d keys, sum %v", mode, a.Choice, want, len(a.Probabilities), sum)
		}
	}
	// default ("off") keeps today's behaviour; in this mock the catch-all reaches the final and the target still wins
	// the final (it is the only TARGET), so check structure only.
	resp, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions())
	if err != nil || len(resp.Answers["intent"].Probabilities) != 151 {
		t.Fatalf("off: %v", err)
	}
}

func TestNoulModeChoice(t *testing.T) {
	var seen []string
	srv := catchAllServer(t, &seen)
	defer srv.Close()
	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)
	req := SystemOneRequest{State: "x", Questions: map[string]SystemOneQuestion{
		"h": {Type: "noul", Instructions: "Hallucination?", Criteria: map[string]string{"true": "unsupported", "false": "supported"}}}}
	opts := DefaultEngineOptions()
	resp, err := ExecuteSystemOne(context.Background(), cli, req, opts)
	if err != nil || resp.Answers["h"].Noul == nil || math.Abs(*resp.Answers["h"].Noul-0.2) > 1e-9 {
		t.Fatalf("noul mode: %v %+v", err, resp)
	}
	opts.NoulMode = "choice"
	resp, err = ExecuteSystemOne(context.Background(), cli, req, opts)
	if err != nil || resp.Answers["h"].Noul == nil || math.Abs(*resp.Answers["h"].Noul-0.8) > 1e-9 {
		t.Fatalf("choice mode: %v %+v", err, resp)
	}
	if resp.Answers["h"].Type != "noul" {
		t.Fatalf("answer type must stay noul, got %s", resp.Answers["h"].Type)
	}
	if seen[len(seen)-1] != "choice" {
		t.Fatalf("choice mode should send a choice question, sent %v", seen)
	}
	q := noulAsChoice("h", req.Questions["h"])
	opt := q["options"].([]map[string]string)
	if opt[0]["description"] != "unsupported" || opt[1]["description"] != "supported" {
		t.Fatalf("descriptions: %v", opt)
	}
}

func TestCatchAllKeys(t *testing.T) {
	got := catchAllKeys(map[string]string{"option_0": "accept reservations", "option_80": "out of scope: none of the listed intents",
		"option_1": "account blocked", "other_none": "None of the above"})
	if len(got) != 2 || got[0] != "option_80" || got[1] != "other_none" {
		t.Fatalf("got %v", got)
	}
	if catchAllKeys(map[string]string{"a": "billing", "b": "shipping"}) != nil {
		t.Fatal("no catch-all expected")
	}
	many := map[string]string{}
	for i := 0; i < 5; i++ {
		many[fmt.Sprintf("o%d", i)] = "not applicable"
	}
	if catchAllKeys(many) != nil {
		t.Fatal("more than 3 matches must not be trusted")
	}
}
