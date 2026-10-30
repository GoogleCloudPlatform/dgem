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
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// sentRead is one upstream read as the mock server saw it.
type sentRead struct {
	questions []sentQ
	images    int
}

type sentQ struct {
	ID      string   `json:"id"`
	Type    string   `json:"type"`
	Levels  []string `json:"levels"`
	Options []struct {
		Name string `json:"name"`
	} `json:"options"`
}

// jevUpstream behaves like the serving image's chat route for these tests: it refuses more than 10 questions in one
// read and questions with fewer than two alternatives, answers choices with the first option, booleans with
// noul 0.8 and score questions with 0.7 on the third level (or the last, if fewer), keyed by level text like the
// server. It records every read.
func jevUpstream(t *testing.T) (*httptest.Server, func() []sentRead) {
	t.Helper()
	var mu sync.Mutex
	var reads []sentRead
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		var schema struct {
			Questions []sentQ `json:"questions"`
		}
		images := 0
		for _, m := range body.Messages {
			var s string
			if json.Unmarshal(m.Content, &s) == nil {
				if m.Role == "system" {
					_ = json.Unmarshal([]byte(s), &schema)
				}
				continue
			}
			var parts []map[string]any
			if json.Unmarshal(m.Content, &parts) == nil {
				for _, p := range parts {
					if p["type"] == "image_url" {
						images++
					}
				}
			}
		}
		mu.Lock()
		reads = append(reads, sentRead{schema.Questions, images})
		mu.Unlock()
		fail := func(msg string) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprintf(w, `{"error": {"message": %q, "type": "invalid_request_error"}}`, msg)
		}
		if len(schema.Questions) > 10 {
			fail("schema: more than 10 questions would use the indexed format")
			return
		}
		answers := map[string]client.QuestionAnswer{}
		for _, q := range schema.Questions {
			switch q.Type {
			case "boolean":
				answers[q.ID] = client.QuestionAnswer{Type: "noul", Noul: 0.8}
			case "score":
				if len(q.Levels) < 2 {
					fail(fmt.Sprintf("question '%s': needs at least two alternatives", q.ID))
					return
				}
				top := 2
				if top >= len(q.Levels) {
					top = len(q.Levels) - 1
				}
				probs := map[string]float64{}
				for i, l := range q.Levels {
					probs[l] = 0.3 / float64(len(q.Levels)-1)
					if i == top {
						probs[l] = 0.7
					}
				}
				answers[q.ID] = client.QuestionAnswer{Type: "score", Level: q.Levels[top], Probabilities: probs}
			default:
				if len(q.Options) < 2 {
					fail(fmt.Sprintf("question '%s': needs at least two alternatives", q.ID))
					return
				}
				probs := map[string]float64{}
				for _, o := range q.Options {
					probs[o.Name] = 0.1 / float64(len(q.Options)-1)
				}
				probs[q.Options[0].Name] = 0.9
				answers[q.ID] = client.QuestionAnswer{Type: "choice", Choice: q.Options[0].Name, Probabilities: probs}
			}
		}
		content, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": string(content)}}}})
	}))
	return srv, func() []sentRead { mu.Lock(); defer mu.Unlock(); return append([]sentRead(nil), reads...) }
}

// mixedRequestJSON builds a Jev request body with the given numbers of choice, noul and 5-level score questions.
// Ids contain dots and end in letters, the shape that broke the server's indexed format.
func mixedRequestJSON(choices, nouls, scores int) string {
	qs := map[string]any{}
	for i := 0; i < choices; i++ {
		qs[fmt.Sprintf("layout.slot%c.kind", 'a'+i%26)+fmt.Sprint(i)] = map[string]any{"type": "choice", "instructions": "Pick one.",
			"criteria": map[string]string{"list": "a list", "grid": "a grid", "form": "a form"}}
	}
	for i := 0; i < nouls; i++ {
		qs[fmt.Sprintf("screen%d.article", i)] = map[string]any{"type": "noul", "instructions": "Is it shown?"}
	}
	for i := 0; i < scores; i++ {
		qs[fmt.Sprintf("rating%d.severity", i)] = map[string]any{"type": "score", "instructions": "Rate it.",
			"criteria": []string{"none", "low", "medium", "high", "critical"}}
	}
	b, _ := json.Marshal(map[string]any{"model": "dgemma", "state": "a planning brief", "questions": qs})
	return string(b)
}

func postAdapter(t *testing.T, h http.HandlerFunc, body string) (int, SystemOneResponse, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	var resp SystemOneResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v: %s", err, rec.Body.String())
		}
	}
	return rec.Code, resp, rec.Body.String()
}

func newTestHandler(srvURL string) http.HandlerFunc {
	cli := client.NewClient(srvURL+"/v1", "dgemma", 5*time.Second)
	cli.MaxRetries = 0
	return NewSystemOneHTTPHandler(cli, DefaultEngineOptions())
}

// Score questions (criteria as an array of levels) are parsed, batched with the rest in reads of at most 8
// questions, and answered in the Jev form. Shapes: 33, 76, 74 and 16 questions with 1-2 score questions.
func TestScoreQuestionsBatched(t *testing.T) {
	for _, tc := range []struct{ choices, nouls, scores int }{{5, 27, 1}, {62, 13, 1}, {58, 15, 1}, {5, 9, 2}} {
		srv, reads := jevUpstream(t)
		code, resp, raw := postAdapter(t, newTestHandler(srv.URL), mixedRequestJSON(tc.choices, tc.nouls, tc.scores))
		srv.Close()
		total := tc.choices + tc.nouls + tc.scores
		if code != http.StatusOK {
			t.Fatalf("%v: %d %s", tc, code, raw)
		}
		if len(resp.Answers) != total {
			t.Fatalf("%v: %d answers, want %d", tc, len(resp.Answers), total)
		}
		want := (total + MaxSlotsPerPass - 1) / MaxSlotsPerPass
		if got := len(reads()); got != want || resp.EvaluationTrace.ForwardPasses != want {
			t.Errorf("%v: %d reads (trace %d), want %d", tc, got, resp.EvaluationTrace.ForwardPasses, want)
		}
		for _, rd := range reads() {
			if len(rd.questions) > MaxSlotsPerPass {
				t.Errorf("%v: a read had %d questions", tc, len(rd.questions))
			}
		}
		for k, a := range resp.Answers {
			if !strings.HasPrefix(k, "rating") {
				continue
			}
			if a.Type != "score" || a.Score == nil || a.Confidence == nil || len(a.Legend) != 5 || a.Legend["4"] != "critical" {
				t.Fatalf("%s: %+v", k, a)
			}
			sum, ev := 0.0, 0.0
			for i := 0; i < 5; i++ {
				p, ok := a.Probabilities[fmt.Sprint(i)]
				if !ok {
					t.Fatalf("%s: probabilities not keyed by index: %v", k, a.Probabilities)
				}
				sum += p
				ev += float64(i) * p
			}
			if math.Abs(sum-1) > 1e-9 || math.Abs(*a.Score-ev) > 1e-9 || math.Abs(*a.Score-2.0) > 1e-6 {
				t.Errorf("%s: sum %v score %v ev %v", k, sum, *a.Score, ev)
			}
			if math.Abs(*a.Confidence-0.7) > 1e-6 {
				t.Errorf("%s: confidence %v, want 0.7", k, *a.Confidence)
			}
		}
	}
}

// One-option choices and one-level scores are answered without a model call; a batch of only those makes none,
// and a mixed batch sends only the real questions.
func TestTrivialQuestionsAnsweredLocally(t *testing.T) {
	srv, reads := jevUpstream(t)
	defer srv.Close()
	h := newTestHandler(srv.URL)

	allTrivial := `{"state": "endgame", "questions": {
		"move_1": {"type": "choice", "criteria": {"r1c1_5": "r1c1=5 forced"}},
		"move_2": {"type": "choice", "criteria": {"r2c7_3": "r2c7=3 forced"}},
		"tier":   {"type": "score", "criteria": ["only level"]}}}`
	code, resp, raw := postAdapter(t, h, allTrivial)
	if code != http.StatusOK || len(reads()) != 0 || resp.EvaluationTrace.ForwardPasses != 0 {
		t.Fatalf("all trivial: %d, %d reads: %s", code, len(reads()), raw)
	}
	m := resp.Answers["move_1"]
	if m.Type != "choice" || m.Choice != "r1c1_5" || m.Probabilities["r1c1_5"] != 1 || m.Confidence == nil || *m.Confidence != 1 {
		t.Errorf("move_1: %+v", m)
	}
	s := resp.Answers["tier"]
	if s.Type != "score" || s.Score == nil || *s.Score != 0 || s.Legend["0"] != "only level" || s.Probabilities["0"] != 1 {
		t.Errorf("tier: %+v", s)
	}

	mixed := `{"state": "midgame", "questions": {
		"move_1": {"type": "choice", "criteria": {"r9c9_2": "forced"}},
		"move_2": {"type": "choice", "criteria": {"r1c2_4": "a", "r1c2_7": "b"}},
		"move_3": {"type": "choice", "criteria": {"r3c3_1": "a", "r3c3_8": "b"}}}}`
	code, resp, raw = postAdapter(t, h, mixed)
	if code != http.StatusOK || len(resp.Answers) != 3 {
		t.Fatalf("mixed: %d %s", code, raw)
	}
	rs := reads()
	if len(rs) != 1 || len(rs[0].questions) != 2 {
		t.Fatalf("mixed: reads %+v, want one read with the two real questions", rs)
	}
	for _, q := range rs[0].questions {
		if q.ID == "move_1" {
			t.Error("the one-option question was sent upstream")
		}
	}
	if resp.Answers["move_1"].Choice != "r9c9_2" {
		t.Errorf("move_1: %+v", resp.Answers["move_1"])
	}
}

// Bodies the adapter can't read are 400 (unknown type, wrong criteria shape); 0-option choices stay the server's 422.
func TestQuestionValidation(t *testing.T) {
	srv, reads := jevUpstream(t)
	defer srv.Close()
	h := newTestHandler(srv.URL)
	for _, tc := range []struct {
		name, q string
		want    int
	}{
		{"unknown type", `{"type": "ranking", "criteria": {"a": "", "b": ""}}`, 400},
		{"score with object criteria", `{"type": "score", "criteria": {"a": "", "b": ""}}`, 400},
		{"choice with array criteria", `{"type": "choice", "criteria": ["a", "b"]}`, 400},
		{"criteria a string", `{"type": "choice", "criteria": "a"}`, 400},
		{"score with no levels", `{"type": "score", "criteria": []}`, 422},
		{"score with repeated levels", `{"type": "score", "criteria": ["low", "low", "high"]}`, 422},
	} {
		code, _, raw := postAdapter(t, h, `{"state": "s", "questions": {"q": `+tc.q+`}}`)
		if code != tc.want {
			t.Errorf("%s: %d %s, want %d", tc.name, code, raw, tc.want)
		}
	}
	if n := len(reads()); n != 0 {
		t.Errorf("%d upstream reads for invalid questions", n)
	}
	code, _, raw := postAdapter(t, h, `{"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}}`)
	if code != http.StatusUnprocessableEntity || strings.Contains(raw, "options per choice") {
		t.Errorf("0-option choice: %d %q, want 422 without the capacity hint", code, raw)
	}
	// An empty type keeps meaning choice (object) and becomes score for an array.
	code, resp, raw := postAdapter(t, h, `{"state": "s", "questions": {"c": {"criteria": {"a": "", "b": ""}}, "s": {"criteria": ["x", "y", "z"]}}}`)
	if code != http.StatusOK || resp.Answers["c"].Type != "choice" || resp.Answers["s"].Type != "score" {
		t.Errorf("empty type: %d %s", code, raw)
	}
}

// The capacity hint is only added for the server's upper limit, worded for both question types.
func TestErrorStatusAlternativesHint(t *testing.T) {
	st, msg := ErrorStatus(&client.HTTPError{StatusCode: 400, Body: `{"error": {"message": "question 'move_1': needs at least two alternatives"}}`})
	if st != 422 || strings.Contains(msg, "at most") {
		t.Errorf("needs at least two: %d %q", st, msg)
	}
	st, msg = ErrorStatus(&client.HTTPError{StatusCode: 400, Body: `{"error": {"message": "question 'q': at most 52 alternatives"}}`})
	if st != 422 || !strings.Contains(msg, "(at most 52 options per choice or levels per score)") {
		t.Errorf("at most 52: %d %q", st, msg)
	}
}

// "media" is an alias for "images": image/* entries (data: URI or raw base64) reach the model, after any "images",
// in order; other media types are 400.
func TestMediaAlias(t *testing.T) {
	r := SystemOneRequest{Images: []string{"data:image/png;base64,AAAA"}, Media: []SystemOneMedia{
		{MimeType: "image/jpeg", Data: "data:image/jpeg;base64,BBBB"},
		{MimeType: "image/png", Data: "CCCC"},
		{Data: "data:image/webp;base64,DDDD"},
	}}
	if err := r.MergeMedia(); err != nil {
		t.Fatal(err)
	}
	want := []string{"data:image/png;base64,AAAA", "data:image/jpeg;base64,BBBB", "data:image/png;base64,CCCC", "data:image/webp;base64,DDDD"}
	if fmt.Sprint(r.Images) != fmt.Sprint(want) || r.Media != nil {
		t.Fatalf("images %v media %v", r.Images, r.Media)
	}
	for _, tc := range []struct {
		m    SystemOneMedia
		want error
	}{
		{SystemOneMedia{MimeType: "audio/mpeg", Data: "AAAA"}, ErrUnsupportedMedia},
		{SystemOneMedia{MimeType: "video/mp4", Data: "AAAA"}, ErrUnsupportedMedia},
		{SystemOneMedia{MimeType: "image/png", Data: "data:audio/wav;base64,AAAA"}, ErrUnsupportedMedia},
		{SystemOneMedia{Data: "AAAA"}, ErrInvalidImage},
		{SystemOneMedia{MimeType: "image/png"}, ErrInvalidImage},
		{SystemOneMedia{MimeType: "image/png", Data: "https://example.com/a.png"}, ErrInvalidImage},
	} {
		r := SystemOneRequest{Media: []SystemOneMedia{tc.m}}
		if err := r.MergeMedia(); !errors.Is(err, tc.want) {
			t.Errorf("%+v: %v, want %v", tc.m, err, tc.want)
		} else if st, _ := ErrorStatus(err); st != http.StatusBadRequest {
			t.Errorf("%+v: status %d", tc.m, st)
		}
	}

	srv, reads := jevUpstream(t)
	defer srv.Close()
	h := newTestHandler(srv.URL)
	body := `{"state": "scene", "media": [{"mime_type": "image/png", "data": "iVBORw0KGgo=", "name": "frame.png"}],
		"questions": {"q": {"type": "noul", "instructions": "Is there a spill?"}}}`
	if code, _, raw := postAdapter(t, h, body); code != http.StatusOK {
		t.Fatalf("media image: %d %s", code, raw)
	}
	if rs := reads(); len(rs) != 1 || rs[0].images != 1 {
		t.Fatalf("media image did not reach the model: %+v", rs)
	}
	body = `{"state": "scene", "media": [{"mime_type": "audio/wav", "data": "UklGRg=="}], "questions": {"q": {"type": "noul"}}}`
	if code, _, raw := postAdapter(t, h, body); code != http.StatusBadRequest {
		t.Fatalf("media audio: %d %s", code, raw)
	}
	if n := len(reads()); n != 1 {
		t.Errorf("audio request reached upstream (%d reads)", n)
	}
}
