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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

const testImage = "data:image/png;base64,iVBORw0KGgo="

// imageRecordingServer answers every decision with the first option and records, per request, the image URLs found
// in the user message.
func imageRecordingServer(t *testing.T) (*httptest.Server, func() [][]string) {
	t.Helper()
	var mu sync.Mutex
	var seen [][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var schema struct {
			Questions []struct {
				ID      string `json:"id"`
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		var imgs []string
		for _, m := range body.Messages {
			switch m.Role {
			case "system":
				var s string
				_ = json.Unmarshal(m.Content, &s)
				_ = json.Unmarshal([]byte(s), &schema)
			case "user":
				var parts []struct {
					Type     string `json:"type"`
					ImageURL *struct {
						URL string `json:"url"`
					} `json:"image_url"`
				}
				if json.Unmarshal(m.Content, &parts) == nil {
					for _, p := range parts {
						if p.Type == "image_url" && p.ImageURL != nil {
							imgs = append(imgs, p.ImageURL.URL)
						}
					}
				}
			}
		}
		mu.Lock()
		seen = append(seen, imgs)
		mu.Unlock()

		answers := map[string]client.QuestionAnswer{}
		for _, q := range schema.Questions {
			probs := map[string]float64{}
			for i, o := range q.Options {
				probs[o.Name] = 0.01
				if i == 0 {
					probs[o.Name] = 0.9
				}
			}
			answers[q.ID] = client.QuestionAnswer{Choice: q.Options[0].Name, Probabilities: probs}
		}
		env, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": string(env)}}},
		})
	}))
	return srv, func() [][]string { mu.Lock(); defer mu.Unlock(); return append([][]string(nil), seen...) }
}

func TestExecuteSystemOne_ImagesReachEverySubRequest(t *testing.T) {
	srv, seen := imageRecordingServer(t)
	defer srv.Close()
	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)

	wide := map[string]string{}
	for i := 0; i < 30; i++ {
		wide[fmt.Sprintf("opt_%02d", i)] = ""
	}
	req := SystemOneRequest{
		State:  "Inspect the screenshot.",
		Images: []string{testImage},
		Questions: map[string]SystemOneQuestion{
			"color": {Type: "choice", Instructions: "Main color", Criteria: map[string]string{"red": "", "blue": ""}},
			"page":  {Type: "choice", Instructions: "Which page is shown", Criteria: wide},
		},
	}
	resp, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions())
	if err != nil {
		t.Fatalf("ExecuteSystemOne: %v", err)
	}
	if len(resp.Answers) != 2 {
		t.Fatalf("want 2 answers, got %d", len(resp.Answers))
	}
	reqs := seen()
	// One standard batch plus bracket rounds and a final for the 30-option question.
	if len(reqs) < 3 {
		t.Fatalf("want at least 3 upstream requests (batch + brackets + final), got %d", len(reqs))
	}
	for i, imgs := range reqs {
		if len(imgs) != 1 || imgs[0] != testImage {
			t.Errorf("upstream request %d: images = %v, want [%s]", i, imgs, testImage)
		}
	}
}

func TestExecuteSystemOne_NoImagesStaysText(t *testing.T) {
	srv, seen := imageRecordingServer(t)
	defer srv.Close()
	cli := client.NewClient(srv.URL, "dgemma", 10*time.Second)
	req := SystemOneRequest{
		State:     "text only",
		Questions: map[string]SystemOneQuestion{"color": {Type: "choice", Criteria: map[string]string{"red": "", "blue": ""}}},
	}
	if _, err := ExecuteSystemOne(context.Background(), cli, req, DefaultEngineOptions()); err != nil {
		t.Fatalf("ExecuteSystemOne: %v", err)
	}
	for i, imgs := range seen() {
		if len(imgs) != 0 {
			t.Errorf("upstream request %d carried images %v", i, imgs)
		}
	}
}

func TestValidateImages(t *testing.T) {
	ok := []string{testImage, "", "https://example.com/a.png", "http://example.com/a.png"}
	if err := ValidateImages(ok); err != nil {
		t.Fatalf("ValidateImages(%v): %v", ok, err)
	}
	for _, bad := range []string{"/etc/passwd", "fixtures/x.png", "file:///etc/passwd"} {
		err := ValidateImages([]string{bad})
		if !errors.Is(err, ErrInvalidImage) {
			t.Errorf("ValidateImages(%q) = %v, want ErrInvalidImage", bad, err)
		}
	}
}

func TestNewSystemOneHTTPHandler_RejectsLocalImagePath(t *testing.T) {
	// The upstream is never contacted: validation fails first.
	cli := client.NewClient("http://127.0.0.1:9/v1", "dgemma", time.Second)
	h := NewSystemOneHTTPHandler(cli, DefaultEngineOptions())
	body := `{"state":"x","images":["/etc/passwd"],"questions":{"q":{"type":"noul","instructions":"ok?"}}}`
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d (%s), want 400", rec.Code, rec.Body.String())
	}
}
