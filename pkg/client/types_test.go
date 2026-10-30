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

package client

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseStructuredContent(t *testing.T) {
	raw := `{
		"answers": {
			"urgent": {
				"type": "noul",
				"label": "yes",
				"confidence": 0.985,
				"stderr": 0.012,
				"agreement": 1.0,
				"probabilities": {
					"yes": 0.985,
					"no": 0.015
				}
			},
			"team": {
				"type": "choice",
				"choice": "engineering",
				"label": "C",
				"confidence": 0.999,
				"stderr": 0.0001,
				"agreement": 1.0,
				"probabilities": {
					"engineering": 0.999
				}
			}
		},
		"diagnostics": {
			"hole": "noise",
			"steps": 1,
			"timing": {
				"denoise_ms": 850.5,
				"prefill_ms": 1200.0,
				"prompt_tokens": 160,
				"reused_tokens": 120,
				"rounds": 1,
				"samples": 1,
				"steps_run": 1
			},
			"samples": {
				"n": 1,
				"policy": {
					"extended": false,
					"first_read_max_entropy": 0.02,
					"max": 4,
					"mode": "auto",
					"pinned_slots": 0,
					"threshold": 0.1
				}
			}
		}
	}`

	parsed, err := ParseStructuredContent(raw)
	if err != nil {
		t.Fatalf("unexpected error parsing structured content: %v", err)
	}

	if len(parsed.Answers) != 2 {
		t.Fatalf("expected 2 answers, got %d", len(parsed.Answers))
	}

	urgent := parsed.Answers["urgent"]
	if urgent.DisplayType() != "boolean" {
		t.Errorf("expected DisplayType 'boolean', got %q", urgent.DisplayType())
	}
	if urgent.DisplayValue() != "yes" {
		t.Errorf("expected DisplayValue 'yes', got %q", urgent.DisplayValue())
	}

	if parsed.Diagnostics.Timing.DenoiseMs != 850.5 {
		t.Errorf("expected DenoiseMs 850.5, got %f", parsed.Diagnostics.Timing.DenoiseMs)
	}
}

func TestBuildMultimodalContent(t *testing.T) {
	// Case 1: No images -> returns pure text string
	textOnly, err := BuildMultimodalContent("plain text", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s, ok := textOnly.(string); !ok || s != "plain text" {
		t.Errorf("expected plain text string, got %v", textOnly)
	}

	// Case 2: Remote URL -> fetched and inlined as a data: URI (serving images refuse URLs), image first, text second
	imgSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))
	}))
	defer imgSrv.Close()
	urlOutput, err := BuildMultimodalContent("query text", []string{imgSrv.URL + "/test.png"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts, ok := urlOutput.([]ContentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("expected 2 content parts, got %v", urlOutput)
	}
	if parts[0].Type != "image_url" || !strings.HasPrefix(parts[0].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("expected first part to be image_url, got %+v", parts[0])
	}
	if parts[1].Type != "text" || parts[1].Text != "query text" {
		t.Errorf("expected second part to be text, got %+v", parts[1])
	}

	// Case 3: Local file -> reads and encodes as base64 data URI
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "sample.png")
	if err := os.WriteFile(tempFile, []byte("fake-png-content"), 0644); err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}

	localOutput, err := BuildMultimodalContent("inspect image", []string{tempFile})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	localParts, ok := localOutput.([]ContentPart)
	if !ok || len(localParts) != 2 {
		t.Fatalf("expected 2 parts, got %v", localOutput)
	}
	if !strings.HasPrefix(localParts[0].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("expected data:image/png;base64 prefix, got %q", localParts[0].ImageURL.URL)
	}
}

func TestParseStructuredContentWithLogprobs(t *testing.T) {
	raw := "thought\n{\"intent\": \"card_arrival\"}"
	lp := &ChoiceLogprobs{
		Content: []TokenLogprob{
			{Token: "thought", Logprob: -0.001},
			{Token: "{\"intent\":", Logprob: -0.002},
			{
				Token:   "card_arrival",
				Logprob: -0.08338, // exp(-0.08338) ≈ 0.920
				TopLogprobs: []TopLogprobItem{
					{Token: "card_arrival", Logprob: -0.08338},
					{Token: "card_delivery_estimate", Logprob: -2.5257},
				},
			},
			{Token: "\"}", Logprob: -0.0001},
		},
	}

	parsed, err := ParseStructuredContentWithLogprobs(raw, lp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	ans := parsed.Answers["intent"]
	if ans.DisplayValue() != "card_arrival" {
		t.Errorf("expected 'card_arrival', got %q", ans.DisplayValue())
	}
	if ans.Confidence < 0.91 || ans.Confidence > 0.93 {
		t.Errorf("expected calibrated confidence ~0.920 from exp(logprob), got %f", ans.Confidence)
	}
	if ans.Entropy <= 0 {
		t.Errorf("expected positive Shannon entropy from top_logprobs, got %f", ans.Entropy)
	}
	if len(ans.Probabilities) != 2 {
		t.Errorf("expected 2 top candidate probabilities, got %d", len(ans.Probabilities))
	}
}
