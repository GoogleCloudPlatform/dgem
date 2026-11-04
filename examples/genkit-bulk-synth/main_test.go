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

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/examples/dgemgenkit"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	systemonex "github.com/firebase/genkit/go/plugins/systemone/exp"
)

func newMockBulkDiffGemmaServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		var userBytes []byte
		if len(req.Messages) > 0 {
			userBytes, _ = json.Marshal(req.Messages[len(req.Messages)-1].Content)
		}
		prompt := string(userBytes)

		w.Header().Set("Content-Type", "application/json")

		var spamLabel, spamYes, kindChoice string
		switch {
		case strings.Contains(prompt, "Update README.md"):
			spamLabel, spamYes, kindChoice = "yes", "0.98", "low_effort_unfilled"
		case strings.Contains(prompt, "spam-airdrop.example.org"):
			spamLabel, spamYes, kindChoice = "yes", "0.99", "promotional_or_scam"
		case strings.Contains(prompt, "SYSTEM OVERRIDE"):
			spamLabel, spamYes, kindChoice = "yes", "0.99", "malicious_or_abuse"
		case strings.Contains(prompt, "bump otel"):
			spamLabel, spamYes, kindChoice = "no", "0.01", "docs_or_code_change"
		case strings.Contains(prompt, "Add support for systemonex"):
			spamLabel, spamYes, kindChoice = "no", "0.02", "feature_request"
		default:
			spamLabel, spamYes, kindChoice = "no", "0.01", "genuine_bug"
		}

		content := `{
			"answers": {
				"spam": {"type": "noul", "label": "` + spamLabel + `", "confidence": 0.99, "noul": ` + spamYes + `, "probabilities": {"yes": ` + spamYes + `, "no": 0.01}},
				"kind": {
					"type": "choice",
					"label": "A",
					"choice": "` + kindChoice + `",
					"confidence": 0.97,
					"probabilities": {
						"` + kindChoice + `": 0.97,
						"question": 0.03
					}
				}
			}
		}`
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":    "chatcmpl-bulk-mock",
			"model": "diffgemma-26b-a4b-it-q4",
			"choices": []map[string]any{
				{
					"index":         0,
					"finish_reason": "stop",
					"message": map[string]any{
						"role":    "assistant",
						"content": content,
					},
				},
			},
		})
	}))
}

func TestSynthesizeWithGeminiAndBulkDecide(t *testing.T) {
	mockUpstream := newMockBulkDiffGemmaServer(t)
	defer mockUpstream.Close()

	adapterSrv := dgemgenkit.StartInProcessSystemOneAdapter(mockUpstream.URL+"/v1", "diffgemma-26b-a4b-it-q4")
	defer adapterSrv.Close()

	ctx := context.Background()
	dgemPlugin := &systemonex.SystemOne{
		Provider: "dgem",
		BaseURL:  adapterSrv.URL,
	}
	g := genkit.Init(ctx, genkit.WithPlugins(dgemPlugin))

	// Register a mock "vertexai/gemini-3.8-flash" synthesis model in Genkit to test
	// Stage 1 (genkit.GenerateData[[]SyntheticGitHubItem]) + Stage 2 (concurrent dgem bulk evaluation).
	synthModelName := "vertexai/gemini-3.8-flash"
	genkit.DefineModel(g, synthModelName, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Constrained: ai.ConstrainedSupportAll},
	}, func(ctx context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		b, _ := json.Marshal(curatedGitHubItems()[:4])
		return &ai.ModelResponse{
			FinishReason: ai.FinishReasonStop,
			Message:      ai.NewModelTextMessage(string(b)),
			Usage:        &ai.GenerationUsage{InputTokens: 120, OutputTokens: 340, TotalTokens: 460},
		}, nil
	})

	summary, err := RunSynthesizeAndBulkDecide(ctx, g, BulkConfig{
		Preset:       "github_spam_triage",
		TemplatesDir: filepath.Join("..", "..", "templates"),
		SynthModel:   synthModelName,
		DGemModel:    "dgem/diffgemma-26b-a4b-it-q4",
		Count:        4,
		Workers:      4,
		CuratedOnly:  false,
	})
	if err != nil {
		t.Fatalf("RunSynthesizeAndBulkDecide failed: %v", err)
	}
	if summary.SynthSource != synthModelName {
		t.Errorf("SynthSource = %q, want %q", summary.SynthSource, synthModelName)
	}
	if summary.SynthTokens != 460 {
		t.Errorf("SynthTokens = %d, want 460", summary.SynthTokens)
	}
	if summary.TotalRows != 4 {
		t.Fatalf("TotalRows = %d, want 4", summary.TotalRows)
	}
	if summary.ExactMatchCount != 4 {
		t.Errorf("ExactMatchCount = %d, want 4", summary.ExactMatchCount)
	}
	if summary.ClearCount != 4 {
		t.Errorf("ClearCount = %d, want 4", summary.ClearCount)
	}
}
