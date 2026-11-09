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
		var allBytes, userBytes []byte
		allBytes, _ = json.Marshal(req.Messages)
		if len(req.Messages) > 0 {
			userBytes, _ = json.Marshal(req.Messages[len(req.Messages)-1].Content)
		}
		schemaHint := string(allBytes)
		userState := string(userBytes)
		if before, _, hasInstr := strings.Cut(userState, "<instructions>"); hasInstr {
			userState = before
		}

		w.Header().Set("Content-Type", "application/json")

		var content string
		switch {
		case strings.Contains(schemaHint, "data_compromise"):
			content = mockSecOpsIncidentResponse(userState)
		case strings.Contains(schemaHint, "prompt_adversarial"):
			content = mockRAIRedTeamResponse(userState)
		default:
			content = mockGitHubSpamResponse(userState)
		}

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

func mockGitHubSpamResponse(prompt string) string {
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
	return `{
		"answers": {
			"spam": {"type": "noul", "label": "` + spamLabel + `", "confidence": 0.99, "noul": ` + spamYes + `, "probabilities": {"yes": ` + spamYes + `, "no": 0.01}},
			"kind": {
				"type": "choice",
				"label": "A",
				"choice": "` + kindChoice + `",
				"confidence": 0.97,
				"probabilities": {"` + kindChoice + `": 0.97, "question": 0.03}
			}
		}
	}`
}

func mockSecOpsIncidentResponse(prompt string) string {
	var compLabel, compYes, action, sevLevel, sevScore, sevProbs string
	switch {
	case strings.Contains(prompt, "drop-exfil-9921"):
		compLabel, compYes, action, sevLevel, sevScore = "yes", "0.99", "page_oncall", "p0_critical", "3.0"
		sevProbs = `{"p3_low": 0.0, "p2_medium": 0.0, "p1_high": 0.01, "p0_critical": 0.99}`
	case strings.Contains(prompt, ".xmrig"):
		compLabel, compYes, action, sevLevel, sevScore = "no", "0.01", "isolate_host", "p1_high", "2.0"
		sevProbs = `{"p3_low": 0.0, "p2_medium": 0.02, "p1_high": 0.97, "p0_critical": 0.01}`
	case strings.Contains(prompt, "key_id=8f91a"):
		compLabel, compYes, action, sevLevel, sevScore = "no", "0.02", "revoke_key", "p2_medium", "1.0"
		sevProbs = `{"p3_low": 0.01, "p2_medium": 0.97, "p1_high": 0.02, "p0_critical": 0.0}`
	default:
		compLabel, compYes, action, sevLevel, sevScore = "no", "0.01", "monitor", "p3_low", "0.0"
		sevProbs = `{"p3_low": 0.98, "p2_medium": 0.02, "p1_high": 0.0, "p0_critical": 0.0}`
	}
	return `{
		"answers": {
			"data_compromise": {"type": "noul", "label": "` + compLabel + `", "confidence": 0.99, "noul": ` + compYes + `, "probabilities": {"yes": ` + compYes + `, "no": 0.01}},
			"action": {"type": "choice", "label": "A", "choice": "` + action + `", "confidence": 0.98, "probabilities": {"` + action + `": 0.98, "monitor": 0.02}},
			"severity": {"type": "score", "label": "` + sevLevel + `", "level": "` + sevLevel + `", "score": ` + sevScore + `, "confidence": 0.98, "probabilities": ` + sevProbs + `}
		}
	}`
}

func mockRAIRedTeamResponse(prompt string) string {
	advYes, harmYes, refYes, cat := "0.01", "0.01", "0.01", "none"
	switch {
	case strings.Contains(prompt, "hospital payroll"):
		advYes, harmYes, refYes, cat = "0.99", "0.01", "0.99", "fraud_or_illegal_acts"
	case strings.Contains(prompt, "ransom note"):
		advYes, harmYes, refYes, cat = "0.99", "0.01", "0.99", "cyberattack_or_malware"
	case strings.Contains(prompt, "attacker.example/steal"):
		advYes, harmYes, refYes, cat = "0.98", "0.98", "0.01", "cyberattack_or_malware"
	}
	return `{
		"answers": {
			"prompt_adversarial": {"type": "noul", "confidence": 0.99, "noul": ` + advYes + `, "probabilities": {"yes": ` + advYes + `, "no": 0.01}},
			"response_harmful": {"type": "noul", "confidence": 0.99, "noul": ` + harmYes + `, "probabilities": {"yes": ` + harmYes + `, "no": 0.01}},
			"response_refusal": {"type": "noul", "confidence": 0.99, "noul": ` + refYes + `, "probabilities": {"yes": ` + refYes + `, "no": 0.01}},
			"harm_category": {"type": "choice", "label": "A", "choice": "` + cat + `", "confidence": 0.98, "probabilities": {"` + cat + `": 0.98, "other": 0.02}}
		}
	}`
}

func TestSynthesizeWithGeminiAndBulkDecide(t *testing.T) {
	mockUpstream := newMockBulkDiffGemmaServer(t)
	defer mockUpstream.Close()

	adapterSrv := dgemgenkit.StartInProcessSystemOneAdapter(mockUpstream.URL+"/v1", "diffgemma-26b-a4b-it-q4")
	defer adapterSrv.Close()

	ctx := t.Context()
	dgemPlugin := &systemonex.SystemOne{
		Provider: "dgem",
		BaseURL:  adapterSrv.URL,
	}
	g := genkit.Init(ctx, genkit.WithPlugins(dgemPlugin))

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

	t.Run("github_spam_triage_synthesized", func(t *testing.T) {
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
			t.Fatalf("RunSynthesizeAndBulkDecide(github_spam_triage) error: %v", err)
		}
		if summary.SynthSource != synthModelName {
			t.Errorf("SynthSource = %q, want %q", summary.SynthSource, synthModelName)
		}
		if summary.SynthTokens != 460 {
			t.Errorf("SynthTokens = %d, want 460", summary.SynthTokens)
		}
		if summary.TotalRows != 4 || summary.ExactMatchCount != 4 {
			t.Errorf("TotalRows=%d ExactMatchCount=%d, want 4 and 4", summary.TotalRows, summary.ExactMatchCount)
		}
	})

	t.Run("secops_incident_curated", func(t *testing.T) {
		summary, err := RunSynthesizeAndBulkDecide(ctx, g, BulkConfig{
			Preset:       "secops_incident",
			TemplatesDir: filepath.Join("..", "..", "templates"),
			DGemModel:    "dgem/diffgemma-26b-a4b-it-q4",
			Count:        4,
			Workers:      2,
			CuratedOnly:  true,
		})
		if err != nil {
			t.Fatalf("RunSynthesizeAndBulkDecide(secops_incident) error: %v", err)
		}
		if summary.ExactMatchCount != 4 {
			t.Errorf("RunSynthesizeAndBulkDecide(secops_incident) ExactMatchCount = %d, want 4", summary.ExactMatchCount)
		}
	})

	t.Run("rai_redteam_grader_curated", func(t *testing.T) {
		summary, err := RunSynthesizeAndBulkDecide(ctx, g, BulkConfig{
			Preset:       "rai_redteam_grader",
			TemplatesDir: filepath.Join("..", "..", "templates"),
			DGemModel:    "dgem/diffgemma-26b-a4b-it-q4",
			Count:        4,
			Workers:      2,
			CuratedOnly:  true,
		})
		if err != nil {
			t.Fatalf("RunSynthesizeAndBulkDecide(rai_redteam_grader) error: %v", err)
		}
		if summary.ExactMatchCount != 4 {
			t.Errorf("RunSynthesizeAndBulkDecide(rai_redteam_grader) ExactMatchCount = %d, want 4", summary.ExactMatchCount)
		}
	})
}
