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

// Package dgemgenkit bridges dgem .json.tmpl decision policies to Genkit Go
// v1.14.0's systemonex (System One) plugin, computing per-slot Shannon entropy
// H, normalized Hesitation % (H / ln K), and request wall-time telemetry.
package dgemgenkit

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	systemonex "github.com/firebase/genkit/go/plugins/systemone/exp"
)

// TemplateDecision holds the Genkit systemonex answers for a rendered .json.tmpl
// policy along with per-slot Shannon entropy H (in nats), normalized
// Hesitation (H / ln K in [0, 1]), and wall-time latency.
type TemplateDecision struct {
	TemplatePath string
	Model        string
	Answers      map[string]systemonex.Answer
	OptionCount  map[string]int
	WallTime     time.Duration
	WallTimeMs   int64
}

// EntropyNats returns the Shannon entropy H = -sum p_k ln(p_k) in nats for slot id.
func (d *TemplateDecision) EntropyNats(id string) float64 {
	ans, ok := d.Answers[id]
	if !ok {
		return 0
	}
	if ans.Type == "noul" {
		p := ans.Probability
		if p <= 0 || p >= 1 {
			return 0
		}
		return -(p*math.Log(p) + (1-p)*math.Log(1-p))
	}
	var h float64
	for _, p := range ans.Probabilities {
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	return h
}

// Hesitation returns normalized entropy H / ln(K) in [0, 1] for slot id
// (< 0.16 is Clear, 0.16-0.50 is Somewhat unsure, > 0.50 is Very unsure).
func (d *TemplateDecision) Hesitation(id string) float64 {
	k := d.OptionCount[id]
	if k < 2 {
		k = 2
	}
	h := d.EntropyNats(id)
	maxH := math.Log(float64(k))
	if maxH <= 0 {
		return 0
	}
	return math.Min(1.0, h/maxH)
}

// MaxHesitation returns the highest normalized hesitation across all slots.
func (d *TemplateDecision) MaxHesitation() float64 {
	var maxH float64
	for id := range d.Answers {
		if h := d.Hesitation(id); h > maxH {
			maxH = h
		}
	}
	return maxH
}

// ResolveTemplatesDir finds the dgem templates/ directory whether the example
// is invoked from examples/<name>, examples/, or the repository root.
func ResolveTemplatesDir(hint string) string {
	for _, candidate := range []string{hint, "../../templates", "../templates", "./templates"} {
		if candidate == "" {
			continue
		}
		if st, err := os.Stat(candidate); err == nil && st.IsDir() {
			return candidate
		}
	}
	return hint
}

// StartInProcessSystemOneAdapter starts an in-process HTTP server exposing
// POST /v1/systemone backed by an upstream dgem / diffgemma /v1 endpoint
// (e.g. local Apple Silicon Metal on http://127.0.0.1:8080/v1).
func StartInProcessSystemOneAdapter(upstreamV1URL, modelID string) *httptest.Server {
	cli := client.NewClient(upstreamV1URL, modelID, 2*time.Minute)
	opts := decisionindex.DefaultEngineOptions()
	mux := http.NewServeMux()
	mux.Handle("/v1/systemone", decisionindex.NewSystemOneHTTPHandler(cli, opts))
	return httptest.NewServer(mux)
}

type parsedTemplateSchema struct {
	Instructions string                   `json:"instructions"`
	Questions    []parsedTemplateQuestion `json:"questions"`
}

type parsedTemplateQuestion struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Instructions string          `json:"instructions"`
	Options      json.RawMessage `json:"options,omitempty"`
	Levels       []string        `json:"levels,omitempty"`
}

type parsedChoiceOption struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// DecideWithTemplate renders a dgem .json.tmpl file with vars, converts its
// questions into a Genkit systemonex.Schema, and executes the decision in a
// single pass via Genkit's systemonex plugin, recording wall-time latency.
func DecideWithTemplate(
	ctx context.Context,
	g *genkit.Genkit,
	modelName string,
	tmplPath string,
	vars map[string]any,
) (*TemplateDecision, error) {
	start := time.Now()
	eng := template.NewEngine()
	rendered, err := eng.RenderFile(tmplPath, vars)
	if err != nil {
		return nil, fmt.Errorf("render template %s: %w", tmplPath, err)
	}
	schemaStr, stateStr, err := template.ParseStructuredPayload(rendered, vars)
	if err != nil {
		return nil, fmt.Errorf("parse template payload %s: %w", tmplPath, err)
	}

	var parsed parsedTemplateSchema
	if err := json.Unmarshal([]byte(schemaStr), &parsed); err != nil {
		return nil, fmt.Errorf("unmarshal template schema %s: %w", tmplPath, err)
	}

	questions := make(map[string]systemonex.Question, len(parsed.Questions))
	optionCount := make(map[string]int, len(parsed.Questions))

	for _, q := range parsed.Questions {
		switch q.Type {
		case "boolean", "bool", "noul":
			questions[q.ID] = systemonex.NoulQuestion{
				Instructions: q.Instructions,
			}
			optionCount[q.ID] = 2

		case "choice":
			opts, err := parseChoiceOptions(q.Options)
			if err != nil {
				return nil, fmt.Errorf("question %q options: %w", q.ID, err)
			}
			questions[q.ID] = systemonex.ChoiceQuestion{
				Instructions: q.Instructions,
				Options:      opts,
			}
			optionCount[q.ID] = len(opts)

		case "score":
			questions[q.ID] = systemonex.ScoreQuestion{
				Instructions: q.Instructions,
				Levels:       q.Levels,
			}
			optionCount[q.ID] = len(q.Levels)

		default:
			return nil, fmt.Errorf("unsupported question type %q on slot %q", q.Type, q.ID)
		}
	}

	var stateObj any
	if err := json.Unmarshal([]byte(stateStr), &stateObj); err != nil {
		stateObj = stateStr
	}

	opts := []ai.GenerateOption{
		ai.WithModelName(modelName),
		ai.WithOutputSchema(systemonex.Schema(questions)),
		ai.WithPromptParts(ai.NewDataPart(stateObj)),
	}
	if parsed.Instructions != "" {
		opts = append(opts, ai.WithSystem(parsed.Instructions))
	}

	answers, resp, err := genkit.GenerateData[map[string]systemonex.Answer](ctx, g, opts...)
	if err != nil {
		return nil, fmt.Errorf("genkit GenerateData (%s): %w", tmplPath, err)
	}
	elapsed := time.Since(start)

	return &TemplateDecision{
		TemplatePath: tmplPath,
		Model:        systemonex.ResponseInfo(resp).Model,
		Answers:      *answers,
		OptionCount:  optionCount,
		WallTime:     elapsed,
		WallTimeMs:   elapsed.Milliseconds(),
	}, nil
}

func parseChoiceOptions(raw json.RawMessage) ([]systemonex.ChoiceOption, error) {
	var objList []parsedChoiceOption
	if err := json.Unmarshal(raw, &objList); err == nil && len(objList) > 0 && objList[0].Name != "" {
		out := make([]systemonex.ChoiceOption, len(objList))
		for i, o := range objList {
			out[i] = systemonex.ChoiceOption{Name: o.Name, Criteria: o.Description}
		}
		return out, nil
	}
	var strList []string
	if err := json.Unmarshal(raw, &strList); err == nil {
		out := make([]systemonex.ChoiceOption, len(strList))
		for i, s := range strList {
			out[i] = systemonex.ChoiceOption{Name: s, Criteria: s}
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot decode choice options: %s", string(raw))
}
