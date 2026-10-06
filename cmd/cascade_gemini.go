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

package cmd

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"google.golang.org/genai"
)

const (
	DefaultCascadeGeminiModel  = "gemini-3.8-flash"
	DefaultCascadeGeminiModels = "gemini-3.8-flash,gemini-3.5-flash-lite"
)

// CascadeSlotTelemetry records Stage 1 (dgemma) vs Stage 2 (Gemini 3.x) per-slot resolution.
type CascadeSlotTelemetry struct {
	QuestionID       string  `json:"question_id"`
	Escalated        bool    `json:"escalated"`
	Reason           string  `json:"reason,omitempty"`
	Stage1Value      string  `json:"stage1_value"`
	Stage1Confidence float64 `json:"stage1_confidence"`
	Stage1Entropy    float64 `json:"stage1_entropy"`
	Stage2Value      string  `json:"stage2_value,omitempty"`
	Stage2Confidence float64 `json:"stage2_confidence,omitempty"`
	Explanation      string  `json:"explanation,omitempty"`
}

// CascadeExecutionSummary summarizes Stage 2 Gemini escalation for a decision call.
type CascadeExecutionSummary struct {
	Mode           string                          `json:"mode"`
	Model          string                          `json:"model"`
	Threshold      float64                         `json:"threshold"`
	Triggered      bool                            `json:"triggered"`
	EscalatedCount int                             `json:"escalated_count"`
	TotalSlots     int                             `json:"total_slots"`
	LatencyMs      int64                           `json:"latency_ms"`
	Error          string                          `json:"error,omitempty"`
	Slots          map[string]CascadeSlotTelemetry `json:"slots,omitempty"`
	// ImageCount is how many images were sent to Stage 2 with the escalated slots.
	ImageCount int `json:"image_count,omitempty"`
	// Prior is how Stage 1's answer was shown to Stage 2: "full", "soft" or "none" (see Stage2PriorModes).
	Prior string `json:"prior,omitempty"`
}

// Stage2PriorModes control how much of Stage 1's answer the Stage-2 prompt shows (issue #75):
//   - full: Stage 1's answer, confidence, entropy and full distribution, presented as "the candidate";
//   - soft: only that a fast first stage leaned toward an answer and may be wrong;
//   - none: the question alone (allowed values are still enforced by the response schema).
var Stage2PriorModes = []string{"full", "soft", "none"}

type stage2PriorKey struct{}

// WithStage2Prior sets the Stage-2 prior mode for cascades run with this context.
func WithStage2Prior(ctx context.Context, mode string) context.Context {
	return context.WithValue(ctx, stage2PriorKey{}, mode)
}

// NormalizeStage2Prior validates a prior mode; "" means the default (DGEM_CASCADE_PRIOR, else "soft": on image
// cascades at 0.10 nats, soft scored +2.6 points over full and was never worse on text, issue #75).
func NormalizeStage2Prior(mode string) (string, error) {
	m := strings.ToLower(strings.TrimSpace(mode))
	if m == "" {
		m = strings.ToLower(strings.TrimSpace(os.Getenv("DGEM_CASCADE_PRIOR")))
	}
	if m == "" {
		return "soft", nil
	}
	for _, v := range Stage2PriorModes {
		if m == v {
			return m, nil
		}
	}
	return "", fmt.Errorf("stage2_prior must be full, soft or none (got %q)", mode)
}

func stage2PriorFrom(ctx context.Context) string {
	m, _ := ctx.Value(stage2PriorKey{}).(string)
	if n, err := NormalizeStage2Prior(m); err == nil {
		return n
	}
	return "soft"
}

var (
	cascadeGenaiMu     sync.Mutex
	cascadeGenaiClient *genai.Client
	cascadeGenaiProj   string
)

// SanitizeCascadeModel keeps Stage 2 on generally available Gemini models (default gemini-3.8-flash).
// Older or pre-GA aliases are remapped: gemini-3.5-flash and gemini-3.7-flash -> gemini-3.8-flash,
// gemini-3.1-flash-lite -> gemini-3.5-flash-lite, and any gemini-1.x / gemini-2.x -> gemini-3.8-flash.
func SanitizeCascadeModel(raw string) string {
	m := strings.TrimSpace(raw)
	if m == "" {
		if envM := strings.TrimSpace(os.Getenv("DGEM_CASCADE_MODEL")); envM != "" {
			m = envM
		} else if serveCascadeModel != "" {
			m = serveCascadeModel
		} else {
			m = DefaultCascadeGeminiModel
		}
	}
	lower := strings.ToLower(m)
	switch lower {
	case "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3-flash-preview":
		return DefaultCascadeGeminiModel
	case "gemini-3.1-flash-lite":
		return "gemini-3.5-flash-lite"
	}
	if strings.HasPrefix(lower, "gemini-2") || strings.HasPrefix(lower, "gemini-1") {
		return DefaultCascadeGeminiModel
	}
	return m
}

// GetConfiguredCascadeModels returns the ordered list of selectable Stage-2 Gemini 3.x models
// from DGEM_CASCADE_MODELS or --cascade-models (defaulting to gemini-3.8-flash, gemini-3.5-flash-lite).
func GetConfiguredCascadeModels() []string {
	raw := strings.TrimSpace(os.Getenv("DGEM_CASCADE_MODELS"))
	if raw == "" {
		raw = strings.TrimSpace(serveCascadeModels)
	}
	if raw == "" {
		raw = DefaultCascadeGeminiModels
	}
	parts := strings.Split(raw, ",")
	seen := make(map[string]bool)
	var out []string
	for _, p := range parts {
		m := SanitizeCascadeModel(p)
		if m != "" && !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	if len(out) == 0 {
		return []string{"gemini-3.8-flash", "gemini-3.5-flash-lite"}
	}
	return out
}

func getSharedGenaiClient(ctx context.Context) (*genai.Client, string, error) {
	cascadeGenaiMu.Lock()
	defer cascadeGenaiMu.Unlock()

	proj := resolveGCPProject(serveVertexProject)
	if proj == "" {
		proj = detectGCPProjectID()
	}
	if proj == "" {
		return nil, "", fmt.Errorf("gemini cascade requires GCP project (set GCP_PROJECT or DGEM_GCP_PROJECT)")
	}
	if cascadeGenaiClient != nil && cascadeGenaiProj == proj {
		return cascadeGenaiClient, proj, nil
	}
	c, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  proj,
		Location: "global",
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		return nil, proj, err
	}
	cascadeGenaiClient = c
	cascadeGenaiProj = proj
	return c, proj, nil
}

func extractAnswerString(qa client.QuestionAnswer) string {
	if dv := strings.TrimSpace(qa.DisplayValue()); dv != "" {
		return normalizeDecisionToken(dv)
	}
	if strings.TrimSpace(qa.Choice) != "" {
		return normalizeDecisionToken(qa.Choice)
	}
	if strings.TrimSpace(qa.Level) != "" {
		return normalizeDecisionToken(qa.Level)
	}
	if strings.TrimSpace(qa.Label) != "" {
		return normalizeDecisionToken(qa.Label)
	}
	if qa.Type == "noul" || qa.Type == "boolean" || qa.Type == "bool" {
		if qa.Noul >= 0.5 {
			return "yes"
		}
		return "no"
	}
	return ""
}

func normalizeDecisionToken(val string) string {
	s := strings.TrimSpace(val)
	low := strings.ToLower(s)
	if low == "true" {
		return "yes"
	}
	if low == "false" {
		return "no"
	}
	return s
}

func decisionValuesMatch(predicted, expected string) bool {
	p := strings.ToLower(normalizeDecisionToken(predicted))
	e := strings.ToLower(normalizeDecisionToken(expected))
	if e == "" || e == "—" || e == "-" {
		return true
	}
	return p != "" && p == e
}

func computeSlotEntropyNats(qID string, qa client.QuestionAnswer, resp *client.StructuredDecisionResponse) float64 {
	if qa.Entropy > 0 {
		return qa.Entropy
	}
	if resp != nil {
		if qd, ok := resp.Diagnostics.Questions[qID]; ok && qd.Entropy > 0 {
			return qd.Entropy
		}
	}
	if len(qa.Probabilities) > 0 {
		var h float64
		for _, p := range qa.Probabilities {
			if p > 1e-12 {
				h -= p * math.Log(p)
			}
		}
		return h
	}
	return 0
}

// ExecuteStage2GeminiCascade evaluates whether any slot in resp.Answers requires Stage 2 escalation
// under cascadeMode ("entropy" or "on_miss") and forwards escalated slots to Vertex AI Gemini 3.x
// (defaulting to gemini-3.8-flash) with Stage 1 DiffusionGemma restricted-softmax priors.
func ExecuteStage2GeminiCascade(
	ctx context.Context,
	cascadeMode string,
	cascadeThreshold float64,
	requestedModel string,
	expectedAnswers map[string]string,
	schemaContent string,
	stateContent string,
	resp *client.StructuredDecisionResponse,
) *CascadeExecutionSummary {
	return ExecuteStage2GeminiCascadeWithImages(ctx, cascadeMode, cascadeThreshold, requestedModel, expectedAnswers,
		schemaContent, stateContent, resp, nil)
}

// ExecuteStage2GeminiCascadeWithImages is ExecuteStage2GeminiCascade for multimodal decisions: the images Stage 1
// saw (local paths, data: URIs or http(s) URLs) are sent to Stage 2 too, so Gemini resolves image questions from
// the image rather than from Stage 1's distribution alone.
func ExecuteStage2GeminiCascadeWithImages(
	ctx context.Context,
	cascadeMode string,
	cascadeThreshold float64,
	requestedModel string,
	expectedAnswers map[string]string,
	schemaContent string,
	stateContent string,
	resp *client.StructuredDecisionResponse,
	images []string,
) *CascadeExecutionSummary {
	mode := strings.ToLower(strings.TrimSpace(cascadeMode))
	if mode == "" || mode == "off" || mode == "none" || mode == "false" || resp == nil {
		return nil
	}
	if cascadeThreshold <= 0 {
		cascadeThreshold = 0.35
	}
	model := SanitizeCascadeModel(requestedModel)

	// Parse schemaContent to discover question definitions (type, options, levels, instructions)
	var schemaMap map[string]interface{}
	_ = json.Unmarshal([]byte(schemaContent), &schemaMap)
	globalInstr, _ := schemaMap["instructions"].(string)

	qDefs := make(map[string]map[string]interface{})
	if qs, ok := schemaMap["questions"].([]interface{}); ok {
		for _, qItem := range qs {
			if qm, ok := qItem.(map[string]interface{}); ok {
				if id, ok := qm["id"].(string); ok && id != "" {
					qDefs[id] = qm
				}
			}
		}
	}

	summary := &CascadeExecutionSummary{
		Mode:       mode,
		Model:      model,
		Threshold:  cascadeThreshold,
		TotalSlots: len(resp.Answers),
		Slots:      make(map[string]CascadeSlotTelemetry, len(resp.Answers)),
	}

	var escalatedIDs []string
	for qID, qa := range resp.Answers {
		stage1Val := extractAnswerString(qa)
		ent := computeSlotEntropyNats(qID, qa, resp)
		conf := qa.Confidence
		if conf <= 0 && (qa.Type == "noul" || qa.Type == "boolean") {
			if qa.Noul >= 0.5 {
				conf = qa.Noul
			} else {
				conf = 1.0 - qa.Noul
			}
		}

		shouldEscalate := false
		reason := ""

		switch mode {
		case "on_miss", "miss":
			expVal := ""
			if expectedAnswers != nil {
				expVal = strings.TrimSpace(expectedAnswers[qID])
			}
			if expVal != "" && expVal != "—" {
				if !decisionValuesMatch(stage1Val, expVal) {
					shouldEscalate = true
					reason = fmt.Sprintf("stage1_miss (dgemma=%q != expected=%q)", stage1Val, expVal)
				}
			} else if ent >= cascadeThreshold {
				shouldEscalate = true
				reason = fmt.Sprintf("entropy (H=%.4f >= %.2f)", ent, cascadeThreshold)
			}
		case "entropy", "auto", "true", "prior_guided":
			if ent >= cascadeThreshold {
				shouldEscalate = true
				reason = fmt.Sprintf("entropy (H=%.4f >= %.2f)", ent, cascadeThreshold)
			}
		}

		slotTel := CascadeSlotTelemetry{
			QuestionID:       qID,
			Escalated:        shouldEscalate,
			Reason:           reason,
			Stage1Value:      stage1Val,
			Stage1Confidence: conf,
			Stage1Entropy:    ent,
		}
		summary.Slots[qID] = slotTel
		if shouldEscalate {
			escalatedIDs = append(escalatedIDs, qID)
		}
	}

	sort.Strings(escalatedIDs)
	if len(escalatedIDs) == 0 {
		return summary
	}

	summary.Triggered = true
	summary.EscalatedCount = len(escalatedIDs)

	genaiClient, _, err := getSharedGenaiClient(ctx)
	if err != nil {
		summary.Error = fmt.Sprintf("Vertex AI client init failed: %v", err)
		return summary
	}

	// Build a structured ResponseSchema and Prior-Guided Prompt for all escalated question slots
	slotProps := make(map[string]*genai.Schema, len(escalatedIDs))
	var priorLines []string
	prior := stage2PriorFrom(ctx)
	summary.Prior = prior

	for _, qID := range escalatedIDs {
		tel := summary.Slots[qID]
		qa := resp.Answers[qID]
		qDef := qDefs[qID]
		qType, _ := qDef["type"].(string)
		if qType == "" {
			qType = qa.Type
		}
		qInstr, _ := qDef["instructions"].(string)
		if qInstr == "" {
			qInstr, _ = qDef["question"].(string)
		}

		// Format restricted-softmax prior distribution from Stage 1 dgemma
		var distParts []string
		for k, p := range qa.Probabilities {
			distParts = append(distParts, fmt.Sprintf("%s: %.1f%%", k, p*100.0))
		}
		sort.Strings(distParts)
		switch prior {
		case "soft":
			priorLines = append(priorLines, fmt.Sprintf(
				"- Slot %q (%s): a fast first-stage model leaned toward %q but may be wrong; decide from the input yourself. Question: %s",
				qID, qType, tel.Stage1Value, qInstr))
		case "none":
			priorLines = append(priorLines, fmt.Sprintf("- Slot %q (%s): Question: %s", qID, qType, qInstr))
		default:
			priorLines = append(priorLines, fmt.Sprintf(
				"- Slot %q (%s): Stage-1 dgemma candidate=%q (conf=%.1f%%, H=%.4f nats, reason=%s). Allowed distribution: {%s}. Question: %s",
				qID, qType, tel.Stage1Value, tel.Stage1Confidence*100.0, tel.Stage1Entropy, tel.Reason, strings.Join(distParts, ", "), qInstr,
			))
		}

		var valSchema *genai.Schema
		switch strings.ToLower(qType) {
		case "boolean", "bool", "noul":
			valSchema = &genai.Schema{
				Type:        genai.TypeString,
				Enum:        []string{"yes", "no"},
				Description: fmt.Sprintf("Resolved boolean decision ('yes' or 'no') for slot %s: %s", qID, qInstr),
			}
		case "choice":
			var enumVals []string
			if opts, ok := qDef["options"].([]interface{}); ok {
				for _, o := range opts {
					switch ov := o.(type) {
					case string:
						enumVals = append(enumVals, ov)
					case map[string]interface{}:
						if n, ok := ov["name"].(string); ok && n != "" {
							enumVals = append(enumVals, n)
						}
					}
				}
			}
			if len(enumVals) == 0 && len(qa.Probabilities) > 0 {
				for k := range qa.Probabilities {
					enumVals = append(enumVals, k)
				}
				sort.Strings(enumVals)
			}
			valSchema = &genai.Schema{
				Type:        genai.TypeString,
				Enum:        enumVals,
				Description: fmt.Sprintf("Resolved option name for slot %s: %s", qID, qInstr),
			}
		default:
			var levelVals []string
			if lvls, ok := qDef["levels"].([]interface{}); ok {
				for _, l := range lvls {
					levelVals = append(levelVals, fmt.Sprintf("%v", l))
				}
			}
			valSchema = &genai.Schema{
				Type:        genai.TypeString,
				Enum:        levelVals,
				Description: fmt.Sprintf("Resolved score/level for slot %s: %s", qID, qInstr),
			}
		}

		slotProps[qID] = &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"decision":    valSchema,
				"confidence":  {Type: genai.TypeNumber, Description: "Calibrated confidence in [0.0, 1.0]."},
				"explanation": {Type: genai.TypeString, Description: "Concise 1-sentence justification."},
			},
			Required: []string{"decision", "confidence", "explanation"},
		}
	}

	respSchema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: slotProps,
		Required:   escalatedIDs,
	}

	slotHeader := "[TIER-1 DISCRETE DIFFUSION PRIOR TELEMETRY]"
	if prior != "full" {
		slotHeader = "[QUESTIONS TO RESOLVE]"
	}
	prompt := fmt.Sprintf(
		"You are Stage-2 (%s) in a 2-Stage DiffusionGemma -> Gemini Decision Cascade.\n"+
			"Global Policy Instructions:\n%s\n\n"+
			"Input State / Context:\n%s\n\n"+
			"%s\n%s\n\n"+
			"Carefully verify the input state against the policy instructions and resolve each escalated slot.",
		model, globalInstr, stateContent, slotHeader, strings.Join(priorLines, "\n"),
	)

	imgParts, imgErr := cascadeImageParts(ctx, images)
	if imgErr != nil {
		summary.Error = fmt.Sprintf("Stage-2 image: %v", imgErr)
		return summary
	}
	summary.ImageCount = len(imgParts)
	parts := append(imgParts, genai.NewPartFromText(prompt))
	contents := []*genai.Content{{Role: "user", Parts: parts}}
	callTimeout := 25 * time.Second
	if len(imgParts) > 0 {
		callTimeout = 90 * time.Second // image reasoning has a long latency tail
	}

	temp := float32(0.0)
	cfg := &genai.GenerateContentConfig{
		Temperature:      &temp,
		ResponseMIMEType: "application/json",
		ResponseSchema:   respSchema,
	}

	t0 := time.Now()
	candidateModels := []string{model}
	// Fall back to the other GA Stage-2 model if the requested one is not available in the project/region.
	for _, fallback := range []string{"gemini-3.8-flash", "gemini-3.5-flash-lite"} {
		if fallback != model {
			candidateModels = append(candidateModels, fallback)
		}
	}

	var genResp *genai.GenerateContentResponse
	var genErr error
	for _, cand := range candidateModels {
		callCtx, cancel := context.WithTimeout(ctx, callTimeout)
		genResp, genErr = genaiClient.Models.GenerateContent(callCtx, cand, contents, cfg)
		cancel()
		if genErr == nil && genResp != nil {
			break
		}
	}
	summary.LatencyMs = time.Since(t0).Milliseconds()
	if genErr != nil {
		summary.Error = fmt.Sprintf("Stage-2 Gemini (%s) call failed: %v", model, genErr)
		return summary
	}

	rawJSON := strings.TrimSpace(genResp.Text())
	var parsed map[string]struct {
		Decision    string  `json:"decision"`
		Confidence  float64 `json:"confidence"`
		Explanation string  `json:"explanation"`
	}
	if err := json.Unmarshal([]byte(rawJSON), &parsed); err != nil {
		summary.Error = fmt.Sprintf("Stage-2 Gemini invalid JSON: %v", err)
		return summary
	}

	for _, qID := range escalatedIDs {
		resSlot, ok := parsed[qID]
		if !ok || strings.TrimSpace(resSlot.Decision) == "" {
			continue
		}
		decisionVal := normalizeDecisionToken(resSlot.Decision)
		conf := resSlot.Confidence
		if conf <= 0 || conf > 1.0 {
			conf = 0.95
		}

		// Update the CascadeSlotTelemetry record
		tel := summary.Slots[qID]
		tel.Stage2Value = decisionVal
		tel.Stage2Confidence = conf
		tel.Explanation = resSlot.Explanation
		summary.Slots[qID] = tel

		// Update resp.Answers[qID] so the caller receives the Stage-2 resolved decision
		qa := resp.Answers[qID]
		qa.Confidence = conf
		switch strings.ToLower(qa.Type) {
		case "boolean", "bool", "noul":
			isYes := decisionVal == "yes" || decisionVal == "true"
			qa.Label = decisionVal
			qa.Choice = decisionVal
			if isYes {
				qa.Noul = conf
			} else {
				qa.Noul = 1.0 - conf
			}
		case "choice":
			qa.Choice = decisionVal
			qa.Label = decisionVal
		default:
			qa.Level = decisionVal
			qa.Label = decisionVal
			if f, err := strconv.ParseFloat(decisionVal, 64); err == nil {
				qa.Score = f
			}
		}
		resp.Answers[qID] = qa
	}

	return summary
}

// generateContentWithRetry retries transient Vertex AI errors (HTTP 429/500/503) up to 3 attempts with
// backoff so benchmark receipts do not score infrastructure failures as wrong answers.
func generateContentWithRetry(ctx context.Context, c *genai.Client, model, prompt string, cfg *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		resp, err := c.Models.GenerateContent(ctx, model, genai.Text(prompt), cfg)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		msg := err.Error()
		if !strings.Contains(msg, "500") && !strings.Contains(msg, "503") && !strings.Contains(msg, "429") && !strings.Contains(msg, "INTERNAL") && !strings.Contains(msg, "UNAVAILABLE") {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Duration(2*(attempt+1)) * time.Second):
		}
	}
	return nil, lastErr
}

// cascadeImageParts turns image references (local path, data: URI, http(s) URL) into inline Gemini parts.
func cascadeImageParts(ctx context.Context, images []string) ([]*genai.Part, error) {
	var parts []*genai.Part
	for _, ref := range images {
		ref = strings.TrimSpace(ref)
		if ref == "" {
			continue
		}
		var data []byte
		mime := ""
		switch {
		case strings.HasPrefix(ref, "data:"):
			comma := strings.Index(ref, ",")
			if comma < 0 {
				return nil, fmt.Errorf("malformed data URI")
			}
			meta := ref[5:comma]
			mime = strings.SplitN(meta, ";", 2)[0]
			raw := ref[comma+1:]
			if strings.Contains(meta, ";base64") {
				b, err := base64.StdEncoding.DecodeString(raw)
				if err != nil {
					return nil, fmt.Errorf("data URI: %w", err)
				}
				data = b
			} else {
				data = []byte(raw)
			}
		case strings.HasPrefix(ref, "http://") || strings.HasPrefix(ref, "https://"):
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, ref, nil)
			if err != nil {
				return nil, err
			}
			hr, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
			if err != nil {
				return nil, err
			}
			data, err = io.ReadAll(io.LimitReader(hr.Body, 20<<20))
			hr.Body.Close()
			if err != nil {
				return nil, err
			}
			mime = strings.SplitN(hr.Header.Get("Content-Type"), ";", 2)[0]
		default:
			b, err := os.ReadFile(ref)
			if err != nil {
				return nil, err
			}
			data = b
		}
		if mime == "" || !strings.HasPrefix(mime, "image/") {
			mime = http.DetectContentType(data)
		}
		parts = append(parts, genai.NewPartFromBytes(data, mime))
	}
	return parts, nil
}
