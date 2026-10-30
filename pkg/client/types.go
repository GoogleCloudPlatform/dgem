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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ChatCompletionRequest is the OpenAI-compatible payload format.
type ChatCompletionRequest struct {
	Model       string        `json:"model,omitempty"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Logprobs    bool          `json:"logprobs,omitempty"`
	TopLogprobs int           `json:"top_logprobs,omitempty"`
}

// ChatMessage represents a single chat turn.
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// ContentPart represents an element in a multimodal content array.
type ContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *ImageURLPart `json:"image_url,omitempty"`
}

// ImageURLPart holds an image URL or base64 data URI.
type ImageURLPart struct {
	URL string `json:"url"`
}

// ChatCompletionResponse is the standard OpenAI-compatible response.
type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`
}

// ChatChoice contains the assistant message and optional token logprobs.
type ChatChoice struct {
	Index        int             `json:"index"`
	Message      ChatMessage     `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs `json:"logprobs,omitempty"`
}

// ChoiceLogprobs holds OpenAI/vLLM token-level log-probabilities.
type ChoiceLogprobs struct {
	Content []TokenLogprob `json:"content"`
}

// TokenLogprob represents the log-probability and top-k alternatives for a single output token.
type TokenLogprob struct {
	Token       string           `json:"token"`
	Logprob     float64          `json:"logprob"`
	TopLogprobs []TopLogprobItem `json:"top_logprobs,omitempty"`
}

// TopLogprobItem represents a single candidate token and its logprob at a given position.
type TopLogprobItem struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// Usage reports token statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ProposedOption represents a ready-to-paste {"name", "description"} choice option for a .json.tmpl policy.
type ProposedOption struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// TaxonomyExpansionSuggestion records an unclassified/high-entropy slot detection and DiffusionGemma's proposed new class.
type TaxonomyExpansionSuggestion struct {
	QuestionID        string             `json:"question_id"`
	TriggerReason     string             `json:"trigger_reason"`
	Stage1Choice      string             `json:"stage1_choice"`
	Stage1Confidence  float64            `json:"stage1_confidence"`
	Stage1Entropy     float64            `json:"stage1_entropy"`
	TopCandidates     map[string]float64 `json:"top_candidates,omitempty"`
	InjectedCatchAll  bool               `json:"injected_catch_all"`
	SuggestedOption   ProposedOption     `json:"suggested_option"`
	RawThought        string             `json:"raw_thought,omitempty"`
	TemplatePatchHint string             `json:"template_patch_hint,omitempty"`
}

// StructuredDecisionResponse represents the parsed structured-decision content.
type StructuredDecisionResponse struct {
	Answers             map[string]QuestionAnswer     `json:"answers"`
	Diagnostics         Diagnostics                   `json:"diagnostics"`
	SuggestedExpansions []TaxonomyExpansionSuggestion `json:"suggested_expansions,omitempty"`
}

// QuestionAnswer holds the evaluated result for a single question.
type QuestionAnswer struct {
	Type          string             `json:"type"`
	Label         string             `json:"label"`
	Confidence    float64            `json:"confidence"`
	Logprob       float64            `json:"logprob,omitempty"`
	Entropy       float64            `json:"entropy,omitempty"`
	Stderr        float64            `json:"stderr"`
	Agreement     float64            `json:"agreement"`
	Probabilities map[string]float64 `json:"probabilities"`

	// Type-specific values:
	Choice string  `json:"choice,omitempty"` // For choice type
	Score  float64 `json:"score,omitempty"`  // For score type
	Level  string  `json:"level,omitempty"`  // For score type
	Noul   float64 `json:"noul,omitempty"`   // For boolean/noul type (probability of true/yes)
}

// DisplayType returns a human-friendly type name, replacing "noul" with "boolean".
func (qa QuestionAnswer) DisplayType() string {
	switch qa.Type {
	case "noul", "bool", "boolean":
		return "boolean"
	default:
		return qa.Type
	}
}

// DisplayValue returns the primary value (e.g. "yes", "engineering", "2 (frustrated)").
func (qa QuestionAnswer) DisplayValue() string {
	switch qa.Type {
	case "noul", "bool", "boolean":
		return qa.Label
	case "choice":
		return qa.Choice
	case "score":
		if qa.Level != "" {
			return qa.Level
		}
		return qa.Label
	default:
		return qa.Label
	}
}

// ThoughtDiagnostic captures DiffusionGemma's optional pre-canvas thought channel output ("think": N).
type ThoughtDiagnostic struct {
	Text   string  `json:"text"`
	Tokens int     `json:"tokens"`
	Ms     float64 `json:"ms"`
	Budget int     `json:"budget,omitempty"`
}

// Diagnostics contains server-side timing, sampling, thought channel, and slot entropy data.
type Diagnostics struct {
	Hole      string                        `json:"hole,omitempty"`
	Steps     int                           `json:"steps"`
	Layout    string                        `json:"layout,omitempty"` // prompt layout the server used (serving v0.2.0+)
	Thought   *ThoughtDiagnostic            `json:"thought,omitempty"`
	Timing    TimingStats                   `json:"timing"`
	Samples   SampleStats                   `json:"samples"`
	Questions map[string]QuestionDiagnostic `json:"questions"`
}

// TimingStats details execution breakdown across local Metal and vLLM remote endpoints.
type TimingStats struct {
	TotalMs      float64 `json:"total_ms"`
	Reads        int     `json:"reads"`
	DenoiseMs    float64 `json:"denoise_ms"`
	PrefillMs    float64 `json:"prefill_ms"`
	PromptTokens int     `json:"prompt_tokens"`
	ReusedTokens int     `json:"reused_tokens"`
	Rounds       int     `json:"rounds"`
	Samples      int     `json:"samples"`
	StepsRun     int     `json:"steps_run"`
}

// SampleStats details the multi-read sampling policy.
type SampleStats struct {
	N      int          `json:"n"`
	Policy SamplePolicy `json:"policy"`
}

func (ss *SampleStats) UnmarshalJSON(data []byte) error {
	type Alias SampleStats
	aux := struct {
		RawN json.RawMessage `json:"n"`
		*Alias
	}{
		Alias: (*Alias)(ss),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if len(aux.RawN) > 0 {
		var n int
		if err := json.Unmarshal(aux.RawN, &n); err == nil {
			ss.N = n
		} else {
			var nl []int
			if err := json.Unmarshal(aux.RawN, &nl); err == nil && len(nl) > 0 {
				ss.N = nl[0]
			}
		}
	}
	return nil
}

// SamplePolicy reveals whether the auto-sampling threshold was triggered.
type SamplePolicy struct {
	Extended            bool    `json:"extended"`
	FirstReadMaxEntropy float64 `json:"first_read_max_entropy"`
	Max                 int     `json:"max"`
	Mode                string  `json:"mode"`
	PinnedSlots         int     `json:"pinned_slots"`
	Threshold           float64 `json:"threshold"`
}

func (sp *SamplePolicy) UnmarshalJSON(data []byte) error {
	type Alias SamplePolicy
	aux := struct {
		RawEnt json.RawMessage `json:"first_read_entropy"`
		*Alias
	}{
		Alias: (*Alias)(sp),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if len(aux.RawEnt) > 0 && sp.FirstReadMaxEntropy == 0 {
		var f float64
		if err := json.Unmarshal(aux.RawEnt, &f); err == nil {
			sp.FirstReadMaxEntropy = f
		} else {
			var entMap map[string]float64
			if err := json.Unmarshal(aux.RawEnt, &entMap); err == nil {
				for _, v := range entMap {
					if v > sp.FirstReadMaxEntropy {
						sp.FirstReadMaxEntropy = v
					}
				}
			}
		}
	}
	return nil
}

// QuestionDiagnostic holds raw token entropy and probability mass.
type QuestionDiagnostic struct {
	ArgmaxIsLabel bool    `json:"argmax_is_label"`
	ArgmaxToken   string  `json:"argmax_token"`
	Entropy       float64 `json:"entropy"`
	LabelMass     float64 `json:"label_mass"`
}

func (qd *QuestionDiagnostic) UnmarshalJSON(data []byte) error {
	type Alias QuestionDiagnostic
	aux := struct {
		RawEntropy json.RawMessage `json:"entropy"`
		*Alias
	}{
		Alias: (*Alias)(qd),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if len(aux.RawEntropy) > 0 {
		var f float64
		if err := json.Unmarshal(aux.RawEntropy, &f); err == nil {
			qd.Entropy = f
		} else {
			var fl []float64
			if err := json.Unmarshal(aux.RawEntropy, &fl); err == nil && len(fl) > 0 {
				qd.Entropy = fl[0]
			}
		}
	}
	return nil
}

// RequestStats summarizes the full lifecycle of a query for CLI display.
type RequestStats struct {
	Model        string
	Endpoint     string
	WallTime     time.Duration
	PrefillMs    float64
	DenoiseMs    float64
	PromptTokens int
	ReusedTokens int
	OutputTokens int
	TotalTokens  int
	DenoiseSteps int
	SamplesN     int
	Extended     bool
	Retries      int
	Diagnostics  *Diagnostics
}

// ProbabilityEntropy returns the Shannon entropy H = -sum p ln p (nats) of a label distribution, renormalized so a
// distribution that doesn't sum exactly to 1 still gives H in [0, ln K].
func ProbabilityEntropy(probs map[string]float64) float64 {
	sum := 0.0
	for _, p := range probs {
		if p > 0 && !math.IsNaN(p) && !math.IsInf(p, 0) {
			sum += p
		}
	}
	if sum <= 0 {
		return 0
	}
	h := 0.0
	for _, p := range probs {
		if p > 0 && !math.IsNaN(p) && !math.IsInf(p, 0) {
			q := p / sum
			h -= q * math.Log(q)
		}
	}
	return h
}

// ParseStructuredContent attempts to unmarshal the raw assistant text as a StructuredDecisionResponse.
func ParseStructuredContent(content string) (*StructuredDecisionResponse, error) {
	return ParseStructuredContentWithLogprobs(content, nil)
}

func normalizeQuestionAnswers(s *StructuredDecisionResponse) *StructuredDecisionResponse {
	if s == nil {
		return nil
	}
	if s.Answers == nil {
		s.Answers = make(map[string]QuestionAnswer)
	}
	if s.Diagnostics.Questions == nil {
		s.Diagnostics.Questions = make(map[string]QuestionDiagnostic)
	}
	for k, qa := range s.Answers {
		if qa.Type == "noul" {
			qa.Type = "bool"
		}
		if qa.Probabilities == nil {
			qa.Probabilities = make(map[string]float64)
		}
		// The vLLM server reports label probabilities but no answer entropy (#121). Fill it from the probabilities so
		// the Studio, the cascade gate and summaries all read one value. With constrained reads the server's
		// diagnostics "entropy" is the same number; with --no-constrained it is over top-k tokens and can differ.
		if qa.Entropy == 0 && len(qa.Probabilities) > 1 {
			qa.Entropy = ProbabilityEntropy(qa.Probabilities)
		}
		s.Answers[k] = qa
		if _, ok := s.Diagnostics.Questions[k]; !ok {
			s.Diagnostics.Questions[k] = QuestionDiagnostic{
				ArgmaxIsLabel: true,
				ArgmaxToken:   qa.DisplayValue(),
				Entropy:       qa.Entropy,
				LabelMass:     qa.Confidence,
			}
		}
	}
	return s
}

// ParseStructuredContentWithLogprobs unmarshals the assistant text and enriches slot confidence,
// top-k candidate probabilities, and Shannon entropy using OpenAI/vLLM token logprobs when present.
func ParseStructuredContentWithLogprobs(content string, logprobs *ChoiceLogprobs) (*StructuredDecisionResponse, error) {
	var structured StructuredDecisionResponse
	if err := json.Unmarshal([]byte(content), &structured); err == nil && len(structured.Answers) > 0 {
		return normalizeQuestionAnswers(&structured), nil
	}

	cleaned := cleanJSON(content)
	if err := json.Unmarshal([]byte(cleaned), &structured); err == nil && len(structured.Answers) > 0 {
		return normalizeQuestionAnswers(&structured), nil
	}

	// Envelope fallback: Handles responses with {"answers": {...}, "diagnostics": {...}}
	var rawEnvelope struct {
		Answers     map[string]QuestionAnswer `json:"answers"`
		Diagnostics json.RawMessage           `json:"diagnostics"`
	}
	if err := json.Unmarshal([]byte(cleaned), &rawEnvelope); err == nil && len(rawEnvelope.Answers) > 0 {
		structured.Answers = rawEnvelope.Answers
		if len(rawEnvelope.Diagnostics) > 0 {
			_ = json.Unmarshal(rawEnvelope.Diagnostics, &structured.Diagnostics)
		}
		return normalizeQuestionAnswers(&structured), nil
	}

	// Fallback: Check if response is a direct JSON key-value map (e.g. from vLLM completions)
	var rawMap map[string]interface{}
	if err := json.Unmarshal([]byte(cleaned), &rawMap); err == nil && len(rawMap) > 0 {
		// If rawMap has an nested "answers" key that is a map, extract answers directly
		if rawAnswers, ok := rawMap["answers"].(map[string]interface{}); ok && len(rawAnswers) > 0 {
			ansBytes, _ := json.Marshal(rawAnswers)
			var ansMap map[string]QuestionAnswer
			if err := json.Unmarshal(ansBytes, &ansMap); err == nil && len(ansMap) > 0 {
				structured.Answers = ansMap
				if rawDiag, ok := rawMap["diagnostics"]; ok {
					diagBytes, _ := json.Marshal(rawDiag)
					_ = json.Unmarshal(diagBytes, &structured.Diagnostics)
				}
				return &structured, nil
			}
		}

		structured.Answers = make(map[string]QuestionAnswer)
		if structured.Diagnostics.Questions == nil {
			structured.Diagnostics.Questions = make(map[string]QuestionDiagnostic)
		}
		for k, v := range rawMap {
			label := fmt.Sprintf("%v", v)
			conf, lp, ent, probs := extractSlotLogprobTelemetry(label, logprobs)
			structured.Answers[k] = QuestionAnswer{
				Type:          "auto",
				Label:         label,
				Confidence:    conf,
				Logprob:       lp,
				Entropy:       ent,
				Probabilities: probs,
			}
			structured.Diagnostics.Questions[k] = QuestionDiagnostic{
				ArgmaxIsLabel: true,
				ArgmaxToken:   label,
				Entropy:       ent,
				LabelMass:     conf,
			}
		}
		return &structured, nil
	}

	return nil, fmt.Errorf("failed to parse structured decision output from: %s", content)
}

// extractSlotLogprobTelemetry locates all subword tokens corresponding to the slot value in ChoiceLogprobs
// and computes the bottleneck branching probability exp(min_logprob), maximum Shannon entropy H = -sum(p * ln(p)),
// and all candidate runner-up subword probabilities across the label span.
func extractSlotLogprobTelemetry(label string, lp *ChoiceLogprobs) (confidence float64, logprob float64, entropy float64, probs map[string]float64) {
	confidence = 1.0
	if lp == nil || len(lp.Content) == 0 {
		return confidence, 0, 0, nil
	}

	normLabel := strings.ToLower(strings.TrimSpace(label))
	if normLabel == "" {
		return 1.0, 0, 0, nil
	}

	// Collect all subword tokens in lp.Content that belong to the generated label value
	// (occurring after the JSON colon ':')
	colonIdx := -1
	for i := len(lp.Content) - 1; i >= 0; i-- {
		if strings.Contains(lp.Content[i].Token, ":") {
			colonIdx = i
			break
		}
	}

	startSearch := 0
	if colonIdx != -1 && colonIdx+1 < len(lp.Content) {
		startSearch = colonIdx + 1
	}

	var matchedTokens []*TokenLogprob
	for i := startSearch; i < len(lp.Content); i++ {
		tokClean := strings.ToLower(strings.Trim(lp.Content[i].Token, " \t\n\r\"',:{}[]"))
		if tokClean == "" {
			continue
		}
		if strings.Contains(normLabel, tokClean) || strings.HasPrefix(tokClean, normLabel) {
			matchedTokens = append(matchedTokens, &lp.Content[i])
		}
	}

	if len(matchedTokens) == 0 {
		for i := len(lp.Content) - 1; i >= 0; i-- {
			tokClean := strings.Trim(lp.Content[i].Token, " \t\n\r\"',:{}[]")
			if len(tokClean) > 0 {
				matchedTokens = append(matchedTokens, &lp.Content[i])
				break
			}
		}
	}
	if len(matchedTokens) == 0 {
		return 1.0, 0, 0, nil
	}

	// Across all subwords of the label, find the bottleneck branching token
	// (lowest logprob / highest Shannon entropy) and aggregate runner-up subword probabilities.
	minLogprob := 0.0
	maxEntropy := 0.0
	probs = make(map[string]float64)

	for idx, mt := range matchedTokens {
		if idx == 0 || mt.Logprob < minLogprob {
			minLogprob = mt.Logprob
		}
		var h float64
		for _, item := range mt.TopLogprobs {
			p := math.Exp(item.Logprob)
			tKey := strings.Trim(item.Token, " \t\n\r\"',:{}[]")
			if tKey != "" {
				if existing, ok := probs[tKey]; !ok || p > existing {
					probs[tKey] = p
				}
			}
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		if h > maxEntropy {
			maxEntropy = h
		}
	}

	logprob = minLogprob
	confidence = math.Exp(minLogprob)
	if confidence > 1.0 {
		confidence = 1.0
	}
	entropy = maxEntropy

	return confidence, logprob, entropy, probs
}

func cleanJSON(content string) string {
	content = strings.TrimSpace(content)
	if idx := strings.Index(content, "```json"); idx != -1 {
		content = content[idx+7:]
		if end := strings.Index(content, "```"); end != -1 {
			content = content[:end]
		}
	} else if idx := strings.Index(content, "```"); idx != -1 {
		content = content[idx+3:]
		if end := strings.Index(content, "```"); end != -1 {
			content = content[:end]
		}
	} else if idx := strings.Index(content, "{"); idx != -1 {
		if end := strings.LastIndex(content, "}"); end != -1 && end > idx {
			content = content[idx : end+1]
		}
	}
	return strings.TrimSpace(content)
}

// RawContent returns the content as a string regardless of whether it was deserialized as string or map.
func (m ChatMessage) RawContent() string {
	if s, ok := m.Content.(string); ok {
		return s
	}
	b, _ := json.Marshal(m.Content)
	return string(b)
}

// ErrUnsupportedImageFormat marks an image the serving image can't decode. vLLM loads images with PIL, which reads
// raster formats (PNG, JPEG, WebP, GIF) but not SVG: an SVG came back as HTTP 502 "cannot identify image file" (#117).
var ErrUnsupportedImageFormat = errors.New("unsupported image format")

// CheckImageFormat rejects image references the serving image can't decode: local .svg files (or files whose
// content is SVG) and data:image/svg+xml URIs. http(s) URLs aren't fetched here, so they pass. data maybe nil; when
// given (a local file's bytes) its content is sniffed too.
func CheckImageFormat(ref string, data []byte) error {
	ref = strings.TrimSpace(ref)
	lower := strings.ToLower(ref)
	isSVG := false
	switch {
	case strings.HasPrefix(lower, "data:"):
		isSVG = strings.HasPrefix(lower, "data:image/svg")
	case strings.HasPrefix(lower, "http://"), strings.HasPrefix(lower, "https://"):
		return nil
	default:
		isSVG = strings.HasSuffix(lower, ".svg") || strings.HasSuffix(lower, ".svgz") || looksLikeSVG(data)
	}
	if isSVG {
		name := ref
		if strings.HasPrefix(lower, "data:") {
			name = "a data:image/svg+xml image"
		}
		return fmt.Errorf("%w: %s is SVG, which the model server can't decode; convert it to PNG first (e.g. rsvg-convert -o image.png image.svg)", ErrUnsupportedImageFormat, name)
	}
	return nil
}

func looksLikeSVG(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	head := strings.ToLower(string(data[:min(len(data), 512)]))
	return strings.Contains(head, "<svg")
}

// BuildMultimodalContent formats user message content with optional image parts.
// If images are provided, it encodes local files to base64 data URIs and places
// image parts BEFORE the text content, conforming to DiffusionGemma best practices.
func BuildMultimodalContent(textContent string, imageInputs []string) (interface{}, error) {
	if len(imageInputs) == 0 {
		return textContent, nil
	}

	var parts []ContentPart

	for _, img := range imageInputs {
		img = strings.TrimSpace(img)
		if img == "" {
			continue
		}

		var imageURI string
		if strings.HasPrefix(img, "data:") {
			if err := CheckImageFormat(img, nil); err != nil {
				return nil, err
			}
			imageURI = img
		} else if strings.HasPrefix(img, "http://") || strings.HasPrefix(img, "https://") {
			// Fetched here, from the caller's own network, and sent as a data: URI: serving images refuse image URLs
			// by default (DGEM_IMAGE_URLS=deny). The gateway inlines URLs before they reach this point.
			uri, err := fetchImageAsDataURI(img)
			if err != nil {
				return nil, err
			}
			imageURI = uri
		} else {
			// Read local file
			fileData, err := os.ReadFile(img)
			if err != nil {
				return nil, fmt.Errorf("failed to read local image file %s: %w", img, err)
			}
			if err := CheckImageFormat(img, fileData); err != nil {
				return nil, err
			}

			mimeType := detectImageMime(img, fileData)
			encoded := base64.StdEncoding.EncodeToString(fileData)
			imageURI = fmt.Sprintf("data:%s;base64,%s", mimeType, encoded)
		}

		parts = append(parts, ContentPart{
			Type:     "image_url",
			ImageURL: &ImageURLPart{URL: imageURI},
		})
	}

	// Place text content after images
	if textContent != "" {
		parts = append(parts, ContentPart{
			Type: "text",
			Text: textContent,
		})
	}

	return parts, nil
}

// maxClientImageBytes caps an image fetched by the client.
const maxClientImageBytes = 20 << 20

// ErrImageFetch marks an image URL that couldn't be fetched or isn't an image (the caller's input).
var ErrImageFetch = errors.New("image URL")

// ImageHTTPClient returns the HTTP client used to fetch image URLs. The default is a plain client (the CLI fetches
// from the caller's own network). Network servers (`dgem serve`, `dgem systemone serve`) replace it with a client that
// only connects to public addresses, since there the URL comes from a request.
var ImageHTTPClient = func(timeout time.Duration) *http.Client { return &http.Client{Timeout: timeout} }

// fetchImageAsDataURI downloads an http(s) image and returns it as a data: URI, refusing non-images and SVG.
func fetchImageAsDataURI(ref string) (string, error) {
	hr, err := ImageHTTPClient(30 * time.Second).Get(ref)
	if err != nil {
		return "", fmt.Errorf("%w %s: %w", ErrImageFetch, ref, err)
	}
	defer hr.Body.Close()
	if hr.StatusCode < 200 || hr.StatusCode > 299 {
		return "", fmt.Errorf("%w %s: HTTP %d", ErrImageFetch, ref, hr.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(hr.Body, maxClientImageBytes+1))
	if err != nil {
		return "", fmt.Errorf("%w %s: %w", ErrImageFetch, ref, err)
	}
	if len(data) > maxClientImageBytes {
		return "", fmt.Errorf("%w %s: larger than %d MB", ErrImageFetch, ref, maxClientImageBytes>>20)
	}
	mime := strings.TrimSpace(strings.SplitN(hr.Header.Get("Content-Type"), ";", 2)[0])
	sniff := http.DetectContentType(data)
	if !strings.HasPrefix(sniff, "image/") {
		if looksLikeSVG(data) {
			return "", CheckImageFormat("x.svg", data)
		}
		return "", fmt.Errorf("%w %s: not an image (%s)", ErrImageFetch, ref, sniff)
	}
	if !strings.HasPrefix(mime, "image/") {
		mime = sniff
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

func detectImageMime(path string, data []byte) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	case ".gif":
		return "image/gif"
	case ".svg":
		return "image/svg+xml"
	default:
		contentType := http.DetectContentType(data)
		if strings.HasPrefix(contentType, "image/") {
			return contentType
		}
		return "image/jpeg"
	}
}
