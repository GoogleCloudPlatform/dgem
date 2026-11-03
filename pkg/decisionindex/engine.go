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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/permutation"
)

const (
	// MaxOptionsPerSlot is the hardware/tokenizer ceiling of single-letter [A-Z] slot mapping in structured_server.py.
	MaxOptionsPerSlot = 26
	// BracketSize is the chunk size used when partitioning wide choices (27..255 options) into Round-1 brackets.
	BracketSize = 20
	// MaxSlotsPerPass is the safe per-pass slot count to prevent 256-token diffusion canvas overflow ("the canvas holds").
	MaxSlotsPerPass = 8
)

// SystemOneQuestion matches the Jev / apolinario/decision-index question specification. "criteria" is an object for
// choice and noul questions and an array of ordered levels for score questions:
//
//	{"type": "choice", "instructions": "...", "criteria": {"opt_key": "description", ...}}
//	{"type": "score",  "instructions": "...", "criteria": ["lowest level", ..., "highest level"]}
type SystemOneQuestion struct {
	Type         string            `json:"type"`
	Instructions any               `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
	// Levels are a score question's levels, lowest first (criteria given as an array; set by UnmarshalJSON).
	Levels []string `json:"-"`
	// Order is the criteria keys in the order the request listed them (set by UnmarshalJSON). Go maps drop order;
	// with EngineOptions.OptionOrder "source" the adapter labels options in this order instead of alphabetically.
	Order []string `json:"-"`
}

// UnmarshalJSON accepts any JSON value as an option description. The Decision Index suite sends objects (POP909
// chords, ChessBench moves) and arrays (cfcolor swatches) as well as strings. Strings pass through unchanged; any
// other value is rendered as compact JSON text for the prompt (a mechanical format conversion, no content change).
// A "criteria" array is read as score levels, in order. Before this a score question failed to parse and the
// gateway silently forwarded the whole request unbatched.
func (q *SystemOneQuestion) UnmarshalJSON(b []byte) error {
	var raw struct {
		Type         string          `json:"type"`
		Instructions any             `json:"instructions"`
		Criteria     json.RawMessage `json:"criteria"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	q.Type, q.Instructions = raw.Type, raw.Instructions
	q.Criteria, q.Levels, q.Order = nil, nil, nil
	c := bytes.TrimSpace(raw.Criteria)
	switch {
	case len(c) == 0 || bytes.Equal(c, []byte("null")):
	case c[0] == '{':
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(c, &obj); err != nil {
			return err
		}
		q.Criteria = make(map[string]string, len(obj))
		for k, v := range obj {
			q.Criteria[k] = criterionText(v)
		}
		q.Order = criteriaKeyOrder(b)
	case c[0] == '[':
		var arr []json.RawMessage
		if err := json.Unmarshal(c, &arr); err != nil {
			return err
		}
		q.Levels = make([]string, len(arr))
		for i, v := range arr {
			q.Levels[i] = criterionText(v)
		}
	default:
		return fmt.Errorf("criteria must be an object (choice, noul) or an array of levels (score)")
	}
	return nil
}

// criteriaKeyOrder returns the keys of the question's "criteria" object in source order, or nil if they can't be
// read. It walks the question JSON with a token decoder because json.Unmarshal into a map loses order.
func criteriaKeyOrder(question []byte) []string {
	dec := json.NewDecoder(bytes.NewReader(question))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return nil
	}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil
		}
		if k, _ := kt.(string); k != "criteria" {
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return nil
			}
			continue
		}
		if t, err := dec.Token(); err != nil || t != json.Delim('{') {
			return nil
		}
		var keys []string
		seen := map[string]bool{}
		for dec.More() {
			kt, err := dec.Token()
			if err != nil {
				return nil
			}
			k, _ := kt.(string)
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return nil
			}
			if !seen[k] {
				seen[k] = true
				keys = append(keys, k)
			}
		}
		return keys
	}
	return nil
}

// Option orders for EngineOptions.OptionOrder.
const (
	OptionOrderAlpha  = "alpha"  // sort option keys (default; the adapter's original behaviour)
	OptionOrderSource = "source" // keep the order the request listed them
)

// Label sets for EngineOptions.Labels (the serving image's "labels" schema key).
const (
	LabelsAZ      = "az"       // A-Z, at most 26 options per read (default)
	LabelsAZAA    = "az_aa"    // A-Z then AA-AZ, at most 52
	LabelsAZLower = "az_lower" // A-Z then a-z, at most 52 (server: requests of up to 10 questions)
)

// optionLimit is how many options one read can label with the given label set.
func optionLimit(labels string) int {
	switch labels {
	case LabelsAZAA, LabelsAZLower:
		return 52
	}
	return MaxOptionsPerSlot
}

func wideLabels(labels string) bool { return optionLimit(labels) > MaxOptionsPerSlot }

// optionKeys returns q's option keys in the order they are labelled A, B, C…: the request's order with
// OptionOrderSource when it is known and complete, otherwise sorted.
func optionKeys(q SystemOneQuestion, order string) []string {
	if order == OptionOrderSource && len(q.Order) == len(q.Criteria) {
		ok := true
		for _, k := range q.Order {
			if _, in := q.Criteria[k]; !in {
				ok = false
				break
			}
		}
		if ok {
			return append([]string(nil), q.Order...)
		}
	}
	return sortedOptionKeys(q.Criteria)
}

// subQuestion is a choice question over a subset of q's options (a bracket, the final, a verify read). Its Order
// follows q's source order, so OptionOrderSource carries through every round.
func subQuestion(q SystemOneQuestion, keys []string) SystemOneQuestion {
	pos := make(map[string]int, len(q.Order))
	for i, k := range q.Order {
		pos[k] = i
	}
	order := append([]string(nil), keys...)
	sort.SliceStable(order, func(i, j int) bool {
		pi, iok := pos[order[i]]
		pj, jok := pos[order[j]]
		if iok && jok {
			return pi < pj
		}
		return iok && !jok
	})
	crit := make(map[string]string, len(keys))
	for _, k := range keys {
		crit[k] = q.Criteria[k]
	}
	return SystemOneQuestion{Type: "choice", Instructions: FormatInstructions(q.Instructions), Criteria: crit, Order: order}
}

func criterionText(v json.RawMessage) string {
	var s string
	if err := json.Unmarshal(v, &s); err == nil {
		return s
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, v); err == nil {
		return buf.String()
	}
	return string(v)
}

// SystemOneRequest matches the POST /v1/systemone payload sent by decision_index.engines.http:HttpSystemOne.
type SystemOneRequest struct {
	Model     string                       `json:"model,omitempty"`
	State     any                          `json:"state"`
	Questions map[string]SystemOneQuestion `json:"questions"`
	// Images are data: URIs or http(s) URLs shown to the model ahead of the state, the same "images" field the
	// serving image's /v1/systemone accepts. Every sub-request (batches and bracket rounds) carries them.
	Images []string `json:"images,omitempty"`
	// Media is an alias for Images in the form some Jev clients send: image/* entries are appended to
	// Images by MergeMedia; any other type is refused.
	Media []SystemOneMedia `json:"media,omitempty"`
}

// SystemOneMedia is one entry of a request's "media" array. Data is a data: URI or raw base64 of MimeType. Other
// fields (e.g. "name") are ignored.
type SystemOneMedia struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

// MergeMedia moves the request's image media into Images (after any images already there, in order) and clears
// Media. Non-image media is ErrUnsupportedMedia; a URL or an empty entry is ErrInvalidImage (URLs go in "images",
// where the gateway fetches them with its public-host checks). Safe to call more than once.
func (r *SystemOneRequest) MergeMedia() error {
	for i, m := range r.Media {
		mime := strings.ToLower(strings.TrimSpace(m.MimeType))
		data := strings.TrimSpace(m.Data)
		if strings.HasPrefix(data, "data:") {
			// A data: URI's own type wins over mime_type: it is what the server will decode.
			if end := strings.IndexAny(data, ";,"); end > len("data:") {
				mime = strings.ToLower(data[len("data:"):end])
			}
		}
		if !strings.HasPrefix(mime, "image/") {
			if mime == "" {
				return fmt.Errorf("%w: media[%d] needs a mime_type", ErrInvalidImage, i)
			}
			return fmt.Errorf("%w: media[%d] is %s; this model reads text and images only: send video as sampled frames and audio as a transcript in \"state\"", ErrUnsupportedMedia, i, mime)
		}
		switch {
		case data == "":
			return fmt.Errorf("%w: media[%d] has no data", ErrInvalidImage, i)
		case strings.HasPrefix(data, "data:"):
		case strings.HasPrefix(data, "http://") || strings.HasPrefix(data, "https://"):
			return fmt.Errorf("%w: media[%d]: send image URLs in \"images\"; media data must be a data: URI or base64", ErrInvalidImage, i)
		default:
			data = "data:" + mime + ";base64," + data
		}
		r.Images = append(r.Images, data)
	}
	r.Media = nil
	return nil
}

// ErrInvalidImage marks an image reference the adapter refuses. Local file paths are rejected because the adapter
// runs as a network server: a path would read the adapter host's own filesystem.
var ErrInvalidImage = errors.New("invalid image")

// ErrUnsupportedMedia marks a request carrying media the model can't read (audio, video). DiffusionGemma is text +
// image only; such fields used to be dropped silently and the request answered as a guess.
var ErrUnsupportedMedia = errors.New("unsupported media")

// ErrInvalidRequest marks a request body or question the adapter can't read (400): an unknown question type, or
// criteria of the wrong shape for the type.
var ErrInvalidRequest = errors.New("invalid request")

// ErrUnsupportedQuestion marks a well-formed question the model can't answer in one read (422, the same class as the
// server's capacity refusals): a score question with no or more than MaxOptionsPerSlot levels, or repeated levels.
var ErrUnsupportedQuestion = errors.New("unsupported question")

// Question types after normalizing (see questionKind).
const (
	kindChoice = "choice"
	kindNoul   = "noul"
	kindScore  = "score"
)

// questionKind normalizes q's type. An empty type is a choice when criteria is an object (the adapter's behaviour
// before score support) and a score when it is an array. Unknown types are ErrInvalidRequest: they used to be read
// as choices.
func questionKind(qKey string, q SystemOneQuestion) (string, error) {
	t := strings.ToLower(strings.TrimSpace(q.Type))
	switch {
	case isNoulType(t):
		t = kindNoul
	case t == "" && q.Levels != nil:
		t = kindScore
	case t == "":
		t = kindChoice
	}
	switch t {
	case kindNoul:
		if q.Levels != nil {
			return "", fmt.Errorf("%w: question %q: noul criteria must be an object ({\"true\": ..., \"false\": ...})", ErrInvalidRequest, qKey)
		}
	case kindChoice:
		if q.Levels != nil {
			return "", fmt.Errorf("%w: question %q: choice criteria must be an object of options; an array is a score question's levels", ErrInvalidRequest, qKey)
		}
	case kindScore:
		if q.Criteria != nil {
			return "", fmt.Errorf("%w: question %q: score criteria must be an array of levels, lowest first", ErrInvalidRequest, qKey)
		}
		if len(q.Levels) == 0 || len(q.Levels) > MaxOptionsPerSlot {
			return "", fmt.Errorf("%w: question %q: a score question needs 1 to %d levels (got %d)", ErrUnsupportedQuestion, qKey, MaxOptionsPerSlot, len(q.Levels))
		}
		seen := make(map[string]bool, len(q.Levels))
		for _, l := range q.Levels {
			l = strings.TrimSpace(l)
			if seen[l] {
				return "", fmt.Errorf("%w: question %q: score levels must be distinct (%q repeats)", ErrUnsupportedQuestion, qKey, l)
			}
			seen[l] = true
		}
	default:
		return "", fmt.Errorf("%w: question %q: unknown type %q (want choice, noul or score)", ErrInvalidRequest, qKey, q.Type)
	}
	return t, nil
}

// trivialAnswer answers a question with a single option or level without a model call: the server refuses them
// ("needs at least two alternatives") and the answer is certain. Same shape other Jev clients use for this case.
func trivialAnswer(kind string, q SystemOneQuestion) (SystemOneAnswer, bool) {
	one := 1.0
	switch {
	case kind == kindChoice && len(q.Criteria) == 1:
		var key string
		for k := range q.Criteria {
			key = k
		}
		return SystemOneAnswer{Type: kindChoice, Choice: key, Probabilities: map[string]float64{key: 1}, Confidence: &one}, true
	case kind == kindScore && len(q.Levels) == 1:
		zero := 0.0
		return SystemOneAnswer{Type: kindScore, Score: &zero, Legend: map[string]string{"0": q.Levels[0]},
			Probabilities: map[string]float64{"0": 1}, Confidence: &one}, true
	}
	return SystemOneAnswer{}, false
}

// UnsupportedMediaKeys are top-level request fields refused with ErrUnsupportedMedia.
var UnsupportedMediaKeys = []string{"audio", "audios", "video", "videos", "input_audio", "video_url", "videos_url"}

// CheckMediaKeys returns ErrUnsupportedMedia when a JSON request body has any of UnsupportedMediaKeys at top level.
// Bodies that aren't JSON objects pass (the caller's own decoding reports those).
func CheckMediaKeys(body []byte) error {
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil {
		return nil
	}
	var found []string
	for _, k := range UnsupportedMediaKeys {
		if _, ok := top[k]; ok {
			found = append(found, k)
		}
	}
	if len(found) == 0 {
		return nil
	}
	return fmt.Errorf("%w: %s not supported; this model reads text and images only: send video as sampled frames in \"images\" and audio as a transcript in \"state\"", ErrUnsupportedMedia, strings.Join(found, ", "))
}

// ValidateImages checks that every image is a data: URI or an http(s) URL (empty entries are skipped, as
// client.BuildMultimodalContent does).
func ValidateImages(images []string) error {
	for i, img := range images {
		img = strings.TrimSpace(img)
		if err := client.CheckImageFormat(img, nil); err != nil {
			return fmt.Errorf("%w: images[%d]: %v", ErrInvalidImage, i, err)
		}
		if img == "" || strings.HasPrefix(img, "data:") || strings.HasPrefix(img, "https://") || strings.HasPrefix(img, "http://") {
			continue
		}
		return fmt.Errorf("%w: images[%d] must be a data: URI or an http(s) URL", ErrInvalidImage, i)
	}
	return nil
}

// SystemOneAnswer matches the per-question answer validated by decision_index.engines.base:validate().
// Score answers follow the serving image's jev_answer: probabilities keyed "0".."n-1" (level index), score the
// 0-indexed expected level sum(i * p_i), legend the level text per index.
type SystemOneAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Noul          *float64           `json:"noul,omitempty"`
	// Confidence is the probability of the answer given (choice: the chosen option; score: the most likely level;
	// noul: max(p, 1-p)), as the serving image's jev_answer reports it.
	Confidence  *float64 `json:"confidence,omitempty"`
	Entropy     float64  `json:"entropy,omitempty"`
	NormalizedH float64  `json:"normalized_entropy,omitempty"`
	PassesUsed  int      `json:"passes_used,omitempty"`
}

// SystemOneResponse matches the response envelope validated by decision_index.
type SystemOneResponse struct {
	Answers         map[string]SystemOneAnswer `json:"answers"`
	EvaluationTrace *EvaluationTrace           `json:"evaluation_trace,omitempty"`
}

// EvaluationTrace records dgem's multi-slot batching and wide-option bracket telemetry.
type EvaluationTrace struct {
	TotalQuestions   int     `json:"total_questions"`
	MaxOptionsSeen   int     `json:"max_options_seen"`
	ForwardPasses    int     `json:"forward_passes"`
	WideBracketedQs  int     `json:"wide_bracketed_questions"`
	MultiSlotBatches int     `json:"multi_slot_batches"`
	WallTimeMs       float64 `json:"wall_time_ms"`
	// ServerMs is the serving image's own time for this request: the sum of diagnostics.timing.total_ms over every
	// pass the adapter made (batches, bracket rounds). Omitted when the server reports no timing.
	ServerMs         float64 `json:"server_ms,omitempty"`
	TemperatureScale float64 `json:"temperature_scale,omitempty"`
}

// serverTime accumulates the upstream server time of one ExecuteSystemOne call across its passes.
type serverTime struct {
	mu sync.Mutex
	ms float64
}

type serverTimeKey struct{}

func addServerTime(ctx context.Context, ms float64) {
	if st, ok := ctx.Value(serverTimeKey{}).(*serverTime); ok && st != nil && ms > 0 {
		st.mu.Lock()
		st.ms += ms
		st.mu.Unlock()
	}
}

// EngineOptions configures the Decision Index execution adapter.
type EngineOptions struct {
	MaxSlotsPerPass   int
	MaxOptionsPerSlot int
	NaiveLimits       bool    // If true, mimics naive 26-option / 10-slot capacity rejections (HTTP 422 Unsupported)
	TemperatureScale  float64 // Post-hoc slot temperature scaling T* (1.0 = unscaled, default)
	MaxConcurrency    int
	DualMirror        bool    // Evaluate forward + reversed option orderings on the same diffusion canvas
	NullPriorDebias   bool    // Divide out content-free positional 'A'-bias prior
	PriorAlpha        float64 // Damping exponent alpha in [0, 1] for null-prior de-biasing
	BracketSize       int     // Max options per Round-1 bracket for wide (>26) choices; <=1 means BracketSize
	CatchAll          string  // Wide-option catch-all handling: "off" (default), "final", "both" or "verify"; see catchAllKeys
	NoulMode          string  // How yes/no questions are read: "noul" (default) or "choice" (2-option choice; see noulAsChoice)
	Labels            string  // Server label set for choice options: "az" (default, 26) or "az_aa"/"az_lower" (52). Wide sets read up to 52 options natively and bracket above that with 52-wide brackets
	OptionOrder       string  // How options are labelled A, B, C…: "alpha" (default, sorted keys) or "source" (request order)
	PromptLayout      string  // Server prompt layout: "document_first" (default: state first, then questions) or "schema_first"/"" (questions as the system prompt)
}

// DefaultEngineOptions returns production settings with Wide-Option Tournament + Multi-Slot Batching enabled (T*=1.0).
func DefaultEngineOptions() EngineOptions {
	return EngineOptions{
		MaxSlotsPerPass:   MaxSlotsPerPass,
		MaxOptionsPerSlot: MaxOptionsPerSlot,
		NaiveLimits:       false,
		TemperatureScale:  1.0,
		MaxConcurrency:    4,
		PriorAlpha:        0.50,
		BracketSize:       BracketSize,
		// document_first: on the wave-2 canary (dev suites, paired) it raised di_catchall 0.790 -> 0.870 and rag_dev
		// 0.705 -> 0.770 with JevBench and di_wide unchanged; out-of-scope recall on di_catchall fell 1.00 -> 0.825.
		PromptLayout: "document_first",
	}
}

func isNoulType(t string) bool {
	lower := strings.ToLower(strings.TrimSpace(t))
	return lower == "noul" || lower == "bool" || lower == "boolean"
}

// FormatState converts an arbitrary Decision Index state (string or JSON object) into a valid JSON string for structured_server.py.
func FormatState(state any) string {
	if state == nil {
		return `{"state":"Evaluate the decision questions based on the provided option criteria."}`
	}
	switch v := state.(type) {
	case string:
		trimmed := strings.TrimSpace(v)
		if (strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")) ||
			(strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]")) {
			if json.Valid([]byte(trimmed)) {
				return trimmed
			}
		}
		b, _ := json.Marshal(map[string]string{"state": v})
		return string(b)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			b2, _ := json.Marshal(map[string]string{"state": fmt.Sprintf("%v", v)})
			return string(b2)
		}
		return string(b)
	}
}

// FormatInstructions converts instructions (string or structured object) into a concise string.
func FormatInstructions(instr any) string {
	if instr == nil {
		return "Select the single best option matching the criteria."
	}
	switch v := instr.(type) {
	case string:
		return v
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(b)
	}
}

// ExecuteSystemOne evaluates a Decision Index request against a dgem Client, automatically applying:
// 1. Wide-Option Tournament Routing when any question has >26 options (up to 255 options)
// 2. Multi-Slot Canvas Batching when a request has >MaxSlotsPerPass simultaneous questions
// 3. Post-hoc Slot Temperature Scaling T* while strictly preserving argmax and sum(p)=1.0.
func ExecuteSystemOne(ctx context.Context, cli *client.Client, req SystemOneRequest, opts EngineOptions) (*SystemOneResponse, error) {
	start := time.Now()
	if opts.MaxSlotsPerPass <= 0 {
		opts.MaxSlotsPerPass = MaxSlotsPerPass
	}
	if opts.MaxOptionsPerSlot <= 0 {
		opts.MaxOptionsPerSlot = MaxOptionsPerSlot
	}
	if wideLabels(opts.Labels) && opts.MaxOptionsPerSlot == MaxOptionsPerSlot {
		// A wide label set reads up to its limit natively; brackets start above it.
		opts.MaxOptionsPerSlot = optionLimit(opts.Labels)
	}
	if opts.TemperatureScale <= 0 {
		opts.TemperatureScale = 1.0
	}

	if err := req.MergeMedia(); err != nil {
		return nil, err
	}
	qKeys := make([]string, 0, len(req.Questions))
	maxOpts := 0
	kinds := make(map[string]string, len(req.Questions))
	for k, q := range req.Questions {
		kind, err := questionKind(k, q)
		if err != nil {
			return nil, err
		}
		kinds[k] = kind
		qKeys = append(qKeys, k)
		maxOpts = maxInt(maxOpts, maxInt(len(q.Criteria), len(q.Levels)))
	}
	sort.Strings(qKeys)

	// If NaiveLimits mode is enabled, enforce raw 26-option and 10-slot canvas ceilings to demonstrate capacity failure rate.
	if opts.NaiveLimits {
		if maxOpts > 26 {
			return nil, fmt.Errorf("HTTP 422 Unsupported: options per choice (%d) exceeds 26-option [A-Z] limit", maxOpts)
		}
		if len(qKeys) > 10 {
			return nil, fmt.Errorf("HTTP 422 Unsupported: the canvas holds at most 10 questions per pass (got %d)", len(qKeys))
		}
	}

	if err := ValidateImages(req.Images); err != nil {
		return nil, err
	}
	stateText := FormatState(req.State)
	st := &serverTime{}
	ctx = context.WithValue(ctx, serverTimeKey{}, st)
	answers := make(map[string]SystemOneAnswer, len(qKeys))
	var mu sync.Mutex
	totalPasses := 0
	wideQs := 0
	multiBatches := 0

	// Partition questions into trivial (one option or level: answered here, no model call), standard (K <= 26,
	// noul, score) and wide (choice with K > 26).
	var standardKeys []string
	var wideKeys []string
	for _, k := range qKeys {
		q := req.Questions[k]
		if a, ok := trivialAnswer(kinds[k], q); ok {
			answers[k] = a
			continue
		}
		if kinds[k] == kindChoice && len(q.Criteria) > opts.MaxOptionsPerSlot {
			wideKeys = append(wideKeys, k)
		} else {
			standardKeys = append(standardKeys, k)
		}
	}

	// 1. Process standard questions (K <= 26) in Multi-Slot Canvas Batches of size opts.MaxSlotsPerPass
	for i := 0; i < len(standardKeys); i += opts.MaxSlotsPerPass {
		end := i + opts.MaxSlotsPerPass
		if end > len(standardKeys) {
			end = len(standardKeys)
		}
		batchKeys := standardKeys[i:end]
		multiBatches++

		batchAns, passes, err := evaluateStandardBatch(ctx, cli, stateText, req.Images, batchKeys, req.Questions, kinds, opts)
		if err != nil {
			return nil, err
		}
		mu.Lock()
		totalPasses += passes
		for k, a := range batchAns {
			answers[k] = a
		}
		mu.Unlock()
	}

	// 2. Process wide questions (27 <= K <= 255) via 2-Stage Bracket Tournament Routing
	for _, k := range wideKeys {
		wideQs++
		q := req.Questions[k]
		ans, passes, err := evaluateWideQuestionTournament(ctx, cli, stateText, req.Images, k, q, opts.TemperatureScale, opts.BracketSize, opts.CatchAll, opts.PromptLayout, opts.OptionOrder, opts.Labels)
		if err != nil {
			return nil, err
		}
		mu.Lock()
		totalPasses += passes
		answers[k] = ans
		mu.Unlock()
	}

	return &SystemOneResponse{
		Answers: answers,
		EvaluationTrace: &EvaluationTrace{
			TotalQuestions:   len(qKeys),
			MaxOptionsSeen:   maxOpts,
			ForwardPasses:    totalPasses,
			WideBracketedQs:  wideQs,
			MultiSlotBatches: multiBatches,
			WallTimeMs:       float64(time.Since(start).Milliseconds()),
			ServerMs:         math.Round(st.ms*10) / 10,
			TemperatureScale: opts.TemperatureScale,
		},
	}, nil
}

// evaluateStandardBatch runs a single forward pass for up to MaxSlotsPerPass questions where each question has <= 26 options.
func evaluateStandardBatch(
	ctx context.Context,
	cli *client.Client,
	stateText string,
	images []string,
	batchKeys []string,
	allQuestions map[string]SystemOneQuestion,
	kinds map[string]string,
	opts EngineOptions,
) (map[string]SystemOneAnswer, int, error) {
	questionsPayload := make([]map[string]any, 0, len(batchKeys))
	for _, qKey := range batchKeys {
		qSpec := allQuestions[qKey]
		if kinds[qKey] == kindScore {
			// "lines" format on the server: batches stay <= MaxSlotsPerPass (8) questions, under its 10-question
			// switch to the "indexed" format, where ids ending in letters can merge with labels.
			questionsPayload = append(questionsPayload, map[string]any{
				"id":           qKey,
				"type":         kindScore,
				"instructions": FormatInstructions(qSpec.Instructions),
				"levels":       qSpec.Levels,
			})
		} else if isNoulType(qSpec.Type) && opts.NoulMode == "choice" {
			questionsPayload = append(questionsPayload, noulAsChoice(qKey, qSpec))
		} else if isNoulType(qSpec.Type) {
			q := map[string]any{
				"id":           qKey,
				"type":         "boolean",
				"instructions": FormatInstructions(qSpec.Instructions),
			}
			// Keep the request's true/false descriptions: the structured server renders them next to yes/no.
			// Dropping them changed the question (e.g. RAGTruth states the claim in `instructions` and defines
			// what yes and no mean only in `criteria`).
			if crit := noulCriteria(qSpec.Criteria); crit != nil {
				q["criteria"] = crit
			}
			questionsPayload = append(questionsPayload, q)
		} else {
			optKeys := optionKeys(qSpec, opts.OptionOrder)
			optObjs := make([]map[string]string, 0, len(optKeys))
			for _, ok := range optKeys {
				optObjs = append(optObjs, map[string]string{
					"name":        ok,
					"description": strings.TrimSpace(qSpec.Criteria[ok]),
				})
			}
			questionsPayload = append(questionsPayload, map[string]any{
				"id":           qKey,
				"type":         "choice",
				"instructions": FormatInstructions(qSpec.Instructions),
				"options":      optObjs,
			})
		}
	}

	schemaEnvelope := map[string]any{
		"instructions": "Evaluate each decision question strictly against the provided state and option descriptions.",
		"samples":      1,
		"think":        0,
		"questions":    questionsPayload,
	}
	if opts.PromptLayout == "document_first" {
		schemaEnvelope["layout"] = "document_first"
	}
	if wideLabels(opts.Labels) {
		schemaEnvelope["labels"] = opts.Labels
	}
	schemaBytes, err := json.Marshal(schemaEnvelope)
	if err != nil {
		return nil, 1, err
	}
	schemaJSON := string(schemaBytes)

	var slotOpts map[string][]permutation.OptionItem
	if opts.DualMirror && len(batchKeys) <= 4 {
		schemaJSON, slotOpts, _ = permutation.InjectDualMirrorSchema(schemaJSON)
	} else if opts.NullPriorDebias {
		slotOpts = permutation.ExtractSchemaSlotOptions(schemaJSON)
	}

	if strings.TrimSpace(stateText) == "" {
		stateText = "Evaluate the decision questions based on the provided option criteria."
	}

	resp, _, err := cli.Decide(ctx, schemaJSON, stateText, images...)
	if err != nil {
		return nil, 1, err
	}
	addServerTime(ctx, resp.Diagnostics.Timing.TotalMs)

	if (opts.DualMirror && len(batchKeys) <= 4) || opts.NullPriorDebias {
		permutation.PostProcessDecisionResponse(resp, slotOpts, opts.DualMirror && len(batchKeys) <= 4, opts.NullPriorDebias, opts.PriorAlpha)
	}

	tempScale := opts.TemperatureScale

	out := make(map[string]SystemOneAnswer, len(batchKeys))
	for _, qKey := range batchKeys {
		qSpec := allQuestions[qKey]
		rawAns, exists := resp.Answers[qKey]

		if isNoulType(qSpec.Type) {
			pYes := 0.5
			if exists && opts.NoulMode == "choice" {
				// Rendered as a yes/no choice (noulAsChoice): read the option probabilities directly.
				if py, ok := rawAns.Probabilities["yes"]; ok {
					pYes = py
				} else if pn, ok := rawAns.Probabilities["no"]; ok {
					pYes = 1.0 - pn
				}
			} else if exists {
				if rawAns.Noul > 0 {
					pYes = rawAns.Noul
				} else if rawAns.Confidence > 0 && (strings.EqualFold(rawAns.Label, "yes") || strings.EqualFold(rawAns.Label, "true")) {
					pYes = rawAns.Confidence
				} else if rawAns.Confidence > 0 && (strings.EqualFold(rawAns.Label, "no") || strings.EqualFold(rawAns.Label, "false")) {
					pYes = 1.0 - rawAns.Confidence
				} else if rawAns.Probabilities != nil {
					if py, ok := rawAns.Probabilities["yes"]; ok {
						pYes = py
					} else if pt, ok := rawAns.Probabilities["true"]; ok {
						pYes = pt
					} else if pn, ok := rawAns.Probabilities["no"]; ok {
						pYes = 1.0 - pn
					}
				}
			}
			if tempScale != 1.0 && tempScale > 0 {
				logitYes := math.Log(math.Max(1e-12, pYes)) / tempScale
				logitNo := math.Log(math.Max(1e-12, 1.0-pYes)) / tempScale
				maxLogit := math.Max(logitYes, logitNo)
				ey := math.Exp(logitYes - maxLogit)
				en := math.Exp(logitNo - maxLogit)
				pYes = ey / (ey + en)
			}
			pYes = math.Max(0.0, math.Min(1.0, pYes))
			normH := 0.0
			if rawAns.Entropy > 0 {
				normH = rawAns.Entropy / math.Ln2
			}
			conf := math.Max(pYes, 1-pYes)
			out[qKey] = SystemOneAnswer{
				Type:        "noul",
				Noul:        &pYes,
				Confidence:  &conf,
				PassesUsed:  1,
				Entropy:     rawAns.Entropy,
				NormalizedH: normH,
			}
			continue
		}

		if kinds[qKey] == kindScore {
			out[qKey] = scoreAnswer(qSpec.Levels, rawAns, exists, tempScale)
			continue
		}

		optKeys := sortedOptionKeys(qSpec.Criteria)
		probs := make(map[string]float64, len(optKeys))

		if exists && len(rawAns.Probabilities) > 0 {
			for _, ok := range optKeys {
				probs[ok] = rawAns.Probabilities[ok]
			}
		} else if exists && (rawAns.Choice != "" || rawAns.Label != "") {
			winner := rawAns.Choice
			if winner == "" {
				winner = rawAns.Label
			}
			for _, ok := range optKeys {
				if ok == winner {
					probs[ok] = 0.90
				} else {
					probs[ok] = 0.10 / float64(maxInt(1, len(optKeys)-1))
				}
			}
		} else {
			u := 1.0 / float64(len(optKeys))
			for _, ok := range optKeys {
				probs[ok] = u
			}
		}

		scaledProbs, winner, h, normH := NormalizeAndScaleProbabilities(probs, optKeys, tempScale)
		conf := scaledProbs[winner]
		out[qKey] = SystemOneAnswer{
			Type:          "choice",
			Choice:        winner,
			Probabilities: scaledProbs,
			Confidence:    &conf,
			Entropy:       h,
			NormalizedH:   normH,
			PassesUsed:    1,
		}
	}

	return out, 1, nil
}

// scoreAnswer maps the server's score answer (probabilities keyed by level text) to the Jev wire form: probabilities
// keyed "0".."n-1", score = sum(i * p_i) (0-indexed), legend, confidence = the most likely level's probability.
// Temperature scaling applies as for choices. Without probabilities it falls back to the reported level (0.9) or
// uniform, like choices.
func scoreAnswer(levels []string, rawAns client.QuestionAnswer, exists bool, tempScale float64) SystemOneAnswer {
	n := len(levels)
	idx := make([]string, n)
	legend := make(map[string]string, n)
	probs := make(map[string]float64, n)
	for i, l := range levels {
		idx[i] = strconv.Itoa(i)
		legend[idx[i]] = l
	}
	switch {
	case exists && len(rawAns.Probabilities) > 0:
		for i, l := range levels {
			probs[idx[i]] = rawAns.Probabilities[l]
		}
	case exists && rawAns.Level != "":
		for i, l := range levels {
			if l == rawAns.Level {
				probs[idx[i]] = 0.90
			} else {
				probs[idx[i]] = 0.10 / float64(maxInt(1, n-1))
			}
		}
	default:
		for i := range levels {
			probs[idx[i]] = 1.0 / float64(n)
		}
	}
	scaled, winner, h, normH := NormalizeAndScaleProbabilities(probs, idx, tempScale)
	score := 0.0
	for i := range levels {
		score += float64(i) * scaled[idx[i]]
	}
	conf := scaled[winner]
	return SystemOneAnswer{
		Type:          kindScore,
		Score:         &score,
		Legend:        legend,
		Probabilities: scaled,
		Confidence:    &conf,
		Entropy:       h,
		NormalizedH:   normH,
		PassesUsed:    1,
	}
}

// evaluateWideQuestionTournament resolves a choice question with 27..255 options by:
// 1. Partitioning the K options into brackets of <= BracketSize (20) options.
// 2. Running Round-1 single-pass readouts on each bracket to extract top-2 contenders + bracket probability masses.
// 3. Running a Round-2 Finals readout over the bracket winners to calibrate the global probability distribution over all K options.
func evaluateWideQuestionTournament(
	ctx context.Context,
	cli *client.Client,
	stateText string,
	images []string,
	qKey string,
	qSpec SystemOneQuestion,
	tempScale float64,
	bracketSize int,
	catchAllMode string,
	layout string,
	optionOrder string,
	labels string,
) (SystemOneAnswer, int, error) {
	optKeys := optionKeys(qSpec, optionOrder)
	numOpts := len(optKeys)
	limit := optionLimit(labels)
	// Catch-all options ("none of the listed", "out of scope", ...) are the correct answer inside any Round-1 bracket
	// that lacks the true option, so a flat tournament sends them to the final too often. Opt-in modes:
	//   final  - catch-alls skip Round 1 and compete only in the final
	//   both   - catch-alls appear in every Round-1 bracket and in the final
	//   verify - the final runs over real options only, then a second read picks between that winner and the catch-alls
	// Measured on CLINC150 validation (dev): final raised macro-F1 0.734 -> 0.775 but lowered out-of-scope recall
	// 0.96 -> 0.87, so the default stays "off".
	var catchAll []string
	if catchAllMode != "" && catchAllMode != "off" && numOpts > limit {
		catchAll = catchAllKeys(qSpec.Criteria)
	}
	isCatchAll := make(map[string]bool, len(catchAll))
	for _, k := range catchAll {
		isCatchAll[k] = true
	}
	roundKeys := optKeys
	if len(catchAll) > 0 {
		roundKeys = make([]string, 0, numOpts-len(catchAll))
		for _, k := range optKeys {
			if !isCatchAll[k] {
				roundKeys = append(roundKeys, k)
			}
		}
	}
	if numOpts <= limit {
		batchMap, passes, err := evaluateStandardBatch(ctx, cli, stateText, images, []string{qKey}, map[string]SystemOneQuestion{qKey: qSpec}, nil, EngineOptions{TemperatureScale: tempScale, PromptLayout: layout, OptionOrder: optionOrder, Labels: labels})
		if err != nil {
			return SystemOneAnswer{}, passes, err
		}
		return batchMap[qKey], passes, nil
	}

	if wideLabels(labels) && (bracketSize <= 1 || bracketSize == BracketSize) {
		bracketSize = limit // wide label sets bracket in 52-wide groups
	}
	if bracketSize <= 1 || bracketSize > limit {
		bracketSize = BracketSize
	}
	brackets := balancedBrackets(roundKeys, bracketSize)

	// Pack up to MaxSlotsPerPass bracket sub-questions into a single Round-1 canvas pass!
	round1Questions := make(map[string]SystemOneQuestion, len(brackets))
	round1Keys := make([]string, 0, len(brackets))
	for bIdx, bOpts := range brackets {
		bKey := fmt.Sprintf("%s_b%02d", qKey, bIdx)
		round1Keys = append(round1Keys, bKey)
		keys := append([]string(nil), bOpts...)
		if catchAllMode == "both" {
			keys = append(keys, catchAll...)
		}
		round1Questions[bKey] = subQuestion(qSpec, keys)
	}

	round1Results := make(map[string]SystemOneAnswer, len(brackets))
	passesUsed := 0
	for i := 0; i < len(round1Keys); i += MaxSlotsPerPass {
		end := i + MaxSlotsPerPass
		if end > len(round1Keys) {
			end = len(round1Keys)
		}
		subKeys := round1Keys[i:end]
		subAns, p, err := evaluateStandardBatch(ctx, cli, stateText, images, subKeys, round1Questions, nil, EngineOptions{TemperatureScale: 1.0, PromptLayout: layout, OptionOrder: optionOrder, Labels: labels})
		if err != nil {
			return SystemOneAnswer{}, passesUsed + p, err
		}
		passesUsed += p
		for k, v := range subAns {
			round1Results[k] = v
		}
	}

	// Collect top-2 contenders from each bracket for Round-2 Finals (capped at 24 finalists)
	type contender struct {
		key        string
		bracketKey string
		localProb  float64
	}
	var finalists []contender
	for bIdx, bKey := range round1Keys {
		bAns := round1Results[bKey]
		bOpts := brackets[bIdx]
		sortedLocal := make([]contender, 0, len(bOpts))
		for _, ok := range bOpts {
			sortedLocal = append(sortedLocal, contender{
				key:        ok,
				bracketKey: bKey,
				localProb:  bAns.Probabilities[ok],
			})
		}
		sort.Slice(sortedLocal, func(i, j int) bool {
			return sortedLocal[i].localProb > sortedLocal[j].localProb
		})
		take := 2
		if len(sortedLocal) < take {
			take = len(sortedLocal)
		}
		finalists = append(finalists, sortedLocal[:take]...)
	}

	maxFinal := limit - 2
	if catchAllMode != "verify" {
		maxFinal -= len(catchAll)
	}
	if len(finalists) > maxFinal {
		sort.Slice(finalists, func(i, j int) bool {
			return finalists[i].localProb > finalists[j].localProb
		})
		finalists = finalists[:maxFinal]
	}
	if catchAllMode == "final" || catchAllMode == "both" {
		for _, k := range catchAll {
			finalists = append(finalists, contender{key: k, localProb: 1})
		}
	}

	finalKeys := make([]string, 0, len(finalists))
	for _, f := range finalists {
		finalKeys = append(finalKeys, f.key)
	}
	finalQ := map[string]SystemOneQuestion{qKey: subQuestion(qSpec, finalKeys)}

	finalBatch, p, err := evaluateStandardBatch(ctx, cli, stateText, images, []string{qKey}, finalQ, nil, EngineOptions{TemperatureScale: 1.0, PromptLayout: layout, OptionOrder: optionOrder, Labels: labels})
	if err != nil {
		return SystemOneAnswer{}, passesUsed + p, err
	}
	passesUsed += p
	finalAns := finalBatch[qKey]

	if catchAllMode == "verify" && len(catchAll) > 0 {
		best, bestP := "", -1.0
		for k, v := range finalAns.Probabilities {
			if v > bestP || (v == bestP && k < best) {
				best, bestP = k, v
			}
		}
		vQ := subQuestion(qSpec, append([]string{best}, catchAll...))
		vBatch, p2, err := evaluateStandardBatch(ctx, cli, stateText, images, []string{qKey},
			map[string]SystemOneQuestion{qKey: vQ}, nil,
			EngineOptions{TemperatureScale: 1.0, PromptLayout: layout, OptionOrder: optionOrder, Labels: labels})
		if err != nil {
			return SystemOneAnswer{}, passesUsed + p2, err
		}
		passesUsed += p2
		pv := vBatch[qKey].Probabilities
		merged := make(map[string]float64, len(finalAns.Probabilities)+len(catchAll))
		for k, v := range finalAns.Probabilities {
			merged[k] = v * pv[best]
		}
		for _, k := range catchAll {
			merged[k] = pv[k]
			finalists = append(finalists, contender{key: k, localProb: 1})
		}
		finalAns.Probabilities = merged
	}

	finalistSet := make(map[string]bool, len(finalists))
	for _, f := range finalists {
		finalistSet[f.key] = true
	}
	round1 := make([]map[string]float64, len(brackets))
	for bIdx, bKey := range round1Keys {
		round1[bIdx] = round1Results[bKey].Probabilities
	}
	// Catch-alls are in no bracket's option list, so fusion treats them as finalists carrying final-round mass only.
	combined := fuseBracketProbabilities(brackets, round1, finalistSet, finalAns.Probabilities)

	scaledProbs, winner, h, normH := NormalizeAndScaleProbabilities(combined, optKeys, tempScale)
	conf := scaledProbs[winner]
	return SystemOneAnswer{
		Type:          "choice",
		Choice:        winner,
		Probabilities: scaledProbs,
		Confidence:    &conf,
		Entropy:       h,
		NormalizedH:   normH,
		PassesUsed:    passesUsed,
	}, passesUsed, nil
}

// NormalizeAndScaleProbabilities applies temperature scaling T > 0, guarantees strict positivity and
// exact normalization (abs(sum(p)-1) < 1e-9) required by decision_index/engines/base.py:validate(),
// and returns (probabilities, argmaxWinner, shannonEntropyNats, normalizedEntropy).
func NormalizeAndScaleProbabilities(raw map[string]float64, optKeys []string, tempScale float64) (map[string]float64, string, float64, float64) {
	if tempScale <= 0 {
		tempScale = 1.0
	}
	kCount := len(optKeys)
	if kCount == 0 {
		return map[string]float64{}, "", 0, 0
	}

	scaled := make(map[string]float64, kCount)
	sum := 0.0
	for _, k := range optKeys {
		p := raw[k]
		if math.IsNaN(p) || math.IsInf(p, 0) || p <= 1e-9 {
			p = 1e-6
		}
		sp := math.Pow(p, 1.0/tempScale)
		scaled[k] = sp
		sum += sp
	}

	winner := optKeys[0]
	bestP := -1.0
	entropy := 0.0
	for _, k := range optKeys {
		p := scaled[k] / sum
		scaled[k] = p
		if p > bestP {
			bestP = p
			winner = k
		}
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}

	normH := 0.0
	if kCount > 1 {
		normH = entropy / math.Log(float64(kCount))
	}
	return scaled, winner, entropy, normH
}

func sortedOptionKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// NewSystemOneHTTPHandler returns an http.HandlerFunc serving POST /v1/systemone compatible with
// `python -m decision_index run --engine http`.
func NewSystemOneHTTPHandler(cli *client.Client, opts EngineOptions) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("invalid body: %v", err), http.StatusBadRequest)
			return
		}
		if err := CheckMediaKeys(raw); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var req SystemOneRequest
		if err := json.Unmarshal(raw, &req); err != nil {
			http.Error(w, fmt.Sprintf("invalid JSON body: %v", err), http.StatusBadRequest)
			return
		}
		resp, err := ExecuteSystemOne(r.Context(), cli, req, opts)
		if err != nil {
			if errors.Is(err, ErrInvalidImage) {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			status, msg := ErrorStatus(err)
			http.Error(w, msg, status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

// upstreamClientError matches the serving image's wrapping of a model-server 4xx ("upstream 422: Failed to fetch
// media"), which images before the fix return as HTTP 502.
var upstreamClientError = regexp.MustCompile(`upstream 4\d\d\b`)

// atMostAlternatives matches the server's upper-limit schema error, capturing the limit.
var atMostAlternatives = regexp.MustCompile(`at most (\d+) alternatives`)

// ErrorStatus maps an ExecuteSystemOne error to the HTTP status and message the adapter answers with. Requests the
// server refuses (bad schema, too many options, an image it can't fetch or decode, context overflow) are the caller's
// to fix: 422 with the upstream message (the Decision Index kit scores 422 as a capacity refusal, not a failure).
// Everything else is 500. Shared by `dgem systemone serve` and the gateway's /v1/systemone route.
func ErrorStatus(err error) (int, string) {
	if errors.Is(err, ErrInvalidImage) || errors.Is(err, ErrUnsupportedMedia) || errors.Is(err, client.ErrUnsupportedImageFormat) ||
		errors.Is(err, ErrInvalidRequest) {
		return http.StatusBadRequest, err.Error()
	}
	if errors.Is(err, ErrUnsupportedQuestion) {
		return http.StatusUnprocessableEntity, err.Error()
	}
	if errors.Is(err, client.ErrImageFetch) {
		return http.StatusUnprocessableEntity, err.Error() // an image URL that can't be fetched or isn't an image
	}
	msg := err.Error()
	var he *client.HTTPError
	if errors.As(err, &he) {
		switch {
		case he.StatusCode == http.StatusBadRequest || he.StatusCode == http.StatusUnprocessableEntity:
			// e.g. "question 'x': at most 26 alternatives" for a score question with more levels (#120). Only the
			// upper limit gets the hint; "needs at least two alternatives" is not a capacity problem.
			if m := atMostAlternatives.FindStringSubmatch(he.Body); m != nil && !strings.Contains(msg, "options per choice") {
				msg += fmt.Sprintf(" (at most %s options per choice or levels per score)", m[1])
			}
			return http.StatusUnprocessableEntity, msg
		case he.StatusCode == http.StatusBadGateway && upstreamClientError.MatchString(he.Body):
			// the model server refused the request itself, e.g. an image URL it can't fetch
			return http.StatusUnprocessableEntity, msg
		}
	}
	for _, marker := range []string{"HTTP 422", "options per choice", "the canvas holds at most", "too many tokens",
		"maximum context length", "context window"} {
		if strings.Contains(msg, marker) {
			return http.StatusUnprocessableEntity, msg
		}
	}
	return http.StatusInternalServerError, msg
}

// balancedBrackets splits keys into ceil(K/size) brackets of near-equal size (differing by at most one).
// Fixed-size chunking left a 1-option tail whenever K % size == 1 (e.g. 41, 61, 101 options), which the
// server rejects ("needs at least two alternatives").
func balancedBrackets(keys []string, size int) [][]string {
	if size < 2 {
		size = 2
	}
	n := len(keys)
	if n == 0 {
		return nil
	}
	if n <= size {
		return [][]string{keys}
	}
	nb := (n-1)/size + 1
	base, extra := n/nb, n%nb
	out := make([][]string, 0, nb)
	i := 0
	for b := 0; b < nb; b++ {
		sz := base
		if b < extra {
			sz++
		}
		out = append(out, keys[i:i+sz])
		i += sz
	}
	return out
}

// fuseBracketProbabilities combines Round-1 bracket readouts and the Round-2 final into one distribution over all K
// options, using the model's own outputs instead of a fixed finalist/non-finalist split (previously 92% / 8%, which
// capped every wide-option confidence at ~0.92):
//
//	P(b)          = Σ_{f finalist in b} p_final(f)          (the final round is the evidence for which bracket wins)
//	non-final o∈b = P(b) · p_b(o)                           (Round-1 share inside that bracket)
//	finalist  f   = p_final(f) · (1 − Σ non-finalist mass)
//
// A uniform bracket prior (w_b = |b|/K) was rejected: Round-1 distributions sum to 1 even in brackets with no good
// option, so it hands most of the mass to brackets the final round already rejected.
// Every option gets a strictly positive probability, the result sums to 1, and the final-round winner stays the
// argmax (a non-finalist is capped just below it before renormalising, which preserves order).
func fuseBracketProbabilities(brackets [][]string, round1 []map[string]float64, finalist map[string]bool, final map[string]float64) map[string]float64 {
	const eps = 1e-6
	out := make(map[string]float64)
	finTot := 0.0
	for o := range finalist {
		finTot += math.Max(eps, final[o])
	}
	if finTot == 0 {
		finTot = 1
	}
	tail := 0.0
	nonFin := make(map[string]float64)
	for bIdx, b := range brackets {
		mass, pb := 0.0, 0.0
		for _, o := range b {
			mass += math.Max(eps, round1[bIdx][o])
			if finalist[o] {
				pb += math.Max(eps, final[o]) / finTot
			}
		}
		pb = math.Max(eps, pb)
		for _, o := range b {
			if !finalist[o] {
				v := pb * math.Max(eps, round1[bIdx][o]) / mass
				nonFin[o] = v
				tail += v
			}
		}
	}
	tail = math.Min(1-eps, tail)
	winner, best := "", -1.0
	for o := range finalist {
		v := (1 - tail) * math.Max(eps, final[o]) / finTot
		out[o] = v
		if v > best || (v == best && o < winner) {
			winner, best = o, v
		}
	}
	for o, v := range nonFin {
		if winner != "" && v >= best {
			v = best * 0.999
		}
		out[o] = math.Max(eps*eps, v)
	}
	sum := 0.0
	for _, v := range out {
		sum += v
	}
	for o := range out {
		out[o] /= sum
	}
	return out
}

// noulCriteria maps a noul question's criteria to the structured server's boolean form ({"true", "false"}),
// accepting "yes"/"no" keys as synonyms. Returns nil when neither description is present.
func noulCriteria(c map[string]string) map[string]string {
	if len(c) == 0 {
		return nil
	}
	out := map[string]string{}
	for k, v := range c {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(k)) {
		case "true", "yes":
			out["true"] = v
		case "false", "no":
			out["false"] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// noulAsChoice renders a yes/no question as a 2-option choice ("yes"/"no") whose descriptions are the question's
// true/false criteria. Opt-in (NoulMode "choice"): on RAGTruth train (dev) it raised hallucination F1 0.441 -> 0.752
// on a held-out confirm split, but lowered accuracy slightly on other yes/no families (JevBench, calibration suite,
// typed-decisions), so it is not the default. The choice answer is mapped back to a noul answer.
func noulAsChoice(qKey string, qSpec SystemOneQuestion) map[string]any {
	yes, no := "yes", "no"
	if crit := noulCriteria(qSpec.Criteria); crit != nil {
		if v := crit["true"]; v != "" {
			yes = v
		}
		if v := crit["false"]; v != "" {
			no = v
		}
	}
	return map[string]any{
		"id":           qKey,
		"type":         "choice",
		"instructions": FormatInstructions(qSpec.Instructions),
		"options":      []map[string]string{{"name": "yes", "description": yes}, {"name": "no", "description": no}},
	}
}

// catchAllPatterns mark an option as "none of the others apply" by its description or name (generic wording only).
var catchAllPatterns = []string{
	"none of the listed", "none of the above", "none of these", "none of the options", "none apply",
	"out of scope", "out-of-scope", "not listed", "no matching", "not applicable",
}

// catchAllKeys returns the options whose description (or name) says "none of the others apply". If more than 3
// options match, the heuristic is not trusted and nothing is returned.
func catchAllKeys(criteria map[string]string) []string {
	var out []string
	for _, k := range sortedOptionKeys(criteria) {
		text := strings.ToLower(strings.TrimSpace(criteria[k]) + " " + strings.ReplaceAll(k, "_", " "))
		for _, p := range catchAllPatterns {
			if strings.Contains(text, p) {
				out = append(out, k)
				break
			}
		}
	}
	if len(out) > 3 {
		return nil
	}
	return out
}

// FitsServingImage reports whether the serving image's own /v1/systemone can answer req in one read exactly as sent:
// at most MaxSlotsPerPass questions (the server keeps its readable "lines" answer format up to 10), every choice with
// 2..MaxOptionsPerSlot options and every score with 2..MaxOptionsPerSlot levels, and only known question types.
// Requests that fit can be forwarded unchanged; the rest need the adapter (batching, brackets, one-option answers).
// Call after MergeMedia.
func FitsServingImage(req SystemOneRequest) bool {
	if len(req.Questions) == 0 || len(req.Questions) > MaxSlotsPerPass || len(req.Media) > 0 {
		return false
	}
	for k, q := range req.Questions {
		kind, err := questionKind(k, q)
		if err != nil {
			return false
		}
		switch kind {
		case kindChoice:
			if len(q.Criteria) < 2 || len(q.Criteria) > MaxOptionsPerSlot {
				return false
			}
		case kindScore:
			if len(q.Levels) < 2 || len(q.Levels) > MaxOptionsPerSlot {
				return false
			}
		case kindNoul:
		default:
			return false
		}
		// The serving image's Jev parser takes "type" literally (noul, choice, score): an empty type or a synonym
		// (bool, boolean) is resolved only here.
		if q.Type != kind {
			return false
		}
	}
	return true
}
