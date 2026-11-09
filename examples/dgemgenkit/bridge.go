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
// v1.14.0's systemonex (System One) plugin, supporting multi-stage conditional
// DAG policies (depends_on / ask_if), post-hoc temperature calibration,
// calibrated threshold helpers, per-slot Shannon entropy H, normalized
// Hesitation % (H / ln K), and request wall-time telemetry.
package dgemgenkit

import (
	"cmp"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	systemonex "github.com/firebase/genkit/go/plugins/systemone/exp"
)

// Default local endpoint and model constants.
const (
	DefaultUpstreamV1URL = "http://127.0.0.1:8080/v1"
	DefaultModelID       = "diffgemma-26b-a4b-it-q4"
	DefaultProvider      = "dgem"
	DefaultTimeout       = 2 * time.Minute
)

// EndpointConfig configures how Genkit's systemonex plugin connects to dgem.
//
// It supports two connection modes seamlessly:
//  1. Direct Gateway / SystemOne mode: when SystemOneURL (env: DGEM_GATEWAY_URL
//     or DGEM_SYSTEMONE_URL, flag: -gateway-url) is set, or when URL points to
//     a gateway / SystemOne endpoint (e.g. http://127.0.0.1:8095, an Apple
//     container host bridge at http://192.168.64.1:8095, or a remote Cloud Run
//     gateway at https://<service>.a.run.app), Genkit's systemonex plugin calls
//     POST /v1/systemone directly over HTTP(S) with appropriate auth headers.
//  2. In-process /v1 adapter mode: when URL points to a raw OpenAI-compatible
//     /v1 endpoint (e.g. local diffgemma on http://127.0.0.1:8080/v1) and
//     SystemOneURL is empty, an in-process /v1/systemone adapter is started
//     automatically in front of it.
type EndpointConfig struct {
	// URL is the upstream /v1 endpoint (env: DGEM_URL) or a direct gateway URL.
	URL string
	// SystemOneURL is an explicit dgem gateway or /v1/systemone endpoint URL
	// (env: DGEM_GATEWAY_URL or DGEM_SYSTEMONE_URL). Takes precedence over URL.
	SystemOneURL string
	// ModelID is the dgem model identifier (env: DGEM_MODEL).
	ModelID string
	// Provider is the Genkit plugin provider prefix (env: DGEM_PROVIDER; default "dgem").
	Provider string
	// Backend selects the gateway backend target via X-DGem-Backend
	// ("vertex_first", "vertex", "cloudrun", or "local"; env: DGEM_BACKEND).
	Backend string
	// AuthToken is an optional Bearer / IAM / OIDC token sent in Authorization
	// (env: DGEM_AUTH_TOKEN).
	AuthToken string
	// GatewayKey is an optional X-DGem-Api-Key for a self-hosted gateway
	// (env: DGEM_GATEWAY_KEY, scoped by DGEM_GATEWAY_URL).
	GatewayKey string
	// ServerKey is an optional X-DGem-Key for a serving image
	// (env: DGEM_SERVER_KEY).
	ServerKey string
	// Timeout is the HTTP request timeout (default 2 minutes).
	Timeout time.Duration
}

// EndpointConfigFromEnv returns an EndpointConfig populated from environment
// variables (DGEM_GATEWAY_URL, DGEM_SYSTEMONE_URL, DGEM_URL, DGEM_MODEL,
// DGEM_PROVIDER, DGEM_BACKEND, DGEM_AUTH_TOKEN, DGEM_GATEWAY_KEY, DGEM_SERVER_KEY)
// with sensible local defaults.
func EndpointConfigFromEnv() EndpointConfig {
	return EndpointConfig{
		URL:          firstNonEmpty(os.Getenv("DGEM_URL"), DefaultUpstreamV1URL),
		SystemOneURL: firstNonEmpty(os.Getenv("DGEM_SYSTEMONE_URL"), os.Getenv("DGEM_GATEWAY_URL")),
		ModelID:      firstNonEmpty(os.Getenv("DGEM_MODEL"), DefaultModelID),
		Provider:     firstNonEmpty(os.Getenv("DGEM_PROVIDER"), DefaultProvider),
		Backend:      strings.TrimSpace(os.Getenv("DGEM_BACKEND")),
		AuthToken:    strings.TrimSpace(os.Getenv("DGEM_AUTH_TOKEN")),
		ServerKey:    client.ServerKeyFromEnv(),
		Timeout:      DefaultTimeout,
	}
}

// BindEndpointFlags registers standard CLI flags (-url, -gateway-url, -model,
// -backend) onto fs, bound to the fields of cfg.
func BindEndpointFlags(fs *flag.FlagSet, cfg *EndpointConfig) {
	cfg.URL = cmp.Or(cfg.URL, DefaultUpstreamV1URL)
	cfg.ModelID = cmp.Or(cfg.ModelID, DefaultModelID)
	cfg.Provider = cmp.Or(cfg.Provider, DefaultProvider)
	fs.StringVar(&cfg.URL, "url", cfg.URL, "Upstream diffgemma /v1 URL or direct dgem gateway URL (env: DGEM_URL)")
	fs.StringVar(&cfg.SystemOneURL, "gateway-url", cfg.SystemOneURL, "Direct dgem gateway or /v1/systemone endpoint URL, e.g. http://127.0.0.1:8095 or https://<gateway>.a.run.app (env: DGEM_GATEWAY_URL or DGEM_SYSTEMONE_URL)")
	fs.StringVar(&cfg.ModelID, "model", cfg.ModelID, "Upstream dgem model ID (env: DGEM_MODEL)")
	fs.StringVar(&cfg.Backend, "backend", cfg.Backend, "Optional X-DGem-Backend header for gateway routing: vertex_first, vertex, cloudrun, or local (env: DGEM_BACKEND)")
}

// GenkitModelName returns the full "<provider>/<model>" identifier for Genkit.
func (c EndpointConfig) GenkitModelName() string {
	prov := firstNonEmpty(c.Provider, DefaultProvider)
	mod := firstNonEmpty(c.ModelID, DefaultModelID)
	return prov + "/" + mod
}

// IsDirectSystemOne reports whether c will connect directly to a remote or
// gateway POST /v1/systemone endpoint without starting an in-process /v1 adapter.
func (c EndpointConfig) IsDirectSystemOne() bool {
	if strings.TrimSpace(c.SystemOneURL) != "" {
		return true
	}
	return !isRawV1ChatURL(c.URL)
}

// DescribeTarget returns a concise human-readable description of the active
// endpoint and connection mode.
func (c EndpointConfig) DescribeTarget() string {
	if c.IsDirectSystemOne() {
		base, path := resolveSystemOneBaseAndPath(firstNonEmpty(c.SystemOneURL, c.URL))
		if path == "/" {
			return fmt.Sprintf("direct(%s)", base)
		}
		if path == "" {
			path = "/v1/systemone"
		}
		return fmt.Sprintf("direct(%s%s)", base, path)
	}
	return fmt.Sprintf("in-process-adapter(%s)", firstNonEmpty(c.URL, DefaultUpstreamV1URL))
}

func isRawV1ChatURL(raw string) bool {
	u := strings.TrimRight(strings.TrimSpace(raw), "/")
	if u == "" {
		return true
	}
	lu := strings.ToLower(u)
	if strings.HasSuffix(lu, "/v1/systemone") ||
		strings.HasSuffix(lu, ":rawpredict") ||
		strings.HasSuffix(lu, ":predict") {
		return false
	}
	return strings.HasSuffix(lu, "/v1") ||
		strings.HasSuffix(lu, "/v1/chat/completions") ||
		strings.HasSuffix(lu, "/chat/completions")
}

func resolveSystemOneBaseAndPath(raw string) (baseURL, path string) {
	u := strings.TrimRight(strings.TrimSpace(raw), "/")
	lu := strings.ToLower(u)
	switch {
	case strings.HasSuffix(lu, ":rawpredict") || strings.HasSuffix(lu, ":streamrawpredict") || strings.HasSuffix(lu, ":predict"):
		return u, "/"
	case strings.HasSuffix(lu, "/v1/systemone"):
		return u[:len(u)-len("/v1/systemone")], "/v1/systemone"
	default:
		return u, ""
	}
}

// NewSystemOnePlugin creates a Genkit systemonex.SystemOne plugin from cfg and
// returns a cleanup function (which stops the in-process adapter if one was
// started, or is a no-op when connecting directly to a gateway/SystemOne URL).
func NewSystemOnePlugin(cfg EndpointConfig) (*systemonex.SystemOne, func()) {
	provider := firstNonEmpty(cfg.Provider, DefaultProvider)
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}

	if cfg.IsDirectSystemOne() {
		target := firstNonEmpty(cfg.SystemOneURL, cfg.URL)
		baseURL, path := resolveSystemOneBaseAndPath(target)

		headers := make(http.Header)
		gwKey := strings.TrimSpace(cfg.GatewayKey)
		if gwKey == "" {
			if strings.TrimSpace(cfg.SystemOneURL) != "" {
				gwKey = client.GatewayKeyForRemote(baseURL)
			} else {
				gwKey = client.GatewayKeyFor(baseURL)
			}
		}
		if gwKey != "" {
			headers.Set(client.GatewayKeyHeader, gwKey)
		}
		if srvKey := firstNonEmpty(cfg.ServerKey, client.ServerKeyFromEnv()); srvKey != "" {
			headers.Set(client.ServerKeyHeader, srvKey)
		}
		if b := strings.TrimSpace(cfg.Backend); b != "" {
			headers.Set("X-DGem-Backend", b)
		}

		apiKey := strings.TrimSpace(cfg.AuthToken)
		if strings.HasPrefix(strings.ToLower(apiKey), "bearer ") {
			apiKey = strings.TrimSpace(apiKey[len("bearer "):])
		}

		plugin := &systemonex.SystemOne{
			Provider:   provider,
			BaseURL:    baseURL,
			Path:       path,
			APIKey:     apiKey,
			HTTPClient: &http.Client{Timeout: timeout, CheckRedirect: client.StripKeysOnCrossOriginRedirect},
		}
		if len(headers) > 0 {
			plugin.Headers = headers
		}
		return plugin, func() {}
	}

	adapterSrv := StartInProcessSystemOneAdapterWithConfig(cfg)
	return &systemonex.SystemOne{
		Provider: provider,
		BaseURL:  adapterSrv.URL,
	}, adapterSrv.Close
}

// StartInProcessSystemOneAdapterWithConfig starts an in-process HTTP server
// exposing POST /v1/systemone backed by cfg.URL (/v1/chat/completions).
func StartInProcessSystemOneAdapterWithConfig(cfg EndpointConfig) *httptest.Server {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	cli := client.NewClient(firstNonEmpty(cfg.URL, DefaultUpstreamV1URL), firstNonEmpty(cfg.ModelID, DefaultModelID), timeout)
	if cfg.AuthToken != "" {
		cli.WithAuthToken(cfg.AuthToken)
	}
	if cfg.ServerKey != "" {
		cli.ServerKey = cfg.ServerKey
	}
	opts := decisionindex.DefaultEngineOptions()
	mux := http.NewServeMux()
	mux.Handle("/v1/systemone", decisionindex.NewSystemOneHTTPHandler(cli, opts))
	return httptest.NewServer(mux)
}

func firstNonEmpty(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return strings.TrimSpace(a)
	}
	return b
}

// TemplateDecision holds the Genkit systemonex answers for a rendered .json.tmpl
// policy along with unscaled RawAnswers, conditional DAG telemetry (Passes,
// Skipped), per-slot Shannon entropy H (in nats), normalized Hesitation
// (H / ln K in [0, 1]), post-hoc temperature calibration helpers, and wall-time
// latency.
type TemplateDecision struct {
	TemplatePath string
	Model        string
	// Answers holds the active per-slot answers (after schema temperature or
	// ApplyTemperature if configured; otherwise identical to RawAnswers).
	Answers map[string]systemonex.Answer
	// RawAnswers preserves the unscaled Stage-1 answers (T = 1.0) so
	// RawHesitation and repeated ApplyTemperature / CalibratedProb calls never
	// double-scale probabilities.
	RawAnswers  map[string]systemonex.Answer
	OptionCount map[string]int
	// Passes records how many systemonex forward passes ran across conditional
	// DAG stages (1 for single-pass policies or when Stage-2 questions are all
	// skipped by ask_if; >=2 when dependent stages execute).
	Passes int
	// Skipped lists question IDs whose ask_if condition evaluated false (or
	// whose upstream dependency was skipped), in template declaration order.
	Skipped []string
	// Temperature records the scalar post-hoc temperature applied to Answers
	// (1.0 when unscaled).
	Temperature float64
	WallTime    time.Duration
	WallTimeMs  int64
}

// IsSkipped reports whether slot was skipped by an ask_if condition.
func (d *TemplateDecision) IsSkipped(slot string) bool {
	return slices.Contains(d.Skipped, slot)
}

// EntropyNats returns the Shannon entropy H = -sum p_k ln(p_k) in nats for
// slotID on the active Answers distribution.
func (d *TemplateDecision) EntropyNats(slotID string) float64 {
	ans, ok := d.Answers[slotID]
	if !ok {
		return 0
	}
	return answerEntropyNats(ans)
}

// Hesitation returns normalized entropy H / ln(K) in [0, 1] for slotID on the
// active Answers distribution (< 0.16 is Clear, 0.16-0.50 is Somewhat unsure,
// > 0.50 is Very unsure).
func (d *TemplateDecision) Hesitation(slotID string) float64 {
	ans, ok := d.Answers[slotID]
	if !ok {
		return 0
	}
	return answerHesitation(ans, d.OptionCount[slotID])
}

// MaxHesitation returns the highest normalized hesitation across all answered slots.
func (d *TemplateDecision) MaxHesitation() float64 {
	var maxH float64
	for slotID := range d.Answers {
		if h := d.Hesitation(slotID); h > maxH {
			maxH = h
		}
	}
	return maxH
}

// RawEntropyNats returns the unscaled Stage-1 Shannon entropy H (T = 1.0) in
// nats for slotID, unaffected by ApplyTemperature or policy temperature.
func (d *TemplateDecision) RawEntropyNats(slotID string) float64 {
	ans, ok := d.rawAnswer(slotID)
	if !ok {
		return 0
	}
	return answerEntropyNats(ans)
}

// RawHesitation returns the unscaled Stage-1 normalized entropy H_raw / ln(K)
// in [0, 1] for slotID, preserved even after ApplyTemperature(t) rescales Answers.
func (d *TemplateDecision) RawHesitation(slotID string) float64 {
	ans, ok := d.rawAnswer(slotID)
	if !ok {
		return 0
	}
	return answerHesitation(ans, d.OptionCount[slotID])
}

// ApplyTemperature rescales every answered slot in d.Answers from the unscaled
// Stage-1 distribution (RawAnswers) by temperature > 0:
//
//	p_k(T) = p_k^(1/T) / sum_j p_j^(1/T)
//
// Argmax choices remain invariant, while Confidence, Noul probability, expected
// Score, EntropyNats, and Hesitation reflect the calibrated distribution.
// Calling ApplyTemperature multiple times with different temperatures is
// idempotent because scaling always starts from RawAnswers.
func (d *TemplateDecision) ApplyTemperature(temperature float64) {
	if d == nil {
		return
	}
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		temperature = 1.0
	}
	d.ApplyPolicyTemperature(client.Temperature{All: temperature})
	d.Temperature = temperature
}

// ApplyPolicyTemperature rescales d.Answers from RawAnswers using a policy
// client.Temperature (either uniform All or per-question-type ByType).
func (d *TemplateDecision) ApplyPolicyTemperature(policyTemp client.Temperature) {
	if d == nil {
		return
	}
	if len(d.RawAnswers) == 0 && len(d.Answers) > 0 {
		d.RawAnswers = cloneAnswersMap(d.Answers)
	}
	scaled := make(map[string]systemonex.Answer, len(d.RawAnswers))
	for slotID, raw := range d.RawAnswers {
		slotTemp := policyTemp.For(raw.Type)
		scaled[slotID] = scaleAnswerTemperature(raw, slotTemp)
	}
	d.Answers = scaled
	if policyTemp.All > 0 {
		d.Temperature = policyTemp.All
	} else if !policyTemp.IsSet() {
		d.Temperature = 1.0
	}
}

// CalibratedProb returns the post-hoc temperature-scaled probability p_T(option)
// in [0, 1] for option on slot without mutating d.Answers.
//
// For boolean/noul slots, option accepts "yes"/"true" or "no"/"false". For
// choice slots, option is the option name. For score slots, option accepts
// either the 0-indexed level key ("0", "1", ...) or the rubric legend label.
func (d *TemplateDecision) CalibratedProb(slot, option string, temperature float64) float64 {
	raw, ok := d.rawAnswer(slot)
	if !ok {
		return 0
	}
	scaled := scaleAnswerTemperature(raw, temperature)
	switch scaled.Type {
	case "noul":
		switch strings.ToLower(strings.TrimSpace(option)) {
		case "yes", "true", "1":
			return scaled.Probability
		case "no", "false", "0":
			return 1.0 - scaled.Probability
		default:
			return 0
		}
	case "score":
		if p, exists := scaled.Probabilities[option]; exists {
			return p
		}
		for idxKey, label := range scaled.Legend {
			if label == option {
				return scaled.Probabilities[idxKey]
			}
		}
		return 0
	default:
		return scaled.Probabilities[option]
	}
}

// ExceedsThreshold reports whether the temperature-scaled probability
// CalibratedProb(slot, option, temperature) is at least threshold (tau).
func (d *TemplateDecision) ExceedsThreshold(slot, option string, temperature, threshold float64) bool {
	return d.CalibratedProb(slot, option, temperature) >= threshold
}

// CalibratedHesitation returns the normalized entropy H(p_T) / ln(K) in [0, 1]
// computed AFTER scaling slot's raw Stage-1 distribution by temperature, while
// leaving RawHesitation(slot) and d.Answers untouched.
func (d *TemplateDecision) CalibratedHesitation(slot string, temperature float64) float64 {
	raw, ok := d.rawAnswer(slot)
	if !ok {
		return 0
	}
	scaled := scaleAnswerTemperature(raw, temperature)
	return answerHesitation(scaled, d.OptionCount[slot])
}

func (d *TemplateDecision) rawAnswer(slotID string) (systemonex.Answer, bool) {
	if d == nil {
		return systemonex.Answer{}, false
	}
	if len(d.RawAnswers) > 0 {
		ans, ok := d.RawAnswers[slotID]
		return ans, ok
	}
	ans, ok := d.Answers[slotID]
	return ans, ok
}

func answerEntropyNats(ans systemonex.Answer) float64 {
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

func answerHesitation(ans systemonex.Answer, optionCount int) float64 {
	k := optionCount
	if k < 2 {
		if ans.Type != "noul" && len(ans.Probabilities) > 2 {
			k = len(ans.Probabilities)
		} else {
			k = 2
		}
	}
	h := answerEntropyNats(ans)
	maxH := math.Log(float64(k))
	if maxH <= 0 {
		return 0
	}
	return math.Min(1.0, h/maxH)
}

func scaleAnswerTemperature(ans systemonex.Answer, temperature float64) systemonex.Answer {
	out := cloneAnswer(ans)
	if temperature <= 0 || temperature == 1.0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return out
	}

	const probFloor = 1e-12
	if out.Type == "noul" {
		pYes := math.Min(math.Max(out.Probability, probFloor), 1.0-probFloor)
		pNo := 1.0 - pYes
		lYes := math.Log(pYes) / temperature
		lNo := math.Log(pNo) / temperature
		mx := math.Max(lYes, lNo)
		eYes := math.Exp(lYes - mx)
		eNo := math.Exp(lNo - mx)
		scaledYes := eYes / (eYes + eNo)
		scaledNo := 1.0 - scaledYes
		out.Probability = scaledYes
		out.Confidence = math.Max(scaledYes, scaledNo)
		if len(out.Probabilities) > 0 {
			for k := range out.Probabilities {
				switch strings.ToLower(strings.TrimSpace(k)) {
				case "yes", "true":
					out.Probabilities[k] = scaledYes
				case "no", "false":
					out.Probabilities[k] = scaledNo
				}
			}
		}
		return out
	}

	if len(out.Probabilities) == 0 {
		return out
	}
	keys := make([]string, 0, len(out.Probabilities))
	for k := range out.Probabilities {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	logits := make([]float64, len(keys))
	mx := math.Inf(-1)
	for i, k := range keys {
		logits[i] = math.Log(math.Max(out.Probabilities[k], probFloor)) / temperature
		if logits[i] > mx {
			mx = logits[i]
		}
	}
	var z float64
	for i := range logits {
		logits[i] = math.Exp(logits[i] - mx)
		z += logits[i]
	}
	scaledProbs := make(map[string]float64, len(keys))
	topProb := 0.0
	for i, k := range keys {
		p := logits[i] / z
		scaledProbs[k] = p
		if p > topProb {
			topProb = p
		}
	}
	out.Probabilities = scaledProbs
	if out.Type == "choice" && out.Choice != "" {
		if p, ok := scaledProbs[out.Choice]; ok {
			out.Confidence = p
		} else {
			out.Confidence = topProb
		}
	} else {
		out.Confidence = topProb
	}
	if out.Type == "score" {
		var weighted float64
		allNumeric := true
		for k, p := range scaledProbs {
			lvl, err := strconv.ParseFloat(k, 64)
			if err != nil {
				allNumeric = false
				break
			}
			weighted += lvl * p
		}
		if allNumeric {
			out.Score = weighted
		}
	}
	return out
}

func cloneAnswer(ans systemonex.Answer) systemonex.Answer {
	out := ans
	if ans.Probabilities != nil {
		out.Probabilities = maps.Clone(ans.Probabilities)
	}
	if ans.Legend != nil {
		out.Legend = maps.Clone(ans.Legend)
	}
	return out
}

func cloneAnswersMap(src map[string]systemonex.Answer) map[string]systemonex.Answer {
	if src == nil {
		return nil
	}
	dst := make(map[string]systemonex.Answer, len(src))
	for k, v := range src {
		dst[k] = cloneAnswer(v)
	}
	return dst
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
	SeedToken    string                   `json:"seed_token,omitempty"`
	Isolate      string                   `json:"isolate,omitempty"`
}

type parsedTemplateQuestion struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Instructions string              `json:"instructions"`
	Options      json.RawMessage     `json:"options,omitempty"`
	Levels       []string            `json:"levels,omitempty"`
	DependsOn    []string            `json:"depends_on,omitempty"`
	AskIf        map[string][]string `json:"ask_if,omitempty"`
}

type parsedChoiceOption struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// DecideWithTemplate renders a dgem .json.tmpl file with vars, schedules its
// questions across topological dependency stages (depends_on / ask_if), and
// executes each active stage via Genkit's systemonex plugin, recording wall-time
// latency, Passes, Skipped, RawAnswers, and optional schema temperature scaling.
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
		return nil, fmt.Errorf("render template %q: %w", tmplPath, err)
	}
	schemaStr, stateStr, err := template.ParseStructuredPayload(rendered, vars)
	if err != nil {
		return nil, fmt.Errorf("parse template payload %q: %w", tmplPath, err)
	}
	schemaTemp, err := client.ParseSchemaTemperature(schemaStr)
	if err != nil {
		return nil, fmt.Errorf("parse template temperature %q: %w", tmplPath, err)
	}

	var parsed parsedTemplateSchema
	if err := json.Unmarshal([]byte(schemaStr), &parsed); err != nil {
		return nil, fmt.Errorf("unmarshal template schema %q: %w", tmplPath, err)
	}

	allQuestions := make(map[string]systemonex.Question, len(parsed.Questions))
	optionCount := make(map[string]int, len(parsed.Questions))
	validAnswers := make(map[string][]string, len(parsed.Questions))
	byID := make(map[string]parsedTemplateQuestion, len(parsed.Questions))

	for i := range parsed.Questions {
		q := &parsed.Questions[i]
		if q.ID == "" {
			return nil, fmt.Errorf("question at index %d has empty id", i)
		}
		if _, dup := byID[q.ID]; dup {
			return nil, fmt.Errorf("duplicate question id %q", q.ID)
		}
		q.DependsOn = mergeDependsOn(q.DependsOn, q.AskIf)

		switch strings.ToLower(strings.TrimSpace(q.Type)) {
		case "boolean", "bool", "noul":
			q.Type = "noul"
			allQuestions[q.ID] = systemonex.NoulQuestion{
				Instructions: q.Instructions,
			}
			optionCount[q.ID] = 2
			validAnswers[q.ID] = []string{"yes", "no"}

		case "choice":
			q.Type = "choice"
			opts, optErr := parseChoiceOptions(q.Options)
			if optErr != nil {
				return nil, fmt.Errorf("question %q options: %w", q.ID, optErr)
			}
			allQuestions[q.ID] = systemonex.ChoiceQuestion{
				Instructions: q.Instructions,
				Options:      opts,
			}
			optionCount[q.ID] = len(opts)
			names := make([]string, len(opts))
			for j, opt := range opts {
				names[j] = opt.Name
			}
			validAnswers[q.ID] = names

		case "score":
			q.Type = "score"
			allQuestions[q.ID] = systemonex.ScoreQuestion{
				Instructions: q.Instructions,
				Levels:       q.Levels,
			}
			optionCount[q.ID] = len(q.Levels)
			validAnswers[q.ID] = slices.Clone(q.Levels)

		default:
			return nil, fmt.Errorf("unsupported question type %q on slot %q", q.Type, q.ID)
		}
		byID[q.ID] = *q
	}

	for _, q := range parsed.Questions {
		for _, dep := range q.DependsOn {
			if _, exists := byID[dep]; !exists || dep == q.ID {
				return nil, fmt.Errorf("question %q depends on unknown question %q", q.ID, dep)
			}
		}
		for dep, vals := range q.AskIf {
			if len(vals) == 0 {
				return nil, fmt.Errorf("question %q ask_if for %q must be non-empty", q.ID, dep)
			}
			allowed := validAnswers[dep]
			for _, v := range vals {
				if !slices.Contains(allowed, v) {
					return nil, fmt.Errorf("question %q ask_if value %q for %q must be among %v", q.ID, v, dep, allowed)
				}
			}
		}
	}

	levels, err := scheduleQuestions(parsed.Questions)
	if err != nil {
		return nil, err
	}

	var stateObj any
	if err := json.Unmarshal([]byte(stateStr), &stateObj); err != nil {
		stateObj = stateStr
	}

	extraConfig := make(map[string]any)
	if st := strings.TrimSpace(parsed.SeedToken); st != "" {
		extraConfig["seed_token"] = st
	}
	if iso := strings.TrimSpace(parsed.Isolate); iso != "" {
		extraConfig["isolate"] = iso
	}

	rawAnswers := make(map[string]systemonex.Answer, len(parsed.Questions))
	skippedSet := make(map[string]bool)
	var skipped []string
	priorNames := make(map[string]string)
	var priorLines []string
	var modelUsed string
	var passes int

	for stageIdx, level := range levels {
		stageQuestions := make(map[string]systemonex.Question, len(level))
		var askedInStage []parsedTemplateQuestion
		for _, q := range level {
			if shouldSkipQuestion(q, skippedSet, priorNames) {
				skippedSet[q.ID] = true
				skipped = append(skipped, q.ID)
				continue
			}
			stageQuestions[q.ID] = allQuestions[q.ID]
			askedInStage = append(askedInStage, q)
		}
		if len(stageQuestions) == 0 {
			continue
		}

		stageState := buildConditionedState(stateObj, stageIdx, priorNames, priorLines)
		opts := []ai.GenerateOption{
			ai.WithModelName(modelName),
			ai.WithOutputSchema(systemonex.Schema(stageQuestions)),
			ai.WithPromptParts(ai.NewDataPart(stageState)),
		}
		if parsed.Instructions != "" {
			opts = append(opts, ai.WithSystem(parsed.Instructions))
		}
		if len(extraConfig) > 0 {
			opts = append(opts, ai.WithConfig(systemonex.Config{Extra: maps.Clone(extraConfig)}))
		}

		stageAns, resp, genErr := genkit.GenerateData[map[string]systemonex.Answer](ctx, g, opts...)
		if genErr != nil {
			return nil, fmt.Errorf("genkit GenerateData (%s stage %d): %w", tmplPath, stageIdx, genErr)
		}
		passes++
		if infoModel := systemonex.ResponseInfo(resp).Model; infoModel != "" {
			modelUsed = infoModel
		}
		for _, q := range askedInStage {
			ans := (*stageAns)[q.ID]
			if q.Type == "score" && len(q.Levels) > 0 && len(ans.Legend) == 0 {
				ans.Legend = make(map[string]string, len(q.Levels))
				for idx, lvl := range q.Levels {
					ans.Legend[strconv.Itoa(idx)] = lvl
				}
			}
			rawAnswers[q.ID] = cloneAnswer(ans)
			lbl := answerLabelName(q, ans)
			priorNames[q.ID] = lbl
			priorLines = append(priorLines, fmt.Sprintf("%s: %s", q.ID, lbl))
		}
	}

	elapsed := time.Since(start)
	dec := &TemplateDecision{
		TemplatePath: tmplPath,
		Model:        modelUsed,
		Answers:      cloneAnswersMap(rawAnswers),
		RawAnswers:   cloneAnswersMap(rawAnswers),
		OptionCount:  optionCount,
		Passes:       passes,
		Skipped:      skipped,
		Temperature:  1.0,
		WallTime:     elapsed,
		WallTimeMs:   elapsed.Milliseconds(),
	}
	if schemaTemp.IsSet() {
		dec.ApplyPolicyTemperature(schemaTemp)
	}
	return dec, nil
}

func mergeDependsOn(deps []string, askIf map[string][]string) []string {
	var merged []string
	seen := make(map[string]bool, len(deps)+len(askIf))
	for _, d := range deps {
		if !seen[d] {
			seen[d] = true
			merged = append(merged, d)
		}
	}
	if len(askIf) > 0 {
		keys := make([]string, 0, len(askIf))
		for k := range askIf {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			if !seen[k] {
				seen[k] = true
				merged = append(merged, k)
			}
		}
	}
	return merged
}

func scheduleQuestions(questions []parsedTemplateQuestion) ([][]parsedTemplateQuestion, error) {
	pending := slices.Clone(questions)
	done := make(map[string]bool, len(questions))
	var levels [][]parsedTemplateQuestion
	for len(pending) > 0 {
		var level []parsedTemplateQuestion
		var nextPending []parsedTemplateQuestion
		for _, q := range pending {
			ready := true
			for _, dep := range q.DependsOn {
				if !done[dep] {
					ready = false
					break
				}
			}
			if ready {
				level = append(level, q)
			} else {
				nextPending = append(nextPending, q)
			}
		}
		if len(level) == 0 {
			ids := make([]string, len(pending))
			for i, q := range pending {
				ids[i] = q.ID
			}
			return nil, fmt.Errorf("dependency cycle among questions %v", ids)
		}
		for _, q := range level {
			done[q.ID] = true
		}
		levels = append(levels, level)
		pending = nextPending
	}
	return levels, nil
}

func shouldSkipQuestion(q parsedTemplateQuestion, skippedSet map[string]bool, priorNames map[string]string) bool {
	for _, dep := range q.DependsOn {
		if skippedSet[dep] {
			return true
		}
	}
	for dep, allowed := range q.AskIf {
		if !slices.Contains(allowed, priorNames[dep]) {
			return true
		}
	}
	return false
}

func answerLabelName(q parsedTemplateQuestion, ans systemonex.Answer) string {
	switch q.Type {
	case "noul":
		if ans.Probability >= 0.5 {
			return "yes"
		}
		return "no"
	case "score":
		if len(q.Levels) == 0 {
			return ""
		}
		idx := int(math.Round(ans.Score))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(q.Levels) {
			idx = len(q.Levels) - 1
		}
		if ans.Legend != nil {
			if lbl, ok := ans.Legend[strconv.Itoa(idx)]; ok && lbl != "" {
				return lbl
			}
		}
		return q.Levels[idx]
	default:
		return ans.Choice
	}
}

func buildConditionedState(baseState any, stageIdx int, priorNames map[string]string, priorLines []string) any {
	if stageIdx == 0 || len(priorNames) == 0 {
		return baseState
	}
	switch v := baseState.(type) {
	case map[string]any:
		cloned := maps.Clone(v)
		cloned["prior_answers"] = maps.Clone(priorNames)
		return cloned
	case string:
		return v + "\n\nAnswers so far:\n" + strings.Join(priorLines, "\n")
	default:
		return map[string]any{
			"state":         baseState,
			"prior_answers": maps.Clone(priorNames),
		}
	}
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
