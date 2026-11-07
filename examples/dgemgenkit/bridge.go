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
	"flag"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
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
	if cfg.URL == "" {
		cfg.URL = DefaultUpstreamV1URL
	}
	if cfg.ModelID == "" {
		cfg.ModelID = DefaultModelID
	}
	if cfg.Provider == "" {
		cfg.Provider = DefaultProvider
	}
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

// IsDirectSystemOne reports whether cfg will connect directly to a remote or
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
