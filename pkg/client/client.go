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
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// Client communicates with the local or remote diffgemma server.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
	Model      string
	AuthToken  string
	// MaxRetries is the number of retries for HTTP 429/503 responses (0 = no retry). Retries use
	// exponential backoff with jitter and are reported in RequestStats.Retries.
	MaxRetries int
	// ServerKey is sent as X-DGem-Key, the serving image's service-to-service secret (DGEM_SERVER_KEY on the
	// server). Empty sends no header. NewClient fills it from the DGEM_SERVER_KEY environment variable.
	ServerKey string
}

// ServerKeyHeader is the header carrying the serving image's service-to-service secret.
const ServerKeyHeader = "X-DGem-Key"

// ServerKeyFromEnv returns the key a caller sends: DGEM_SERVER_KEY, or its first entry when it holds a
// comma-separated rotation list (servers accept every listed key; callers send one).
func ServerKeyFromEnv() string {
	k, _, _ := strings.Cut(os.Getenv("DGEM_SERVER_KEY"), ",")
	return strings.TrimSpace(k)
}

// NewClient creates a new DiffGemma client.
func NewClient(baseURL, defaultModel string, timeout time.Duration) *Client {
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8080/v1"
	}
	baseURL = strings.TrimRight(baseURL, "/")

	if timeout <= 0 {
		timeout = 120 * time.Second
	}

	return &Client{
		ServerKey:  ServerKeyFromEnv(),
		BaseURL:    baseURL,
		HTTPClient: &http.Client{Timeout: timeout},
		Model:      defaultModel,
	}
}

// WithAuthToken sets the Authorization Bearer / IAM token.
func (c *Client) WithAuthToken(token string) *Client {
	c.AuthToken = strings.TrimSpace(token)
	return c
}

// IsVertexEndpointURL returns true if rawURL targets a Google Cloud Vertex AI Online Prediction Endpoint
// (either a Dedicated Endpoint *.prediction.vertexai.goog with /invoke/* custom routes or *.aiplatform.googleapis.com).
func IsVertexEndpointURL(rawURL string) bool {
	u := strings.ToLower(strings.TrimSpace(rawURL))
	return strings.Contains(u, "prediction.vertexai.goog") ||
		strings.Contains(u, "aiplatform.googleapis.com") ||
		strings.Contains(u, "/invoke/") ||
		strings.HasSuffix(u, "/invoke") ||
		strings.HasSuffix(u, ":rawpredict") ||
		strings.HasSuffix(u, ":streamrawpredict") ||
		strings.HasSuffix(u, ":predict")
}

// NormalizeVertexEndpointURL normalizes a Vertex AI Endpoint URL:
//   - For Dedicated Endpoints using arbitrary custom routes (invokeRoutePrefix="/*"),
//     routes to .../invoke/v1/chat/completions (which Vertex forwards as /v1/chat/completions to structured_server.py).
//   - For standard regional endpoints (aiplatform.googleapis.com without /invoke), routes to :rawPredict.
func NormalizeVertexEndpointURL(rawURL string) string {
	u := strings.TrimRight(strings.TrimSpace(rawURL), "/")
	if strings.Contains(strings.ToLower(u), "prediction.vertexai.goog") || strings.Contains(strings.ToLower(u), "/invoke") {
		if strings.HasSuffix(u, "/invoke/v1/chat/completions") ||
			strings.HasSuffix(u, "/invoke/v1/raw/chat/completions") ||
			strings.HasSuffix(u, "/invoke/v1/systemone") {
			return u
		}
		u = strings.TrimSuffix(u, "/chat/completions")
		u = strings.TrimSuffix(u, "/v1")
		if !strings.HasSuffix(u, "/invoke") {
			u += "/invoke"
		}
		return u + "/v1/chat/completions"
	}
	u = strings.TrimSuffix(u, "/v1/chat/completions")
	u = strings.TrimSuffix(u, "/chat/completions")
	u = strings.TrimSuffix(u, "/v1")
	if !strings.HasSuffix(u, ":rawPredict") && !strings.HasSuffix(u, ":streamRawPredict") && !strings.HasSuffix(u, ":predict") {
		u += ":rawPredict"
	}
	return u
}

// Complete executes an OpenAI-compatible chat completion (or Vertex AI /invoke/v1/chat/completions or :rawPredict request).
func (c *Client) Complete(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, *RequestStats, error) {
	if req.Model == "" {
		req.Model = c.Model
	}

	endpoint := fmt.Sprintf("%s/chat/completions", c.BaseURL)
	if IsVertexEndpointURL(c.BaseURL) {
		endpoint = NormalizeVertexEndpointURL(c.BaseURL)
	}
	payloadBytes, err := json.Marshal(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var (
		resp      *http.Response
		bodyBytes []byte
		wallTime  time.Duration
		retries   int
	)
	for attempt := 0; ; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payloadBytes))
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create http request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if c.ServerKey != "" {
			httpReq.Header.Set(ServerKeyHeader, c.ServerKey)
		}

		otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(httpReq.Header))
		if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
			spanBytes := sc.SpanID()
			spanDec := binary.BigEndian.Uint64(spanBytes[:])
			httpReq.Header.Set("X-Cloud-Trace-Context", fmt.Sprintf("%s/%d;o=1", sc.TraceID().String(), spanDec))
		}

		if c.AuthToken != "" {
			token := c.AuthToken
			if !strings.HasPrefix(strings.ToLower(token), "bearer ") {
				token = "Bearer " + token
			}
			httpReq.Header.Set("Authorization", token)
		}

		start := time.Now()
		resp, err = c.HTTPClient.Do(httpReq)
		if err != nil {
			return nil, nil, fmt.Errorf("http request failed to %s: %w", endpoint, err)
		}
		bodyBytes, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		wallTime = time.Since(start)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read response body: %w", err)
		}
		if (resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable) && attempt < c.MaxRetries {
			retries++
			backoff := time.Duration(150*(1<<attempt)) * time.Millisecond
			backoff += time.Duration(rand.Int63n(int64(backoff / 2)))
			select {
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			case <-time.After(backoff):
			}
			continue
		}
		break
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, nil, &HTTPError{StatusCode: resp.StatusCode, Body: string(bodyBytes), Retries: retries}
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	stats := &RequestStats{
		Retries:      retries,
		Model:        chatResp.Model,
		Endpoint:     endpoint,
		WallTime:     wallTime,
		PromptTokens: chatResp.Usage.PromptTokens,
		OutputTokens: chatResp.Usage.CompletionTokens,
		TotalTokens:  chatResp.Usage.TotalTokens,
	}

	// If the response contains structured diagnostics, extract timing telemetry
	if len(chatResp.Choices) > 0 {
		content := chatResp.Choices[0].Message.RawContent()
		if structured, err := ParseStructuredContent(content); err == nil && (structured.Diagnostics.Steps > 0 || structured.Diagnostics.Timing.TotalMs > 0 || len(structured.Answers) > 0) {
			stats.Diagnostics = &structured.Diagnostics
			stats.PrefillMs = structured.Diagnostics.Timing.PrefillMs
			if structured.Diagnostics.Timing.DenoiseMs > 0 {
				stats.DenoiseMs = structured.Diagnostics.Timing.DenoiseMs
			} else {
				stats.DenoiseMs = structured.Diagnostics.Timing.TotalMs
			}
			stats.ReusedTokens = structured.Diagnostics.Timing.ReusedTokens
			stats.DenoiseSteps = structured.Diagnostics.Timing.StepsRun
			stats.SamplesN = structured.Diagnostics.Samples.N
			stats.Extended = structured.Diagnostics.Samples.Policy.Extended
		}
	}

	return &chatResp, stats, nil
}

// Decide executes a discrete diffusion slot readout decision query with optional multimodal images.
func (c *Client) Decide(ctx context.Context, schemaContent, userStateContent string, images ...string) (*StructuredDecisionResponse, *RequestStats, error) {
	userPayload, err := BuildMultimodalContent(userStateContent, images)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to process multimodal content: %w", err)
	}

	req := ChatCompletionRequest{
		Messages: []ChatMessage{
			{Role: "system", Content: NormalizeSchemaSamples(schemaContent)},
			{Role: "user", Content: userPayload},
		},
		Logprobs:    true,
		TopLogprobs: 5,
	}

	chatResp, stats, err := c.Complete(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	if len(chatResp.Choices) == 0 {
		return nil, stats, fmt.Errorf("no response choices returned by model")
	}

	rawText := chatResp.Choices[0].Message.RawContent()
	structured, err := ParseStructuredContentWithLogprobs(rawText, chatResp.Choices[0].Logprobs)
	if err != nil {
		return nil, stats, fmt.Errorf("failed to parse structured decision output: %w (raw content: %s)", err, rawText)
	}

	return structured, stats, nil
}

// HTTPError is returned for non-2xx responses so callers can record the status code.
type HTTPError struct {
	StatusCode int
	Body       string
	Retries    int
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("server returned error HTTP %d: %s", e.StatusCode, e.Body)
}

// NormalizeSchemaSamples turns a digit-string "samples" value (template variables such as `-v samples=4`
// render as strings) into a JSON number. Servers built on the upstream structured server reject "4" with
// `samples must be a positive count or "auto"`. Any other content is returned unchanged.
func NormalizeSchemaSamples(schemaContent string) string {
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(schemaContent), &m); err != nil {
		return schemaContent
	}
	raw, ok := m["samples"]
	if !ok {
		return schemaContent
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return schemaContent // already a number (or "auto" handled below)
	}
	s = strings.TrimSpace(s)
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 {
		return schemaContent
	}
	m["samples"] = json.RawMessage(strconv.Itoa(n))
	out, err := json.Marshal(m)
	if err != nil {
		return schemaContent
	}
	return string(out)
}
