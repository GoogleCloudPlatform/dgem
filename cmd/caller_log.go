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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
	"go.opentelemetry.io/otel/attribute"
)

// callerEmail is the caller identity for logs: the account IAP verified (or "apikey:<id>"), as callerAuthHandler
// leaves it in X-Goog-Authenticated-User-Email. With caller authentication off it is whatever IAP or the client sent.
func callerEmail(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("X-Goog-Authenticated-User-Email"), "accounts.google.com:")
}

// maxMCPLogBody caps how much of an MCP request body is read to find the JSON-RPC method and tool name.
const maxMCPLogBody = 1 << 20

// mcpSpanHandler wraps the MCP endpoint in a dgem.gateway.mcp span carrying the caller, the JSON-RPC method, the tool
// name for tools/call and the HTTP status, so MCP use is attributable in the logs like /v1/systemone and /api/decide.
// Tool arguments are never recorded. The MCP handler is stateless with JSON responses: one span per HTTP request.
func mcpSpanHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, span := gatewayTracer().Start(extractTraceContextFromRequest(r), "dgem.gateway.mcp")
		defer span.End()
		// Same response header as /v1/systemone and /api/decide, so body-log entries and callers can link to the span.
		if tid := span.SpanContext().TraceID(); tid.IsValid() {
			w.Header().Set("X-Dgem-Trace-Id", tid.String())
		}
		attrs := []attribute.KeyValue{attribute.String("dgem.route", "/mcp"), attribute.String("http.method", r.Method)}
		if u := callerEmail(r); u != "" {
			attrs = append(attrs, attribute.String("dgem.user", u))
		}
		if r.Method == http.MethodPost && r.Body != nil {
			body, err := io.ReadAll(io.LimitReader(r.Body, maxMCPLogBody+1))
			if err == nil && len(body) <= maxMCPLogBody {
				method, tool := mcpRPCMethod(body)
				if method != "" {
					attrs = append(attrs, attribute.String("dgem.mcp.method", method))
				}
				if tool != "" {
					attrs = append(attrs, attribute.String("dgem.mcp.tool", tool))
				}
			}
			r.Body = io.NopCloser(io.MultiReader(bytes.NewReader(body), r.Body))
		}
		span.SetAttributes(attrs...)
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(ctx))
		if rec.status >= 400 {
			failSpan(span, rec.status, http.StatusText(rec.status))
		} else {
			span.SetAttributes(attribute.Int("http.status_code", rec.status))
		}
	})
}

// mcpRPCMethod returns a JSON-RPC body's method and, for tools/call, the tool name (first message of a batch).
func mcpRPCMethod(body []byte) (string, string) {
	type msg struct {
		Method string `json:"method"`
		Params struct {
			Name string `json:"name"`
		} `json:"params"`
	}
	var m msg
	if json.Unmarshal(body, &m) != nil {
		var batch []msg
		if json.Unmarshal(body, &batch) != nil || len(batch) == 0 {
			return "", ""
		}
		m = batch[0]
	}
	if m.Method != "tools/call" {
		return m.Method, ""
	}
	return m.Method, m.Params.Name
}

// statusRecorder records the status code written through it; Flush and Unwrap keep streaming responses working.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *statusRecorder) Unwrap() http.ResponseWriter { return s.ResponseWriter }

// decideUpstreamStatus is the status /api/decide answers when the serving image call fails: the adapter's mapping
// (decisionindex.ErrorStatus) for requests the server refused (400/422: schema, too many options, image it can't
// read, context overflow), 502 for everything else (unreachable, 5xx, timeouts). It used to be 502 for all of them,
// so a caller's mistake looked like a server fault and logged at ERROR.
func decideUpstreamStatus(err error) int {
	if status, _ := decisionindex.ErrorStatus(err); status >= 400 && status < 500 {
		return status
	}
	return http.StatusBadGateway
}

// systemOneRawUsers lists callers (dgem.user) whose /v1/systemone requests are forwarded to the serving image unchanged
// when it can answer them as sent (decisionindex.FitsServingImage); "*" means every caller. Set with
// --systemone-raw-users / DGEM_SYSTEMONE_RAW_USERS.
var systemOneRawUsers []string

func systemOneRawCaller(user string) bool {
	for _, u := range systemOneRawUsers {
		if u == "*" || (user != "" && strings.EqualFold(u, user)) {
			return true
		}
	}
	return false
}

// systemOneForwardBody is the body forwarded on the pass-through path: the caller's bytes unchanged, except that
// "media" entries (already merged into req.Images) are replaced by "images", which is what the serving image reads.
func systemOneForwardBody(body []byte, req decisionindex.SystemOneRequest) ([]byte, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, err
	}
	if _, ok := top["media"]; !ok {
		return body, nil
	}
	delete(top, "media")
	imgs, err := json.Marshal(req.Images)
	if err != nil {
		return nil, err
	}
	top["images"] = imgs
	return json.Marshal(top)
}

// seedTokenUsers lists callers (dgem.user) whose requests default to seed_token="fixed" when the request body
// does not explicitly set seed_token; "*" means every caller. Set with --seed-token-users / DGEM_SEED_TOKEN_USERS.
var seedTokenUsers []string

func seedTokenCaller(user string) bool {
	for _, u := range seedTokenUsers {
		if u == "*" || (user != "" && strings.EqualFold(u, user)) {
			return true
		}
	}
	return false
}

// normalizeSeedToken validates a seed_token value ("random" or "fixed"; empty means unset).
func normalizeSeedToken(v string) (string, error) {
	s := strings.ToLower(strings.TrimSpace(v))
	switch s {
	case "", "random", "fixed":
		return s, nil
	default:
		return "", fmt.Errorf("seed_token must be random or fixed (got %q)", v)
	}
}

// applySystemOneSeedToken injects defaultSeedToken (e.g. "fixed") into a JSON /v1/systemone body only when
// "seed_token" is not already set by the caller. Returns the (possibly updated) body and the effective seed_token
// ("" if unset). When no change is needed, the original body bytes are returned verbatim.
func applySystemOneSeedToken(body []byte, defaultSeedToken string) ([]byte, string, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return body, "", nil
	}
	if raw, ok := top["seed_token"]; ok && len(bytes.TrimSpace(raw)) > 0 && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		var existing string
		if err := json.Unmarshal(raw, &existing); err != nil {
			return nil, "", fmt.Errorf("seed_token must be a string (\"random\" or \"fixed\")")
		}
		norm, err := normalizeSeedToken(existing)
		if err != nil {
			return nil, "", err
		}
		if norm != "" {
			return body, norm, nil
		}
	}
	if defaultSeedToken == "" {
		return body, "", nil
	}
	val, err := json.Marshal(defaultSeedToken)
	if err != nil {
		return nil, "", err
	}
	top["seed_token"] = val
	out, err := json.Marshal(top)
	if err != nil {
		return nil, "", err
	}
	return out, defaultSeedToken, nil
}

// applySchemaSeedToken resolves seed_token on a /api/decide schema: explicit caller override wins,
// otherwise an existing template seed_token is kept, otherwise defaultSeedToken (from DGEM_SEED_TOKEN_USERS)
// is injected. Returns the (possibly updated) schema JSON and the effective seed_token ("" if unset).
func applySchemaSeedToken(schemaJSON, explicit, defaultSeedToken string) (string, string, error) {
	exp, err := normalizeSeedToken(explicit)
	if err != nil {
		return schemaJSON, "", err
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(schemaJSON), &m); err != nil {
		return schemaJSON, "", fmt.Errorf("seed_token: schema is not a JSON object: %w", err)
	}
	if exp != "" {
		m["seed_token"] = exp
		b, err := json.Marshal(m)
		if err != nil {
			return schemaJSON, "", err
		}
		return string(b), exp, nil
	}
	if raw, ok := m["seed_token"]; ok && raw != nil {
		if s, ok := raw.(string); ok {
			if norm, nErr := normalizeSeedToken(s); nErr == nil && norm != "" {
				return schemaJSON, norm, nil
			}
		}
	}
	if defaultSeedToken == "" {
		return schemaJSON, "", nil
	}
	m["seed_token"] = defaultSeedToken
	b, err := json.Marshal(m)
	if err != nil {
		return schemaJSON, "", err
	}
	return string(b), defaultSeedToken, nil
}
