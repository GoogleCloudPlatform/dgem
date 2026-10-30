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
	"io"
	"net/http"
	"strings"

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
