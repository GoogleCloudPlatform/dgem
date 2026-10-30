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
	"net/http/httptest"
	"strings"
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// The MCP endpoint logs one span per request with the caller, JSON-RPC method and tool, never the arguments, and the
// wrapped handler still sees the full body.
func TestMCPSpanHandlerLogsCaller(t *testing.T) {
	var buf bytes.Buffer
	oldW := spanLogWriter
	spanLogWriter = &buf
	defer func() { spanLogWriter = oldW }()
	oldTP := otel.GetTracerProvider()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(&gatewaySpanProcessor{}))
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(oldTP)

	const args = `{"policy":"support_triage","variables":{"input":"SECRET-ARGUMENT"}}`
	body := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"decide_policy","arguments":` + args + `}}`
	var seen string
	h := mcpSpanHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seen = string(b)
		w.WriteHeader(http.StatusAccepted)
	}))
	req := httptest.NewRequest("POST", "/mcp", strings.NewReader(body))
	req.Header.Set("X-Goog-Authenticated-User-Email", "accounts.google.com:client-sa@example.iam.gserviceaccount.com")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if seen != body {
		t.Fatalf("handler body = %q", seen)
	}
	var line map[string]any
	for _, l := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		var m map[string]any
		if json.Unmarshal([]byte(l), &m) == nil && m["span_name"] == "dgem.gateway.mcp" {
			line = m
		}
	}
	if line == nil {
		t.Fatalf("no dgem.gateway.mcp line in %q", buf.String())
	}
	a := line["attributes"].(map[string]any)
	if a["dgem.user"] != "client-sa@example.iam.gserviceaccount.com" || a["dgem.mcp.method"] != "tools/call" ||
		a["dgem.mcp.tool"] != "decide_policy" || a["http.status_code"] != float64(202) || line["severity"] != "INFO" {
		t.Errorf("line: %v", line)
	}
	if strings.Contains(buf.String(), "SECRET-ARGUMENT") {
		t.Error("tool arguments were logged")
	}
}

func TestMCPRPCMethod(t *testing.T) {
	for _, tc := range []struct{ body, method, tool string }{
		{`{"method":"tools/list"}`, "tools/list", ""},
		{`[{"method":"tools/call","params":{"name":"decide_custom_questions"}}]`, "tools/call", "decide_custom_questions"},
		{`not json`, "", ""},
	} {
		if m, tl := mcpRPCMethod([]byte(tc.body)); m != tc.method || tl != tc.tool {
			t.Errorf("%s: %q %q", tc.body, m, tl)
		}
	}
}
