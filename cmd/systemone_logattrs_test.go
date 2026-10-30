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
	"context"
	"encoding/json"
	"strings"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// spanLines runs fn with a span from a provider that logs through gatewaySpanProcessor and returns the log lines.
func spanLines(t *testing.T, fn func(span trace.Span)) []map[string]any {
	t.Helper()
	var buf bytes.Buffer
	old := spanLogWriter
	spanLogWriter = &buf
	defer func() { spanLogWriter = old }()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(&gatewaySpanProcessor{}))
	defer func() { _ = tp.Shutdown(context.Background()) }()
	_, span := tp.Tracer("test").Start(context.Background(), "dgem.gateway.proxy")
	fn(span)
	span.End()
	var out []map[string]any
	for _, l := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatalf("not JSON: %q", l)
		}
		out = append(out, m)
	}
	return out
}

const secretState = "CONFIDENTIAL-STATE-TEXT"
const secretInstr = "CONFIDENTIAL-INSTRUCTION-TEXT"

var shapeBody = []byte(`{"state": "` + secretState + `", "media": [{"mime_type": "image/png", "data": "AAAA"}], "questions": {
	"a": {"type": "choice", "instructions": "` + secretInstr + `", "criteria": {"x": "OPTTEXT", "y": "OPTTEXT", "z": "OPTTEXT"}},
	"b": {"type": "noul", "instructions": "` + secretInstr + `"},
	"c": {"type": "score", "instructions": "` + secretInstr + `", "criteria": ["l0", "l1", "l2", "l3", "l4"]}}}`)

// A refused request logs WARNING with the reason; shape attributes are there and no request text is.
func TestSystemOneFailureLogged(t *testing.T) {
	lines := spanLines(t, func(span trace.Span) {
		span.SetAttributes(systemOneBodyAttrs(shapeBody)...)
		req, _, err := decodeSystemOneAdapterRequest(shapeBody)
		if err != nil {
			t.Fatal(err)
		}
		span.SetAttributes(systemOneShapeAttrs(req)...)
		failSpan(span, 422, "server returned error HTTP 400: question 'move_1': needs at least two alternatives")
	})
	if len(lines) != 1 {
		t.Fatalf("%d lines", len(lines))
	}
	l := lines[0]
	if l["severity"] != "WARNING" || !strings.Contains(l["error"].(string), "question 'move_1'") {
		t.Errorf("severity/error: %v", l)
	}
	a := l["attributes"].(map[string]any)
	want := map[string]float64{"http.status_code": 422, "dgem.systemone.questions": 3, "dgem.systemone.q_choice": 1,
		"dgem.systemone.q_noul": 1, "dgem.systemone.q_score": 1, "dgem.systemone.max_options": 5,
		"dgem.systemone.media_entries": 1, "dgem.image_count": 1, "dgem.systemone.body_bytes": float64(len(shapeBody))}
	for k, v := range want {
		if a[k] != v {
			t.Errorf("%s = %v, want %v", k, a[k], v)
		}
	}
	raw, _ := json.Marshal(l)
	for _, s := range []string{secretState, secretInstr, "AAAA", "l3", "OPTTEXT"} {
		if strings.Contains(string(raw), s) {
			t.Errorf("log line contains request content %q", s)
		}
	}
}

func TestSpanSeverity(t *testing.T) {
	for _, tc := range []struct {
		status int
		err    string
		sev    string
	}{{200, "", "INFO"}, {400, "bad", "WARNING"}, {502, "upstream 500: x", "ERROR"}, {0, "dial tcp: refused", "ERROR"}} {
		lines := spanLines(t, func(span trace.Span) {
			if tc.err != "" {
				failSpan(span, tc.status, tc.err)
			}
		})
		if lines[0]["severity"] != tc.sev || (tc.err != "" && lines[0]["error"] != tc.err) || (tc.err == "" && lines[0]["error"] != nil) {
			t.Errorf("%+v: %v", tc, lines[0])
		}
	}
	long := strings.Repeat("x", 1000)
	lines := spanLines(t, func(span trace.Span) { failSpan(span, 422, long) })
	if n := len([]rune(lines[0]["error"].(string))); n != maxSpanErrorLen+3 {
		t.Errorf("error not truncated: %d runes", n)
	}
}
