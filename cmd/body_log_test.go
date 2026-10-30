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
	"time"
)

var bodyLogT0 = time.Date(2026, 10, 21, 12, 0, 0, 0, time.UTC)

func envOf(m map[string]string) func(string) string { return func(k string) string { return m[k] } }

func TestParseBodyLogConfig(t *testing.T) {
	if c := parseBodyLogConfig(envOf(nil), bodyLogT0); c != nil {
		t.Fatalf("default must be off, got %+v", c)
	}
	ok := map[string]string{"DGEM_LOG_BODIES": "both", "DGEM_LOG_BODIES_UNTIL": "2026-10-22T12:00:00Z", "DGEM_LOG_BODIES_USERS": "a@x.com, b@x.com"}
	if c := parseBodyLogConfig(envOf(ok), bodyLogT0); !c.on() || len(c.users) != 2 || c.mode != "both" {
		t.Fatalf("valid config refused: %+v", c)
	}
	for name, change := range map[string][2]string{
		"bad mode":       {"DGEM_LOG_BODIES", "all"},
		"missing until":  {"DGEM_LOG_BODIES_UNTIL", ""},
		"until not 3339": {"DGEM_LOG_BODIES_UNTIL", "tomorrow"},
		"until past":     {"DGEM_LOG_BODIES_UNTIL", "2026-10-20T12:00:00Z"},
		"until > 7 days": {"DGEM_LOG_BODIES_UNTIL", "2026-10-29T12:00:00Z"},
		"no users":       {"DGEM_LOG_BODIES_USERS", " , "},
		"bad sample":     {"DGEM_LOG_BODIES_SAMPLE", "2"},
	} {
		m := map[string]string{}
		for k, v := range ok {
			m[k] = v
		}
		m[change[0]] = change[1]
		if c := parseBodyLogConfig(envOf(m), bodyLogT0); c == nil || c.on() || c.err == "" {
			t.Errorf("%s: should be refused, got %+v", name, c)
		}
	}
}

// runBodyLog sends one request through bodyLogHandler at time now and returns the logged entries.
func runBodyLog(t *testing.T, c *bodyLogConfig, now time.Time, path, user, body string, respond http.HandlerFunc) []map[string]any {
	t.Helper()
	var buf bytes.Buffer
	oldW, oldNow := bodyLogWriter, bodyLogNow
	bodyLogWriter, bodyLogNow = &buf, func() time.Time { return now }
	defer func() { bodyLogWriter, bodyLogNow = oldW, oldNow }()
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if user != "" {
		req.Header.Set("X-Goog-Authenticated-User-Email", "accounts.google.com:"+user)
	}
	bodyLogHandler(respond, c).ServeHTTP(httptest.NewRecorder(), req)
	var out []map[string]any
	for _, l := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if l == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatalf("not JSON: %q", l)
		}
		out = append(out, m)
	}
	return out
}

func echoJSON(seen *string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if seen != nil {
			*seen = string(b)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Dgem-Trace-Id", "abc123")
		_, _ = w.Write([]byte(`{"answers": {"q": {"type": "noul", "noul": 0.7}}}`))
	}
}

func TestBodyLogHandler(t *testing.T) {
	c := &bodyLogConfig{mode: "both", until: bodyLogT0.Add(time.Hour), users: []string{"client@x.iam.gserviceaccount.com"}, sample: 1}
	img := "data:image/png;base64," + strings.Repeat("QUJD", 300)
	body := `{"state": "board text", "images": ["` + img + `"], "media": [{"mime_type": "image/jpeg", "data": "` + strings.Repeat("A", 600) + `"}], "questions": {"q": {"type": "noul"}}}`
	var seen string
	got := runBodyLog(t, c, bodyLogT0, "/v1/systemone", "client@x.iam.gserviceaccount.com", body, echoJSON(&seen))
	if seen != body {
		t.Fatal("handler did not receive the full body")
	}
	if len(got) != 1 {
		t.Fatalf("%d entries", len(got))
	}
	e := got[0]
	raw, _ := json.Marshal(e)
	if e["log_type"] != "dgem.bodies" || e["dgem_user"] != "client@x.iam.gserviceaccount.com" || e["trace_id"] != "abc123" ||
		e["http_status"] != float64(200) || !strings.Contains(string(raw), "board text") || !strings.Contains(string(raw), `"noul":0.7`) {
		t.Errorf("entry: %s", raw)
	}
	if strings.Contains(string(raw), "QUJDQUJD") || strings.Contains(string(raw), strings.Repeat("A", 600)) ||
		!strings.Contains(string(raw), `"redacted":"data_uri"`) || !strings.Contains(string(raw), `"redacted":"base64"`) {
		t.Errorf("media not redacted: %s", raw)
	}

	if n := len(runBodyLog(t, c, bodyLogT0, "/v1/systemone", "other@x.com", body, echoJSON(nil))); n != 0 {
		t.Error("unlisted caller logged")
	}
	if n := len(runBodyLog(t, c, bodyLogT0.Add(2*time.Hour), "/v1/systemone", "client@x.iam.gserviceaccount.com", body, echoJSON(nil))); n != 0 {
		t.Error("logged after UNTIL")
	}
	if n := len(runBodyLog(t, c, bodyLogT0, "/api/auth/me", "client@x.iam.gserviceaccount.com", body, echoJSON(nil))); n != 0 {
		t.Error("logged a non-decision route")
	}
	if n := len(runBodyLog(t, c, bodyLogT0, "/mcp", "client@x.iam.gserviceaccount.com", `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`, echoJSON(nil))); n != 0 {
		t.Error("logged an MCP tools/list")
	}
	if n := len(runBodyLog(t, c, bodyLogT0, "/mcp", "client@x.iam.gserviceaccount.com", `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"decide_policy"}}`, echoJSON(nil))); n != 1 {
		t.Error("MCP tools/call not logged")
	}

	resp := &bodyLogConfig{mode: "responses", until: c.until, users: []string{"*"}, sample: 1}
	e = runBodyLog(t, resp, bodyLogT0, "/api/decide/x", "anyone@x.com", body, echoJSON(nil))[0]
	if _, has := e["request"]; has || e["response"] == nil {
		t.Errorf("responses mode: %v", e)
	}

	big := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"text": "` + strings.Repeat("x ", bodyLogCap) + `"}`))
	}
	e = runBodyLog(t, resp, bodyLogT0, "/v1/systemone", "anyone@x.com", `{}`, big)[0]
	if e["response_truncated"] != true || len(e["response"].(string)) != bodyLogCap {
		t.Errorf("not capped: truncated=%v", e["response_truncated"])
	}
	if bodyLogHandler(echoJSON(nil), nil) == nil || bodyLogHandler(echoJSON(nil), &bodyLogConfig{err: "x"}) == nil {
		t.Error("off config must return the handler")
	}
}
