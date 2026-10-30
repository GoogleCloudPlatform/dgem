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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Opt-in request/response body logging for debugging callers. Off by default. When on, it is limited in
// time (DGEM_LOG_BODIES_UNTIL, at most bodyLogMaxWindow ahead) and to listed callers (DGEM_LOG_BODIES_USERS, "*" must
// be written out), and media is replaced by a digest. Entries go to stdout with "log_type": "dgem.bodies" so a log
// router sink can send them to a separate bucket with its own retention and access (docs/operate/observability.md).

const (
	bodyLogMaxWindow = 7 * 24 * time.Hour
	bodyLogCap       = 96 << 10 // per body; a Cloud Logging entry holds at most 256 KB
	bodyLogReadLimit = 32 << 20 // never buffer more than this of a request body for logging
	bodyLogType      = "dgem.bodies"
)

type bodyLogConfig struct {
	mode   string // "responses" or "both"
	until  time.Time
	users  []string
	sample float64
	err    string // set when the env asked for logging but the config was refused (logging stays off)
}

var (
	bodyLogWriter io.Writer = os.Stdout
	bodyLogNow              = time.Now
	bodyLogMu     sync.Mutex
	bodyLog       *bodyLogConfig
)

// parseBodyLogConfig reads the DGEM_LOG_BODIES* settings. It returns nil when logging is off; a config with err set
// when the settings are invalid (logging stays off and the reason is reported at startup and in /health).
func parseBodyLogConfig(getenv func(string) string, now time.Time) *bodyLogConfig {
	mode := strings.ToLower(strings.TrimSpace(getenv("DGEM_LOG_BODIES")))
	switch mode {
	case "", "off", "false", "0":
		return nil
	case "responses", "both":
	default:
		return &bodyLogConfig{err: fmt.Sprintf("DGEM_LOG_BODIES must be off, responses or both (got %q)", mode)}
	}
	c := &bodyLogConfig{mode: mode, sample: 1}
	u := strings.TrimSpace(getenv("DGEM_LOG_BODIES_UNTIL"))
	if u == "" {
		c.err = "DGEM_LOG_BODIES_UNTIL (RFC3339) is required"
		return c
	}
	t, err := time.Parse(time.RFC3339, u)
	if err != nil {
		c.err = "DGEM_LOG_BODIES_UNTIL must be RFC3339, e.g. 2026-10-22T18:00:00Z"
		return c
	}
	if !t.After(now) {
		c.err = "DGEM_LOG_BODIES_UNTIL is in the past"
		return c
	}
	if t.Sub(now) > bodyLogMaxWindow {
		c.err = fmt.Sprintf("DGEM_LOG_BODIES_UNTIL is more than %s ahead", bodyLogMaxWindow)
		return c
	}
	c.until = t
	for _, x := range strings.Split(getenv("DGEM_LOG_BODIES_USERS"), ",") {
		if x = strings.TrimSpace(x); x != "" {
			c.users = append(c.users, x)
		}
	}
	if len(c.users) == 0 {
		c.err = `DGEM_LOG_BODIES_USERS is required (caller identities, or "*" for every caller)`
		return c
	}
	if s := strings.TrimSpace(getenv("DGEM_LOG_BODIES_SAMPLE")); s != "" {
		var f float64
		if _, err := fmt.Sscanf(s, "%g", &f); err != nil || f <= 0 || f > 1 {
			c.err = "DGEM_LOG_BODIES_SAMPLE must be in (0, 1]"
			return c
		}
		c.sample = f
	}
	return c
}

func (c *bodyLogConfig) on() bool { return c != nil && c.err == "" }

func (c *bodyLogConfig) logs(user string, now time.Time) bool {
	if !c.on() || !now.Before(c.until) {
		return false
	}
	for _, u := range c.users {
		if u == "*" || (user != "" && strings.EqualFold(u, user)) {
			return c.sample >= 1 || rand.Float64() < c.sample
		}
	}
	return false
}

// bodyLogStatus is the /health view of body logging.
func bodyLogStatus() map[string]any {
	bodyLogMu.Lock()
	c := bodyLog
	bodyLogMu.Unlock()
	switch {
	case c == nil:
		return map[string]any{"enabled": false}
	case c.err != "":
		return map[string]any{"enabled": false, "error": c.err}
	}
	return map[string]any{"enabled": bodyLogNow().Before(c.until), "mode": c.mode, "until": c.until.UTC().Format(time.RFC3339),
		"callers": len(c.users), "sample": c.sample}
}

// bodyLogRoute reports whether a request is one whose bodies may be logged: decisions only, never auth or admin routes.
func bodyLogRoute(r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}
	p := r.URL.Path
	return p == "/v1/systemone" || p == "/api/decide" || strings.HasPrefix(p, "/api/decide/") || p == "/mcp" || p == "/mcp/"
}

// bodyLogHandler logs request/response bodies for matching requests. It sits inside callerAuthHandler, so the
// caller is the verified identity. Without a valid config it returns next unchanged.
func bodyLogHandler(next http.Handler, c *bodyLogConfig) http.Handler {
	if !c.on() {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user := callerEmail(r)
		if !bodyLogRoute(r) || !c.logs(user, bodyLogNow()) {
			next.ServeHTTP(w, r)
			return
		}
		var reqBody []byte
		if r.Body != nil {
			reqBody, _ = io.ReadAll(io.LimitReader(r.Body, bodyLogReadLimit))
			r.Body = io.NopCloser(io.MultiReader(bytes.NewReader(reqBody), r.Body))
		}
		if strings.HasPrefix(r.URL.Path, "/mcp") {
			if m, _ := mcpRPCMethod(reqBody); m != "tools/call" {
				next.ServeHTTP(w, r) // MCP handshakes and listings carry no decision
				return
			}
		}
		rec := &bodyRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		entry := map[string]any{
			"severity":     "INFO",
			"log_type":     bodyLogType,
			"message":      "[dgem.bodies] " + r.URL.Path,
			"dgem_user":    user,
			"route":        r.URL.Path,
			"http_status":  rec.status,
			"trace_id":     rec.Header().Get("X-Dgem-Trace-Id"),
			"logged_until": c.until.UTC().Format(time.RFC3339),
		}
		if tid := rec.Header().Get("X-Dgem-Trace-Id"); tid != "" && gcpProjectID != "" {
			entry["logging.googleapis.com/trace"] = fmt.Sprintf("projects/%s/traces/%s", gcpProjectID, tid)
		}
		if c.mode == "both" {
			entry["request"], entry["request_truncated"] = bodyForLog(reqBody, r.Header.Get("Content-Type"))
		}
		entry["response"], entry["response_truncated"] = bodyForLog(rec.buf.Bytes(), rec.Header().Get("Content-Type"))
		entry["response_bytes"] = rec.n
		if b, err := json.Marshal(entry); err == nil {
			fmt.Fprintln(bodyLogWriter, string(b))
		}
	})
}

// bodyRecorder passes the response through and keeps up to bodyLogCap+1 bytes of it.
type bodyRecorder struct {
	http.ResponseWriter
	status int
	buf    bytes.Buffer
	n      int
}

func (b *bodyRecorder) WriteHeader(code int) {
	b.status = code
	b.ResponseWriter.WriteHeader(code)
}

func (b *bodyRecorder) Write(p []byte) (int, error) {
	if room := bodyLogCap + 1 - b.buf.Len(); room > 0 {
		b.buf.Write(p[:min(room, len(p))])
	}
	b.n += len(p)
	return b.ResponseWriter.Write(p)
}

func (b *bodyRecorder) Flush() {
	if f, ok := b.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (b *bodyRecorder) Unwrap() http.ResponseWriter { return b.ResponseWriter }

// bodyForLog returns a body for the log entry: JSON with media replaced by digests, capped at bodyLogCap (then as a
// truncated string); non-JSON bodies (multipart uploads, plain-text errors) as text, or only their size if binary.
func bodyForLog(body []byte, contentType string) (any, bool) {
	if len(body) == 0 {
		return nil, false
	}
	var v any
	if json.Unmarshal(body, &v) == nil {
		out, _ := json.Marshal(redactMedia(v))
		if len(out) <= bodyLogCap {
			return json.RawMessage(out), false
		}
		return string(out[:bodyLogCap]), true
	}
	if strings.HasPrefix(contentType, "multipart/") || !isText(body) {
		return map[string]any{"content_type": contentType, "bytes": len(body), "omitted": "not JSON"}, false
	}
	if len(body) > bodyLogCap {
		return string(body[:bodyLogCap]), true
	}
	return string(body), false
}

// redactMedia replaces data: URIs and long base64 strings anywhere in a JSON value with {redacted, mime, chars, sha256}.
func redactMedia(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			x[k] = redactMedia(e)
		}
		return x
	case []any:
		for i, e := range x {
			x[i] = redactMedia(e)
		}
		return x
	case string:
		sum := sha256.Sum256([]byte(x))
		digest := hex.EncodeToString(sum[:8])
		if strings.HasPrefix(x, "data:") {
			mime := ""
			if end := strings.IndexAny(x, ";,"); end > len("data:") {
				mime = x[len("data:"):end]
			}
			return map[string]any{"redacted": "data_uri", "mime": mime, "chars": len(x), "sha256": digest}
		}
		if len(x) >= 512 && looksBase64(x) {
			return map[string]any{"redacted": "base64", "chars": len(x), "sha256": digest}
		}
	}
	return v
}

func looksBase64(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '+' || c == '/' || c == '=' || c == '-' || c == '_' || c == '\n' || c == '\r') {
			return false
		}
	}
	return true
}

func isText(b []byte) bool {
	for _, c := range b[:min(len(b), 512)] {
		if c == 0 {
			return false
		}
	}
	return true
}
