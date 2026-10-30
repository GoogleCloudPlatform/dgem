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
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// echoIdentity reports the identity header the handlers would see.
var echoIdentity = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	_, _ = w.Write([]byte(r.Header.Get("X-Goog-Authenticated-User-Email") + "|" + r.Header.Get(client.GatewayKeyHeader)))
})

func do(h http.Handler, method, path string, hdr map[string]string) (int, string) {
	req := httptest.NewRequest(method, path, nil)
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec.Code, rec.Body.String()
}

func TestCallerAuthDisabledPassesThrough(t *testing.T) {
	h := callerAuthHandler(echoIdentity, parseCallerAuth("", ""))
	if code, body := do(h, "POST", "/api/decide", map[string]string{"X-Goog-Authenticated-User-Email": "accounts.google.com:a@x"}); code != 200 || !strings.HasPrefix(body, "accounts.google.com:a@x") {
		t.Fatalf("%d %q", code, body)
	}
}

func TestCallerAuthAPIKeys(t *testing.T) {
	h := callerAuthHandler(echoIdentity, parseCallerAuth("", "new-key, old-key"))
	for _, k := range []string{"new-key", "old-key"} {
		code, body := do(h, "POST", "/api/decide", map[string]string{client.GatewayKeyHeader: k})
		if code != 200 || !strings.HasPrefix(body, "apikey:") || strings.HasSuffix(body, k) {
			t.Fatalf("key %s: %d %q (identity must be set and the key stripped)", k, code, body)
		}
	}
	for _, hdr := range []map[string]string{
		nil,
		{client.GatewayKeyHeader: "wrong"},
		{"X-Goog-Authenticated-User-Email": "accounts.google.com:admin@x"}, // spoofed identity, no key
		{"Authorization": "Bearer new-key"},                                // keys only in X-DGem-Api-Key
	} {
		if code, _ := do(h, "POST", "/mcp", hdr); code != http.StatusUnauthorized {
			t.Fatalf("%v: %d, want 401", hdr, code)
		}
	}
	if code, _ := do(h, "GET", "/health", nil); code != 200 {
		t.Fatalf("/health: %d", code)
	}
	if code, _ := do(h, "POST", "/health", nil); code != http.StatusUnauthorized {
		t.Fatalf("POST /health must not be exempt: %d", code)
	}
	if code, _ := do(h, "GET", "/", nil); code != http.StatusUnauthorized {
		t.Fatalf("Studio without a key: %d", code)
	}
}

func TestCallerAuthIAP(t *testing.T) {
	const aud = "/projects/123/locations/us-central1/services/dgemma-gateway"
	old := iapValidate
	iapValidate = func(_ context.Context, tok, a string) (string, error) {
		if a != aud {
			return "", errors.New("audience")
		}
		if tok == "good" {
			return "user@example.com", nil
		}
		return "", errors.New("bad signature")
	}
	t.Cleanup(func() { iapValidate = old })
	h := callerAuthHandler(echoIdentity, parseCallerAuth(aud, ""))
	code, body := do(h, "GET", "/", map[string]string{iapAssertionHeader: "good", "X-Goog-Authenticated-User-Email": "accounts.google.com:spoof@x"})
	if code != 200 || !strings.HasPrefix(body, "accounts.google.com:user@example.com|") {
		t.Fatalf("valid JWT: %d %q", code, body)
	}
	for _, hdr := range []map[string]string{nil, {iapAssertionHeader: "forged"}, {"X-Goog-Authenticated-User-Email": "accounts.google.com:spoof@x"}} {
		if code, _ := do(h, "POST", "/api/decide", hdr); code != http.StatusUnauthorized {
			t.Fatalf("%v: %d", hdr, code)
		}
	}
	// IAP and keys together: either is enough.
	h2 := callerAuthHandler(echoIdentity, parseCallerAuth(aud, "k1"))
	if code, _ := do(h2, "POST", "/api/decide", map[string]string{client.GatewayKeyHeader: "k1"}); code != 200 {
		t.Fatalf("key with IAP configured: %d", code)
	}
}

// A 401 goes to callers that haven't authenticated: the same body whatever is configured, without the IAP audience
// (the project number on Cloud Run), the number of keys or the accepted methods.
func TestCallerAuth401RevealsNoConfiguration(t *testing.T) {
	const aud = "/projects/987654321/locations/us-central1/services/my-gateway"
	var bodies []string
	for _, a := range []callerAuth{parseCallerAuth(aud, ""), parseCallerAuth("", "k1,k2"), parseCallerAuth(aud, "k1")} {
		code, body := do(callerAuthHandler(echoIdentity, a), "POST", "/api/decide", map[string]string{client.GatewayKeyHeader: "wrong"})
		if code != http.StatusUnauthorized {
			t.Fatalf("%d, want 401", code)
		}
		for _, leak := range []string{"987654321", "my-gateway", "audience", "IAP", "key(s)", client.GatewayKeyHeader} {
			if strings.Contains(body, leak) {
				t.Errorf("401 body reveals %q: %s", leak, body)
			}
		}
		bodies = append(bodies, body)
	}
	if bodies[0] != bodies[1] || bodies[1] != bodies[2] {
		t.Errorf("401 bodies differ with the configuration: %q", bodies)
	}
}

// The gateway never forwards a caller key to the serving image.
func TestGatewayKeyNotSentWhenDisabled(t *testing.T) {
	t.Setenv("DGEM_GATEWAY_KEY", "k1,k2")
	if client.GatewayKeyFromEnv() != "k1" {
		t.Fatal("client should send the first key")
	}
	client.SendGatewayKey = false
	t.Cleanup(func() { client.SendGatewayKey = true })
	if client.GatewayKeyFromEnv() != "" {
		t.Fatal("gateway must not send a caller key")
	}
}
