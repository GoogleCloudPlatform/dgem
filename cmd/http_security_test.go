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
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type testAddr string

func (a testAddr) Network() string { return "tcp" }
func (a testAddr) String() string  { return string(a) }

func secureTestHandler(t *testing.T, origins, hosts []string) http.Handler {
	t.Helper()
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	h, err := gatewaySecurityHandler(ok, origins, hosts)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// request builds a request as received on localAddr with the given Host header.
func request(method, target, localAddr, host string, hdr map[string]string) *http.Request {
	r := httptest.NewRequest(method, target, strings.NewReader("{}"))
	r.Host = host
	for k, v := range hdr {
		r.Header.Set(k, v)
	}
	if localAddr != "" {
		r = r.WithContext(context.WithValue(r.Context(), http.LocalAddrContextKey, net.Addr(testAddr(localAddr))))
	}
	return r
}

func TestGatewayNoWildcardCORS(t *testing.T) {
	h := secureTestHandler(t, nil, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodGet, "/api/templates", "127.0.0.1:8090", "localhost:8090",
		map[string]string{"Origin": "https://evil.example"}))
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("unexpected Access-Control-Allow-Origin %q", got)
	}
	// Preflight from an unknown origin gets no allow headers.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodOptions, "/mcp", "127.0.0.1:8090", "localhost:8090",
		map[string]string{"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}))
	if rec.Header().Get("Access-Control-Allow-Origin") != "" || rec.Header().Get("Access-Control-Allow-Methods") != "" {
		t.Fatalf("preflight from unknown origin got CORS headers: %v", rec.Header())
	}
}

func TestGatewayRejectsCrossSitePOST(t *testing.T) {
	h := secureTestHandler(t, nil, nil)
	for _, hdr := range []map[string]string{
		{"Origin": "https://evil.example", "Content-Type": "text/plain"},
		{"Sec-Fetch-Site": "cross-site", "Content-Type": "application/x-www-form-urlencoded"},
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, request(http.MethodPost, "/mcp", "127.0.0.1:8090", "localhost:8090", hdr))
		if rec.Code != http.StatusForbidden {
			t.Fatalf("cross-site POST %v: code %d, want 403", hdr, rec.Code)
		}
	}
}

func TestGatewayAllowsNonBrowserAndSameOrigin(t *testing.T) {
	h := secureTestHandler(t, nil, nil)
	cases := []map[string]string{
		nil, // curl, SDKs, MCP clients
		{"Origin": "http://localhost:8090"},
		{"Sec-Fetch-Site": "same-origin", "Origin": "http://localhost:8090"},
	}
	for _, hdr := range cases {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, request(http.MethodPost, "/api/decide", "127.0.0.1:8090", "localhost:8090", hdr))
		if rec.Code != http.StatusOK {
			t.Fatalf("%v: code %d, want 200", hdr, rec.Code)
		}
	}
	// Remote deployment (Cloud Run): local address is not loopback, so any Host is fine.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodPost, "/api/decide", "10.0.0.5:8080", "dgem-gw.example.run.app", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("remote Host: code %d", rec.Code)
	}
}

func TestGatewayAllowedOrigin(t *testing.T) {
	origins, err := parseCORSOrigins("https://studio.example.com/, http://localhost:5173")
	if err != nil {
		t.Fatal(err)
	}
	h := secureTestHandler(t, origins, nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodOptions, "/api/decide", "127.0.0.1:8090", "localhost:8090",
		map[string]string{"Origin": "https://studio.example.com", "Access-Control-Request-Method": "POST",
			"Access-Control-Request-Headers": "content-type,x-dgem-backend"}))
	if rec.Code != http.StatusNoContent || rec.Header().Get("Access-Control-Allow-Origin") != "https://studio.example.com" ||
		rec.Header().Get("Access-Control-Allow-Headers") != "content-type,x-dgem-backend" {
		t.Fatalf("preflight: %d %v", rec.Code, rec.Header())
	}
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodPost, "/api/decide", "127.0.0.1:8090", "localhost:8090",
		map[string]string{"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site"}))
	if rec.Code != http.StatusOK || rec.Header().Get("Access-Control-Allow-Origin") != "http://localhost:5173" {
		t.Fatalf("trusted origin POST: %d %v", rec.Code, rec.Header())
	}
}

func TestGatewayDNSRebinding(t *testing.T) {
	h := secureTestHandler(t, nil, nil)
	for _, host := range []string{"attacker.example:8090", "attacker.example", "192.168.1.10:8090"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, request(http.MethodGet, "/api/templates", "127.0.0.1:8090", host, nil))
		if rec.Code != http.StatusForbidden {
			t.Fatalf("Host %q on loopback: code %d, want 403", host, rec.Code)
		}
	}
	for _, host := range []string{"localhost:8090", "127.0.0.1:8090", "[::1]:8090", "localhost"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, request(http.MethodGet, "/api/templates", "[::1]:8090", host, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("Host %q on loopback: code %d, want 200", host, rec.Code)
		}
	}
	hosts, err := parseAllowedHosts("dgem.internal.example")
	if err != nil {
		t.Fatal(err)
	}
	h = secureTestHandler(t, nil, hosts)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodGet, "/api/templates", "127.0.0.1:8090", "dgem.internal.example:443", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("--allowed-hosts host: code %d", rec.Code)
	}
}

func TestParseCORSOrigins(t *testing.T) {
	for _, bad := range []string{"*", "studio.example.com", "https://x.example/path", "ftp://x.example", "https://u@x.example"} {
		if _, err := parseCORSOrigins(bad); err == nil {
			t.Errorf("%q: expected error", bad)
		}
	}
	got, err := parseCORSOrigins("HTTPS://Studio.Example.com")
	if err != nil || len(got) != 1 || got[0] != "https://studio.example.com" {
		t.Fatalf("got %v %v", got, err)
	}
}

// The /mcp handler keeps the SDK's DNS rebinding protection unless extra hosts are configured.
func TestMCPHandlerLocalhostProtection(t *testing.T) {
	h := newMCPHTTPHandler(false)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, request(http.MethodPost, "/mcp", "127.0.0.1:8090", "attacker.example:8090",
		map[string]string{"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}))
	if rec.Code != http.StatusForbidden {
		t.Fatalf("SDK localhost protection off: code %d", rec.Code)
	}
}
