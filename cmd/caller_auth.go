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
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"google.golang.org/api/idtoken"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// Caller authentication for `dgem serve`. With neither option set the gateway trusts its edge (IAP or IAM in front,
// or a loopback bind), as before. With either set, every request except GET /health must carry:
//   - a valid IAP JWT (X-Goog-IAP-JWT-Assertion, signature, audience and expiry checked), when --iap-audience is set; or
//   - one of the --api-keys in X-DGem-Api-Key (for self-hosters without IAP).
//
// The caller's identity then comes from the verified JWT ("accounts.google.com:<email>") or from the key
// ("apikey:<8 hex of its sha256>"), written to X-Goog-Authenticated-User-Email for the handlers and logs. Client-supplied
// identity headers are discarded, so they can't be spoofed once verification is on. Refused requests get the same 401
// whatever is configured; the accepted methods and the IAP audience appear only in the startup log.

var (
	serveIAPAudience string // --iap-audience / DGEM_IAP_AUDIENCE
	serveAPIKeys     string // --api-keys / DGEM_GATEWAY_KEYS
)

const iapAssertionHeader = "X-Goog-IAP-JWT-Assertion"

// iapValidate verifies an IAP JWT and returns the caller's email (a variable so tests can stub Google's key fetch).
var iapValidate = func(ctx context.Context, token, audience string) (string, error) {
	p, err := idtoken.Validate(ctx, token, audience)
	if err != nil {
		return "", err
	}
	if p.Issuer != "https://cloud.google.com/iap" {
		return "", fmt.Errorf("unexpected issuer %q", p.Issuer)
	}
	email, _ := p.Claims["email"].(string)
	if email == "" {
		return "", fmt.Errorf("token has no email claim")
	}
	return email, nil
}

type callerAuth struct {
	iapAudience string
	keys        [][]byte
}

func parseCallerAuth(iapAudience, keys string) callerAuth {
	a := callerAuth{iapAudience: strings.TrimSpace(iapAudience)}
	for _, k := range strings.Split(keys, ",") {
		if k = strings.TrimSpace(k); k != "" {
			a.keys = append(a.keys, []byte(k))
		}
	}
	return a
}

func (a callerAuth) enabled() bool { return a.iapAudience != "" || len(a.keys) > 0 }

// describe lists the accepted methods for the startup log. Never put it in a reply: it includes the IAP audience,
// which holds the project number on Cloud Run.
func (a callerAuth) describe() string {
	var parts []string
	if a.iapAudience != "" {
		parts = append(parts, "IAP JWT (audience "+a.iapAudience+")")
	}
	if len(a.keys) > 0 {
		parts = append(parts, fmt.Sprintf("%d API key(s) in %s", len(a.keys), client.GatewayKeyHeader))
	}
	return strings.Join(parts, " or ")
}

// keyIdentity matches a presented key against the configured ones in constant time and returns its identity.
func (a callerAuth) keyIdentity(presented string) (string, bool) {
	got := []byte(strings.TrimSpace(presented))
	match := false
	for _, k := range a.keys {
		match = hmac.Equal(got, k) || match
	}
	if !match {
		return "", false
	}
	sum := sha256.Sum256(got)
	return "apikey:" + hex.EncodeToString(sum[:4]), true
}

func callerAuthHandler(next http.Handler, a callerAuth) http.Handler {
	if !a.enabled() {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never trust identity headers from the caller once verification is on.
		r.Header.Del("X-Goog-Authenticated-User-Email")
		r.Header.Del("X-Goog-Authenticated-User-Id")
		if r.Method == http.MethodGet && (r.URL.Path == "/health" || r.URL.Path == "/healthz") {
			next.ServeHTTP(w, r)
			return
		}
		if k := r.Header.Get(client.GatewayKeyHeader); k != "" && len(a.keys) > 0 {
			if id, ok := a.keyIdentity(k); ok {
				r.Header.Del(client.GatewayKeyHeader)
				r.Header.Set("X-Goog-Authenticated-User-Email", id)
				next.ServeHTTP(w, r)
				return
			}
		}
		if tok := r.Header.Get(iapAssertionHeader); tok != "" && a.iapAudience != "" {
			if email, err := iapValidate(r.Context(), tok, a.iapAudience); err == nil {
				r.Header.Set("X-Goog-Authenticated-User-Email", "accounts.google.com:"+email)
				next.ServeHTTP(w, r)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("WWW-Authenticate", `DGem realm="dgem gateway"`)
		w.WriteHeader(http.StatusUnauthorized)
		// The same body whatever is configured: the caller hasn't authenticated, so it learns nothing about the setup.
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
			"message": "caller authentication required",
			"type":    "authentication_error",
		}})
	})
}
