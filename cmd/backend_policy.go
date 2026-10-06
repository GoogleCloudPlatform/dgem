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
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/spf13/viper"
)

// Backend allow-list and gateway admin controls (issue #1).
//
//	--backends vertex_first,cloudrun,local   (DGEM_BACKENDS)          explicit allow-list; default: derived from URLs
//	--enable-admin-api                       (DGEM_ADMIN_API=1)       allow POST /api/backend-config and
//	                                                                  /api/vertex/deploy|teardown (off by default)
//	--allowed-vertex-endpoints id,id         (DGEM_ALLOWED_VERTEX_ENDPOINTS)
//	                                                                  extra Vertex endpoints a request may name in
//	                                                                  vertex_url / X-DGem-Vertex-Url (the configured
//	                                                                  endpoint is always allowed)
var (
	serveBackendsFlag        string
	serveAdminAPI            bool
	serveAllowedVertexFlag   string
	serveExplicitBackends    []string // validated --backends, nil when derived
	serveAllowedVertexExtras []string // normalized invoke URLs
)

var knownBackends = []string{"vertex_first", "vertex", "cloudrun", "local"}

// backendRequestError is a client error (HTTP 400): unknown, disabled or disallowed backend/endpoint.
type backendRequestError struct{ msg string }

func (e *backendRequestError) Error() string { return e.msg }

func isKnownBackend(b string) bool {
	for _, k := range knownBackends {
		if b == k {
			return true
		}
	}
	return false
}

func parseBackendList(s string) ([]string, error) {
	var out []string
	seen := map[string]bool{}
	for _, part := range splitList(s) {
		b := strings.ToLower(strings.TrimSpace(part))
		if b == "" {
			continue
		}
		if !isKnownBackend(b) {
			return nil, fmt.Errorf("unknown backend %q in --backends (valid: %s)", b, strings.Join(knownBackends, ", "))
		}
		if !seen[b] {
			seen[b] = true
			out = append(out, b)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("--backends is empty (valid: %s)", strings.Join(knownBackends, ", "))
	}
	return out, nil
}

// configuredBackends is the gateway's allow-list: the explicit --backends list, or the list derived
// from the configured endpoint URLs.
func configuredBackends() []string {
	backendConfigMu.RLock()
	explicit := serveExplicitBackends
	loc, locMode := serveLocalURL, serveLocalMode
	backendConfigMu.RUnlock()
	if explicit != nil {
		return append([]string(nil), explicit...)
	}
	return deriveAvailableBackends(effectiveVertexURL(), viper.GetString("url"), loc, locMode)
}

// effectiveVertexURL is the Vertex endpoint the gateway routes to: --vertex-url, or the DGEM_VERTEX_ENDPOINT_ID /
// DGEM_VERTEX_URL environment fallback. `dgem mcp` (stdio) never copies the environment into serveVertexURL, so
// reading only the flag hid Vertex from the allow-list there.
//
// DGEM_VERTEX_URL is returned as given, not reduced to its endpoint ID: a full /invoke URL must not need
// GCP_PROJECT_NUMBER to be rebuilt (issue #21), matching --vertex-url and `dgem serve`.
func effectiveVertexURL() string {
	backendConfigMu.RLock()
	vx := serveVertexURL
	backendConfigMu.RUnlock()
	if strings.TrimSpace(vx) != "" {
		return vx
	}
	if ep := strings.TrimSpace(os.Getenv("DGEM_VERTEX_ENDPOINT_ID")); ep != "" {
		return ep
	}
	return strings.TrimSpace(os.Getenv("DGEM_VERTEX_URL"))
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// validateBackendSetup checks --backends against the configured endpoints and the default backend.
// Called once at startup; returns an error instead of failing silently at request time.
func validateBackendSetup(vx, crURL, loc string, locMode bool, defaultBackend string, defaultExplicit bool) ([]string, error) {
	if strings.TrimSpace(serveBackendsFlag) == "" {
		return nil, nil
	}
	list, err := parseBackendList(serveBackendsFlag)
	if err != nil {
		return nil, err
	}
	var missing []string
	for _, b := range list {
		switch b {
		case "vertex", "vertex_first":
			if strings.TrimSpace(vx) == "" {
				missing = append(missing, b+" needs --vertex-url (DGEM_VERTEX_URL)")
			}
			if b == "vertex_first" && (strings.TrimSpace(crURL) == "" || isLoopbackURL(crURL)) {
				missing = append(missing, "vertex_first needs a Cloud Run failover URL (-u)")
			}
		case "cloudrun":
			if strings.TrimSpace(crURL) == "" || isLoopbackURL(crURL) {
				missing = append(missing, "cloudrun needs a remote -u URL")
			}
		case "local":
			if strings.TrimSpace(loc) == "" && !locMode && !isLoopbackURL(crURL) {
				missing = append(missing, "local needs --local or --local-url")
			}
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("--backends %s: %s", strings.Join(list, ","), strings.Join(missing, "; "))
	}
	if !containsString(list, defaultBackend) {
		if defaultExplicit {
			return nil, fmt.Errorf("--default-backend %q is not in --backends (%s)", defaultBackend, strings.Join(list, ","))
		}
	}
	return list, nil
}

// normalizedVertexIdentity reduces an invoke URL to its endpoint ID for comparisons.
func normalizedVertexIdentity(u string) string {
	if id := extractEndpointIDFromURL(u); id != "" {
		return id
	}
	return strings.TrimRight(strings.TrimSpace(u), "/")
}

// checkVertexOverride allows a request-supplied Vertex endpoint only if it is the configured endpoint or
// in --allowed-vertex-endpoints. Without this, any caller could make the gateway call an arbitrary
// endpoint with the gateway's service-account token.
func checkVertexOverride(requested, configured string) error {
	req := strings.TrimSpace(requested)
	if req == "" {
		return nil
	}
	norm, err := expandAndValidateVertexURL(req)
	if err != nil {
		return &backendRequestError{fmt.Sprintf("invalid vertex_url: %v", err)}
	}
	want := normalizedVertexIdentity(norm)
	if configured != "" && normalizedVertexIdentity(configured) == want {
		return nil
	}
	backendConfigMu.RLock()
	extras := serveAllowedVertexExtras
	backendConfigMu.RUnlock()
	for _, a := range extras {
		if normalizedVertexIdentity(a) == want {
			return nil
		}
	}
	return &backendRequestError{fmt.Sprintf("vertex_url %q is not allowed on this gateway (use the configured endpoint or add it to --allowed-vertex-endpoints)", req)}
}

func parseAllowedVertexEndpoints(s string) ([]string, error) {
	var out []string
	for _, part := range splitList(s) {
		p := strings.TrimSpace(part)
		if p == "" {
			continue
		}
		norm, err := expandAndValidateVertexURL(p)
		if err != nil {
			return nil, fmt.Errorf("--allowed-vertex-endpoints %q: %w", p, err)
		}
		out = append(out, norm)
	}
	return out, nil
}

// checkRequestedBackend validates an explicitly requested backend against the allow-list.
func checkRequestedBackend(mode string, available []string) error {
	if !isKnownBackend(mode) {
		return &backendRequestError{fmt.Sprintf("unknown backend %q (available on this gateway: %s)", mode, strings.Join(available, ", "))}
	}
	if !containsString(available, mode) {
		return &backendRequestError{fmt.Sprintf("backend %q is not enabled on this gateway (available: %s)", mode, strings.Join(available, ", "))}
	}
	return nil
}

func envBool(name string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	return v == "1" || v == "true" || v == "yes"
}

// requireAdminAPI writes 403 and returns false unless the gateway was started with --enable-admin-api.
func requireAdminAPI(w http.ResponseWriter) bool {
	if serveAdminAPI {
		return true
	}
	w.WriteHeader(http.StatusForbidden)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error": "admin API disabled: this endpoint changes gateway-wide settings or Vertex deployments; start dgem serve with --enable-admin-api (DGEM_ADMIN_API=1) to allow it",
	})
	return false
}

// splitList splits on commas or semicolons (semicolons let lists pass through gcloud --set-env-vars).
func splitList(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ';' })
}
