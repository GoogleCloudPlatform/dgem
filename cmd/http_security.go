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
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"strings"
)

var (
	serveCORSOriginsFlag  string // --cors-origins / DGEM_CORS_ORIGINS
	serveAllowedHostsFlag string // --allowed-hosts / DGEM_ALLOWED_HOSTS
)

// parseCORSOrigins validates --cors-origins: a comma-separated list of exact origins (scheme://host[:port]).
// "*" is rejected: the gateway acts with its own cloud credentials, so any website could otherwise drive it.
func parseCORSOrigins(s string) ([]string, error) {
	var out []string
	for _, part := range splitList(s) {
		p := strings.TrimRight(strings.TrimSpace(part), "/")
		if p == "" {
			continue
		}
		if p == "*" {
			return nil, fmt.Errorf("--cors-origins: \"*\" is not allowed; list the exact origins (e.g. https://studio.example.com)")
		}
		u, err := url.Parse(p)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" ||
			u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.User != nil {
			return nil, fmt.Errorf("--cors-origins %q: must be an origin like https://studio.example.com or http://localhost:5173", p)
		}
		out = append(out, strings.ToLower(u.Scheme+"://"+u.Host))
	}
	return out, nil
}

// parseAllowedHosts validates --allowed-hosts: extra Host header values (host or host:port) accepted on loopback
// connections, for a reverse proxy on the same machine that forwards a public name to 127.0.0.1.
func parseAllowedHosts(s string) ([]string, error) {
	var out []string
	for _, part := range splitList(s) {
		p := strings.ToLower(strings.TrimSpace(part))
		if p == "" {
			continue
		}
		if p == "*" || strings.ContainsAny(p, "/ ") {
			return nil, fmt.Errorf("--allowed-hosts %q: must be a host name or host:port", p)
		}
		out = append(out, p)
	}
	return out, nil
}

// isLoopbackHost reports whether a host or host:port names the loopback interface (localhost, 127.0.0.0/8, ::1).
func isLoopbackHost(hostport string) bool {
	host, _, err := net.SplitHostPort(hostport)
	if err != nil {
		host = hostport
	}
	host = strings.ToLower(strings.Trim(host, "[]"))
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip, err := netip.ParseAddr(host)
	return err == nil && ip.Unmap().IsLoopback()
}

// gatewaySecurityHandler wraps the gateway mux with three browser-facing protections:
//
//  1. DNS rebinding: a request that arrived on a loopback address but carries a non-loopback Host header (a
//     website whose name was re-pointed at 127.0.0.1) is rejected with 403, unless the host is in allowedHosts.
//  2. CORS: Access-Control-Allow-* headers are sent only for origins listed in corsOrigins (none by default). The
//     embedded Studio is same-origin and needs none.
//  3. Cross-site request forgery: http.CrossOriginProtection rejects cross-origin non-GET/HEAD/OPTIONS browser
//     requests (including "simple" form or text/plain POSTs that skip preflight). Requests without Origin or
//     Sec-Fetch-Site headers, such as curl, SDKs and MCP clients, are unaffected.
func gatewaySecurityHandler(next http.Handler, corsOrigins, allowedHosts []string) (http.Handler, error) {
	allowedOrigin := make(map[string]bool, len(corsOrigins))
	cop := http.NewCrossOriginProtection()
	for _, o := range corsOrigins {
		if err := cop.AddTrustedOrigin(o); err != nil {
			return nil, fmt.Errorf("--cors-origins %q: %w", o, err)
		}
		allowedOrigin[o] = true
	}
	hostOK := make(map[string]bool, len(allowedHosts))
	for _, h := range allowedHosts {
		hostOK[h] = true
	}
	cop.SetDenyHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Forbidden: cross-origin request (allow the origin with --cors-origins)", http.StatusForbidden)
	}))
	protected := cop.Handler(next)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if la, ok := r.Context().Value(http.LocalAddrContextKey).(net.Addr); ok && la != nil {
			if isLoopbackHost(la.String()) && !isLoopbackHost(r.Host) {
				h := strings.ToLower(r.Host)
				name, _, err := net.SplitHostPort(h)
				if err != nil {
					name = h
				}
				if !hostOK[h] && !hostOK[name] {
					http.Error(w, fmt.Sprintf("Forbidden: invalid Host header %q (allow it with --allowed-hosts)", r.Host), http.StatusForbidden)
					return
				}
			}
		}
		if origin := r.Header.Get("Origin"); origin != "" {
			w.Header().Add("Vary", "Origin")
			if allowedOrigin[strings.ToLower(origin)] {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				allowHeaders := "Content-Type, Authorization, X-Requested-With, Accept, Mcp-Session-Id, Mcp-Protocol-Version"
				if reqHeaders := r.Header.Get("Access-Control-Request-Headers"); reqHeaders != "" {
					allowHeaders = reqHeaders // trusted origin: allow the X-DGem-* and trace headers it asks for
				}
				w.Header().Set("Access-Control-Allow-Headers", allowHeaders)
				w.Header().Set("Access-Control-Expose-Headers", "Mcp-Session-Id, X-Dgem-Trace-Id")
			}
			if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
				w.WriteHeader(http.StatusNoContent) // preflight; the browser enforces the (absent) allow headers
				return
			}
		}
		protected.ServeHTTP(w, r)
	}), nil
}

// defaultBindHost is the default --host for dgem serve and dgem systemone serve: loopback, so a laptop gateway isn't
// reachable from the network by accident. On Cloud Run (K_SERVICE) and Vertex (AIP_HTTP_PORT) the container must
// listen on every interface for the platform's front end, so the default there is 0.0.0.0.
func defaultBindHost() string {
	if os.Getenv("K_SERVICE") != "" || os.Getenv("AIP_HTTP_PORT") != "" {
		return "0.0.0.0"
	}
	return "127.0.0.1"
}
