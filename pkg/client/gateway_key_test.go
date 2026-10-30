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

package client

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// roundTripFunc answers requests in memory, so these tests open no sockets. Redirects are still followed by
// http.Client, through the client's CheckRedirect policy.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

const okChatBody = `{"choices":[{"message":{"role":"assistant","content":"{}"}}]}`

func reply(r *http.Request, code int, body string, hdr http.Header) *http.Response {
	if hdr == nil {
		hdr = http.Header{}
	}
	return &http.Response{StatusCode: code, Header: hdr, Body: io.NopCloser(strings.NewReader(body)), Request: r}
}

// DGEM_GATEWAY_KEY goes only to the scheme, host and port of DGEM_GATEWAY_URL, and nowhere without it.
func TestGatewayKeyFor(t *testing.T) {
	t.Setenv("DGEM_GATEWAY_KEY", " k1 , k2")
	for _, tc := range []struct{ name, gateway, target, want string }{
		{"same origin", "https://gw.example.com", "https://gw.example.com/v1/chat/completions", "k1"},
		{"default port and host case", "https://GW.example.com:443/v1", "https://gw.example.com/v1/chat/completions", "k1"},
		{"http with a port", "http://127.0.0.1:8090", "http://127.0.0.1:8090/v1/chat/completions", "k1"},
		{"other scheme", "https://gw.example.com", "http://gw.example.com/v1/chat/completions", ""},
		{"other scheme, same port", "https://gw.example.com:8443", "http://gw.example.com:8443/v1/chat/completions", ""},
		{"other port", "https://gw.example.com", "https://gw.example.com:8443/v1/chat/completions", ""},
		{"other host", "https://gw.example.com", "https://serving.example.com/v1/chat/completions", ""},
		{"host suffix", "https://gw.example.com", "https://gw.example.com.example.net/v1/chat/completions", ""},
		{"gateway in the query", "https://gw.example.com", "https://example.net/v1?next=https://gw.example.com", ""},
		{"gateway as user info", "https://gw.example.com", "https://gw.example.com@example.net/v1", ""},
		{"vertex endpoint", "https://gw.example.com", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/1:rawPredict", ""},
		{"gateway not a url", "gw.example.com", "https://gw.example.com/v1/chat/completions", ""},
		{"no gateway configured", "", "https://gw.example.com/v1/chat/completions", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(GatewayURLEnv, tc.gateway)
			if got := GatewayKeyFor(tc.target); got != tc.want {
				t.Errorf("GatewayKeyFor(%q) with %s=%q: %q, want %q", tc.target, GatewayURLEnv, tc.gateway, got, tc.want)
			}
		})
	}
}

// `dgem mcp --remote` sends the key to its remote unless DGEM_GATEWAY_URL names another gateway. The gateway itself
// (SendGatewayKey false) sends it nowhere.
func TestGatewayKeyForRemote(t *testing.T) {
	t.Setenv("DGEM_GATEWAY_KEY", "k1")
	remote := "https://gw.example.com/mcp"
	for _, tc := range []struct{ gateway, want string }{
		{"", "k1"},
		{"https://gw.example.com", "k1"},
		{"https://other.example.com", ""},
	} {
		t.Setenv(GatewayURLEnv, tc.gateway)
		if got := GatewayKeyForRemote(remote); got != tc.want {
			t.Errorf("GatewayKeyForRemote(%q) with %s=%q: %q, want %q", remote, GatewayURLEnv, tc.gateway, got, tc.want)
		}
	}

	SendGatewayKey = false
	t.Cleanup(func() { SendGatewayKey = true })
	t.Setenv(GatewayURLEnv, "https://gw.example.com")
	if k := GatewayKeyFor(remote) + GatewayKeyForRemote(remote); k != "" {
		t.Errorf("SendGatewayKey=false still sends %q", k)
	}
}

// Complete sends X-DGem-Api-Key to the gateway in DGEM_GATEWAY_URL and to no other endpoint.
func TestCompleteSendsGatewayKeyOnlyToGateway(t *testing.T) {
	t.Setenv("DGEM_GATEWAY_KEY", "k1")
	t.Setenv(GatewayURLEnv, "https://gw.example.com")
	for _, tc := range []struct{ base, want string }{
		{"https://gw.example.com/v1", "k1"},
		{"https://serving.example.com/v1", ""},
		{"https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/1", ""},
	} {
		var got []string
		cli := NewClient(tc.base, "dgemma", 5*time.Second)
		cli.HTTPClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
			got = append(got, r.Header.Values(GatewayKeyHeader)...)
			return reply(r, http.StatusOK, okChatBody, nil), nil
		})
		if _, _, err := cli.Complete(context.Background(), ChatCompletionRequest{}); err != nil {
			t.Fatalf("%s: %v", tc.base, err)
		}
		if strings.Join(got, ",") != tc.want {
			t.Errorf("%s: sent %s %q, want %q", tc.base, GatewayKeyHeader, got, tc.want)
		}
	}
}

// A redirect to another origin drops X-DGem-Api-Key and X-DGem-Key, also on later redirects back to the gateway; a
// redirect within the origin keeps them.
func TestCompleteDropsKeysOnCrossOriginRedirect(t *testing.T) {
	t.Setenv("DGEM_GATEWAY_KEY", "k1")
	t.Setenv("DGEM_SERVER_KEY", "s1")
	t.Setenv(GatewayURLEnv, "https://gw.example.com")
	const first = "https://gw.example.com/v1/chat/completions"
	for _, tc := range []struct {
		name string
		hops []string // redirect targets, in order
		keep bool
	}{
		{"same origin", []string{"https://gw.example.com/v2/chat/completions"}, true},
		{"other host", []string{"https://collector.example.net/v1/chat/completions"}, false},
		{"subdomain", []string{"https://eu.gw.example.com/v1/chat/completions"}, false},
		{"http", []string{"http://gw.example.com/v1/chat/completions"}, false},
		{"back to the gateway", []string{"https://collector.example.net/r", "https://gw.example.com/v2/chat/completions"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var firstHdr, final http.Header
			n := 0
			cli := NewClient("https://gw.example.com/v1", "dgemma", 5*time.Second)
			cli.HTTPClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if n == 0 {
					if r.URL.String() != first {
						t.Errorf("first request to %s, want %s", r.URL, first)
					}
					firstHdr = r.Header.Clone()
				}
				if n < len(tc.hops) {
					n++
					return reply(r, http.StatusTemporaryRedirect, "", http.Header{"Location": {tc.hops[n-1]}}), nil
				}
				final = r.Header.Clone()
				return reply(r, http.StatusOK, okChatBody, nil), nil
			})
			if _, _, err := cli.Complete(context.Background(), ChatCompletionRequest{}); err != nil {
				t.Fatal(err)
			}
			if final == nil {
				t.Fatalf("redirects to %v were not followed", tc.hops)
			}
			for h, v := range map[string]string{GatewayKeyHeader: "k1", ServerKeyHeader: "s1"} {
				if firstHdr.Get(h) != v {
					t.Errorf("first request: %s %q, want %q", h, firstHdr.Get(h), v)
				}
				if kept := final.Get(h) != ""; kept != tc.keep {
					t.Errorf("redirects to %v: %s kept %v, want %v", tc.hops, h, kept, tc.keep)
				}
			}
			if final.Get("Content-Type") != "application/json" {
				t.Errorf("redirects to %v dropped Content-Type", tc.hops)
			}
		})
	}
}

// The redirect policy itself: keys stay within the first request's origin, are not restored by a redirect back to it,
// and at most 10 redirects are followed.
func TestStripKeysOnCrossOriginRedirect(t *testing.T) {
	first, _ := http.NewRequest(http.MethodPost, "https://gw.example.com/v1/chat/completions", nil)
	next := func(u string) *http.Request {
		r, _ := http.NewRequest(http.MethodPost, u, nil)
		r.Header.Set(GatewayKeyHeader, "k1")
		r.Header.Set(ServerKeyHeader, "s1")
		return r
	}
	for _, tc := range []struct {
		target string
		keep   bool
	}{
		{"https://gw.example.com:443/v2", true},
		{"https://GW.EXAMPLE.COM/v2", true},
		{"https://gw.example.com:8443/v2", false},
		{"http://gw.example.com:443/v2", false},
		{"https://example.net/v2", false},
	} {
		r := next(tc.target)
		if err := StripKeysOnCrossOriginRedirect(r, []*http.Request{first}); err != nil {
			t.Fatalf("%s: %v", tc.target, err)
		}
		for _, h := range []string{GatewayKeyHeader, ServerKeyHeader} {
			if kept := r.Header.Get(h) != ""; kept != tc.keep {
				t.Errorf("redirect to %s: %s kept %v, want %v", tc.target, h, kept, tc.keep)
			}
		}
	}

	for _, tc := range []struct {
		name string
		via  []*http.Request
		keep bool
	}{
		{"within the origin", []*http.Request{first, next("https://gw.example.com/v2")}, true},
		{"back from another host", []*http.Request{first, next("https://example.net/r")}, false},
		{"back after two hops", []*http.Request{first, next("https://example.net/r"), next("https://gw.example.com/v2")}, false},
	} {
		r := next("https://gw.example.com/v3")
		if err := StripKeysOnCrossOriginRedirect(r, tc.via); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		for _, h := range []string{GatewayKeyHeader, ServerKeyHeader} {
			if kept := r.Header.Get(h) != ""; kept != tc.keep {
				t.Errorf("%s: %s kept %v, want %v", tc.name, h, kept, tc.keep)
			}
		}
	}

	via := make([]*http.Request, 10)
	for i := range via {
		via[i] = first
	}
	if err := StripKeysOnCrossOriginRedirect(next("https://gw.example.com/v3"), via); err == nil {
		t.Error("followed an 11th redirect")
	}
}

// A dgem gateway's 401 explains a DGEM_GATEWAY_KEY that wasn't sent; other 401s are reported as they are.
func TestGatewayKeyHintOn401(t *testing.T) {
	t.Setenv("DGEM_GATEWAY_KEY", "k1")
	dgemChallenge := http.Header{"Www-Authenticate": {`DGem realm="dgem gateway"`}}
	for _, tc := range []struct {
		name, gateway string
		hdr           http.Header
		hint          bool
	}{
		{"no gateway configured", "", dgemChallenge, true},
		{"another gateway configured", "https://other.example.com", dgemChallenge, true},
		{"key sent", "https://gw.example.com", dgemChallenge, false},
		{"not a dgem gateway", "", http.Header{"Www-Authenticate": {`Bearer realm="example"`}}, false},
		{"no challenge", "", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(GatewayURLEnv, tc.gateway)
			cli := NewClient("https://gw.example.com/v1", "dgemma", 5*time.Second)
			cli.HTTPClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
				return reply(r, http.StatusUnauthorized, "caller authentication required\n", tc.hdr.Clone()), nil
			})
			_, _, err := cli.Complete(context.Background(), ChatCompletionRequest{})
			var he *HTTPError
			if !errors.As(err, &he) || he.StatusCode != http.StatusUnauthorized {
				t.Fatalf("got %v, want an HTTP 401 error", err)
			}
			if got := he.Hint != "" && strings.Contains(err.Error(), GatewayURLEnv); got != tc.hint {
				t.Errorf("hint shown %v, want %v: %q", got, tc.hint, err.Error())
			}
			if strings.Contains(err.Error(), "k1") {
				t.Errorf("error shows the key: %q", err.Error())
			}
		})
	}
}
