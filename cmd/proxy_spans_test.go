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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type pipeAddr struct{}

func (pipeAddr) Network() string { return "pipe" }
func (pipeAddr) String() string  { return "pipe" }

type pipeListener struct {
	ch     chan net.Conn
	closed chan struct{}
	once   sync.Once
}

func newPipeListener() *pipeListener {
	return &pipeListener{
		ch:     make(chan net.Conn),
		closed: make(chan struct{}),
	}
}

func (l *pipeListener) Accept() (net.Conn, error) {
	select {
	case c := <-l.ch:
		return c, nil
	case <-l.closed:
		return nil, net.ErrClosed
	}
}

func (l *pipeListener) Close() error {
	l.once.Do(func() { close(l.closed) })
	return nil
}

func (l *pipeListener) Addr() net.Addr { return pipeAddr{} }

func (l *pipeListener) DialContext(ctx context.Context, _, _ string) (net.Conn, error) {
	c1, c2 := net.Pipe()
	select {
	case l.ch <- c2:
		return c1, nil
	case <-l.closed:
		_ = c1.Close()
		_ = c2.Close()
		return nil, net.ErrClosed
	case <-ctx.Done():
		_ = c1.Close()
		_ = c2.Close()
		return nil, ctx.Err()
	}
}

func TestProxyTransportPoolSizing(t *testing.T) {
	hc := newProxyHTTPClient()
	tr, ok := hc.Transport.(*http.Transport)
	if !ok || tr == nil {
		t.Fatalf("expected *http.Transport, got %T", hc.Transport)
	}
	if tr.MaxIdleConns < 128 {
		t.Errorf("MaxIdleConns = %d, want >= 128", tr.MaxIdleConns)
	}
	if tr.MaxIdleConnsPerHost < 64 {
		t.Errorf("MaxIdleConnsPerHost = %d, want >= 64", tr.MaxIdleConnsPerHost)
	}
	if tr.IdleConnTimeout != 90*time.Second {
		t.Errorf("IdleConnTimeout = %v, want 90s", tr.IdleConnTimeout)
	}
	if hc.CheckRedirect == nil {
		t.Error("expected CheckRedirect to strip credentials on cross-origin redirects")
	}
}

func TestV1ProxySubSpansTokenCacheAndConnReuse(t *testing.T) {
	t.Setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "")

	var logBuf bytes.Buffer
	oldW := spanLogWriter
	spanLogWriter = &logBuf
	defer func() { spanLogWriter = oldW }()

	oldTP := otel.GetTracerProvider()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(&gatewaySpanProcessor{}))
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(oldTP)

	ln := newPipeListener()
	defer func() { _ = ln.Close() }()

	var upstreamCalls atomic.Int32
	var seenAuth atomic.Value
	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.ReadAll(r.Body)
			_ = r.Body.Close()
			upstreamCalls.Add(1)
			seenAuth.Store(r.Header.Get("Authorization"))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"answers":{"q1":{"choice":"yes","probabilities":{"yes":0.9,"no":0.1}}}}`))
		}),
	}
	go func() { _ = srv.Serve(ln) }()
	defer func() { _ = srv.Close() }()

	oldClient := proxyHTTPClient
	testClient := newProxyHTTPClient()
	testClient.Transport.(*http.Transport).DialContext = ln.DialContext
	proxyHTTPClient = testClient
	defer func() { proxyHTTPClient = oldClient }()

	const pscURL = "http://123-us-central1-456.prediction.p.vertexai.goog/v1/projects/456/locations/us-central1/endpoints/123:rawPredict"
	backendConfigMu.Lock()
	oldDefaultBackend := serveDefaultBackend
	oldVertexURL := serveVertexURL
	oldExplicit := serveExplicitBackends
	oldMode := serveSystemOneMode
	oldRawUsers := systemOneRawUsers
	serveDefaultBackend = "vertex"
	serveVertexURL = pscURL
	serveExplicitBackends = []string{"vertex"}
	serveSystemOneMode = "adapter"
	systemOneRawUsers = []string{"*"}
	backendConfigMu.Unlock()
	defer func() {
		backendConfigMu.Lock()
		serveDefaultBackend = oldDefaultBackend
		serveVertexURL = oldVertexURL
		serveExplicitBackends = oldExplicit
		serveSystemOneMode = oldMode
		systemOneRawUsers = oldRawUsers
		backendConfigMu.Unlock()
	}()

	vertexStatusCacheMu.Lock()
	oldCached := vertexStatusCached
	oldExpires := vertexStatusCacheExpires
	vertexStatusCached = vertexEndpointLiveStatus{
		EndpointID:   "123",
		State:        "deployed",
		ReplicaCount: 1,
	}
	vertexStatusCacheExpires = time.Now().Add(5 * time.Minute)
	vertexStatusCacheMu.Unlock()
	defer func() {
		vertexStatusCacheMu.Lock()
		vertexStatusCached = oldCached
		vertexStatusCacheExpires = oldExpires
		vertexStatusCacheMu.Unlock()
	}()

	var mintCalls atomic.Int32
	oldMint := mintGCPAccessToken
	mintGCPAccessToken = func() (string, time.Duration) {
		mintCalls.Add(1)
		return "minted-test-token", 30 * time.Minute
	}
	defer func() { mintGCPAccessToken = oldMint }()

	tokenCacheMu.Lock()
	delete(tokenCache, "oauth2_access_token|cloud-platform")
	tokenCacheMu.Unlock()
	defer func() {
		tokenCacheMu.Lock()
		delete(tokenCache, "oauth2_access_token|cloud-platform")
		tokenCacheMu.Unlock()
	}()

	sendTurn := func(state string) (*httptest.ResponseRecorder, []TraceSpanRecord) {
		t.Helper()
		payload := `{"state":"` + state + `","questions":{"q1":{"type":"noul"}}}`
		req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Goog-Authenticated-User-Email", "accounts.google.com:caller@example.com")
		rec := httptest.NewRecorder()
		v1ProxyHandler(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
		}
		tid := rec.Header().Get("X-Dgem-Trace-Id")
		if len(tid) != 32 {
			t.Fatalf("missing X-Dgem-Trace-Id = %q", tid)
		}
		return rec, getTraceSpansByTraceID(tid)
	}

	// Turn 1: fresh token mint (cache_hit=false) and fresh upstream connection (conn.reused=false).
	_, spans1 := sendTurn("turn-1-secret-state")
	wantNames := []struct {
		name  string
		depth int
	}{
		{"dgem.gateway.proxy", 0},
		{"dgem.gateway.read_body", 1},
		{"dgem.gateway.resolve_backend", 1},
		{"dgem.gateway.prepare_body", 1},
		{"dgem.gateway.auth_token", 1},
		{"dgem.gpu.proxy_forward", 1},
		{"dgem.gateway.write_response", 1},
	}
	if len(spans1) != len(wantNames) {
		t.Fatalf("turn 1 spans = %d (%v), want %d", len(spans1), spans1, len(wantNames))
	}
	byName1 := make(map[string]TraceSpanRecord, len(spans1))
	for i, w := range wantNames {
		if spans1[i].Name != w.name || spans1[i].Depth != w.depth {
			t.Errorf("turn 1 span[%d] = (%s, depth %d), want (%s, depth %d)", i, spans1[i].Name, spans1[i].Depth, w.name, w.depth)
		}
		byName1[spans1[i].Name] = spans1[i]
	}

	if got := byName1["dgem.gateway.read_body"].Attributes["http.request_content_length"]; got == nil || got.(int64) <= 0 {
		t.Errorf("read_body content_length = %v", got)
	}
	if got := byName1["dgem.gateway.resolve_backend"].Attributes["dgem.vertex_status.cache_hit"]; got != true {
		t.Errorf("resolve_backend vertex_status.cache_hit = %v, want true", got)
	}
	if got := byName1["dgem.gateway.prepare_body"].Attributes["dgem.systemone.mode"]; got != "passthrough" {
		t.Errorf("prepare_body systemone.mode = %v, want passthrough", got)
	}
	if got := byName1["dgem.gateway.auth_token"].Attributes["dgem.token.cache_hit"]; got != false {
		t.Errorf("turn 1 token.cache_hit = %v, want false", got)
	}
	if got := byName1["dgem.gpu.proxy_forward"].Attributes["dgem.conn.reused"]; got != false {
		t.Errorf("turn 1 conn.reused = %v, want false", got)
	}
	if got := byName1["dgem.gateway.write_response"].Attributes["dgem.written_bytes"]; got == nil || got.(int64) <= 0 {
		t.Errorf("turn 1 written_bytes = %v", got)
	}
	if got, _ := seenAuth.Load().(string); got != "Bearer minted-test-token" {
		t.Errorf("upstream Authorization = %q, want Bearer minted-test-token", got)
	}

	// Turn 2: cached token (cache_hit=true) and reused keep-alive connection (conn.reused=true).
	_, spans2 := sendTurn("turn-2-secret-state")
	byName2 := make(map[string]TraceSpanRecord, len(spans2))
	for _, sp := range spans2 {
		byName2[sp.Name] = sp
	}
	if got := byName2["dgem.gateway.auth_token"].Attributes["dgem.token.cache_hit"]; got != true {
		t.Errorf("turn 2 token.cache_hit = %v, want true", got)
	}
	if got := byName2["dgem.gpu.proxy_forward"].Attributes["dgem.conn.reused"]; got != true {
		t.Errorf("turn 2 conn.reused = %v, want true", got)
	}
	if got := byName2["dgem.gpu.proxy_forward"].Attributes["dgem.conn.was_idle"]; got != true {
		t.Errorf("turn 2 conn.was_idle = %v, want true", got)
	}
	if mintCalls.Load() != 1 {
		t.Errorf("mintGCPAccessToken called %d times across 2 turns, want 1", mintCalls.Load())
	}
	if upstreamCalls.Load() != 2 {
		t.Errorf("upstream called %d times, want 2", upstreamCalls.Load())
	}
	if strings.Contains(logBuf.String(), "secret-state") {
		t.Error("span logs leaked caller request content")
	}
}
