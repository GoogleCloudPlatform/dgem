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
	"sync"
	"testing"

	"github.com/spf13/viper"
)

func TestSeedTokenHelpers(t *testing.T) {
	old := seedTokenUsers
	defer func() { seedTokenUsers = old }()

	seedTokenUsers = []string{"control-loop@example.iam.gserviceaccount.com"}
	if !seedTokenCaller("Control-Loop@example.iam.gserviceaccount.com") {
		t.Error("expected case-insensitive caller match")
	}
	if seedTokenCaller("other@example.com") || seedTokenCaller("") {
		t.Error("unexpected match for unlisted caller")
	}
	seedTokenUsers = []string{"*"}
	if !seedTokenCaller("anyone@example.com") {
		t.Error("expected wildcard * match")
	}

	// Pass-through JSON body helper: listed caller gets fixed when unset, explicit value respected, unlisted untouched.
	plain := []byte(`{"state":"board\nrow 2","questions":{"q":{"type":"noul"}}}`)
	got, eff, err := applySystemOneSeedToken(plain, "")
	if err != nil || eff != "" || string(got) != string(plain) {
		t.Fatalf("unlisted: got=(%q, %q, %v), want unchanged bytes", got, eff, err)
	}

	got, eff, err = applySystemOneSeedToken(plain, "fixed")
	if err != nil || eff != "fixed" || !strings.Contains(string(got), `"seed_token":"fixed"`) {
		t.Fatalf("listed unset: got=(%q, %q, %v), want fixed injected", got, eff, err)
	}

	explicitRandom := []byte(`{"state":"s","seed_token":"random","questions":{"q":{"type":"noul"}}}`)
	got, eff, err = applySystemOneSeedToken(explicitRandom, "fixed")
	if err != nil || eff != "random" || string(got) != string(explicitRandom) {
		t.Fatalf("listed explicit random: got=(%q, %q, %v), want explicit random untouched", got, eff, err)
	}

	invalid := []byte(`{"state":"s","seed_token":"bogus","questions":{"q":{"type":"noul"}}}`)
	if _, _, err := applySystemOneSeedToken(invalid, "fixed"); err == nil {
		t.Fatal("expected error for invalid seed_token")
	}

	// /api/decide schema helper.
	baseSchema := `{"questions":[{"id":"q","type":"noul"}]}`
	sOut, sEff, err := applySchemaSeedToken(baseSchema, "", "")
	if err != nil || sEff != "" || sOut != baseSchema {
		t.Fatalf("schema unlisted: got=(%q, %q, %v)", sOut, sEff, err)
	}
	sOut, sEff, err = applySchemaSeedToken(baseSchema, "", "fixed")
	if err != nil || sEff != "fixed" || !strings.Contains(sOut, `"seed_token":"fixed"`) {
		t.Fatalf("schema listed unset: got=(%q, %q, %v)", sOut, sEff, err)
	}
	tmplRandom := `{"seed_token":"random","questions":[{"id":"q","type":"noul"}]}`
	sOut, sEff, err = applySchemaSeedToken(tmplRandom, "", "fixed")
	if err != nil || sEff != "random" || sOut != tmplRandom {
		t.Fatalf("schema template random should beat per-caller default: got=(%q, %q, %v)", sOut, sEff, err)
	}
	sOut, sEff, err = applySchemaSeedToken(tmplRandom, "fixed", "")
	if err != nil || sEff != "fixed" || !strings.Contains(sOut, `"seed_token":"fixed"`) {
		t.Fatalf("schema explicit override should win: got=(%q, %q, %v)", sOut, sEff, err)
	}
}

func TestGatewaySystemOneSeedTokenUsers_PassThroughAndAdapter(t *testing.T) {
	shutdown := initGatewayTracer(t.Context())
	defer func() { _ = shutdown(t.Context()) }()

	var mu sync.Mutex
	var upstreamBodies []string
	var upstreamPaths []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		upstreamBodies = append(upstreamBodies, string(b))
		upstreamPaths = append(upstreamPaths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/chat/completions") {
			// Adapter path response.
			content := `{"answers":{"q":{"answer":"yes","confidence":0.9,"probabilities":{"yes":0.9,"no":0.1}}},"diagnostics":{"timing":{"total_ms":12.3}}}`
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": content}}},
			})
			return
		}
		// Pass-through /v1/systemone response.
		_, _ = w.Write([]byte(`{"answers":{"q":{"type":"noul","noul":0.9,"probabilities":{"true":0.9,"false":0.1},"confidence":0.9}},"diagnostics":{"timing":{"total_ms":11.1}}}`))
	}))
	defer upstream.Close()

	oldClient := proxyHTTPClient
	proxyHTTPClient = upstream.Client()
	defer func() { proxyHTTPClient = oldClient }()

	backendConfigMu.Lock()
	oldDefault := serveDefaultBackend
	oldVertexURL := serveVertexURL
	oldExplicit := serveExplicitBackends
	oldMode := serveSystemOneMode
	oldRawUsers := systemOneRawUsers
	oldSeedUsers := seedTokenUsers
	serveDefaultBackend = "cloudrun"
	serveVertexURL = ""
	serveExplicitBackends = []string{"cloudrun"}
	serveSystemOneMode = "adapter"
	backendConfigMu.Unlock()
	defer func() {
		backendConfigMu.Lock()
		serveDefaultBackend = oldDefault
		serveVertexURL = oldVertexURL
		serveExplicitBackends = oldExplicit
		serveSystemOneMode = oldMode
		systemOneRawUsers = oldRawUsers
		seedTokenUsers = oldSeedUsers
		backendConfigMu.Unlock()
	}()

	// Point cloudrun upstream URL at our httptest server.
	oldURL := viper.GetString("url")
	viper.Set("url", upstream.URL+"/v1")
	defer viper.Set("url", oldURL)

	send := func(caller, body string, rawUsers, seedUsers []string) (string, []TraceSpanRecord) {
		t.Helper()
		systemOneRawUsers = rawUsers
		seedTokenUsers = seedUsers
		req := httptest.NewRequest(http.MethodPost, "/v1/systemone", bytes.NewBufferString(body))
		req.Header.Set("Content-Type", "application/json")
		if caller != "" {
			req.Header.Set("X-Goog-Authenticated-User-Email", "accounts.google.com:"+caller)
		}
		rec := httptest.NewRecorder()
		v1ProxyHandler(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
		}
		mu.Lock()
		last := upstreamBodies[len(upstreamBodies)-1]
		mu.Unlock()
		return last, getTraceSpansByTraceID(rec.Header().Get("X-Dgem-Trace-Id"))
	}

	listed := []string{"gev-runner@example.iam.gserviceaccount.com"}
	unsetBody := `{"state":"turn 1","questions":{"q":{"type":"noul"}}}`
	explicitRandom := `{"state":"turn 1","seed_token":"random","questions":{"q":{"type":"noul"}}}`

	// 1. Pass-through path: listed caller gets seed_token=fixed when unset, span logs dgem.seed_token=fixed.
	gotBody, spans := send("gev-runner@example.iam.gserviceaccount.com", unsetBody, []string{"*"}, listed)
	if !strings.Contains(gotBody, `"seed_token":"fixed"`) {
		t.Errorf("pass-through listed unset: upstream body = %s, want seed_token=fixed", gotBody)
	}
	if !hasSpanAttr(spans, "dgem.seed_token", "fixed") {
		t.Errorf("pass-through listed unset: missing dgem.seed_token=fixed span attribute")
	}

	// 2. Pass-through path: listed caller with explicit random keeps random.
	gotBody, spans = send("gev-runner@example.iam.gserviceaccount.com", explicitRandom, []string{"*"}, listed)
	if gotBody != explicitRandom {
		t.Errorf("pass-through explicit random modified: got %s, want %s", gotBody, explicitRandom)
	}
	if !hasSpanAttr(spans, "dgem.seed_token", "random") {
		t.Errorf("pass-through explicit random: missing dgem.seed_token=random span attribute")
	}

	// 3. Pass-through path: unlisted caller stays byte-for-byte unchanged.
	gotBody, spans = send("other@example.com", unsetBody, []string{"*"}, listed)
	if gotBody != unsetBody {
		t.Errorf("pass-through unlisted modified: got %s, want %s", gotBody, unsetBody)
	}
	if hasSpanAttr(spans, "dgem.seed_token", "fixed") {
		t.Errorf("pass-through unlisted: unexpected dgem.seed_token=fixed")
	}

	// 4. Adapter path: listed caller gets seed_token=fixed in schema, explicit random respected, unlisted omitted.
	gotBody, spans = send("gev-runner@example.iam.gserviceaccount.com", unsetBody, nil, listed)
	if !strings.Contains(gotBody, `\"seed_token\":\"fixed\"`) {
		t.Errorf("adapter listed unset: upstream body = %s, want schema seed_token=fixed", gotBody)
	}
	if !hasSpanAttr(spans, "dgem.seed_token", "fixed") {
		t.Errorf("adapter listed unset: missing dgem.seed_token=fixed span attribute")
	}

	gotBody, spans = send("gev-runner@example.iam.gserviceaccount.com", explicitRandom, nil, listed)
	if !strings.Contains(gotBody, `\"seed_token\":\"random\"`) {
		t.Errorf("adapter listed explicit random: upstream body = %s, want schema seed_token=random", gotBody)
	}
	if !hasSpanAttr(spans, "dgem.seed_token", "random") {
		t.Errorf("adapter listed explicit random: missing dgem.seed_token=random span attribute")
	}

	gotBody, _ = send("other@example.com", unsetBody, nil, listed)
	if strings.Contains(gotBody, "seed_token") {
		t.Errorf("adapter unlisted: unexpected seed_token in %s", gotBody)
	}
}

func hasSpanAttr(spans []TraceSpanRecord, key, want string) bool {
	for _, s := range spans {
		if v, ok := s.Attributes[key]; ok && v == want {
			return true
		}
	}
	return false
}
