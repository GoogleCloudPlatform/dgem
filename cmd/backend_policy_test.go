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

	"github.com/spf13/viper"
)

// withBackendConfig sets the gateway's backend state for one test and restores it afterwards.
func withBackendConfig(t *testing.T, vx, cr, loc string, locMode bool, def string, explicit []string, extras []string) {
	t.Helper()
	backendConfigMu.Lock()
	o := struct {
		vx, loc, def   string
		locMode        bool
		explicit, extr []string
	}{serveVertexURL, serveLocalURL, serveDefaultBackend, serveLocalMode, serveExplicitBackends, serveAllowedVertexExtras}
	serveVertexURL, serveLocalURL, serveLocalMode, serveDefaultBackend = vx, loc, locMode, def
	serveExplicitBackends, serveAllowedVertexExtras = explicit, extras
	backendConfigMu.Unlock()
	oURL := viper.GetString("url")
	viper.Set("url", cr)
	t.Cleanup(func() {
		backendConfigMu.Lock()
		serveVertexURL, serveLocalURL, serveDefaultBackend, serveLocalMode = o.vx, o.loc, o.def, o.locMode
		serveExplicitBackends, serveAllowedVertexExtras = o.explicit, o.extr
		backendConfigMu.Unlock()
		viper.Set("url", oURL)
	})
}

const (
	testVX    = "https://111.us-central1-222.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions"
	testOther = "https://999.us-central1-222.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/999/invoke/v1/chat/completions"
	testCR    = "https://dgemma-test.a.run.app/v1"
)

func isBackendRequestError(err error) bool {
	var e *backendRequestError
	return errors.As(err, &e)
}

func TestParseBackendList(t *testing.T) {
	got, err := parseBackendList(" cloudrun; LOCAL ,cloudrun")
	if err != nil || strings.Join(got, ",") != "cloudrun,local" {
		t.Fatalf("got %v, %v", got, err)
	}
	if _, err := parseBackendList("cloudrun,banana"); err == nil {
		t.Fatal("expected error for unknown backend")
	}
	if _, err := parseBackendList(" , "); err == nil {
		t.Fatal("expected error for empty list")
	}
}

func TestValidateBackendSetup(t *testing.T) {
	orig := serveBackendsFlag
	t.Cleanup(func() { serveBackendsFlag = orig })

	serveBackendsFlag = ""
	if list, err := validateBackendSetup("", testCR, "", false, "vertex_first", false); err != nil || list != nil {
		t.Fatalf("unset --backends should derive: %v %v", list, err)
	}

	serveBackendsFlag = "cloudrun,local"
	if list, err := validateBackendSetup("", testCR, "http://127.0.0.1:8080/v1", false, "cloudrun", true); err != nil || len(list) != 2 {
		t.Fatalf("valid config rejected: %v %v", list, err)
	}

	serveBackendsFlag = "vertex,cloudrun"
	if _, err := validateBackendSetup("", testCR, "", false, "cloudrun", true); err == nil || !strings.Contains(err.Error(), "vertex-url") {
		t.Fatalf("expected missing vertex-url error, got %v", err)
	}

	serveBackendsFlag = "cloudrun"
	if _, err := validateBackendSetup("", "http://127.0.0.1:8080/v1", "", false, "cloudrun", true); err == nil {
		t.Fatal("expected error: cloudrun with a loopback -u")
	}

	serveBackendsFlag = "cloudrun"
	if _, err := validateBackendSetup("", testCR, "", false, "vertex_first", true); err == nil || !strings.Contains(err.Error(), "--default-backend") {
		t.Fatalf("expected default-not-in-list error, got %v", err)
	}
}

func TestResolveRejectsUnknownAndDisabledBackends(t *testing.T) {
	withBackendConfig(t, "", testCR, "", false, "cloudrun", []string{"cloudrun"}, nil)
	ctx := context.Background()

	if _, _, err := resolveBackendTargetFromParams(ctx, "banana", ""); err == nil || !isBackendRequestError(err) {
		t.Fatalf("unknown backend: want backendRequestError, got %v", err)
	}
	_, _, err := resolveBackendTargetFromParams(ctx, "local", "")
	if err == nil || !isBackendRequestError(err) || !strings.Contains(err.Error(), "available: cloudrun") {
		t.Fatalf("disabled backend: want error listing cloudrun, got %v", err)
	}
	target, u, err := resolveBackendTargetFromParams(ctx, "", "")
	if err != nil || target != "cloudrun" || u != testCR {
		t.Fatalf("default: got %q %q %v", target, u, err)
	}
}

func TestResolveDefaultFallsBackToAvailable(t *testing.T) {
	// A stale default (e.g. vertex_first with no Vertex configured) routes to the first available backend.
	withBackendConfig(t, "", testCR, "", false, "vertex_first", nil, nil)
	target, _, err := resolveBackendTargetFromParams(context.Background(), "", "")
	if err != nil || target != "cloudrun" {
		t.Fatalf("got %q %v", target, err)
	}
}

func TestVertexOverrideAllowList(t *testing.T) {
	withBackendConfig(t, testVX, testCR, "", false, "cloudrun", nil, nil)
	ctx := context.Background()

	_, _, err := resolveBackendTargetFromParams(ctx, "cloudrun", testOther)
	if err == nil || !isBackendRequestError(err) || !strings.Contains(err.Error(), "not allowed") {
		t.Fatalf("foreign vertex_url must be rejected, got %v", err)
	}
	if err := checkVertexOverride(testVX, testVX); err != nil {
		t.Fatalf("configured endpoint must be allowed: %v", err)
	}
	if err := checkVertexOverride("", testVX); err != nil {
		t.Fatalf("no override must be allowed: %v", err)
	}

	backendConfigMu.Lock()
	serveAllowedVertexExtras = []string{testOther}
	backendConfigMu.Unlock()
	if err := checkVertexOverride(testOther, testVX); err != nil {
		t.Fatalf("allow-listed endpoint rejected: %v", err)
	}
}

func TestRequireAdminAPI(t *testing.T) {
	orig := serveAdminAPI
	t.Cleanup(func() { serveAdminAPI = orig })

	serveAdminAPI = false
	rec := httptest.NewRecorder()
	if requireAdminAPI(rec) || rec.Code != http.StatusForbidden || !strings.Contains(rec.Body.String(), "--enable-admin-api") {
		t.Fatalf("disabled admin API: code %d body %s", rec.Code, rec.Body.String())
	}
	serveAdminAPI = true
	rec = httptest.NewRecorder()
	if !requireAdminAPI(rec) || rec.Code != http.StatusOK {
		t.Fatalf("enabled admin API should pass, code %d", rec.Code)
	}
}

// Regression (PR #18): `dgem mcp` over stdio with the endpoint only in DGEM_VERTEX_URL must still offer
// vertex_first / vertex, as request routing always did.
func TestAllowListUsesVertexEnvFallback(t *testing.T) {
	withBackendConfig(t, "", "http://127.0.0.1:8080/v1", "", false, "vertex_first", nil, nil)
	t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "")
	t.Setenv("DGEM_VERTEX_URL", testVX)
	got := configuredBackends()
	if !containsString(got, "vertex_first") || !containsString(got, "vertex") {
		t.Fatalf("DGEM_VERTEX_URL ignored by the allow-list: %v", got)
	}
	if err := checkRequestedBackend("vertex_first", got); err != nil {
		t.Fatalf("vertex_first rejected: %v", err)
	}
	// The env endpoint counts as the configured endpoint for vertex_url checks.
	if err := checkVertexOverride(testVX, effectiveVertexURL()); err != nil {
		t.Fatalf("env endpoint rejected as override: %v", err)
	}

	t.Setenv("DGEM_VERTEX_URL", "")
	if got := configuredBackends(); containsString(got, "vertex") {
		t.Fatalf("no Vertex configured but allow-list has it: %v", got)
	}
}
