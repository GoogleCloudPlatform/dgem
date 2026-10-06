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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/spf13/viper"
)

func TestIsLoopbackURL(t *testing.T) {
	loopbacks := []string{
		"http://127.0.0.1:8080/v1",
		"http://localhost:8090/api/decide",
		"http://[::1]:8080",
		"http://0.0.0.0:8080",
		"127.0.0.1:8080",
	}
	for _, u := range loopbacks {
		if !isLoopbackURL(u) {
			t.Errorf("expected %q to be recognized as loopback URL", u)
		}
	}

	nonLoopbacks := []string{
		"https://dgemma-example-uc.a.run.app/v1",
		"https://gateway.example.com/v1",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/foo/locations/us-central1/endpoints/123:rawPredict",
	}
	for _, u := range nonLoopbacks {
		if isLoopbackURL(u) {
			t.Errorf("expected %q NOT to be recognized as loopback URL", u)
		}
	}
}

func TestResolveBackendTargetLocal(t *testing.T) {
	backendConfigMu.Lock()
	origDefB := serveDefaultBackend
	origLocURL := serveLocalURL
	serveDefaultBackend = "vertex_first"
	serveLocalURL = "http://127.0.0.1:8080/v1"
	backendConfigMu.Unlock()
	defer func() {
		backendConfigMu.Lock()
		serveDefaultBackend = origDefB
		serveLocalURL = origLocURL
		backendConfigMu.Unlock()
	}()

	ctx := context.Background()

	// 1. Explicit "local" request
	target, targetURL, err := resolveBackendTargetFromParams(ctx, "local", "")
	if err != nil {
		t.Fatalf("unexpected error resolving local backend: %v", err)
	}
	if target != "local" {
		t.Errorf("expected target 'local', got %q", target)
	}
	if targetURL != "http://127.0.0.1:8080/v1" {
		t.Errorf("expected targetURL 'http://127.0.0.1:8080/v1', got %q", targetURL)
	}

	// 2. Default backend set to "local"
	backendConfigMu.Lock()
	serveDefaultBackend = "local"
	backendConfigMu.Unlock()

	target, targetURL, err = resolveBackendTargetFromParams(ctx, "", "")
	if err != nil {
		t.Fatalf("unexpected error resolving empty backend with local default: %v", err)
	}
	if target != "local" {
		t.Errorf("expected target 'local' from default, got %q", target)
	}
	if targetURL != "http://127.0.0.1:8080/v1" {
		t.Errorf("expected targetURL 'http://127.0.0.1:8080/v1', got %q", targetURL)
	}

	// 3. Fallback to Cloud Run when unknown requested mode and default is cloudrun
	backendConfigMu.Lock()
	serveDefaultBackend = "cloudrun"
	backendConfigMu.Unlock()
	viper.Set("url", "https://dgemma-test.run.app/v1")

	target, targetURL, err = resolveBackendTargetFromParams(ctx, "cloudrun", "")
	if err != nil {
		t.Fatalf("unexpected error resolving cloudrun: %v", err)
	}
	if target != "cloudrun" {
		t.Errorf("expected target 'cloudrun', got %q", target)
	}
	if targetURL != "https://dgemma-test.run.app/v1" {
		t.Errorf("expected cloudrun targetURL, got %q", targetURL)
	}
}

func TestCheckHealthAndGPUStatusForBackendLocal(t *testing.T) {
	// Start mock local diffgemma server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" || r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]string{{"id": "diffgemma-26b-a4b-it-q4"}},
			})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer mockServer.Close()

	backendConfigMu.Lock()
	origDefB := serveDefaultBackend
	origLocURL := serveLocalURL
	serveDefaultBackend = "local"
	serveLocalURL = mockServer.URL + "/v1"
	backendConfigMu.Unlock()
	defer func() {
		backendConfigMu.Lock()
		serveDefaultBackend = origDefB
		serveLocalURL = origLocURL
		backendConfigMu.Unlock()
	}()

	gpuStateMu.Lock()
	initialWarm := lastWarmTimestamp
	gpuStateMu.Unlock()

	out := CheckHealthAndGPUStatusForBackend(context.Background(), "user@example.com", "local")

	if !out.GPUAvailable {
		t.Errorf("expected GPUAvailable true, got false")
	}
	if out.GPUState != "warm_and_ready" {
		t.Errorf("expected GPUState 'warm_and_ready', got %q", out.GPUState)
	}
	if out.ActiveBackend != "local" {
		t.Errorf("expected ActiveBackend 'local', got %q", out.ActiveBackend)
	}
	if out.GPUTier != "Apple Silicon Metal (Unified Memory)" {
		t.Errorf("expected Apple Silicon Metal tier, got %q", out.GPUTier)
	}

	// Verify Cloud Run warm timestamp was NOT modified by checking local backend
	gpuStateMu.Lock()
	afterWarm := lastWarmTimestamp
	gpuStateMu.Unlock()
	if !afterWarm.Equal(initialWarm) {
		t.Errorf("expected Cloud Run lastWarmTimestamp to remain %v, but was mutated to %v", initialWarm, afterWarm)
	}
}

func TestRecordLocalReadoutLatency(t *testing.T) {
	RecordLocalReadoutLatency(678)
	gpuStateMu.RLock()
	val := lastLocalReadoutLatencyMs
	gpuStateMu.RUnlock()
	if val != 678 {
		t.Errorf("expected lastLocalReadoutLatencyMs 678, got %d", val)
	}

	// Negative or zero shouldn't overwrite
	RecordLocalReadoutLatency(0)
	gpuStateMu.RLock()
	val2 := lastLocalReadoutLatencyMs
	gpuStateMu.RUnlock()
	if val2 != 678 {
		t.Errorf("expected lastLocalReadoutLatencyMs to stay 678, got %d", val2)
	}
}

func TestDeriveAvailableBackends(t *testing.T) {
	// 1. Local-only mode
	b1 := deriveAvailableBackends("https://endpoint.prediction.vertexai.goog/v1", "https://dgemma.run.app/v1", "", true)
	if len(b1) != 1 || b1[0] != "local" {
		t.Errorf("expected [local] when locMode=true, got %v", b1)
	}

	// 2. No vertex, loopback Cloud Run URL -> local only
	b2 := deriveAvailableBackends("", "http://127.0.0.1:8080/v1", "", false)
	if len(b2) != 1 || b2[0] != "local" {
		t.Errorf("expected [local] when unconfigured, got %v", b2)
	}

	// 3. Remote Cloud Run only
	b3 := deriveAvailableBackends("", "https://dgemma-test-uc.a.run.app/v1", "", false)
	if len(b3) != 1 || b3[0] != "cloudrun" {
		t.Errorf("expected [cloudrun] when Cloud Run configured, got %v", b3)
	}

	// 4. Vertex and remote Cloud Run
	b4 := deriveAvailableBackends("https://endpoint.prediction.vertexai.goog/v1", "https://dgemma-test-uc.a.run.app/v1", "", false)
	if len(b4) != 3 || b4[0] != "vertex_first" || b4[1] != "vertex" || b4[2] != "cloudrun" {
		t.Errorf("expected [vertex_first, vertex, cloudrun], got %v", b4)
	}

	// 5. Remote Cloud Run and local Metal
	b5 := deriveAvailableBackends("", "https://dgemma-test-uc.a.run.app/v1", "http://127.0.0.1:8080/v1", false)
	if len(b5) != 2 || b5[0] != "cloudrun" || b5[1] != "local" {
		t.Errorf("expected [cloudrun, local], got %v", b5)
	}
}

func TestServePortDefault(t *testing.T) {
	flag := serveCmd.Flags().Lookup("port")
	if flag == nil {
		t.Fatalf("serve flag --port not found")
	}
	if flag.DefValue != "8090" {
		t.Errorf("expected default serve port '8090', got %q", flag.DefValue)
	}
}
