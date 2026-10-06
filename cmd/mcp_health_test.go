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
	"strings"
	"testing"
	"time"
)

// seedVertexStatus pre-populates the Vertex endpoint status cache so
// CheckHealthAndGPUStatusForBackend never calls Vertex or ADC in tests.
func seedVertexStatus(t *testing.T, st vertexEndpointLiveStatus) {
	t.Helper()
	vertexStatusCacheMu.Lock()
	oSt, oExp := vertexStatusCached, vertexStatusCacheExpires
	vertexStatusCached, vertexStatusCacheExpires = st, time.Now().Add(time.Hour)
	vertexStatusCacheMu.Unlock()
	t.Cleanup(func() {
		vertexStatusCacheMu.Lock()
		vertexStatusCached, vertexStatusCacheExpires = oSt, oExp
		vertexStatusCacheMu.Unlock()
	})
}

// Issue #21: with the Vertex endpoint only in DGEM_VERTEX_URL (how `dgem mcp`
// over stdio is configured), get_health_and_gpu_status must report the same
// upstream_url as when it is passed with --vertex-url.
func TestHealthReportsVertexEndpointFromEnv(t *testing.T) {
	for _, state := range []string{"quiesced", "deployed"} {
		t.Run(state, func(t *testing.T) {
			withBackendConfig(t, "", testCR, "", false, "vertex_first", nil, nil)
			t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "")
			t.Setenv("DGEM_VERTEX_URL", testVX)
			st := vertexEndpointLiveStatus{EndpointID: "111", State: state, MachineType: "g4-standard-48", Message: "seeded " + state}
			if state == "deployed" {
				st.ReplicaCount = 1
			}
			seedVertexStatus(t, st)

			out := CheckHealthAndGPUStatusForBackend(context.Background(), "", "vertex")
			if out.ActiveBackend != "vertex" {
				t.Fatalf("active_backend = %q, want vertex (detail: %s)", out.ActiveBackend, out.Detail)
			}
			if out.UpstreamURL != testVX {
				t.Errorf("upstream_url = %q, want the DGEM_VERTEX_URL endpoint %q", out.UpstreamURL, testVX)
			}
		})
	}
}

// Issue #21: a full /invoke URL in DGEM_VERTEX_URL must route without GCP_PROJECT_NUMBER, as --vertex-url does.
// Previously the env fallback reduced it to the bare endpoint ID, which then needed the project number to expand.
func TestRoutingUsesFullVertexURLFromEnv(t *testing.T) {
	withBackendConfig(t, "", testCR, "", false, "vertex_first", nil, nil)
	t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "")
	t.Setenv("DGEM_VERTEX_URL", testVX)
	t.Setenv("GCP_PROJECT_NUMBER", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT_NUMBER", "")
	t.Setenv("DGEM_GCP_PROJECT_NUMBER", "")
	if detectGCPProjectNumber() != "" {
		t.Skip("a GCP project number is still discoverable in this environment")
	}
	seedVertexStatus(t, vertexEndpointLiveStatus{EndpointID: "111", State: "quiesced", Message: "seeded"})

	_, target, err := resolveBackendTargetFromParams(context.Background(), "vertex", "")
	if err != nil && !strings.Contains(err.Error(), "quiesced") {
		t.Fatalf("routing to the DGEM_VERTEX_URL endpoint failed: %v", err)
	}
	if target != testVX {
		t.Errorf("target = %q, want %q", target, testVX)
	}
}

// The flag keeps precedence over the environment.
func TestHealthPrefersVertexFlagOverEnv(t *testing.T) {
	withBackendConfig(t, testVX, testCR, "", false, "vertex_first", nil, nil)
	t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "")
	t.Setenv("DGEM_VERTEX_URL", testOther)
	seedVertexStatus(t, vertexEndpointLiveStatus{EndpointID: "111", State: "quiesced", Message: "seeded"})

	out := CheckHealthAndGPUStatusForBackend(context.Background(), "", "vertex")
	if out.UpstreamURL != testVX {
		t.Errorf("upstream_url = %q, want the --vertex-url endpoint %q", out.UpstreamURL, testVX)
	}
}
