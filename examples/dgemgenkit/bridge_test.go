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

package dgemgenkit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/firebase/genkit/go/genkit"
)

func TestNewSystemOnePlugin_DirectGatewayMode(t *testing.T) {
	var (
		gotPath    string
		gotBackend string
		gotGwKey   string
		gotAuth    string
	)
	mockGateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBackend = r.Header.Get("X-DGem-Backend")
		gotGwKey = r.Header.Get("X-DGem-Api-Key")
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "diffgemma-26b-a4b-it-q4",
			"answers": map[string]any{
				"urgent": map[string]any{
					"type": "noul",
					"noul": 0.99,
				},
				"team": map[string]any{
					"type":          "choice",
					"choice":        "billing",
					"confidence":    0.98,
					"probabilities": map[string]float64{"billing": 0.98, "support": 0.01, "engineering": 0.01},
				},
				"sentiment": map[string]any{
					"type":          "score",
					"score":         2.0,
					"confidence":    0.95,
					"probabilities": map[string]float64{"0": 0.01, "1": 0.04, "2": 0.95},
				},
			},
		})
	}))
	defer mockGateway.Close()

	cfg := EndpointConfig{
		SystemOneURL: mockGateway.URL,
		ModelID:      "diffgemma-26b-a4b-it-q4",
		Provider:     "dgem",
		Backend:      "vertex_first",
		GatewayKey:   "gw-secret-123",
		AuthToken:    "Bearer iam-token-xyz",
	}
	if !cfg.IsDirectSystemOne() {
		t.Fatalf("cfg.IsDirectSystemOne() = false, want true")
	}

	plugin, closeFn := NewSystemOnePlugin(cfg)
	defer closeFn()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	tmplDir := ResolveTemplatesDir("../../templates")
	dec, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), filepath.Join(tmplDir, "support_triage.json.tmpl"), map[string]any{
		"ticket":  "Double billed for renewal",
		"samples": 1,
	})
	if err != nil {
		t.Fatalf("DecideWithTemplate failed: %v", err)
	}
	if gotPath != "/v1/systemone" {
		t.Errorf("gotPath = %q, want /v1/systemone", gotPath)
	}
	if gotBackend != "vertex_first" {
		t.Errorf("gotBackend = %q, want vertex_first", gotBackend)
	}
	if gotGwKey != "gw-secret-123" {
		t.Errorf("gotGwKey = %q, want gw-secret-123", gotGwKey)
	}
	if gotAuth != "Bearer iam-token-xyz" {
		t.Errorf("gotAuth = %q, want Bearer iam-token-xyz", gotAuth)
	}
	if dec.Answers["team"].Choice != "billing" {
		t.Errorf("team choice = %q, want billing", dec.Answers["team"].Choice)
	}
}

func TestEndpointConfigModeDetection(t *testing.T) {
	cases := []struct {
		name       string
		cfg        EndpointConfig
		wantDirect bool
	}{
		{
			name:       "default local diffgemma /v1",
			cfg:        EndpointConfig{URL: "http://127.0.0.1:8080/v1"},
			wantDirect: false,
		},
		{
			name:       "explicit SystemOneURL (DGEM_GATEWAY_URL)",
			cfg:        EndpointConfig{URL: "http://127.0.0.1:8080/v1", SystemOneURL: "http://127.0.0.1:8095"},
			wantDirect: true,
		},
		{
			name:       "gateway root URL in URL field",
			cfg:        EndpointConfig{URL: "https://dgemma-gateway.a.run.app"},
			wantDirect: true,
		},
		{
			name:       "apple container host bridge gateway URL",
			cfg:        EndpointConfig{URL: "http://192.168.64.1:8095"},
			wantDirect: true,
		},
		{
			name:       "explicit /v1/systemone path in URL field",
			cfg:        EndpointConfig{URL: "http://192.168.64.1:8095/v1/systemone"},
			wantDirect: true,
		},
		{
			name:       "vertex :rawPredict endpoint",
			cfg:        EndpointConfig{URL: "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/123:rawPredict"},
			wantDirect: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.IsDirectSystemOne(); got != tc.wantDirect {
				t.Errorf("IsDirectSystemOne() = %v, want %v (desc=%s)", got, tc.wantDirect, tc.cfg.DescribeTarget())
			}
		})
	}
}
