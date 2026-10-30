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
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
)

// GET /health is open to callers that haven't authenticated (callerAuthHandler lets it through), so it reports
// liveness and version only, never backends or upstream URLs.
func TestGatewayHealthRevealsNoConfiguration(t *testing.T) {
	withBackendConfig(t, testVX, testCR, "", false, "vertex_first", nil, nil)
	auth := parseCallerAuth("/projects/987654321/locations/us-central1/services/my-gateway", "k1")
	rec := httptest.NewRecorder()
	callerAuthHandler(http.HandlerFunc(gatewayHealth), auth).ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /health without credentials: %d", rec.Code)
	}
	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	var keys []string
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if want := "body_logging,revision,service,status,version"; strings.Join(keys, ",") != want {
		t.Errorf("/health fields %v, want %s", keys, want)
	}
	for _, leak := range []string{"vertexai.goog", "run.app", "987654321", "vertex_first"} {
		if strings.Contains(rec.Body.String(), leak) {
			t.Errorf("/health reveals %q: %s", leak, rec.Body.String())
		}
	}
}
