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
	"strings"
	"testing"
)

func TestParseVertexEndpointIdentity(t *testing.T) {
	good := map[string]vertexIdentity{
		testVX: {Host: "111.us-central1-222.prediction.vertexai.goog", Project: "222", Location: "us-central1", EndpointID: "111", Dedicated: true, ProjectNum: "222"},
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111:rawPredict": {
			Host: "us-central1-aiplatform.googleapis.com", Project: "222", Location: "us-central1", EndpointID: "111", ProjectNum: "222"},
		"https://111.europe-west4-222.prediction.vertexai.goog/v1/projects/my-proj/locations/europe-west4/endpoints/111/invoke/v1/systemone": {
			Host: "111.europe-west4-222.prediction.vertexai.goog", Project: "my-proj", Location: "europe-west4", EndpointID: "111", Dedicated: true, ProjectNum: "222"},
	}
	for raw, want := range good {
		got, err := parseVertexEndpointIdentity(raw)
		if err != nil || got != want {
			t.Errorf("%s:\n got %+v %v\nwant %+v", raw, got, err, want)
		}
	}
	for _, bad := range []string{
		"http://111.us-central1-222.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		// No endpoint path (used to fall back to the configured endpoint ID).
		"https://111.us-central1-999.prediction.vertexai.goog/invoke/v1/chat/completions",
		"https://attacker.prediction.vertexai.goog/",
		// /endpoints/<id> smuggled into a query, fragment or extra segment.
		"https://999.us-central1-999.prediction.vertexai.goog/v1/projects/999/locations/us-central1/endpoints/999/invoke/v1/chat/completions?x=/endpoints/111",
		"https://evil.us-central1-999.prediction.vertexai.goog/x/../v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111/../../../999:rawPredict",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111%2F..%2F999:rawPredict",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111:rawPredict#/endpoints/1",
		// Host and path disagree.
		"https://999.us-central1-999.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		"https://111.us-central1-999.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		"https://111.europe-west4-222.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		"https://europe-west4-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111:rawPredict",
		"https://a.b.111.us-central1-222.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		// Other hosts, ports, user info.
		"https://evil.example/v1/projects/222/locations/us-central1/endpoints/111:rawPredict",
		"https://evil.example/.aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111:rawPredict",
		"https://us-central1-aiplatform.googleapis.com:8443/v1/projects/222/locations/us-central1/endpoints/111:rawPredict",
		"https://user@us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111:rawPredict",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111",
	} {
		if got, err := parseVertexEndpointIdentity(bad); err == nil {
			t.Errorf("%s: expected rejection, got %+v", bad, got)
		}
	}
}

// Regression for the endpoint-override bypasses: each of these used to normalise to the configured endpoint ID.
func TestVertexOverrideRejectsForeignEndpoints(t *testing.T) {
	withBackendConfig(t, testVX, testCR, "", false, "cloudrun", nil, nil)
	t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "111")
	t.Setenv("GCP_PROJECT_NUMBER", "222")
	for _, req := range []string{
		// Same endpoint ID in an attacker's project.
		"https://111.us-central1-999.prediction.vertexai.goog/v1/projects/999/locations/us-central1/endpoints/111/invoke/v1/chat/completions",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/999/locations/us-central1/endpoints/111:rawPredict",
		// Same endpoint ID and project, other region.
		"https://111.europe-west4-222.prediction.vertexai.goog/v1/projects/222/locations/europe-west4/endpoints/111/invoke/v1/chat/completions",
		// No /endpoints/ path: used to fall back to defaultVertexEndpointID().
		"https://999.us-central1-999.prediction.vertexai.goog/invoke/v1/chat/completions",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/999/models/x:rawPredict",
		// "/endpoints/111" injected after the real endpoint.
		"https://999.us-central1-999.prediction.vertexai.goog/v1/projects/999/locations/us-central1/endpoints/999/invoke/v1/chat/completions?q=/endpoints/111",
		"https://999.us-central1-999.prediction.vertexai.goog/v1/projects/999/locations/us-central1/endpoints/999/invoke/endpoints/111",
		"https://999.us-central1-999.prediction.vertexai.goog/x/endpoints/111/invoke/v1/chat/completions",
	} {
		err := checkVertexOverride(req, testVX)
		if err == nil || !isBackendRequestError(err) {
			t.Errorf("%s: must be rejected, got %v", req, err)
		}
	}
	// The configured endpoint, in any of its accepted spellings, is still allowed.
	for _, req := range []string{
		testVX,
		"111",
		"projects/222/locations/us-central1/endpoints/111",
		"https://111.us-central1-222.prediction.vertexai.goog/v1/projects/222/locations/us-central1/endpoints/111",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/111:rawPredict",
	} {
		if err := checkVertexOverride(req, testVX); err != nil {
			t.Errorf("%s: configured endpoint rejected: %v", req, err)
		}
	}
}

func TestExtractEndpointIDNoSubstringMatch(t *testing.T) {
	t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "111")
	cases := map[string]string{
		testVX: "111",
		"https://us-central1-aiplatform.googleapis.com/v1/projects/222/locations/us-central1/endpoints/333:rawPredict":                  "333",
		"https://999.us-central1-999.prediction.vertexai.goog/v1/projects/999/locations/us-central1/endpoints/999/invoke/endpoints/111": "999",
		"https://999.us-central1-999.prediction.vertexai.goog/invoke/v1/chat/completions":                                               "",
		"https://evil.example/?/endpoints/111":             "",
		"projects/222/locations/us-central1/endpoints/444": "444",
		"555": "555",
		"":    "111",
	}
	for in, want := range cases {
		if got := extractEndpointIDFromURL(in); got != want {
			t.Errorf("%q: got %q, want %q", in, got, want)
		}
	}
	if got := extractEndpointIDFromURL("https://a/b?/endpoints/x"); strings.Contains(got, "x") {
		t.Errorf("query smuggling: %q", got)
	}
}
