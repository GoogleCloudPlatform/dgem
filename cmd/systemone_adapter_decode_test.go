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
	"net/http"
	"testing"
)

// The gateway's adapter mode refuses bodies it can't read instead of forwarding them unbatched, and merges
// "media" into "images" before the image checks.
func TestDecodeSystemOneAdapterRequest(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		want       int
	}{
		{"score question", `{"state": "s", "questions": {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}}`, http.StatusOK},
		{"not JSON", `questions: yes`, http.StatusBadRequest},
		{"serving-image schema (questions as a list)", `{"questions": [{"id": "q", "type": "boolean"}]}`, http.StatusBadRequest},
		{"no questions", `{"state": "s"}`, http.StatusBadRequest},
		{"criteria of the wrong shape", `{"questions": {"q": {"type": "choice", "criteria": 3}}}`, http.StatusBadRequest},
		{"audio media", `{"questions": {"q": {"type": "noul"}}, "media": [{"mime_type": "audio/wav", "data": "AA=="}]}`, http.StatusBadRequest},
	} {
		req, status, err := decodeSystemOneAdapterRequest([]byte(tc.body))
		if status != tc.want || (tc.want == http.StatusOK) != (err == nil) {
			t.Errorf("%s: status %d err %v, want %d", tc.name, status, err, tc.want)
		}
		if tc.name == "score question" && len(req.Questions["q"].Levels) != 3 {
			t.Errorf("levels not parsed: %+v", req.Questions["q"])
		}
	}
	req, status, err := decodeSystemOneAdapterRequest([]byte(`{"questions": {"q": {"type": "noul"}}, "images": ["data:image/png;base64,AA"], "media": [{"mime_type": "image/jpeg", "data": "BB", "name": "f.jpg"}]}`))
	if err != nil || status != http.StatusOK || len(req.Images) != 2 || req.Images[1] != "data:image/jpeg;base64,BB" {
		t.Fatalf("media merge: %d %v %v", status, err, req.Images)
	}
}
