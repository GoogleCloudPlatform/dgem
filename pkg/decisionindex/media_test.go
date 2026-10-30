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

package decisionindex

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestCheckMediaKeys(t *testing.T) {
	for _, body := range []string{`{"videos": []}`, `{"audio": ["data:audio/mpeg;base64,AA"]}`, `{"video": "x", "audio": 1}`} {
		if err := CheckMediaKeys([]byte(body)); !errors.Is(err, ErrUnsupportedMedia) || !strings.Contains(err.Error(), "text and images only") {
			t.Errorf("%s: %v", body, err)
		}
	}
	for _, body := range []string{`{"state": "x", "images": ["data:image/png;base64,AA"]}`, `not json`, `[1]`} {
		if err := CheckMediaKeys([]byte(body)); err != nil {
			t.Errorf("%s: unexpected %v", body, err)
		}
	}
}

// `dgem systemone serve`: a body with videos or audio is refused with 400 before any upstream call.
func TestSystemOneHandlerRefusesMedia(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { called = true }))
	defer srv.Close()
	h := NewSystemOneHTTPHandler(client.NewClient(srv.URL+"/v1", "dgemma", 2*time.Second), DefaultEngineOptions())
	for _, key := range []string{"videos", "audio"} {
		rec := httptest.NewRecorder()
		body := `{"state": "x", "questions": {"q": {"type": "noul"}}, "` + key + `": ["data:x;base64,AA"]}`
		h(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), key) {
			t.Errorf("%s: %d %q", key, rec.Code, rec.Body.String())
		}
	}
	if called {
		t.Error("upstream was called")
	}
	if st, _ := ErrorStatus(CheckMediaKeys([]byte(`{"videos": []}`))); st != http.StatusBadRequest {
		t.Errorf("ErrorStatus = %d", st)
	}
}
