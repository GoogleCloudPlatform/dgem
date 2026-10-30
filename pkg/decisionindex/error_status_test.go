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
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestErrorStatus(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want int
	}{
		{"unfetchable image (server wraps vLLM 422 as 502)", &client.HTTPError{StatusCode: 502, Body: `{"error": {"message": "upstream 422: Failed to fetch media", "type": "server_error"}}`}, 422},
		{"vLLM 400 wrapped", &client.HTTPError{StatusCode: 502, Body: `{"error": {"message": "upstream 400: maximum context length"}}`}, 422},
		{"server 400 schema error", &client.HTTPError{StatusCode: 400, Body: `question 'q': at most 26 alternatives`}, 422},
		{"server 422 (fixed images)", &client.HTTPError{StatusCode: 422, Body: `{"error": {"message": "Failed to fetch media"}}`}, 422},
		{"upstream 5xx stays a server error", &client.HTTPError{StatusCode: 502, Body: `{"error": {"message": "upstream 500: engine dead"}}`}, 500},
		{"server crash", &client.HTTPError{StatusCode: 500, Body: `boom`}, 500},
		{"invalid image ref", fmt.Errorf("%w: local path", ErrInvalidImage), 400},
		{"network error", errors.New("dial tcp: connection refused"), 500},
	} {
		if got, msg := ErrorStatus(tc.err); got != tc.want || msg == "" {
			t.Errorf("%s: status %d (%q), want %d", tc.name, got, msg, tc.want)
		}
	}
}

// End to end through `dgem systemone serve`'s handler: an image URL the model server can't fetch is the caller's
// error (422 with the upstream message), not a 500.
func TestSystemOneHandlerUnfetchableImage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error": {"message": "upstream 422: Failed to fetch media", "type": "server_error"}}`))
	}))
	defer srv.Close()
	cli := client.NewClient(srv.URL+"/v1", "dgemma", 5*time.Second)
	cli.MaxRetries = 0
	body := `{"state": "x", "images": ["https://example.com/x.png"], "questions": {"q": {"type": "choice", "criteria": {"a": "", "b": ""}}}}`
	rec := httptest.NewRecorder()
	NewSystemOneHTTPHandler(cli, DefaultEngineOptions())(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	if rec.Code != http.StatusUnprocessableEntity || !strings.Contains(rec.Body.String(), "Failed to fetch media") {
		t.Fatalf("got %d %q, want 422 with the upstream message", rec.Code, rec.Body.String())
	}
}
