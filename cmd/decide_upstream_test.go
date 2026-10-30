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
	"errors"
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// /api/decide: a request the serving image refused is the caller's (4xx, WARNING) and is never cold-start retried;
// unreachable or 5xx upstreams stay 502.
func TestDecideUpstreamStatus(t *testing.T) {
	refused := &client.HTTPError{StatusCode: 400, Body: `{"error": {"message": "question 'wide_choice': at most 26 alternatives", "type": "invalid_request_error"}}`}
	for _, tc := range []struct {
		name  string
		err   error
		want  int
		retry bool
	}{
		{"server refused (>26 options)", refused, 422, false},
		{"server refused, wrapped", fmt.Errorf("decide: %w", refused), 422, false},
		{"server 422", &client.HTTPError{StatusCode: 422, Body: `{"error": {"message": "Failed to fetch media"}}`}, 422, false},
		{"rate limited", &client.HTTPError{StatusCode: 429, Body: `slow down`}, 502, true},
		{"server 503 (cold start)", &client.HTTPError{StatusCode: 503, Body: `Service Unavailable`}, 502, true},
		{"server crash", &client.HTTPError{StatusCode: 500, Body: `boom`}, 502, false},
		{"unreachable", errors.New("dial tcp: Connection refused"), 502, true},
	} {
		if got := decideUpstreamStatus(tc.err); got != tc.want {
			t.Errorf("%s: status %d, want %d", tc.name, got, tc.want)
		}
		if got := isColdStartRetryable(tc.err); got != tc.retry {
			t.Errorf("%s: retryable %v, want %v", tc.name, got, tc.retry)
		}
	}
}
