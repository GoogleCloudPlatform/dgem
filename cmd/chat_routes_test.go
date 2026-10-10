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
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRawChatGone(t *testing.T) {
	rec := httptest.NewRecorder()
	rawChatGoneHandler(rec, httptest.NewRequest("POST", "/v1/raw/chat/completions", strings.NewReader(`{"messages":[]}`)))
	if rec.Code != http.StatusGone {
		t.Fatalf("status = %d, want 410", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "/api/decide") {
		t.Fatalf("body should point to the decision routes: %s", rec.Body.String())
	}
}

func TestChatCompletionsDeprecationHeaders(t *testing.T) {
	rec := httptest.NewRecorder()
	setChatCompletionsDeprecation(rec)
	if got := rec.Header().Get("Deprecation"); got != chatCompletionsDeprecatedAt {
		t.Fatalf("Deprecation = %q", got)
	}
	links := strings.Join(rec.Header().Values("Link"), ", ")
	if !strings.Contains(links, `</v1/systemone>; rel="successor-version"`) {
		t.Fatalf("Link = %q", links)
	}
}
