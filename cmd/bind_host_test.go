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

import "testing"

func TestDefaultBindHost(t *testing.T) {
	for _, tc := range []struct{ kService, aip, want string }{
		{"", "", "127.0.0.1"},
		{"dgemma-gateway", "", "0.0.0.0"},
		{"", "8080", "0.0.0.0"},
	} {
		t.Setenv("K_SERVICE", tc.kService)
		t.Setenv("AIP_HTTP_PORT", tc.aip)
		if got := defaultBindHost(); got != tc.want {
			t.Errorf("K_SERVICE=%q AIP_HTTP_PORT=%q: %q, want %q", tc.kService, tc.aip, got, tc.want)
		}
	}
}
