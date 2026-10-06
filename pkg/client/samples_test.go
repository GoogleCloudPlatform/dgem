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

package client

import (
	"encoding/json"
	"testing"
)

func TestNormalizeSchemaSamples(t *testing.T) {
	cases := map[string]any{
		`{"questions":[],"samples":"4"}`:    float64(4),
		`{"questions":[],"samples":" 2 "}`:  float64(2),
		`{"questions":[],"samples":4}`:      float64(4),
		`{"questions":[],"samples":"auto"}`: "auto",
		`{"questions":[],"samples":"0"}`:    "0",
	}
	for in, want := range cases {
		var m map[string]any
		if err := json.Unmarshal([]byte(NormalizeSchemaSamples(in)), &m); err != nil {
			t.Fatalf("%s: %v", in, err)
		}
		if m["samples"] != want {
			t.Errorf("%s: samples = %#v, want %#v", in, m["samples"], want)
		}
	}
	if got := NormalizeSchemaSamples(`{"questions":[]}`); got != `{"questions":[]}` {
		t.Errorf("no samples: changed to %s", got)
	}
	if got := NormalizeSchemaSamples(`not json`); got != `not json` {
		t.Errorf("invalid json changed: %s", got)
	}
}
