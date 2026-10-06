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

package template

import (
	"testing"
)

func TestRenderString(t *testing.T) {
	e := NewEngine()
	data := map[string]interface{}{
		"Name":   "DiffusionGemma",
		"Active": true,
	}

	tmpl := `Model: {{ .Name }}, Active: {{ .Active }}, Upper: {{ upper .Name }}`
	out, err := e.RenderString("test", tmpl, data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "Model: DiffusionGemma, Active: true, Upper: DIFFUSIONGEMMA"
	if out != expected {
		t.Errorf("expected %q, got %q", expected, out)
	}
}

func TestParseStructuredPayload(t *testing.T) {
	input := `{
		"schema": {
			"instructions": "Test",
			"questions": [{"id": "q1", "type": "boolean"}]
		},
		"state": {
			"ticket": "sample ticket"
		}
	}`

	schema, state, err := ParseStructuredPayload(input, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if schema == "" || state == "" {
		t.Fatalf("expected non-empty schema and state, got schema=%q state=%q", schema, state)
	}
}
