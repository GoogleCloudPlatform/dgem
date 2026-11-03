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
	"encoding/json"
	"testing"
)

// Every shipped multimodal template renders to valid JSON (the call-quality template starts with a template comment).
func TestMultimodalTemplatesRender(t *testing.T) {
	for _, name := range []string{"call_quality_audio", "workplace_hazard_video", "ui_design_review", "pcb_defect_triage", "kyc_document_audit"} {
		out, err := NewEngine().RenderFile("../../templates/multimodal/"+name+".json.tmpl", map[string]interface{}{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var v map[string]any
		if err := json.Unmarshal([]byte(out), &v); err != nil {
			t.Fatalf("%s: rendered output is not JSON: %v", name, err)
		}
	}
}
