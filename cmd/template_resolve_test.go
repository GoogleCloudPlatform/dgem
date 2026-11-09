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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/template"
)

func writeTemplateTree(t *testing.T) (root, outside string) {
	t.Helper()
	base := t.TempDir()
	root = filepath.Join(base, "templates")
	for _, rel := range []string{
		"support_triage.json.tmpl",
		"calibration/grounding_claim_check.json.tmpl",
		"multimodal/bbox_localization.json.tmpl",
		"rai/prompt_safety_guardrail.json.tmpl",
		"custom/nested/deep_policy.json.tmpl",
	} {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// A template-shaped file next to (outside) the templates directory.
	outside = filepath.Join(base, "secret.json.tmpl")
	if err := os.WriteFile(outside, []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	return root, outside
}

func TestResolveTemplateFileValidNames(t *testing.T) {
	root, _ := writeTemplateTree(t)
	cases := map[string]string{
		"":                                  "support_triage",
		"support_triage":                    "support_triage",
		"/support_triage":                   "support_triage",
		"support_triage.json.tmpl":          "support_triage",
		"grounding_claim_check":             "calibration/grounding_claim_check",
		"calibration/grounding_claim_check": "calibration/grounding_claim_check",
		"factuality_grounding":              "calibration/grounding_claim_check",
		"bbox_single":                       "multimodal/bbox_localization",
		"prompt_safety_guardrail":           "rai/prompt_safety_guardrail",
		"rai/prompt_safety_guardrail":       "rai/prompt_safety_guardrail",
		"deep_policy":                       "custom/nested/deep_policy",
		"custom/nested/deep_policy":         "custom/nested/deep_policy",
	}
	for name, wantID := range cases {
		path, id, err := resolveTemplateFile(root, name)
		if err != nil {
			t.Errorf("%q: unexpected error %v", name, err)
			continue
		}
		if id != wantID {
			t.Errorf("%q: id %q, want %q", name, id, wantID)
		}
		if !isWithinDir(root, path) {
			t.Errorf("%q: path %q escapes %q", name, path, root)
		}
	}
}

func TestResolveTemplateFileRejectsTraversal(t *testing.T) {
	root, outside := writeTemplateTree(t)
	for _, name := range []string{
		"../secret",
		"../secret.json.tmpl",
		"/../secret",
		"calibration/../../secret",
		"calibration/../support_triage",
		"../../../../etc/passwd",
		"..",
		"..\\secret",
		"calibration\\..\\..\\secret",
		"//etc/passwd",
		"support_triage\x00",
	} {
		path, _, err := resolveTemplateFile(root, name)
		if err == nil {
			t.Errorf("%q: expected rejection, resolved to %q", name, path)
		}
	}
	// An absolute path loses its leading slash and is looked up under root; it never resolves to the real file.
	if path, _, err := resolveTemplateFile(root, outside); err == nil && (path == outside || !isWithinDir(root, path)) {
		t.Errorf("absolute path resolved outside root: %q", path)
	}
}

func TestIsWithinDir(t *testing.T) {
	root := filepath.Join(string(filepath.Separator), "srv", "templates")
	in := []string{root, filepath.Join(root, "a.json.tmpl"), filepath.Join(root, "x", "..b.json.tmpl")}
	out := []string{filepath.Join(root, ".."), filepath.Join(root, "..", "x"), filepath.Join(string(filepath.Separator), "etc")}
	for _, p := range in {
		if !isWithinDir(root, p) {
			t.Errorf("%q should be within %q", p, root)
		}
	}
	for _, p := range out {
		if isWithinDir(root, p) {
			t.Errorf("%q should not be within %q", p, root)
		}
	}
}

func TestSecOpsAndRAITemplatesRenderValidJSON(t *testing.T) {
	eng := template.NewEngine()
	policies := []string{
		"secops_rag_quarantine.json.tmpl",
		"secops_iam_change_audit.json.tmpl",
		"rai/prompt_safety_guardrail.json.tmpl",
		"rai/response_safety_grader.json.tmpl",
		"rai/custom_content_policy.json.tmpl",
		"rai/unbiased_context_qa.json.tmpl",
		"rai/provenance_claim_check.json.tmpl",
	}
	for _, rel := range policies {
		t.Run(rel, func(t *testing.T) {
			tmplPath := filepath.Join("..", "templates", filepath.FromSlash(rel))
			rendered, err := eng.RenderFile(tmplPath, nil)
			if err != nil {
				t.Fatalf("RenderFile(%q, nil) unexpected error: %v", tmplPath, err)
			}
			schemaStr, stateStr, err := template.ParseStructuredPayload(rendered, nil)
			if err != nil {
				t.Fatalf("ParseStructuredPayload(%q) unexpected error: %v", tmplPath, err)
			}
			var schema struct {
				SeedToken string            `json:"seed_token"`
				Samples   int               `json:"samples"`
				Questions []json.RawMessage `json:"questions"`
			}
			if err := json.Unmarshal([]byte(schemaStr), &schema); err != nil {
				t.Fatalf("json.Unmarshal(schema of %q) error: %v", tmplPath, err)
			}
			if schema.SeedToken != "fixed" {
				t.Errorf("%q seed_token = %q, want %q", rel, schema.SeedToken, "fixed")
			}
			if schema.Samples != 1 {
				t.Errorf("%q samples = %d, want 1", rel, schema.Samples)
			}
			if len(schema.Questions) < 2 || len(schema.Questions) > 8 {
				t.Errorf("%q len(questions) = %d, want in [2, 8]", rel, len(schema.Questions))
			}
			var state map[string]any
			if err := json.Unmarshal([]byte(stateStr), &state); err != nil || len(state) == 0 {
				t.Errorf("json.Unmarshal(state of %q) = (%v, %v), want non-empty object", tmplPath, state, err)
			}
		})
	}
}

func TestDiscoverTemplatesCatalogAndSampleVars(t *testing.T) {
	catalog, err := discoverTemplates(filepath.Join("..", "templates"))
	if err != nil {
		t.Fatalf("discoverTemplates(../templates) error: %v", err)
	}
	if len(catalog) < 35 {
		t.Fatalf("discoverTemplates returned %d entries, want at least 35", len(catalog))
	}

	eng := template.NewEngine()
	byID := make(map[string]TemplateCatalogEntry, len(catalog))
	for _, entry := range catalog {
		byID[entry.ID] = entry
		if len(entry.Examples) < 3 {
			t.Errorf("template %q has %d examples in catalog, want >= 3", entry.ID, len(entry.Examples))
		}
		vars := make(map[string]any, len(entry.SampleVars))
		for k, v := range entry.SampleVars {
			vars[k] = v
		}
		vars = template.CoerceJSONVariableStrings(vars)
		rendered, err := eng.RenderFile(entry.Path, vars)
		if err != nil {
			t.Errorf("RenderFile(%q, SampleVars) error: %v", entry.ID, err)
			continue
		}
		if err := template.ValidateRenderedPayload(rendered, vars, false, nil); err != nil {
			t.Errorf("ValidateRenderedPayload(%q, SampleVars) error: %v", entry.ID, err)
		}
	}

	if iam, ok := byID["secops_iam_change_audit"]; !ok {
		t.Error("missing secops_iam_change_audit in catalog")
	} else {
		for _, v := range iam.Variables {
			if v == "com" || v == "iam" || v == "gserviceaccount" {
				t.Errorf("secops_iam_change_audit extracted bogus variable %q from string literal: %v", v, iam.Variables)
			}
		}
	}

	if audio, ok := byID["multimodal/call_quality_audio"]; !ok {
		t.Error("missing multimodal/call_quality_audio in catalog")
	} else if audio.Multimodal {
		t.Error("multimodal/call_quality_audio should have Multimodal=false (text-only transcript template)")
	}
}
