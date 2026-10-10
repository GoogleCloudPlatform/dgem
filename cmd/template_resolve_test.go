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
	"os"
	"path/filepath"
	"testing"
)

func writeTemplateTree(t *testing.T) (root, outside string) {
	t.Helper()
	base := t.TempDir()
	root = filepath.Join(base, "templates")
	for _, rel := range []string{
		"support_triage.json.tmpl",
		"calibration/grounding_claim_check.json.tmpl",
		"multimodal/bbox_localization.json.tmpl",
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
