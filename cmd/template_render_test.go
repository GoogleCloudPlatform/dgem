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
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTemplateRenderDataFileAndOverrides(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "vars.json")
	if err := os.WriteFile(dataPath, []byte(`{"ticket":"Invoice double charge on #4412","samples":2}`), 0o644); err != nil {
		t.Fatal(err)
	}

	origTmpl, origData, origVars := renderTemplate, renderDataFile, renderVars
	t.Cleanup(func() {
		renderTemplate = origTmpl
		renderDataFile = origData
		renderVars = origVars
	})

	renderTemplate = filepath.Join("..", "templates", "support_triage.json.tmpl")
	renderDataFile = dataPath
	renderVars = []string{"samples=1"}

	var out bytes.Buffer
	templateRenderCmd.SetOut(&out)
	if err := templateRenderCmd.RunE(templateRenderCmd, nil); err != nil {
		t.Fatalf("templateRenderCmd.RunE error: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "Invoice double charge on #4412") {
		t.Errorf("rendered output missing -d data file value:\n%s", got)
	}
	if !strings.Contains(got, `"samples": "1"`) && !strings.Contains(got, `"samples": 1`) {
		t.Errorf("rendered output missing -v override for samples:\n%s", got)
	}
}

func TestLoadTemplateRenderVarsErrors(t *testing.T) {
	if _, err := loadTemplateRenderVars(filepath.Join(t.TempDir(), "missing.json"), nil); err == nil {
		t.Error("expected error for missing -d file, got nil")
	}

	badJSON := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(badJSON, []byte(`{not json`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadTemplateRenderVars(badJSON, nil); err == nil {
		t.Error("expected error for invalid JSON in -d file, got nil")
	}

	if _, err := loadTemplateRenderVars("", []string{"missing_equals_sign"}); err == nil {
		t.Error("expected error for malformed -v entry, got nil")
	}
}
