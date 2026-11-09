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
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAllShippedTemplatesRenderAndValidate(t *testing.T) {
	templatesRoot := filepath.Join("..", "..", "templates")
	var tmplPaths []string
	err := filepath.WalkDir(templatesRoot, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".json.tmpl") {
			tmplPaths = append(tmplPaths, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", templatesRoot, err)
	}
	if len(tmplPaths) < 35 {
		t.Fatalf("discovered %d .json.tmpl files under %s, want at least 35", len(tmplPaths), templatesRoot)
	}

	eng := NewEngine()
	for _, tmplPath := range tmplPaths {
		rel, err := filepath.Rel(templatesRoot, tmplPath)
		if err != nil {
			t.Fatalf("rel path for %s: %v", tmplPath, err)
		}
		tmplID := strings.TrimSuffix(filepath.ToSlash(rel), ".json.tmpl")

		t.Run(tmplID, func(t *testing.T) {
			exPath := ExamplesFilePath(tmplPath)
			ef, err := LoadExamplesFile(exPath)
			if err != nil {
				t.Fatalf("missing or invalid companion examples file %s: %v", exPath, err)
			}
			if len(ef.Examples) < 3 {
				t.Fatalf("%s has %d examples, want at least 3 (typical, edge, minimal)", exPath, len(ef.Examples))
			}

			seenNames := make(map[string]bool, len(ef.Examples))
			for _, ex := range ef.Examples {
				if strings.TrimSpace(ex.Name) == "" {
					t.Fatalf("%s contains an example with an empty name", exPath)
				}
				if seenNames[ex.Name] {
					t.Fatalf("%s contains duplicate example name %q", exPath, ex.Name)
				}
				seenNames[ex.Name] = true
				if ex.Unambiguous && len(ex.ExpectedAnswers) == 0 {
					t.Errorf("%s example %q is marked unambiguous=true but defines no expected_answers", exPath, ex.Name)
				}

				ex := ex
				t.Run(ex.Name, func(t *testing.T) {
					rendered, err := eng.RenderFile(tmplPath, ex.Variables)
					if err != nil {
						t.Fatalf("RenderFile(%s, %s) error: %v", tmplID, ex.Name, err)
					}
					if err := ValidateRenderedPayload(rendered, ex.Variables, ef.AllowWideChoices, ex.ExpectedAnswers); err != nil {
						t.Fatalf("ValidateRenderedPayload(%s, %s) error: %v\nrendered:\n%s", tmplID, ex.Name, err, rendered)
					}
				})
			}

			for _, required := range []string{"typical", "edge", "minimal"} {
				if !seenNames[required] {
					t.Errorf("%s must include a %q example (got %v)", exPath, required, seenNames)
				}
			}

			// Also verify stringified SampleVars round-trip cleanly through CoerceJSONVariableStrings,
			// matching GET /api/templates -> POST /api/decide/{id}.
			t.Run("sample_vars_roundtrip", func(t *testing.T) {
				sv := SampleVarsFromExamples(ef)
				asAny := make(map[string]any, len(sv))
				for k, v := range sv {
					asAny[k] = v
				}
				coerced := CoerceJSONVariableStrings(asAny)
				rendered, err := eng.RenderFile(tmplPath, coerced)
				if err != nil {
					t.Fatalf("RenderFile(%s, sample_vars) error: %v", tmplID, err)
				}
				if err := ValidateRenderedPayload(rendered, coerced, ef.AllowWideChoices, nil); err != nil {
					t.Fatalf("ValidateRenderedPayload(%s, sample_vars) error: %v", tmplID, err)
				}
			})
		})
	}
}

func TestValidateRenderedPayloadRejectsInvalidSchemas(t *testing.T) {
	var thirtyOpts []string
	for i := 0; i < 30; i++ {
		thirtyOpts = append(thirtyOpts, fmt.Sprintf(`{"name":"opt_%02d"}`, i))
	}
	wideJSON := fmt.Sprintf(`{"schema":{"questions":[{"id":"q1","type":"choice","instructions":"Pick one","options":[%s]}]},"state":{"x":1}}`, strings.Join(thirtyOpts, ","))
	wideWithLabelsJSON := fmt.Sprintf(`{"schema":{"labels":"az_aa","questions":[{"id":"q1","type":"choice","instructions":"Pick one","options":[%s]}]},"state":{"x":1}}`, strings.Join(thirtyOpts, ","))

	cases := []struct {
		name      string
		payload   string
		allowWide bool
		expected  map[string]string
		wantErr   string
	}{
		{
			name:    "exceeds_26_options_without_az_aa",
			payload: wideJSON,
			wantErr: "exceeding limit 26",
		},
		{
			name:      "allows_30_options_with_allow_wide_flag",
			payload:   wideJSON,
			allowWide: true,
		},
		{
			name:    "allows_30_options_with_az_aa_label_set",
			payload: wideWithLabelsJSON,
		},
		{
			name:    "duplicate_question_id",
			payload: `{"schema":{"questions":[{"id":"dup","type":"boolean","instructions":"First"},{"id":"dup","type":"boolean","instructions":"Second"}]},"state":{}}`,
			wantErr: "duplicate question id",
		},
		{
			name:    "empty_instructions",
			payload: `{"schema":{"questions":[{"id":"q1","type":"boolean","instructions":"   "}]},"state":{}}`,
			wantErr: "non-empty instructions",
		},
		{
			name:    "unknown_question_type",
			payload: `{"schema":{"questions":[{"id":"q1","type":"free_text","instructions":"Write prose"}]},"state":{}}`,
			wantErr: "unknown type",
		},
		{
			name:    "cyclic_depends_on",
			payload: `{"schema":{"questions":[{"id":"a","type":"boolean","instructions":"A?","depends_on":["b"]},{"id":"b","type":"boolean","instructions":"B?","depends_on":["a"]}]},"state":{}}`,
			wantErr: "cyclic depends_on",
		},
		{
			name:     "invalid_expected_answer_value",
			payload:  `{"schema":{"questions":[{"id":"flag","type":"boolean","instructions":"Is it valid?"}]},"state":{}}`,
			expected: map[string]string{"flag": "true"},
			wantErr:  "not a valid answer option",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateRenderedPayload(tc.payload, nil, tc.allowWide, tc.expected)
			if tc.wantErr == "" && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("got error %v, want substring %q", err, tc.wantErr)
				}
			}
		})
	}
}
