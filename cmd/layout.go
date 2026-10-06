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
	"fmt"
	"strings"
)

// PromptLayouts are the serving image's prompt layouts (schema key "layout", serving v0.2.0+):
//   - document_first: the state first, then the questions, both in the user turn (the server default from v0.2.0)
//   - schema_first:   the questions as the system prompt, the state as the user turn (the layout before v0.2.0)
//
// See docs/policies/prompt-layout.md.
var PromptLayouts = []string{"document_first", "schema_first"}

// NormalizePromptLayout validates a layout name. An empty value means "the server's default" and is returned as "".
func NormalizePromptLayout(layout string) (string, error) {
	l := strings.ToLower(strings.TrimSpace(layout))
	switch l {
	case "", "default", "server":
		return "", nil
	case "document_first", "schema_first":
		return l, nil
	}
	return "", fmt.Errorf("layout must be document_first or schema_first (got %q)", layout)
}

// ApplyPromptLayout sets the "layout" key of a rendered schema. An empty layout leaves the schema untouched (the
// template's own "layout", if any, or the server default), so servers and backends that predate the key never see it.
func ApplyPromptLayout(schemaJSON, layout string) (string, error) {
	l, err := NormalizePromptLayout(layout)
	if err != nil || l == "" {
		return schemaJSON, err
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(schemaJSON), &m); err != nil {
		return schemaJSON, fmt.Errorf("layout: schema is not a JSON object: %w", err)
	}
	m["layout"] = l
	b, err := json.Marshal(m)
	if err != nil {
		return schemaJSON, err
	}
	return string(b), nil
}
