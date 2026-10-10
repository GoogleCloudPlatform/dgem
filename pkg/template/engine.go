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
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/template"
)

// Engine parses and executes Go templates with rich helper functions.
type Engine struct {
	funcMap template.FuncMap
}

// NewEngine creates a new template engine with default helper functions.
func NewEngine() *Engine {
	e := &Engine{}
	e.funcMap = template.FuncMap{
		"toJson": func(v interface{}) (string, error) {
			b, err := json.Marshal(v)
			if err != nil {
				return "", err
			}
			return string(b), nil
		},
		"toPrettyJson": func(v interface{}) (string, error) {
			b, err := json.MarshalIndent(v, "", "  ")
			if err != nil {
				return "", err
			}
			return string(b), nil
		},
		"default": func(def, val interface{}) interface{} {
			if val == nil {
				return def
			}
			switch v := val.(type) {
			case string:
				if strings.TrimSpace(v) == "" {
					return def
				}
			case bool:
				return v
			}
			return val
		},
		"indent": func(spaces int, s string) string {
			pad := strings.Repeat(" ", spaces)
			lines := strings.Split(s, "\n")
			for i, line := range lines {
				if line != "" {
					lines[i] = pad + line
				}
			}
			return strings.Join(lines, "\n")
		},
		"upper": strings.ToUpper,
		"lower": strings.ToLower,
		"trim":  strings.TrimSpace,
	}
	return e
}

// RenderFile loads a template file from disk and executes it with the given data.
func (e *Engine) RenderFile(filePath string, data map[string]interface{}) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to read template file %s: %w", filePath, err)
	}

	name := filepath.Base(filePath)
	return e.RenderString(name, string(content), data)
}

// RenderString parses and executes an in-memory template string.
func (e *Engine) RenderString(name, tmplStr string, data map[string]interface{}) (string, error) {
	if data == nil {
		data = make(map[string]interface{})
	}

	tmpl, err := template.New(name).Funcs(e.funcMap).Parse(tmplStr)
	if err != nil {
		return "", fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

// StructuredPayload represents the rendered JSON package containing a schema and state.
type StructuredPayload struct {
	Schema json.RawMessage `json:"schema"`
	State  json.RawMessage `json:"state"`
}

// ParseStructuredPayload parses rendered output that has separate "schema" and "state" envelopes,
// or treats the full payload as the schema if not wrapped, and normalizes legacy question fields
// ("name" -> "id", "question"/"prompt" -> "instructions", "items" -> "levels", dict "options" -> list)
// so that all templates and ad-hoc schemas conform to structured_server.py's parse_schema contract.
func ParseStructuredPayload(rendered string, fallbackState map[string]interface{}) (string, string, error) {
	trimmed := strings.TrimSpace(rendered)
	if strings.HasPrefix(trimmed, "{") {
		var envelope StructuredPayload
		if err := json.Unmarshal([]byte(trimmed), &envelope); err == nil && len(envelope.Schema) > 0 {
			schemaStr := normalizeSchemaJSON(string(envelope.Schema))

			var stateStr string
			if len(envelope.State) > 0 && string(envelope.State) != "null" {
				stateStr = string(envelope.State)
			} else {
				sb, _ := json.Marshal(fallbackState)
				stateStr = string(sb)
			}
			return schemaStr, stateStr, nil
		}

		// Check if it's a flat {"input": ..., "questions": [...]} or {"context": ..., "questions": [...]} object
		decoded, order, err := decodeOrdered([]byte(trimmed))
		if rawObj, isObj := decoded.(map[string]interface{}); err == nil && isObj {
			stateMap := make(map[string]interface{})
			for k, v := range fallbackState {
				stateMap[k] = v
			}
			if inp, ok := rawObj["input"]; ok {
				stateMap["input"] = inp
				delete(rawObj, "input")
			}
			if ctxVal, ok := rawObj["context"]; ok {
				stateMap["context"] = ctxVal
				delete(rawObj, "context")
			}
			if stVal, ok := rawObj["state"]; ok {
				if stMap, isMap := stVal.(map[string]interface{}); isMap {
					for k, v := range stMap {
						stateMap[k] = v
					}
				} else {
					stateMap["input"] = stVal
				}
				delete(rawObj, "state")
			}
			normalizeSchemaMap(rawObj, order)
			schemaBytes, _ := json.Marshal(rawObj)
			stateBytes, _ := json.Marshal(stateMap)
			return string(schemaBytes), string(stateBytes), nil
		}
	}

	// Fallback if not a JSON object
	stateBytes, err := json.Marshal(fallbackState)
	if err != nil {
		return "", "", fmt.Errorf("failed to serialize fallback state: %w", err)
	}

	return trimmed, string(stateBytes), nil
}

func normalizeSchemaJSON(schemaStr string) string {
	decoded, order, err := decodeOrdered([]byte(schemaStr))
	if err != nil {
		return schemaStr
	}
	m, ok := decoded.(map[string]interface{})
	if !ok {
		return schemaStr
	}
	normalizeSchemaMap(m, order)
	b, err := json.Marshal(m)
	if err != nil {
		return schemaStr
	}
	return string(b)
}

// normalizeSchemaMap rewrites legacy question fields in place. Dict-form questions and options become lists in their
// source order (order, from decodeOrdered), or sorted by key when the order is unknown.
func normalizeSchemaMap(m map[string]interface{}, order keyOrder) {
	if qMap, isMap := m["questions"].(map[string]interface{}); isMap {
		qList := make([]interface{}, 0, len(qMap))
		for _, k := range order.keysOf(qMap) {
			v := qMap[k]
			if qObj, ok := v.(map[string]interface{}); ok {
				if _, hasID := qObj["id"]; !hasID {
					qObj["id"] = k
				}
				qList = append(qList, qObj)
			}
		}
		m["questions"] = qList
	}
	rawQs, ok := m["questions"].([]interface{})
	if !ok {
		return
	}
	for _, item := range rawQs {
		q, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		// 0. Normalize "bool" / "noul" -> "boolean"
		if tStr, ok := q["type"].(string); ok {
			if tStr == "bool" || tStr == "noul" {
				q["type"] = "boolean"
			}
		}
		// 1. "name" -> "id"
		if _, hasID := q["id"]; !hasID {
			if nameVal, hasName := q["name"]; hasName {
				q["id"] = nameVal
				delete(q, "name")
			}
		}
		// 2. "question" or "prompt" -> "instructions"
		if _, hasInst := q["instructions"]; !hasInst {
			if qText, hasQ := q["question"]; hasQ {
				q["instructions"] = qText
				delete(q, "question")
			} else if pText, hasP := q["prompt"]; hasP {
				q["instructions"] = pText
				delete(q, "prompt")
			}
		}
		// 3. "choices" or "criteria" -> "options"
		if _, hasOpts := q["options"]; !hasOpts {
			if chVal, hasChoices := q["choices"]; hasChoices {
				q["options"] = chVal
				delete(q, "choices")
			} else if critVal, hasCrit := q["criteria"]; hasCrit {
				q["options"] = critVal
				delete(q, "criteria")
			}
		}
		// 4. Dictionary "options": {"yes": "desc"} -> [{"name": "yes", "description": "desc"}]
		if optsMap, isMap := q["options"].(map[string]interface{}); isMap {
			optList := make([]map[string]interface{}, 0, len(optsMap))
			for _, k := range order.keysOf(optsMap) {
				v := optsMap[k]
				optList = append(optList, map[string]interface{}{
					"name":        k,
					"description": fmt.Sprintf("%v", v),
				})
			}
			q["options"] = optList
		}
		// 5. "items" -> "levels" (and ensure string elements for score levels)
		if _, hasLevels := q["levels"]; !hasLevels {
			if itemsVal, hasItems := q["items"]; hasItems {
				q["levels"] = itemsVal
				delete(q, "items")
			} else if q["type"] == "score" {
				q["levels"] = []string{"1", "2", "3", "4", "5"}
			}
		}
		if lvls, isSlice := q["levels"].([]interface{}); isSlice {
			strLvls := make([]string, 0, len(lvls))
			for _, l := range lvls {
				strLvls = append(strLvls, fmt.Sprintf("%v", l))
			}
			q["levels"] = strLvls
		}
	}
}
