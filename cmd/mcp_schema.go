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
	"fmt"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// geminiSafeInputSchema infers the JSON Schema for an MCP tool input type T and
// normalizes it so Gemini / Vertex AI function calling accepts it.
//
// jsonschema-go marks Go slices and pointers as nullable (`"type": ["null",
// "array"]`). Gemini rewrites a type array into `anyOf`, then rejects the
// declaration because `anyOf` sits next to sibling keys such as `description`
// and `items` ("When using any_of, it must be the only field set"). A single
// invalid declaration fails every request from the MCP client, not just calls
// to this tool, so every tool registers its input schema through this helper.
//
// Output schemas are intentionally not normalized: Gemini does not validate
// them, and go-sdk validates handler results against them, where nil
// slices/pointers legitimately serialize as null.
func geminiSafeInputSchema[T any]() *jsonschema.Schema {
	s, err := jsonschema.For[T](nil)
	if err != nil {
		// Input types are static Go structs; failure is a programming error.
		panic(fmt.Sprintf("dgem mcp: inferring input schema for %T: %v", *new(T), err))
	}
	stripNullability(s)
	return s
}

// stripNullability recursively collapses `["null", X]` type arrays to `X` and
// drops null-only branches from `anyOf`/`oneOf`.
func stripNullability(s *jsonschema.Schema) {
	if s == nil {
		return
	}

	if len(s.Types) > 0 {
		nonNull := make([]string, 0, len(s.Types))
		for _, t := range s.Types {
			if t != "null" {
				nonNull = append(nonNull, t)
			}
		}
		if len(nonNull) == 1 {
			s.Type, s.Types = nonNull[0], nil
		} else {
			s.Types = nonNull
		}
	}
	if s.Type == "null" && len(s.Types) == 0 {
		// A bare null-only schema carries no useful constraint for Gemini.
		s.Type = ""
	}

	s.AnyOf = dropNullBranches(s.AnyOf)
	s.OneOf = dropNullBranches(s.OneOf)

	for _, c := range s.Properties {
		stripNullability(c)
	}
	for _, c := range s.PatternProperties {
		stripNullability(c)
	}
	for _, c := range s.Defs {
		stripNullability(c)
	}
	for _, c := range s.Definitions {
		stripNullability(c)
	}
	for _, c := range s.PrefixItems {
		stripNullability(c)
	}
	for _, c := range s.ItemsArray {
		stripNullability(c)
	}
	for _, c := range append(append(append([]*jsonschema.Schema{}, s.AllOf...), s.AnyOf...), s.OneOf...) {
		stripNullability(c)
	}
	for _, c := range []*jsonschema.Schema{
		s.Items, s.AdditionalItems, s.Contains, s.UnevaluatedItems,
		s.AdditionalProperties, s.PropertyNames, s.UnevaluatedProperties,
		s.Not, s.If, s.Then, s.Else,
	} {
		stripNullability(c)
	}
}

// dropNullBranches removes `{"type":"null"}` branches from an anyOf/oneOf
// list. jsonschema-go does not emit anyOf/oneOf for struct inference today;
// TestMCPInputSchemasAreGeminiSafe fails if a future type introduces one.
func dropNullBranches(branches []*jsonschema.Schema) []*jsonschema.Schema {
	var kept []*jsonschema.Schema
	for _, b := range branches {
		if b != nil && b.Type == "null" && len(b.Types) == 0 {
			continue
		}
		kept = append(kept, b)
	}
	return kept
}

// backendChoicesSchema returns the input schema for T with its "backend" property restricted to the backends
// this server actually has (configuredBackends at registration time), so agents are never told to use a
// backend that would return "not enabled" (issue #1). The description names the default.
func backendChoicesSchema[T any]() *jsonschema.Schema {
	s := geminiSafeInputSchema[T]()
	applyBackendChoices(s, configuredBackends(), currentDefaultBackend())
	return s
}

// visionBackendChoicesSchema is backendChoicesSchema without 'local' (diffgemma has no vision tower). When
// only 'local' is configured the tool keeps an unrestricted backend field and fails with a clear error.
func visionBackendChoicesSchema[T any]() *jsonschema.Schema {
	s := geminiSafeInputSchema[T]()
	var avail []string
	for _, b := range configuredBackends() {
		if b != "local" {
			avail = append(avail, b)
		}
	}
	applyBackendChoices(s, avail, currentDefaultBackend())
	return s
}

func currentDefaultBackend() string {
	backendConfigMu.RLock()
	defer backendConfigMu.RUnlock()
	return serveDefaultBackend
}

var backendMeaning = map[string]string{
	"vertex_first": "Vertex AI endpoint, failing over to Cloud Run",
	"vertex":       "Vertex AI endpoint only",
	"cloudrun":     "Cloud Run GPU only",
	"local":        "local diffgemma engine",
}

func applyBackendChoices(s *jsonschema.Schema, available []string, def string) {
	if s == nil || s.Properties == nil || len(available) == 0 {
		return
	}
	prop, ok := s.Properties["backend"]
	if !ok || prop == nil {
		return
	}
	if !containsString(available, def) {
		def = available[0]
	}
	parts := make([]string, 0, len(available))
	enum := make([]any, 0, len(available))
	for _, b := range available {
		parts = append(parts, fmt.Sprintf("'%s' (%s)", b, backendMeaning[b]))
		enum = append(enum, b)
	}
	prop.Description = fmt.Sprintf("Optional inference backend. Available on this server: %s. Omit to use the default, '%s'.",
		strings.Join(parts, ", "), def)
	prop.Enum = enum
}
