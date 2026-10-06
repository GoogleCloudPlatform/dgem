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
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestMCPInputSchemasAreGeminiSafe guards against tool input schemas that
// Gemini / Vertex AI function calling rejects (issue #12): type arrays such as
// ["null","array"] and anyOf/oneOf combinators alongside sibling keys. One bad
// declaration fails every request from a Gemini-backed MCP client.
func TestMCPInputSchemasAreGeminiSafe(t *testing.T) {
	ctx := context.Background()
	serverT, clientT := mcp.NewInMemoryTransports()
	if _, err := buildMCPServer().Connect(ctx, serverT, nil); err != nil {
		t.Fatalf("server connect: %v", err)
	}
	cs, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "0"}, nil).Connect(ctx, clientT, nil)
	if err != nil {
		t.Fatalf("client connect: %v", err)
	}
	defer cs.Close()

	res, err := cs.ListTools(ctx, nil)
	if err != nil {
		t.Fatalf("tools/list: %v", err)
	}
	if len(res.Tools) == 0 {
		t.Fatal("no tools registered")
	}
	for _, tool := range res.Tools {
		raw, err := json.Marshal(tool.InputSchema)
		if err != nil {
			t.Fatalf("%s: marshal input schema: %v", tool.Name, err)
		}
		var schema any
		if err := json.Unmarshal(raw, &schema); err != nil {
			t.Fatalf("%s: unmarshal input schema: %v", tool.Name, err)
		}
		for _, v := range findGeminiUnsafe(schema, tool.Name+".inputSchema") {
			t.Errorf("Gemini-unsafe schema: %s", v)
		}
	}
}

func findGeminiUnsafe(node any, path string) []string {
	var out []string
	switch n := node.(type) {
	case map[string]any:
		if ty, ok := n["type"].([]any); ok {
			out = append(out, fmt.Sprintf("%s.type is an array %v", path, ty))
		}
		for _, k := range []string{"anyOf", "oneOf"} {
			if _, ok := n[k]; ok {
				out = append(out, fmt.Sprintf("%s uses %s", path, k))
			}
		}
		for k, v := range n {
			out = append(out, findGeminiUnsafe(v, path+"."+k)...)
		}
	case []any:
		for i, v := range n {
			out = append(out, findGeminiUnsafe(v, fmt.Sprintf("%s[%d]", path, i))...)
		}
	}
	return out
}

func TestStripNullability(t *testing.T) {
	type inner struct {
		Tags []string `json:"tags,omitempty"`
	}
	type in struct {
		List  []inner          `json:"list"`
		Ptr   *inner           `json:"ptr,omitempty"`
		Map   map[string]int   `json:"map,omitempty"`
		Multi map[string][]int `json:"multi,omitempty"`
	}
	s := geminiSafeInputSchema[in]()
	raw, _ := json.Marshal(s)
	var schema any
	_ = json.Unmarshal(raw, &schema)
	if bad := findGeminiUnsafe(schema, "in"); len(bad) > 0 {
		t.Fatalf("unexpected unsafe nodes: %v\nschema: %s", bad, raw)
	}
	if got := s.Properties["list"].Type; got != "array" {
		t.Errorf("list.type = %q, want array", got)
	}
	if got := s.Properties["list"].Items.Properties["tags"].Type; got != "array" {
		t.Errorf("list.items.tags.type = %q, want array", got)
	}
	if got := s.Properties["ptr"].Type; got != "object" {
		t.Errorf("ptr.type = %q, want object", got)
	}
	if got := s.Properties["multi"].AdditionalProperties.Type; got != "array" {
		t.Errorf("multi.additionalProperties.type = %q, want array", got)
	}
}
