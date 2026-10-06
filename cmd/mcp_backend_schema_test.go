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
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// listToolSchemas starts an in-memory MCP server with the current backend configuration and returns each
// tool's "backend" property (nil when absent).
func listBackendProps(t *testing.T) map[string]map[string]any {
	t.Helper()
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
	out := map[string]map[string]any{}
	for _, tool := range res.Tools {
		raw, _ := json.Marshal(tool.InputSchema)
		var s struct {
			Properties map[string]map[string]any `json:"properties"`
		}
		_ = json.Unmarshal(raw, &s)
		out[tool.Name] = s.Properties["backend"]
	}
	return out
}

func enumOf(p map[string]any) []string {
	var out []string
	if arr, ok := p["enum"].([]any); ok {
		for _, v := range arr {
			out = append(out, v.(string))
		}
	}
	return out
}

// Issue #1: MCP tools advertise only the backends this server has.
func TestMCPBackendSchemaListsOnlyConfiguredBackends(t *testing.T) {
	t.Setenv("DGEM_VERTEX_URL", "")
	t.Setenv("DGEM_VERTEX_ENDPOINT_ID", "")

	// Local only (dgem mcp --local / dgem serve --local)
	withBackendConfig(t, "", "http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1", true, "local", nil, nil)
	props := listBackendProps(t)
	for _, name := range []string{"decide_policy", "decide_custom_questions", "get_health_and_gpu_status"} {
		p := props[name]
		if p == nil {
			t.Fatalf("%s: no backend property", name)
		}
		if got := strings.Join(enumOf(p), ","); got != "local" {
			t.Errorf("%s: enum = %q, want local", name, got)
		}
		desc, _ := p["description"].(string)
		if strings.Contains(desc, "vertex") || !strings.Contains(desc, "'local'") {
			t.Errorf("%s: description should mention only local: %q", name, desc)
		}
	}
	// Vision tool: local has no vision tower, so the field stays unrestricted (tool errors clearly).
	if e := enumOf(props["locate_bounding_boxes"]); len(e) != 0 {
		t.Errorf("locate_bounding_boxes on local-only: enum = %v, want none", e)
	}

	// Vertex from the environment + Cloud Run (stdio dgem mcp with DGEM_VERTEX_URL)
	withBackendConfig(t, "", testCR, "", false, "vertex_first", nil, nil)
	t.Setenv("DGEM_VERTEX_URL", testVX)
	props = listBackendProps(t)
	if got := strings.Join(enumOf(props["decide_policy"]), ","); got != "vertex_first,vertex,cloudrun" {
		t.Errorf("decide_policy enum = %q", got)
	}
	if got := strings.Join(enumOf(props["locate_bounding_boxes"]), ","); got != "vertex_first,vertex,cloudrun" {
		t.Errorf("locate_bounding_boxes enum = %q", got)
	}
	if desc, _ := props["decide_policy"]["description"].(string); !strings.Contains(desc, "default, 'vertex_first'") {
		t.Errorf("default not named: %q", desc)
	}

	// Explicit --backends list
	withBackendConfig(t, "", testCR, "", false, "cloudrun", []string{"cloudrun"}, nil)
	props = listBackendProps(t)
	if got := strings.Join(enumOf(props["decide_custom_questions"]), ","); got != "cloudrun" {
		t.Errorf("explicit list: enum = %q, want cloudrun", got)
	}
}
