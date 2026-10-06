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

package permutation

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestDualMirrorWarning(t *testing.T) {
	var buf bytes.Buffer
	WarnWriter = &buf
	letterCollisionSeen.Store(false)

	boolSchema := `{"questions":[{"id":"q","type":"boolean","instructions":"x","options":[{"name":"yes"},{"name":"no"}]}]}`
	if _, _, ok := InjectDualMirrorSchema(boolSchema); ok && DualMirrorLetterCollisionSeen() {
		t.Fatalf("boolean slot should not trigger the letter-collision warning")
	}

	choiceSchema := `{"questions":[{"id":"q","type":"choice","instructions":"x","options":[{"name":"a","description":"A"},{"name":"b","description":"B"}]}]}`
	if _, _, ok := InjectDualMirrorSchema(choiceSchema); !ok {
		t.Fatalf("expected mirror injection for choice slot")
	}
	if !DualMirrorLetterCollisionSeen() {
		t.Fatalf("choice slot should trigger the letter-collision warning")
	}
	if !strings.Contains(buf.String(), "Research use only") {
		t.Fatalf("warning not written: %q", buf.String())
	}
}

func TestMirrorModes(t *testing.T) {
	WarnWriter = &bytes.Buffer{}
	defer func() { MirrorMode = "reversed" }()
	schema := `{"questions":[{"id":"q","type":"choice","instructions":"x","options":[{"name":"a","description":"A"},{"name":"b","description":"B"},{"name":"c","description":"C"}]}]}`
	for _, mode := range []string{"reversed", "copy", "reversed-digits", "reversed-first"} {
		MirrorMode = mode
		out, _, ok := InjectDualMirrorSchema(schema)
		if !ok {
			t.Fatalf("%s: not injected", mode)
		}
		var m map[string]any
		_ = json.Unmarshal([]byte(out), &m)
		qs := m["questions"].([]any)
		first := qs[0].(map[string]any)["id"].(string)
		second := qs[1].(map[string]any)
		switch mode {
		case "reversed-first":
			if first != "q__rev" {
				t.Fatalf("reversed-first: first slot %s", first)
			}
		case "reversed-digits":
			if second["type"] != "score" {
				t.Fatalf("digits: type %v", second["type"])
			}
			lv := second["levels"].([]any)
			if lv[0].(string) != "c: C" {
				t.Fatalf("digits: first level %v", lv[0])
			}
		case "copy":
			opts := second["options"].([]any)
			if opts[0].(map[string]any)["name"] != "a" {
				t.Fatalf("copy: order %v", opts[0])
			}
		}
	}
}
