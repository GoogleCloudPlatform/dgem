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

package decisionindex

import (
	"encoding/json"
	"testing"
)

func TestSystemOneQuestionAcceptsNonStringCriteria(t *testing.T) {
	body := `{"model":"dgem","state":{"x":1},"questions":{
		"chord":{"type":"choice","instructions":"Which chord?","criteria":{"chord_0":{"root":"C","quality":"maj"},"chord_1":{"root":"A","quality":"min"}}},
		"swatch":{"type":"choice","instructions":["Pick", "one"],"criteria":{"A":[255,0,0],"B":[0,0,255]}},
		"move":{"type":"choice","instructions":"Best move?","criteria":{"a3b3":"rook to b3","a3a4":{"san":"Ra4"}}},
		"urgent":{"type":"noul","instructions":"Urgent?"}}}`
	var req SystemOneRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	want := map[string]map[string]string{
		"chord":  {"chord_0": `{"root":"C","quality":"maj"}`, "chord_1": `{"root":"A","quality":"min"}`},
		"swatch": {"A": "[255,0,0]", "B": "[0,0,255]"},
		"move":   {"a3b3": "rook to b3", "a3a4": `{"san":"Ra4"}`},
	}
	for q, opts := range want {
		for k, v := range opts {
			if got := req.Questions[q].Criteria[k]; got != v {
				t.Fatalf("%s.%s = %q, want %q", q, k, got, v)
			}
		}
	}
	if req.Questions["urgent"].Type != "noul" || req.Questions["urgent"].Criteria != nil {
		t.Fatalf("noul question mangled: %+v", req.Questions["urgent"])
	}
}
