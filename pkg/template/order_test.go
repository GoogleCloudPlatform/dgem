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
	"encoding/json"
	"reflect"
	"testing"
)

func schemaOrder(t *testing.T, schema string) (qIDs []string, opts [][]string) {
	t.Helper()
	var s struct {
		Questions []struct {
			ID      string `json:"id"`
			Options []struct {
				Name string `json:"name"`
			} `json:"options"`
		} `json:"questions"`
	}
	if err := json.Unmarshal([]byte(schema), &s); err != nil {
		t.Fatalf("schema JSON: %v\n%s", err, schema)
	}
	for _, q := range s.Questions {
		qIDs = append(qIDs, q.ID)
		var names []string
		for _, o := range q.Options {
			names = append(names, o.Name)
		}
		opts = append(opts, names)
	}
	return qIDs, opts
}

// TestDictFormOrderIsSourceOrder: dict-form questions and options keep the order they were written in, on every
// render (issue #118). Go map iteration is random, so 50 renders would almost surely disagree without the fix.
func TestDictFormOrderIsSourceOrder(t *testing.T) {
	payloads := map[string]string{
		"flat": `{"input": "x", "questions": {
			"zeta": {"type": "choice", "options": {"refund": "a", "billing": "b", "outage": "c", "account": "d", "other": "e"}},
			"alpha": {"type": "choice", "criteria": {"m": 1, "c": 2, "x": 3, "a": 4}},
			"mid": {"type": "boolean"}}}`,
		"envelope": `{"schema": {"questions": {
			"zeta": {"type": "choice", "options": {"refund": "a", "billing": "b", "outage": "c", "account": "d", "other": "e"}},
			"alpha": {"type": "choice", "criteria": {"m": 1, "c": 2, "x": 3, "a": 4}},
			"mid": {"type": "boolean"}}}, "state": {"input": "x"}}`,
	}
	wantQ := []string{"zeta", "alpha", "mid"}
	wantO := [][]string{{"refund", "billing", "outage", "account", "other"}, {"m", "c", "x", "a"}, nil}
	for name, p := range payloads {
		for i := 0; i < 50; i++ {
			schema, _, err := ParseStructuredPayload(p, nil)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			q, o := schemaOrder(t, schema)
			if !reflect.DeepEqual(q, wantQ) || !reflect.DeepEqual(o, wantO) {
				t.Fatalf("%s render %d: questions %v options %v, want %v %v", name, i, q, o, wantQ, wantO)
			}
		}
	}
}

func TestDecodeOrderedMatchesUnmarshal(t *testing.T) {
	src := `{"b": [1, 2.5, "s", true, null, {"y": 1, "x": 2}], "a": {"k": "v"}, "b2": -3e2}`
	got, _, err := decodeOrdered([]byte(src))
	if err != nil {
		t.Fatal(err)
	}
	var want interface{}
	if err := json.Unmarshal([]byte(src), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decodeOrdered = %#v, want %#v", got, want)
	}
	if _, _, err := decodeOrdered([]byte(`{"a": 1} {"b": 2}`)); err == nil {
		t.Fatal("trailing data accepted")
	}
}

// TestKeysOfFallbackIsSorted: maps decoded without order information still give a deterministic order.
func TestKeysOfFallbackIsSorted(t *testing.T) {
	var o keyOrder
	if got := o.keysOf(map[string]interface{}{"c": 1, "a": 2, "b": 3}); !reflect.DeepEqual(got, []string{"a", "b", "c"}) {
		t.Fatalf("keysOf = %v", got)
	}
}
