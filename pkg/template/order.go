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
	"reflect"
	"sort"
)

// keyOrder records the source order of the keys of each JSON object decoded by decodeOrdered, keyed by the map's
// identity. Go maps don't keep insertion order, and the order of dict-form questions and options matters: the server
// labels options A, B, C… in list order and the model has a first-shown bias (issue #118).
type keyOrder map[uintptr][]string

// decodeOrdered decodes JSON like json.Unmarshal into interface{} (objects as map[string]interface{}, numbers as
// float64) and also returns the source key order of every object.
func decodeOrdered(data []byte) (interface{}, keyOrder, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	order := keyOrder{}
	v, err := decodeValue(dec, order)
	if err != nil {
		return nil, nil, err
	}
	if _, err := dec.Token(); err == nil {
		return nil, nil, fmt.Errorf("unexpected data after JSON value")
	}
	return v, order, nil
}

func decodeValue(dec *json.Decoder, order keyOrder) (interface{}, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			m := map[string]interface{}{}
			var keys []string
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				k, ok := kt.(string)
				if !ok {
					return nil, fmt.Errorf("object key is %T, want string", kt)
				}
				v, err := decodeValue(dec, order)
				if err != nil {
					return nil, err
				}
				if _, dup := m[k]; !dup {
					keys = append(keys, k)
				}
				m[k] = v // last value wins, as in json.Unmarshal
			}
			if _, err := dec.Token(); err != nil { // '}'
				return nil, err
			}
			order[reflect.ValueOf(m).Pointer()] = keys
			return m, nil
		case '[':
			arr := []interface{}{}
			for dec.More() {
				v, err := decodeValue(dec, order)
				if err != nil {
					return nil, err
				}
				arr = append(arr, v)
			}
			if _, err := dec.Token(); err != nil { // ']'
				return nil, err
			}
			return arr, nil
		}
		return nil, fmt.Errorf("unexpected delimiter %q", t)
	default:
		return t, nil // string, float64, bool or nil
	}
}

// keysOf returns m's keys in source order when known, otherwise sorted, so the result never depends on Go's map
// iteration order.
func (o keyOrder) keysOf(m map[string]interface{}) []string {
	if o != nil {
		if keys, ok := o[reflect.ValueOf(m).Pointer()]; ok && len(keys) == len(m) {
			return keys
		}
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
