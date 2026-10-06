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

package client

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Temperature is a post-hoc temperature for a policy's answers: one value for every question, or one per question
// type ("noul", "choice", "score"; an absent type keeps T = 1). JSON accepts a number or an object, e.g.
// "temperature": 1.4 or "temperature": {"noul": 3.0, "choice": 1.3}. A single served default doesn't transfer across
// domains, so fit it on a labelled sample of your own decisions (docs/confidence/calibrate-your-policy.md).
type Temperature struct {
	All    float64
	ByType map[string]float64
}

// UnmarshalJSON accepts a number or a {type: T} object.
func (t *Temperature) UnmarshalJSON(b []byte) error {
	s := strings.TrimSpace(string(b))
	if s == "" || s == "null" {
		return nil
	}
	if s[0] == '{' {
		m := map[string]float64{}
		if err := json.Unmarshal(b, &m); err != nil {
			return fmt.Errorf("temperature: %w", err)
		}
		t.ByType = map[string]float64{}
		for k, v := range m {
			switch strings.ToLower(strings.TrimSpace(k)) {
			case "noul", "bool", "boolean", "yes_no", "yesno", "choice", "score", "scale", "rating":
				t.ByType[normalizeAnswerType(k)] = v
			default:
				return fmt.Errorf("temperature: unknown question type %q (use noul/boolean, choice or score)", k)
			}
		}
		return t.Validate()
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fmt.Errorf("temperature must be a number or an object of numbers per question type")
	}
	t.All = v
	return t.Validate()
}

// MarshalJSON writes the number form when only All is set.
func (t Temperature) MarshalJSON() ([]byte, error) {
	if len(t.ByType) > 0 {
		return json.Marshal(t.ByType)
	}
	if t.All == 0 {
		return []byte("null"), nil
	}
	return json.Marshal(t.All)
}

// IsSet reports whether any temperature other than 1 is configured.
func (t Temperature) IsSet() bool {
	if t.All > 0 && t.All != 1 {
		return true
	}
	for _, v := range t.ByType {
		if v > 0 && v != 1 {
			return true
		}
	}
	return false
}

// Validate checks every value is in (0, 20].
func (t Temperature) Validate() error {
	check := func(v float64) error {
		if v < 0 || v > 20 || math.IsNaN(v) {
			return fmt.Errorf("temperature values must be between 0 and 20 (got %v)", v)
		}
		return nil
	}
	if err := check(t.All); err != nil {
		return err
	}
	for k, v := range t.ByType {
		if k != "noul" && k != "choice" && k != "score" {
			return fmt.Errorf("temperature: unknown question type %q (use noul, choice or score)", k)
		}
		if err := check(v); err != nil {
			return err
		}
	}
	return nil
}

// For returns the temperature for a question type (1 when unset).
func (t Temperature) For(answerType string) float64 {
	if v, ok := t.ByType[normalizeAnswerType(answerType)]; ok && v > 0 {
		return v
	}
	if t.All > 0 {
		return t.All
	}
	return 1
}

func normalizeAnswerType(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "noul", "bool", "boolean", "yes_no", "yesno":
		return "noul"
	case "score", "scale", "rating":
		return "score"
	default:
		return "choice"
	}
}

// ParseSchemaTemperature reads a policy's "temperature" key from a rendered schema (absent: zero value).
func ParseSchemaTemperature(schemaJSON string) (Temperature, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(schemaJSON), &m); err != nil {
		return Temperature{}, nil // not a JSON object: no policy temperature
	}
	raw, ok := m["temperature"]
	if !ok {
		return Temperature{}, nil
	}
	var t Temperature
	if err := t.UnmarshalJSON(raw); err != nil {
		return Temperature{}, err
	}
	return t, nil
}

// ApplyTemperature rescales each answer's probabilities by p_k^(1/T) (renormalised) and recomputes confidence,
// entropy, the yes probability (noul) and the expected score. The argmax answer does not change.
func (r *StructuredDecisionResponse) ApplyTemperature(t Temperature) {
	if r == nil || !t.IsSet() {
		return
	}
	for id, a := range r.Answers {
		T := t.For(a.Type)
		if T == 1 || len(a.Probabilities) == 0 {
			continue
		}
		keys := make([]string, 0, len(a.Probabilities))
		for k := range a.Probabilities {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		lp := make([]float64, len(keys))
		mx := math.Inf(-1)
		for i, k := range keys {
			lp[i] = math.Log(math.Max(a.Probabilities[k], 1e-12)) / T
			mx = math.Max(mx, lp[i])
		}
		z := 0.0
		for i := range lp {
			lp[i] = math.Exp(lp[i] - mx)
			z += lp[i]
		}
		p := make(map[string]float64, len(keys))
		ent, top := 0.0, -1.0
		for i, k := range keys {
			v := lp[i] / z
			p[k] = v
			if v > 0 {
				ent -= v * math.Log(v)
			}
			if v > top {
				top = v
			}
		}
		a.Probabilities = p
		a.Entropy = ent
		if lbl := firstNonEmpty(a.Label, a.Choice, a.Level); lbl != "" {
			if v, ok := p[lbl]; ok {
				a.Confidence = v
			} else {
				a.Confidence = top
			}
		} else {
			a.Confidence = top
		}
		switch normalizeAnswerType(a.Type) {
		case "noul":
			for _, k := range []string{"yes", "true"} {
				if v, ok := p[k]; ok {
					a.Noul = v
					break
				}
			}
		case "score":
			sum, ok := 0.0, true
			for k, v := range p {
				f, err := strconv.ParseFloat(k, 64)
				if err != nil {
					ok = false
					break
				}
				sum += f * v
			}
			if ok {
				a.Score = sum
			}
		}
		r.Answers[id] = a
		if q, ok := r.Diagnostics.Questions[id]; ok {
			q.Entropy = ent
			r.Diagnostics.Questions[id] = q
		}
	}
}

func firstNonEmpty(s ...string) string {
	for _, v := range s {
		if v != "" {
			return v
		}
	}
	return ""
}
