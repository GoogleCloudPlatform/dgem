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

// ParseSchemaScoreLevels extracts ordered score level names per question id from a policy schema JSON
// (supporting both chat-completions questions arrays and /v1/systemone questions maps).
func ParseSchemaScoreLevels(schemaJSON string) map[string][]string {
	var top map[string]json.RawMessage
	if err := json.Unmarshal([]byte(schemaJSON), &top); err != nil {
		return nil
	}
	rawQs, ok := top["questions"]
	if !ok {
		return nil
	}
	trimmed := strings.TrimSpace(string(rawQs))
	if trimmed == "" || trimmed == "null" {
		return nil
	}
	extractStringSlice := func(raw json.RawMessage) []string {
		var vals []interface{}
		if err := json.Unmarshal(raw, &vals); err != nil || len(vals) == 0 {
			return nil
		}
		out := make([]string, 0, len(vals))
		for _, v := range vals {
			out = append(out, fmt.Sprintf("%v", v))
		}
		return out
	}
	out := make(map[string][]string)
	if trimmed[0] == '[' {
		var items []map[string]json.RawMessage
		if err := json.Unmarshal(rawQs, &items); err != nil {
			return nil
		}
		for _, q := range items {
			var qID, qType string
			_ = json.Unmarshal(q["id"], &qID)
			if qID == "" {
				_ = json.Unmarshal(q["name"], &qID)
			}
			_ = json.Unmarshal(q["type"], &qType)
			if qID == "" || normalizeAnswerType(qType) != "score" {
				continue
			}
			for _, field := range []string{"levels", "items", "criteria"} {
				if rawLvls, has := q[field]; has {
					if lvls := extractStringSlice(rawLvls); len(lvls) >= 2 {
						out[qID] = lvls
						break
					}
				}
			}
		}
		return out
	}
	if trimmed[0] == '{' {
		var qMap map[string]map[string]json.RawMessage
		if err := json.Unmarshal(rawQs, &qMap); err != nil {
			return nil
		}
		for qID, q := range qMap {
			var qType string
			_ = json.Unmarshal(q["type"], &qType)
			if qID == "" || normalizeAnswerType(qType) != "score" {
				continue
			}
			for _, field := range []string{"criteria", "levels", "items"} {
				if rawLvls, has := q[field]; has {
					if lvls := extractStringSlice(rawLvls); len(lvls) >= 2 {
						out[qID] = lvls
						break
					}
				}
			}
		}
	}
	return out
}

func lookupDiagLogprob(diagLogprobs map[string]float64, answerType, optionKey string, legend map[string]string) (float64, bool) {
	_, v, ok := lookupDiagLogprobKey(diagLogprobs, answerType, optionKey, legend)
	return v, ok
}

func lookupDiagLogprobKey(diagLogprobs map[string]float64, answerType, optionKey string, legend map[string]string) (string, float64, bool) {
	if len(diagLogprobs) == 0 {
		return "", 0, false
	}
	if v, ok := diagLogprobs[optionKey]; ok {
		return optionKey, v, true
	}
	if len(legend) > 0 {
		if levelName, ok := legend[optionKey]; ok && levelName != "" {
			if v, ok := diagLogprobs[levelName]; ok {
				return levelName, v, true
			}
		}
	}
	if normalizeAnswerType(answerType) == "noul" {
		switch strings.ToLower(strings.TrimSpace(optionKey)) {
		case "yes", "true":
			if v, ok := diagLogprobs["yes"]; ok {
				return "yes", v, true
			}
			if v, ok := diagLogprobs["true"]; ok {
				return "true", v, true
			}
		case "no", "false":
			if v, ok := diagLogprobs["no"]; ok {
				return "no", v, true
			}
			if v, ok := diagLogprobs["false"]; ok {
				return "false", v, true
			}
		}
	}
	return "", 0, false
}

func hasValidScoreLevels(levels []string, probs map[string]float64) bool {
	if len(levels) != len(probs) || len(levels) == 0 {
		return false
	}
	seen := make(map[string]bool, len(levels))
	for _, lvl := range levels {
		if seen[lvl] {
			return false
		}
		if _, ok := probs[lvl]; !ok {
			return false
		}
		seen[lvl] = true
	}
	return true
}

func scoreLabelOrdinalIndex(label string, count int) int {
	lbl := strings.TrimSpace(label)
	if lbl == "" {
		return -1
	}
	if count <= 9 {
		if n, err := strconv.Atoi(lbl); err == nil && n >= 1 && n <= count {
			return n - 1
		}
	}
	if len(lbl) == 1 && lbl[0] >= 'A' && lbl[0] <= 'Z' {
		idx := int(lbl[0] - 'A')
		if idx >= 0 && idx < count {
			return idx
		}
	}
	return -1
}

// ScoreValueForAnswer computes the expected ordinal score from a.Probabilities:
//   - 0-indexed sum(i * p[strconv.Itoa(i)]) on /v1/systemone answers carrying a non-empty Legend;
//   - 1-indexed sum((i + 1) * p[level_i]) on /v1/chat/completions answers (matching structured_server.py).
func ScoreValueForAnswer(a QuestionAnswer) (float64, bool) {
	val, _, ok := recomputeScoreAnswer(a, a.Probabilities)
	return val, ok
}

func recomputeScoreAnswer(orig QuestionAnswer, scaledProbs map[string]float64) (float64, []string, bool) {
	if len(scaledProbs) == 0 {
		return 0, nil, false
	}
	if len(orig.Legend) > 0 {
		sum := 0.0
		for i := 0; i < len(scaledProbs); i++ {
			v, ok := scaledProbs[strconv.Itoa(i)]
			if !ok {
				return 0, nil, false
			}
			sum += float64(i) * v
		}
		return sum, orig.Levels, true
	}
	if hasValidScoreLevels(orig.Levels, scaledProbs) {
		sum := 0.0
		for i, lvl := range orig.Levels {
			sum += float64(i+1) * scaledProbs[lvl]
		}
		return sum, orig.Levels, true
	}
	type numericKey struct {
		key string
		val float64
	}
	numKeys := make([]numericKey, 0, len(scaledProbs))
	allNumeric := true
	seenNumeric := make(map[float64]bool, len(scaledProbs))
	for k := range scaledProbs {
		f, err := strconv.ParseFloat(strings.TrimSpace(k), 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || seenNumeric[f] {
			allNumeric = false
			break
		}
		seenNumeric[f] = true
		numKeys = append(numKeys, numericKey{key: k, val: f})
	}
	if allNumeric {
		sort.Slice(numKeys, func(i, j int) bool { return numKeys[i].val < numKeys[j].val })
		ordered := make([]string, len(numKeys))
		sum := 0.0
		for i, nk := range numKeys {
			ordered[i] = nk.key
			sum += float64(i+1) * scaledProbs[nk.key]
		}
		return sum, ordered, true
	}
	// Named levels constructed as a raw Go struct literal without Levels: recover the unique 1-indexed
	// level permutation consistent with pre-temperature orig.Score (and Label/Level when present).
	if n := len(orig.Probabilities); n >= 2 && n <= 8 && orig.Score > 0 {
		perm := make([]string, 0, n)
		for k := range scaledProbs {
			if _, ok := orig.Probabilities[k]; !ok {
				return 0, nil, false
			}
			perm = append(perm, k)
		}
		sort.Strings(perm)
		anchorIdx := scoreLabelOrdinalIndex(orig.Label, n)
		anchorLevel := strings.TrimSpace(orig.Level)
		if anchorLevel != "" {
			if _, ok := scaledProbs[anchorLevel]; !ok {
				anchorIdx = -1
			}
		}
		var (
			matched      bool
			ambiguous    bool
			matchedScore float64
			matchedOrder []string
			walk         func(pos int)
		)
		walk = func(pos int) {
			if ambiguous {
				return
			}
			if pos == n {
				if anchorIdx >= 0 && anchorLevel != "" && perm[anchorIdx] != anchorLevel {
					return
				}
				origSum := 0.0
				for i, k := range perm {
					origSum += float64(i+1) * orig.Probabilities[k]
				}
				if math.Abs(origSum-orig.Score) > 1e-6 {
					return
				}
				candSum := 0.0
				for i, k := range perm {
					candSum += float64(i+1) * scaledProbs[k]
				}
				if !matched {
					matched = true
					matchedScore = candSum
					matchedOrder = append([]string(nil), perm...)
				} else if math.Abs(candSum-matchedScore) > 1e-9 {
					ambiguous = true
				}
				return
			}
			for i := pos; i < n; i++ {
				perm[pos], perm[i] = perm[i], perm[pos]
				walk(pos + 1)
				perm[pos], perm[i] = perm[i], perm[pos]
			}
		}
		walk(0)
		if matched && !ambiguous {
			return matchedScore, matchedOrder, true
		}
	}
	return 0, nil, false
}

// ApplyTemperature rescales each answer's probabilities by p_k^(1/T) (renormalised) and recomputes confidence,
// entropy, the yes probability (noul) and the expected score, reading lossless pre-softmax log-probabilities from
// Diagnostics.Questions[id].Logprobs when present so answers saturated near 1.0 in float64 retain exact precision.
// Diagnostics.Questions[id].Logprobs is updated to the normalized scaled log-probabilities so sequential
// ApplyTemperature calls compose losslessly. The argmax answer does not change.
func (r *StructuredDecisionResponse) ApplyTemperature(t Temperature) {
	if r == nil || !t.IsSet() {
		return
	}
	for id, a := range r.Answers {
		T := t.For(a.Type)
		if T == 1 || len(a.Probabilities) == 0 {
			continue
		}
		orig := a
		keys := make([]string, 0, len(a.Probabilities))
		for k := range a.Probabilities {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var diagLP map[string]float64
		if qd, ok := r.Diagnostics.Questions[id]; ok {
			diagLP = qd.Logprobs
		}
		useDiagLP := len(diagLP) > 0
		baseLP := make([]float64, len(keys))
		matchedDiagKeys := make([]string, len(keys))
		for i, k := range keys {
			if useDiagLP {
				if dk, v, ok := lookupDiagLogprobKey(diagLP, a.Type, k, a.Legend); ok && !math.IsNaN(v) && !math.IsInf(v, 0) {
					baseLP[i] = v
					matchedDiagKeys[i] = dk
					continue
				}
				useDiagLP = false
				break
			}
		}
		if !useDiagLP {
			for i, k := range keys {
				baseLP[i] = math.Log(math.Max(a.Probabilities[k], 1e-12))
			}
		}
		scaledUnnorm := make([]float64, len(keys))
		mx := math.Inf(-1)
		for i := range keys {
			scaledUnnorm[i] = baseLP[i] / T
			mx = math.Max(mx, scaledUnnorm[i])
		}
		expShifted := make([]float64, len(keys))
		z := 0.0
		for i := range scaledUnnorm {
			expShifted[i] = math.Exp(scaledUnnorm[i] - mx)
			z += expShifted[i]
		}
		logZ := mx + math.Log(z)
		scaledLP := make([]float64, len(keys))
		p := make(map[string]float64, len(keys))
		ent, top := 0.0, -1.0
		for i, k := range keys {
			scaledLP[i] = scaledUnnorm[i] - logZ
			v := expShifted[i] / z
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
		if lbl := firstNonEmpty(a.Choice, a.Level, a.Label); lbl != "" {
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
			if scoreVal, resolvedLevels, ok := recomputeScoreAnswer(orig, p); ok {
				a.Score = scoreVal
				if len(a.Levels) == 0 && len(resolvedLevels) > 0 {
					a.Levels = resolvedLevels
				}
			}
		}
		r.Answers[id] = a
		if r.Diagnostics.Questions == nil {
			r.Diagnostics.Questions = make(map[string]QuestionDiagnostic, len(r.Answers))
		}
		q := r.Diagnostics.Questions[id]
		q.Entropy = ent
		newDiagLP := make(map[string]float64, len(keys))
		for i, k := range keys {
			dk := k
			if useDiagLP && matchedDiagKeys[i] != "" {
				dk = matchedDiagKeys[i]
			} else if len(a.Legend) > 0 {
				if lvlName := a.Legend[k]; lvlName != "" {
					dk = lvlName
				}
			}
			newDiagLP[dk] = scaledLP[i]
		}
		q.Logprobs = newDiagLP
		r.Diagnostics.Questions[id] = q
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
