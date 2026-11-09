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
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

const (
	maxQuestionsPerSchema = 64
	maxOptionsDefault     = 26
	maxOptionsWideLabels  = 52
	maxLevelsPerScore     = 26
)

// ExampleSpec defines a single synthetic input case (for example "typical", "edge", or "minimal")
// for a policy template (.json.tmpl) along with optional unambiguous expected slot answers.
type ExampleSpec struct {
	Name            string            `json:"name"`
	Unambiguous     bool              `json:"unambiguous,omitempty"`
	Variables       map[string]any    `json:"variables"`
	ExpectedAnswers map[string]string `json:"expected_answers,omitempty"`
}

// ExamplesFile holds the sidecar <template>.examples.json metadata and test inputs for a .json.tmpl file.
type ExamplesFile struct {
	Multimodal       *bool         `json:"multimodal,omitempty"`
	AllowWideChoices bool          `json:"allow_wide_choices,omitempty"`
	Examples         []ExampleSpec `json:"examples"`
}

// ExamplesFilePath returns the companion .examples.json path for a .json.tmpl file path.
func ExamplesFilePath(tmplPath string) string {
	return strings.TrimSuffix(tmplPath, ".json.tmpl") + ".examples.json"
}

// LoadExamplesFile reads and decodes a <template>.examples.json file.
func LoadExamplesFile(path string) (*ExamplesFile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read examples file %s: %w", path, err)
	}
	var ef ExamplesFile
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&ef); err != nil {
		return nil, fmt.Errorf("parse examples file %s: %w", path, err)
	}
	return &ef, nil
}

// SampleVarsFromExamples converts the "typical" example (or first example) in ef into string-valued
// sample variables suitable for GET /api/templates and Studio form inputs.
func SampleVarsFromExamples(ef *ExamplesFile) map[string]string {
	if ef == nil || len(ef.Examples) == 0 {
		return nil
	}
	chosen := ef.Examples[0]
	for _, ex := range ef.Examples {
		if ex.Name == "typical" {
			chosen = ex
			break
		}
	}
	out := make(map[string]string, len(chosen.Variables))
	for k, v := range chosen.Variables {
		switch val := v.(type) {
		case nil:
			out[k] = ""
		case string:
			out[k] = val
		case bool:
			out[k] = strconv.FormatBool(val)
		case float64:
			if val == math.Trunc(val) {
				out[k] = strconv.FormatInt(int64(val), 10)
			} else {
				out[k] = strconv.FormatFloat(val, 'f', -1, 64)
			}
		default:
			if jb, err := json.Marshal(val); err == nil {
				out[k] = string(jb)
			}
		}
	}
	return out
}

// CoerceJSONVariableStrings converts string values in vars that encode JSON arrays ("[...]") into
// parsed slices so templates iterating with {{ range .options }} work when fed stringified SampleVars.
func CoerceJSONVariableStrings(vars map[string]any) map[string]any {
	if len(vars) == 0 {
		return vars
	}
	out := make(map[string]any, len(vars))
	for k, v := range vars {
		if s, ok := v.(string); ok {
			trimmed := strings.TrimSpace(s)
			if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") {
				var parsed []any
				if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
					out[k] = parsed
					continue
				}
			}
		}
		out[k] = v
	}
	return out
}

// ValidateRenderedPayload parses rendered through ParseStructuredPayload and checks that the resulting
// schema and state satisfy structured_server.py's parse_schema contract, option limits, DAG rules,
// and any expectedAnswers keys/values.
func ValidateRenderedPayload(rendered string, fallbackVars map[string]any, allowWideChoices bool, expectedAnswers map[string]string) error {
	var rawCheck any
	if err := json.Unmarshal([]byte(strings.TrimSpace(rendered)), &rawCheck); err != nil {
		return fmt.Errorf("rendered template is not valid JSON: %w", err)
	}

	schemaStr, stateStr, err := ParseStructuredPayload(rendered, fallbackVars)
	if err != nil {
		return fmt.Errorf("parse structured payload: %w", err)
	}

	var schema map[string]any
	if err := json.Unmarshal([]byte(schemaStr), &schema); err != nil {
		return fmt.Errorf("schema is not a JSON object: %w", err)
	}
	var state map[string]any
	if err := json.Unmarshal([]byte(stateStr), &state); err != nil {
		return fmt.Errorf("state is not a JSON object: %w", err)
	}

	rawQuestions, ok := schema["questions"].([]any)
	if !ok || len(rawQuestions) == 0 {
		return fmt.Errorf("schema.questions must be a non-empty array")
	}
	if len(rawQuestions) > maxQuestionsPerSchema {
		return fmt.Errorf("schema.questions count %d exceeds maximum %d", len(rawQuestions), maxQuestionsPerSchema)
	}

	labelSet := "az"
	if rawLabels, has := schema["labels"]; has && rawLabels != nil {
		ls, isStr := rawLabels.(string)
		if !isStr {
			return fmt.Errorf("schema.labels must be a string")
		}
		switch ls {
		case "az", "az_aa", "az_lower":
			labelSet = ls
		default:
			return fmt.Errorf("schema.labels %q must be az, az_aa, or az_lower", ls)
		}
	}
	if labelSet == "az_lower" && len(rawQuestions) > 10 {
		return fmt.Errorf("schema.labels az_lower requires at most 10 questions (got %d)", len(rawQuestions))
	}

	if rawLayout, has := schema["layout"]; has && rawLayout != nil {
		ly, isStr := rawLayout.(string)
		if !isStr || (ly != "document_first" && ly != "schema_first") {
			return fmt.Errorf("schema.layout must be document_first or schema_first (got %v)", rawLayout)
		}
	}
	if rawIsolate, has := schema["isolate"]; has && rawIsolate != nil {
		iso, isStr := rawIsolate.(string)
		if !isStr || (iso != "auto" && iso != "noul" && iso != "none" && iso != "all") {
			return fmt.Errorf("schema.isolate must be auto, noul, none, or all (got %v)", rawIsolate)
		}
	}
	if rawSeed, has := schema["seed_token"]; has && rawSeed != nil {
		st, isStr := rawSeed.(string)
		if !isStr || (st != "random" && st != "fixed") {
			return fmt.Errorf("schema.seed_token must be random or fixed (got %v)", rawSeed)
		}
	}
	if rawSamples, has := schema["samples"]; has && rawSamples != nil {
		if err := validateSamplesField(rawSamples); err != nil {
			return err
		}
	}
	if rawThink, has := schema["think"]; has && rawThink != nil {
		tf, isNum := rawThink.(float64)
		if !isNum || tf != math.Trunc(tf) || tf < 0 || tf > 4096 {
			return fmt.Errorf("schema.think must be an integer in [0, 4096] (got %v)", rawThink)
		}
	}

	maxChoice := maxOptionsDefault
	if labelSet == "az_aa" || labelSet == "az_lower" {
		maxChoice = maxOptionsWideLabels
	}

	seenIDs := make(map[string]bool, len(rawQuestions))
	validDomain := make(map[string]map[string]bool, len(rawQuestions))
	depsByID := make(map[string][]string, len(rawQuestions))
	askIfByID := make(map[string]map[string][]string, len(rawQuestions))

	for idx, item := range rawQuestions {
		qm, isObj := item.(map[string]any)
		if !isObj {
			return fmt.Errorf("question[%d] must be an object", idx)
		}
		qid, _ := qm["id"].(string)
		qid = strings.TrimSpace(qid)
		if qid == "" || strings.ContainsAny(qid, ":\n") {
			return fmt.Errorf("question[%d] id %q must be non-empty without ':' or newline", idx, qid)
		}
		if seenIDs[qid] {
			return fmt.Errorf("duplicate question id %q", qid)
		}
		seenIDs[qid] = true

		instr, _ := qm["instructions"].(string)
		if strings.TrimSpace(instr) == "" {
			return fmt.Errorf("question %q must have non-empty instructions", qid)
		}

		qType, _ := qm["type"].(string)
		domain := make(map[string]bool)
		switch qType {
		case "boolean", "bool", "noul":
			domain["yes"] = true
			domain["no"] = true
		case "choice":
			opts, isList := qm["options"].([]any)
			if !isList || len(opts) < 2 {
				return fmt.Errorf("choice question %q needs at least 2 options (got %d)", qid, len(opts))
			}
			if !allowWideChoices && len(opts) > maxChoice {
				return fmt.Errorf("choice question %q has %d options, exceeding limit %d for label set %q", qid, len(opts), maxChoice, labelSet)
			}
			for oi, rawOpt := range opts {
				var optName string
				switch ov := rawOpt.(type) {
				case map[string]any:
					optName, _ = ov["name"].(string)
				case string:
					optName = ov
				default:
					return fmt.Errorf("question %q option[%d] must be an object or string", qid, oi)
				}
				optName = strings.TrimSpace(optName)
				if optName == "" {
					return fmt.Errorf("question %q option[%d] has empty name", qid, oi)
				}
				if domain[optName] {
					return fmt.Errorf("question %q has duplicate option name %q", qid, optName)
				}
				domain[optName] = true
			}
		case "score":
			lvls, isList := qm["levels"].([]any)
			if !isList || len(lvls) < 2 || len(lvls) > maxLevelsPerScore {
				return fmt.Errorf("score question %q needs 2..%d levels (got %d)", qid, maxLevelsPerScore, len(lvls))
			}
			for li, rawLvl := range lvls {
				ls, isStr := rawLvl.(string)
				if !isStr || strings.TrimSpace(ls) == "" {
					return fmt.Errorf("score question %q level[%d] must be a non-empty string", qid, li)
				}
				domain[ls] = true
			}
		default:
			return fmt.Errorf("question %q has unknown type %q", qid, qType)
		}
		validDomain[qid] = domain

		deps, askIf, err := parseQuestionDeps(qid, qm)
		if err != nil {
			return err
		}
		depsByID[qid] = deps
		askIfByID[qid] = askIf
	}

	if err := validateDependencyDAG(seenIDs, validDomain, depsByID, askIfByID); err != nil {
		return err
	}

	for qid, want := range expectedAnswers {
		dom, exists := validDomain[qid]
		if !exists {
			return fmt.Errorf("expected_answers references unknown question id %q", qid)
		}
		if !dom[want] {
			return fmt.Errorf("expected_answers[%q] = %q is not a valid answer option", qid, want)
		}
	}
	return nil
}

func validateSamplesField(raw any) error {
	switch sv := raw.(type) {
	case float64:
		if sv != math.Trunc(sv) || sv < 1 {
			return fmt.Errorf("schema.samples must be a positive integer or auto (got %v)", raw)
		}
	case string:
		s := strings.TrimSpace(sv)
		if s == "auto" {
			return nil
		}
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			return fmt.Errorf("schema.samples must be a positive integer or auto (got %q)", sv)
		}
	default:
		return fmt.Errorf("schema.samples must be a positive integer or auto (got %v)", raw)
	}
	return nil
}

func parseQuestionDeps(qid string, qm map[string]any) ([]string, map[string][]string, error) {
	var deps []string
	seen := make(map[string]bool)
	if rawDeps, has := qm["depends_on"]; has && rawDeps != nil {
		dl, isList := rawDeps.([]any)
		if !isList {
			return nil, nil, fmt.Errorf("question %q depends_on must be a list of strings", qid)
		}
		for _, item := range dl {
			ds, isStr := item.(string)
			if !isStr || strings.TrimSpace(ds) == "" {
				return nil, nil, fmt.Errorf("question %q depends_on entries must be non-empty strings", qid)
			}
			if !seen[ds] {
				seen[ds] = true
				deps = append(deps, ds)
			}
		}
	}
	askIf := make(map[string][]string)
	if rawAsk, has := qm["ask_if"]; has && rawAsk != nil {
		am, isMap := rawAsk.(map[string]any)
		if !isMap {
			return nil, nil, fmt.Errorf("question %q ask_if must be an object", qid)
		}
		for dep, rawVals := range am {
			vl, isList := rawVals.([]any)
			if !isList || len(vl) == 0 {
				return nil, nil, fmt.Errorf("question %q ask_if[%q] must be a non-empty list", qid, dep)
			}
			var vals []string
			for _, v := range vl {
				vs, isStr := v.(string)
				if !isStr {
					return nil, nil, fmt.Errorf("question %q ask_if[%q] values must be strings", qid, dep)
				}
				vals = append(vals, vs)
			}
			askIf[dep] = vals
			if !seen[dep] {
				seen[dep] = true
				deps = append(deps, dep)
			}
		}
	}
	return deps, askIf, nil
}

func validateDependencyDAG(
	seenIDs map[string]bool,
	validDomain map[string]map[string]bool,
	depsByID map[string][]string,
	askIfByID map[string]map[string][]string,
) error {
	for qid, deps := range depsByID {
		for _, dep := range deps {
			if !seenIDs[dep] || dep == qid {
				return fmt.Errorf("question %q depends on unknown or self question %q", qid, dep)
			}
		}
		for dep, vals := range askIfByID[qid] {
			dom := validDomain[dep]
			for _, v := range vals {
				if !dom[v] {
					return fmt.Errorf("question %q ask_if[%q] value %q is not in %q's answer domain", qid, dep, v, dep)
				}
			}
		}
	}

	const (
		unvisited = 0
		visiting  = 1
		visited   = 2
	)
	state := make(map[string]int, len(seenIDs))
	var visit func(string) error
	visit = func(id string) error {
		state[id] = visiting
		for _, dep := range depsByID[id] {
			if state[dep] == visiting {
				return fmt.Errorf("cyclic depends_on involving question %q and %q", id, dep)
			}
			if state[dep] == unvisited {
				if err := visit(dep); err != nil {
					return err
				}
			}
		}
		state[id] = visited
		return nil
	}
	for id := range seenIDs {
		if state[id] == unvisited {
			if err := visit(id); err != nil {
				return err
			}
		}
	}
	return nil
}
