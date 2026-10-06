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
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"testing"
	"text/template"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func loadJevScoreTasks(t *testing.T) []JevTask {
	t.Helper()
	f, err := os.Open("../benchmarks/jevbench/jevbench_public.jsonl")
	if err != nil {
		t.Skip("jevbench_public.jsonl not available")
	}
	defer f.Close()
	var out []JevTask
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		var task JevTask
		if err := json.Unmarshal(sc.Bytes(), &task); err != nil {
			t.Fatal(err)
		}
		if task.Question.Type == "score" {
			out = append(out, task)
		}
	}
	return out
}

func TestBuildJevQuestionScoreIsOrdinal(t *testing.T) {
	tasks := loadJevScoreTasks(t)
	if len(tasks) != 18 {
		t.Fatalf("want 18 JevBench score items, got %d", len(tasks))
	}
	for _, task := range tasks {
		qType, opts, levels := buildJevQuestion(task, task.Labels, false)
		if qType != "score" || opts != nil || len(levels) != len(task.Labels) {
			t.Fatalf("%s: got %s, %d opts, %d levels", task.ID, qType, len(opts), len(levels))
		}
		crit := task.Question.Criteria.([]interface{})
		if levels[0] != crit[0].(string) {
			t.Fatalf("%s: first level %q is not the rubric text %q", task.ID, levels[0], crit[0])
		}
		// the old path: bare index names whose descriptions are the indices (the rubric never reached the model)
		qType, opts, levels = buildJevQuestion(task, task.Labels, true)
		if qType != "choice" || levels != nil || opts[0].Description != task.Labels[0] {
			t.Fatalf("%s: flattened path changed: %s %v", task.ID, qType, opts[0])
		}
	}
}

func TestJevTemplateRendersLevels(t *testing.T) {
	raw, err := os.ReadFile("../templates/jevbench_generic.json.tmpl")
	if err != nil {
		t.Fatal(err)
	}
	tmpl := template.Must(template.New("j").Parse(string(raw)))
	var b bytes.Buffer
	err = tmpl.Execute(&b, map[string]interface{}{"Instructions": "i", "Samples": "1", "ThinkTokens": 0,
		"SlotID": "decision", "QuestionType": "score", "QuestionPrompt": "rate", "Levels": []string{"low \"a\"", "high"}})
	if err != nil {
		t.Fatal(err)
	}
	var schema struct {
		Questions []struct {
			Type   string   `json:"type"`
			Levels []string `json:"levels"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(b.Bytes(), &schema); err != nil {
		t.Fatalf("rendered schema is not JSON: %v\n%s", err, b.String())
	}
	if q := schema.Questions[0]; q.Type != "score" || len(q.Levels) != 2 || q.Levels[0] != "low \"a\"" {
		t.Fatalf("bad question: %+v", q)
	}
}

func TestMapJevScoreAnswer(t *testing.T) {
	levels := []string{"none", "some", "all"}
	labels := []string{"0", "1", "2"}
	// keyed by level text (structured server answers)
	act, probs, conf := mapJevScoreAnswer(client.QuestionAnswer{Type: "score",
		Probabilities: map[string]float64{"none": 0.1, "some": 0.7, "all": 0.2}}, levels, labels)
	if act != "1" || conf != 0.7 || probs["2"] != 0.2 {
		t.Fatalf("text keys: %s %v %v", act, probs, conf)
	}
	// keyed by the server's digit labels 1..n
	act, _, _ = mapJevScoreAnswer(client.QuestionAnswer{Type: "score",
		Probabilities: map[string]float64{"1": 0.2, "2": 0.1, "3": 0.7}}, levels, labels)
	if act != "2" {
		t.Fatalf("digit keys: %s", act)
	}
	// reversed by --flip-options: levels follow the reversed labels
	act, _, _ = mapJevScoreAnswer(client.QuestionAnswer{Type: "score",
		Probabilities: map[string]float64{"all": 0.9, "some": 0.05, "none": 0.05}},
		[]string{"all", "some", "none"}, []string{"2", "1", "0"})
	if act != "2" {
		t.Fatalf("flipped: %s", act)
	}
	// no probabilities: the reported level
	act, _, _ = mapJevScoreAnswer(client.QuestionAnswer{Type: "score", Level: "all", Confidence: 0.8}, levels, labels)
	if act != "2" {
		t.Fatalf("level fallback: %s", act)
	}
}
