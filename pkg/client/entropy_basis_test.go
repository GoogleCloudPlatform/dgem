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
	"math"
	"testing"
)

// A structured_server.py reply: label probabilities in answers, no answer entropy, and a diagnostics entropy computed
// over the top-k tokens at the slot (here deliberately different). The parsed answer must carry the entropy of its
// probabilities (#121).
const vllmReply = `{"answers": {
  "team": {"type": "choice", "label": "A", "choice": "billing", "confidence": 0.7,
           "probabilities": {"billing": 0.7, "technical": 0.2, "account": 0.1}},
  "urgent": {"type": "noul", "label": "yes", "noul": 0.9, "confidence": 0.9,
             "probabilities": {"yes": 0.9, "no": 0.1}},
  "sure": {"type": "noul", "label": "yes", "noul": 1, "confidence": 1,
           "probabilities": {"yes": 1, "no": 0}}},
 "diagnostics": {"questions": {
  "team": {"pos": 7, "entropy": [1.9], "label_mass": 0.98, "argmax_is_label": true},
  "urgent": {"pos": 12, "entropy": [0.05], "label_mass": 0.99, "argmax_is_label": true}}}}`

func TestAnswerEntropyFromProbabilities(t *testing.T) {
	r, err := ParseStructuredContent(vllmReply)
	if err != nil {
		t.Fatal(err)
	}
	h := func(ps ...float64) float64 {
		s := 0.0
		for _, p := range ps {
			if p > 0 {
				s -= p * math.Log(p)
			}
		}
		return s
	}
	for id, want := range map[string]float64{"team": h(0.7, 0.2, 0.1), "urgent": h(0.9, 0.1), "sure": 0} {
		if got := r.Answers[id].Entropy; math.Abs(got-want) > 1e-9 {
			t.Errorf("%s: answer entropy %.6f, want %.6f (from probabilities)", id, got, want)
		}
	}
	// Server diagnostics are kept as reported.
	if got := r.Diagnostics.Questions["team"].Entropy; got != 1.9 {
		t.Errorf("diagnostic entropy changed to %v", got)
	}
}

func TestProbabilityEntropyRenormalizes(t *testing.T) {
	if got, want := ProbabilityEntropy(map[string]float64{"a": 2, "b": 2}), math.Log(2); math.Abs(got-want) > 1e-12 {
		t.Fatalf("got %v want %v", got, want)
	}
	if ProbabilityEntropy(map[string]float64{"a": 0, "b": math.NaN()}) != 0 {
		t.Fatal("degenerate distribution should give 0")
	}
}
