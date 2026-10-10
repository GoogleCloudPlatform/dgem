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
	"fmt"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestSplitBalanced(t *testing.T) {
	opts := make([]string, 30)
	for i := range opts {
		opts[i] = fmt.Sprintf("o%02d", i)
	}
	g := splitBalanced(opts, 20)
	if len(g) != 2 || len(g[0]) != 15 || len(g[1]) != 15 {
		t.Fatalf("30/20 -> %d groups %v", len(g), []int{len(g[0]), len(g[1])})
	}
	if g := splitBalanced(opts[:77%30], 20); len(g) != 1 {
		t.Fatalf("17 options -> %d groups", len(g))
	}
}

// fake decide: picks the option named "o21" when present, otherwise the first option; rejects > 26 options.
func fakeDecide(calls *[]int) intentDecideFunc {
	return func(v map[string]interface{}) (*client.StructuredDecisionResponse, *client.RequestStats, error) {
		opts := v["options"].([]string)
		*calls = append(*calls, len(opts))
		if len(opts) > maxChoiceOptions {
			return nil, nil, fmt.Errorf("at most 26 alternatives")
		}
		pick := opts[0]
		for _, o := range opts {
			if o == "o21" {
				pick = o
			}
		}
		probs := map[string]float64{}
		for _, o := range opts {
			probs[o] = 0.01
		}
		probs[pick] = 0.9
		return &client.StructuredDecisionResponse{Answers: map[string]client.QuestionAnswer{
				"intent": {Type: "choice", Label: pick, Choice: pick, Probabilities: probs}}},
			&client.RequestStats{WallTime: 10 * time.Millisecond}, nil
	}
}

func TestDecideIntentBracketed(t *testing.T) {
	opts := make([]string, 30)
	for i := range opts {
		opts[i] = fmt.Sprintf("o%02d", i)
	}
	var calls []int
	resp, stats, err := decideIntentBracketed(context.Background(), "", map[string]interface{}{"text": "x"}, opts, fakeDecide(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if got := intentAnswer(resp).DisplayValue(); got != "o21" {
		t.Fatalf("final answer %q, want o21", got)
	}
	if fmt.Sprint(calls) != "[15 15 10]" {
		t.Fatalf("calls %v, want two groups of 15 then 10 finalists", calls)
	}
	if stats.WallTime != 30*time.Millisecond {
		t.Fatalf("wall time %v, want summed 30ms", stats.WallTime)
	}

	calls = nil
	if _, _, err := decideIntentBracketed(context.Background(), "", nil, opts[:26], fakeDecide(&calls)); err != nil || fmt.Sprint(calls) != "[26]" {
		t.Fatalf("26 options should be one call: %v %v", calls, err)
	}
}

// TestDecideIntentBracketedWide: 151 options (full CLINC150) keep 8 x 5 = 40 intents after round 1, more than one
// question holds, so they are bracketed again instead of failing the final (#119).
func TestDecideIntentBracketedWide(t *testing.T) {
	opts := make([]string, 151)
	for i := range opts {
		opts[i] = fmt.Sprintf("o%02d", i)
	}
	var calls []int
	resp, stats, err := decideIntentBracketed(context.Background(), "", nil, opts, fakeDecide(&calls))
	if err != nil {
		t.Fatalf("151 options: %v (calls %v)", err, calls)
	}
	if got := intentAnswer(resp).DisplayValue(); got != "o21" {
		t.Fatalf("final answer %q, want o21", got)
	}
	for _, n := range calls {
		if n > maxChoiceOptions {
			t.Fatalf("a round sent %d options: %v", n, calls)
		}
	}
	// 8 groups of 18-19, then 40 kept -> 2 groups of 20, then 10 finalists.
	if len(calls) != 11 || calls[len(calls)-1] != 10 {
		t.Fatalf("calls %v, want 8 + 2 group reads then a final of 10", calls)
	}
	if stats.WallTime != time.Duration(len(calls))*10*time.Millisecond {
		t.Fatalf("wall time %v, want %d rounds summed", stats.WallTime, len(calls))
	}
}
