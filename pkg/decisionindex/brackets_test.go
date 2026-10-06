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
	"fmt"
	"math"
	"testing"
)

func TestBalancedBracketsNoSingletons(t *testing.T) {
	for k := 2; k <= 255; k++ {
		keys := make([]string, k)
		for i := range keys {
			keys[i] = fmt.Sprintf("o%d", i)
		}
		bs := balancedBrackets(keys, BracketSize)
		total := 0
		for _, b := range bs {
			if len(b) > BracketSize || (k > 1 && len(b) < 2) {
				t.Fatalf("K=%d: bracket of size %d", k, len(b))
			}
			total += len(b)
		}
		if total != k {
			t.Fatalf("K=%d: covered %d", k, total)
		}
	}
}

func TestFuseBracketProbabilities(t *testing.T) {
	// 3 brackets of 20; bracket 0 is confident in o0, others diffuse.
	var brackets [][]string
	var round1 []map[string]float64
	for b := 0; b < 3; b++ {
		var keys []string
		p := map[string]float64{}
		for i := 0; i < 20; i++ {
			k := fmt.Sprintf("o%d", b*20+i)
			keys = append(keys, k)
			p[k] = 0.01
		}
		if b == 0 {
			p["o0"] = 0.81
		}
		brackets = append(brackets, keys)
		round1 = append(round1, p)
	}
	fin := map[string]bool{"o0": true, "o1": true, "o20": true, "o21": true, "o40": true, "o41": true}
	final := map[string]float64{"o0": 0.99, "o1": 0.002, "o20": 0.002, "o21": 0.002, "o40": 0.002, "o41": 0.002}
	got := fuseBracketProbabilities(brackets, round1, fin, final)
	if len(got) != 60 {
		t.Fatalf("covers %d options", len(got))
	}
	sum, top, arg := 0.0, 0.0, ""
	for k, v := range got {
		if v <= 0 {
			t.Fatalf("%s non-positive", k)
		}
		sum += v
		if v > top {
			top, arg = v, k
		}
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("sum %v", sum)
	}
	if arg != "o0" {
		t.Fatalf("argmax %s", arg)
	}
	// Final picks o0 at 0.99, so bracket 0 carries ~all the evidence; its 18 non-finalists hold 0.18 of Round-1 mass.
	// Expected top ≈ 0.99 × (1 − 0.99·0.18) ≈ 0.81, i.e. derived from the readouts, not the old 0.92 constant.
	if math.Abs(top-0.814) > 0.01 {
		t.Fatalf("top %v, want ≈0.814", top)
	}

	// Confident brackets: finalists hold almost all Round-1 mass, so the fused top tracks the final.
	for b := range round1 {
		for k := range round1[b] {
			round1[b][k] = 1e-4
		}
	}
	round1[0]["o0"], round1[0]["o1"] = 0.9, 0.09
	round1[1]["o20"], round1[1]["o21"] = 0.5, 0.49
	round1[2]["o40"], round1[2]["o41"] = 0.5, 0.49
	got = fuseBracketProbabilities(brackets, round1, fin, final)
	if got["o0"] < 0.95 {
		t.Fatalf("confident case capped: %v", got["o0"])
	}
}
