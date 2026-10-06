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

import "testing"

// Bench commands must not write receipts unless asked: their old defaults pointed at committed reference
// receipts, so every live run silently overwrote them.
func TestBenchReceiptFlagsDefaultEmpty(t *testing.T) {
	for _, c := range []struct {
		name string
		flag string
		get  func() string
	}{
		{"bench-permutation", "output", func() string { return benchPermutationCmd.Flags().Lookup("output").DefValue }},
		{"bench-permutation", "out", func() string { return benchPermutationCmd.Flags().Lookup("out").DefValue }},
		{"bench-decision-index", "output", func() string { return benchDecisionIndexCmd.Flags().Lookup("output").DefValue }},
		{"bench-decision-index", "out", func() string { return benchDecisionIndexCmd.Flags().Lookup("out").DefValue }},
	} {
		if got := c.get(); got != "" {
			t.Errorf("%s --%s default = %q, want empty", c.name, c.flag, got)
		}
	}
	if benchPermutationCmd.Flags().ShorthandLookup("o") == nil || benchDecisionIndexCmd.Flags().ShorthandLookup("o") == nil {
		t.Error("bench-permutation and bench-decision-index need -o")
	}
}
