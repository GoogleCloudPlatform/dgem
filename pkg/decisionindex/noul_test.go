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

import "testing"

func TestNoulCriteriaKept(t *testing.T) {
	got := noulCriteria(map[string]string{"true": " The response is unsupported. ", "false": "All content is supported."})
	if got["true"] != "The response is unsupported." || got["false"] != "All content is supported." {
		t.Fatalf("got %v", got)
	}
	if got := noulCriteria(map[string]string{"Yes": "a", "no": "b"}); got["true"] != "a" || got["false"] != "b" {
		t.Fatalf("yes/no synonyms: %v", got)
	}
	if noulCriteria(nil) != nil || noulCriteria(map[string]string{"maybe": "x"}) != nil {
		t.Fatal("expected nil")
	}
}
