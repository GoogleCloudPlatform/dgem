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
	"testing"
)

func TestStage2Prior(t *testing.T) {
	t.Setenv("DGEM_CASCADE_PRIOR", "")
	for in, want := range map[string]string{"": "soft", "Soft": "soft", "none": "none", "full": "full"} {
		if got, err := NormalizeStage2Prior(in); err != nil || got != want {
			t.Fatalf("%q -> %q %v (want %q)", in, got, err, want)
		}
	}
	if _, err := NormalizeStage2Prior("loud"); err == nil {
		t.Fatal("invalid prior must be rejected")
	}
	t.Setenv("DGEM_CASCADE_PRIOR", "soft")
	if got := stage2PriorFrom(context.Background()); got != "soft" {
		t.Fatalf("env default: got %q", got)
	}
	if got := stage2PriorFrom(WithStage2Prior(context.Background(), "none")); got != "none" {
		t.Fatalf("context value: got %q", got)
	}
}
