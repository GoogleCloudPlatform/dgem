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
	"math"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// wideAnswer is a 26-option answer at 0.95 with the rest spread evenly: H ≈ 0.359 nats (over 0.35) but
// H / ln 26 ≈ 0.11 (under 0.16).
func wideAnswer() client.QuestionAnswer {
	p := map[string]float64{"o00": 0.95}
	for i := 1; i < 26; i++ {
		p[fmt.Sprintf("o%02d", i)] = 0.002
	}
	return client.QuestionAnswer{Type: "choice", Label: "A", Choice: "o00", Confidence: 0.95, Probabilities: p}
}

// yesNoAnswer at 0.93: H ≈ 0.254 nats (under 0.35) but H / ln 2 ≈ 0.37 (over 0.16).
func yesNoAnswer() client.QuestionAnswer {
	return client.QuestionAnswer{Type: "noul", Label: "yes", Noul: 0.93, Confidence: 0.93,
		Probabilities: map[string]float64{"yes": 0.93, "no": 0.07}}
}

func runGate(t *testing.T, mode string, qa client.QuestionAnswer) *CascadeExecutionSummary {
	t.Helper()
	resp := &client.StructuredDecisionResponse{Answers: map[string]client.QuestionAnswer{"q": qa}}
	ctx := WithCascadeThresholdMode(context.Background(), mode)
	s := ExecuteStage2GeminiCascadeWithImages(ctx, "entropy", 0, "", nil, `{"questions":[{"id":"q"}]}`, "{}", resp, nil)
	if s == nil {
		t.Fatal("nil summary")
	}
	return s
}

// The cases below never escalate, so no Stage-2 call is made.
func TestCascadeThresholdModeGate(t *testing.T) {
	s := runGate(t, "normalized", wideAnswer())
	if s.ThresholdMode != "normalized" || s.Threshold != defaultCascadeThresholdNormalized {
		t.Fatalf("mode %q threshold %v", s.ThresholdMode, s.Threshold)
	}
	sl := s.Slots["q"]
	if sl.Escalated {
		t.Fatalf("26-option slot at H/lnK=%.3f escalated under the normalized gate", sl.Stage1NormalizedEntropy)
	}
	if math.Abs(sl.Stage1NormalizedEntropy-sl.Stage1Entropy/math.Log(26)) > 1e-12 || sl.Stage1Entropy < 0.35 {
		t.Fatalf("entropy %v normalized %v", sl.Stage1Entropy, sl.Stage1NormalizedEntropy)
	}

	s = runGate(t, "", yesNoAnswer())
	if s.ThresholdMode != "nats" || s.Threshold != defaultCascadeThresholdNats {
		t.Fatalf("default mode %q threshold %v", s.ThresholdMode, s.Threshold)
	}
	if sl := s.Slots["q"]; sl.Escalated || sl.Stage1NormalizedEntropy < 0.16 {
		t.Fatalf("yes/no slot: escalated %v normalized %v", sl.Escalated, sl.Stage1NormalizedEntropy)
	}
}

func TestNormalizeCascadeThresholdMode(t *testing.T) {
	for in, want := range map[string]string{"": "nats", "NATS": "nats", "raw": "nats", "normalized": "normalized", "hesitation": "normalized"} {
		if got, err := NormalizeCascadeThresholdMode(in); err != nil || got != want {
			t.Errorf("%q -> %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := NormalizeCascadeThresholdMode("percent"); err == nil {
		t.Error("unknown mode accepted")
	}
}

func TestSlotOptionCount(t *testing.T) {
	if k := slotOptionCount(client.QuestionAnswer{}, map[string]interface{}{"options": []interface{}{"a", "b", "c"}}); k != 3 {
		t.Errorf("from schema options: %d", k)
	}
	if k := slotOptionCount(client.QuestionAnswer{}, nil); k != 2 {
		t.Errorf("fallback: %d", k)
	}
	if k := slotOptionCount(wideAnswer(), nil); k != 26 {
		t.Errorf("from probabilities: %d", k)
	}
}
