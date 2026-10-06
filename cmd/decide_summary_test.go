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
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestSummarizeDecision(t *testing.T) {
	tests := []struct {
		name        string
		resp        *client.StructuredDecisionResponse
		stats       *client.RequestStats
		orchElapsed time.Duration
		want        decideSummary
	}{
		{
			name: "entropy only in diagnostics.questions (local diffgemma)",
			resp: &client.StructuredDecisionResponse{
				Answers: map[string]client.QuestionAnswer{"a": {}, "b": {}},
				Diagnostics: client.Diagnostics{Questions: map[string]client.QuestionDiagnostic{
					"a": {Entropy: 0.0007}, "b": {Entropy: 0.5},
				}},
			},
			stats:       &client.RequestStats{WallTime: 16 * time.Second},
			orchElapsed: 16 * time.Second,
			want:        decideSummary{MaxEntropy: 0.5, GpuForwardMs: 16000, ColdStartWaitMs: 0},
		},
		{
			name: "entropy only on answers (vLLM logprobs)",
			resp: &client.StructuredDecisionResponse{
				Answers: map[string]client.QuestionAnswer{"a": {Entropy: 0.2}, "b": {Entropy: 0.9}},
			},
			stats:       &client.RequestStats{WallTime: 150 * time.Millisecond},
			orchElapsed: 150 * time.Millisecond,
			want:        decideSummary{MaxEntropy: 0.9, GpuForwardMs: 150},
		},
		{
			name: "max across both sources",
			resp: &client.StructuredDecisionResponse{
				Answers:     map[string]client.QuestionAnswer{"a": {Entropy: 0.3}},
				Diagnostics: client.Diagnostics{Questions: map[string]client.QuestionDiagnostic{"a": {Entropy: 0.4}}},
			},
			stats:       &client.RequestStats{WallTime: time.Second},
			orchElapsed: time.Second,
			want:        decideSummary{MaxEntropy: 0.4, GpuForwardMs: 1000},
		},
		{
			name: "cold start: orchestration minus winning forward pass",
			resp: &client.StructuredDecisionResponse{},
			stats: &client.RequestStats{
				WallTime: 900 * time.Millisecond,
			},
			orchElapsed: 150 * time.Second,
			want:        decideSummary{GpuForwardMs: 900, ColdStartWaitMs: 149100},
		},
		{
			name: "falls back to server timing.total_ms",
			resp: &client.StructuredDecisionResponse{
				Diagnostics: client.Diagnostics{Timing: client.TimingStats{TotalMs: 712.6}},
			},
			stats:       &client.RequestStats{},
			orchElapsed: 800 * time.Millisecond,
			want:        decideSummary{GpuForwardMs: 712, ColdStartWaitMs: 88},
		},
		{
			name:        "cold wait clamps at zero",
			resp:        &client.StructuredDecisionResponse{},
			stats:       &client.RequestStats{WallTime: 500 * time.Millisecond},
			orchElapsed: 499 * time.Millisecond,
			want:        decideSummary{GpuForwardMs: 500},
		},
		{
			name:        "nil response and stats",
			orchElapsed: 10 * time.Millisecond,
			want:        decideSummary{ColdStartWaitMs: 10},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := summarizeDecision(tt.resp, tt.stats, tt.orchElapsed); got != tt.want {
				t.Errorf("summarizeDecision() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
