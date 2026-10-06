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
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// decideSummary holds the top-level summary fields of GatewayDecideResponse
// that are derived from an upstream decision rather than copied from it.
type decideSummary struct {
	// MaxEntropy is the largest per-slot Shannon entropy (nats) across
	// diagnostics.questions and answers. Backends disagree on where entropy
	// lives: local diffgemma reports it only under diagnostics.questions,
	// while logprob-derived vLLM responses set it on answers.
	MaxEntropy float64
	// GpuForwardMs is the winning upstream round trip (client wall time),
	// falling back to the server-reported diagnostics.timing.total_ms.
	GpuForwardMs int64
	// ColdStartWaitMs is orchestration time not spent in the winning forward
	// pass (scale-from-zero retries and backoff), clamped at 0.
	ColdStartWaitMs int64
}

// summarizeDecision computes decideSummary for a decision returned by
// executeDecideWithWarmup. orchElapsed is the wall time of the
// executeDecideWithWarmup call. POST /api/decide and the MCP decide tools
// share this so their summary fields cannot drift apart (issue #14).
func summarizeDecision(resp *client.StructuredDecisionResponse, stats *client.RequestStats, orchElapsed time.Duration) decideSummary {
	var s decideSummary
	if resp != nil {
		s.MaxEntropy = maxSlotEntropy(resp)
	}
	if stats != nil {
		s.GpuForwardMs = stats.WallTime.Milliseconds()
	}
	if s.GpuForwardMs <= 0 && resp != nil && resp.Diagnostics.Timing.TotalMs > 0 {
		s.GpuForwardMs = int64(resp.Diagnostics.Timing.TotalMs)
	}
	s.ColdStartWaitMs = orchElapsed.Milliseconds() - s.GpuForwardMs
	if s.ColdStartWaitMs < 0 {
		s.ColdStartWaitMs = 0
	}
	return s
}

func maxSlotEntropy(resp *client.StructuredDecisionResponse) float64 {
	maxEntropy := 0.0
	for _, q := range resp.Diagnostics.Questions {
		if q.Entropy > maxEntropy {
			maxEntropy = q.Entropy
		}
	}
	for _, a := range resp.Answers {
		if a.Entropy > maxEntropy {
			maxEntropy = a.Entropy
		}
	}
	return maxEntropy
}

// recordBackendReadoutLatency routes a readout latency sample to the telemetry
// for the backend that served it. Only Cloud Run marks the GPU warm (see
// AGENTS.md: never call MarkGPUWarm for Vertex or local readouts).
func recordBackendReadoutLatency(backendTarget string, readoutMs int64) {
	switch backendTarget {
	case "cloudrun":
		MarkGPUWarm(readoutMs)
	case "vertex":
		RecordVertexReadoutLatency(readoutMs)
	case "local":
		RecordLocalReadoutLatency(readoutMs)
	}
}
