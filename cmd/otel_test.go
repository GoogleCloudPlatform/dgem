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
	"time"

	"go.opentelemetry.io/otel/trace"
)

func TestGetTraceSpansByTraceIDHierarchicalWaterfall(t *testing.T) {
	shutdown := initGatewayTracer(context.Background())
	defer func() { _ = shutdown(context.Background()) }()

	t0 := time.Now()
	ctx, rootSpan := gatewayTracer().Start(context.Background(), "dgem.gateway.decide", trace.WithTimestamp(t0))
	traceID := rootSpan.SpanContext().TraceID().String()

	// Child 1: template.render (0ms -> 2ms)
	tRenderEnd := t0.Add(2 * time.Millisecond)
	_, renderSpan := gatewayTracer().Start(ctx, "dgem.template.render", trace.WithTimestamp(t0))
	renderSpan.End(trace.WithTimestamp(tRenderEnd))

	// Child 2: gpu.orchestrate (2ms -> 502ms)
	orchCtx, orchSpan := gatewayTracer().Start(ctx, "dgem.gpu.orchestrate", trace.WithTimestamp(tRenderEnd))
	tAttemptEnd := tRenderEnd.Add(500 * time.Millisecond)

	// Grandchild: gpu.forward_pass (2ms -> 502ms)
	attemptCtx, attemptSpan := gatewayTracer().Start(orchCtx, "dgem.gpu.forward_pass", trace.WithTimestamp(tRenderEnd))

	// Leaf 1: network_and_auth (2ms -> 22ms)
	tNetEnd := tRenderEnd.Add(20 * time.Millisecond)
	_, netSpan := gatewayTracer().Start(attemptCtx, "dgem.gpu.network_and_auth", trace.WithTimestamp(tRenderEnd))
	netSpan.End(trace.WithTimestamp(tNetEnd))

	// Leaf 2: prefill (22ms -> 172ms)
	tPrefillEnd := tNetEnd.Add(150 * time.Millisecond)
	_, prefillSpan := gatewayTracer().Start(attemptCtx, "dgem.gpu.prefill", trace.WithTimestamp(tNetEnd))
	prefillSpan.End(trace.WithTimestamp(tPrefillEnd))

	// Leaf 3: denoise (172ms -> 502ms)
	_, denoiseSpan := gatewayTracer().Start(attemptCtx, "dgem.gpu.denoise", trace.WithTimestamp(tPrefillEnd))
	denoiseSpan.End(trace.WithTimestamp(tAttemptEnd))

	// End parent spans in reverse order (just like defer in production)
	attemptSpan.End(trace.WithTimestamp(tAttemptEnd))
	orchSpan.End(trace.WithTimestamp(tAttemptEnd))
	rootSpan.End(trace.WithTimestamp(tAttemptEnd))

	spans := getTraceSpansByTraceID(traceID)
	expectedOrder := []struct {
		name  string
		depth int
	}{
		{"dgem.gateway.decide", 0},
		{"dgem.template.render", 1},
		{"dgem.gpu.orchestrate", 1},
		{"dgem.gpu.forward_pass", 2},
		{"dgem.gpu.network_and_auth", 3},
		{"dgem.gpu.prefill", 3},
		{"dgem.gpu.denoise", 3},
	}

	if len(spans) != len(expectedOrder) {
		t.Fatalf("expected %d spans, got %d", len(expectedOrder), len(spans))
	}
	for i, exp := range expectedOrder {
		if spans[i].Name != exp.name {
			t.Errorf("span[%d]: expected name %q, got %q", i, exp.name, spans[i].Name)
		}
		if spans[i].Depth != exp.depth {
			t.Errorf("span[%d] (%s): expected depth %d, got %d", i, exp.name, exp.depth, spans[i].Depth)
		}
	}

	// Verify sequential offsets for leaf phases inside forward_pass
	if spans[4].OffsetMs >= spans[5].OffsetMs || spans[5].OffsetMs >= spans[6].OffsetMs {
		t.Errorf("expected increasing offsets for network (%.2f) < prefill (%.2f) < denoise (%.2f)",
			spans[4].OffsetMs, spans[5].OffsetMs, spans[6].OffsetMs)
	}
}
