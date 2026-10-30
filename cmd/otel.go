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
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	cloudtrace "github.com/GoogleCloudPlatform/opentelemetry-operations-go/exporter/trace"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

const gatewayTracerName = "github.com/GoogleCloudPlatform/dgem/gateway"

// TraceSpanRecord represents a serialized OpenTelemetry span for API responses & /api/traces.
type TraceSpanRecord struct {
	TraceID      string                 `json:"trace_id"`
	SpanID       string                 `json:"span_id"`
	ParentSpanID string                 `json:"parent_span_id,omitempty"`
	Name         string                 `json:"name"`
	StartTime    string                 `json:"start_time"`
	EndTime      string                 `json:"end_time"`
	OffsetMs     float64                `json:"offset_ms"`
	DurationMs   float64                `json:"duration_ms"`
	Depth        int                    `json:"depth"`
	Status       string                 `json:"status"`
	Attributes   map[string]interface{} `json:"attributes,omitempty"`
}

var (
	traceRingMu   sync.Mutex
	traceRing     []TraceSpanRecord
	maxTraceItems = 250
	gcpProjectID  string
)

func detectGCPProjectID() string {
	for _, env := range []string{"GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "GCLOUD_PROJECT", "DGEM_GCP_PROJECT"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			return v
		}
	}
	req, err := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/project/project-id", nil)
	if err == nil {
		req.Header.Set("Metadata-Flavor", "Google")
		hc := &http.Client{Timeout: 800 * time.Millisecond}
		if resp, err := hc.Do(req); err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				if b, err := io.ReadAll(resp.Body); err == nil {
					if p := strings.TrimSpace(string(b)); p != "" {
						return p
					}
				}
			}
		}
	}
	return ""
}

func detectGCPProjectNumber() string {
	for _, env := range []string{"GCP_PROJECT_NUMBER", "GOOGLE_CLOUD_PROJECT_NUMBER", "DGEM_GCP_PROJECT_NUMBER"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			return v
		}
	}
	req, err := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/project/numeric-project-id", nil)
	if err == nil {
		req.Header.Set("Metadata-Flavor", "Google")
		hc := &http.Client{Timeout: 800 * time.Millisecond}
		if resp, err := hc.Do(req); err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				if b, err := io.ReadAll(resp.Body); err == nil {
					if p := strings.TrimSpace(string(b)); p != "" {
						return p
					}
				}
			}
		}
	}
	return ""
}

type gatewaySpanProcessor struct {
	projectID string
}

func (p *gatewaySpanProcessor) OnStart(parent context.Context, s sdktrace.ReadWriteSpan) {}

func (p *gatewaySpanProcessor) OnEnd(s sdktrace.ReadOnlySpan) {
	if strings.HasPrefix(s.Name(), "google.devtools.cloudtrace") {
		return
	}
	sc := s.SpanContext()
	traceID := sc.TraceID().String()
	spanID := sc.SpanID().String()
	parentID := ""
	if s.Parent().IsValid() {
		parentID = s.Parent().SpanID().String()
	}

	durMs := float64(s.EndTime().Sub(s.StartTime()).Microseconds()) / 1000.0
	attrs := make(map[string]interface{}, len(s.Attributes()))
	for _, kv := range s.Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}

	rec := TraceSpanRecord{
		TraceID:      traceID,
		SpanID:       spanID,
		ParentSpanID: parentID,
		Name:         s.Name(),
		StartTime:    s.StartTime().UTC().Format(time.RFC3339Nano),
		EndTime:      s.EndTime().UTC().Format(time.RFC3339Nano),
		DurationMs:   durMs,
		Status:       s.Status().Code.String(),
		Attributes:   attrs,
	}

	traceRingMu.Lock()
	traceRing = append(traceRing, rec)
	if len(traceRing) > maxTraceItems {
		traceRing = traceRing[len(traceRing)-maxTraceItems:]
	}
	traceRingMu.Unlock()

	// Emit Cloud Run structured JSON log entry correlated with Cloud Trace
	logEntry := map[string]interface{}{
		"severity":    "INFO",
		"message":     fmt.Sprintf("[OTel Span] %s completed in %.2f ms", s.Name(), durMs),
		"span_name":   s.Name(),
		"duration_ms": durMs,
		"attributes":  attrs,
	}
	if s.Name() == "dgem.gateway.decide" {
		logEntry["event_type"] = "dgem.decision.completed"
		if v, ok := attrs["dgem.surface"]; ok {
			logEntry["dgem_surface"] = v
		} else {
			logEntry["dgem_surface"] = "rest_api"
		}
		// "none" when the request failed before a backend was chosen (e.g. 400 unknown backend); defaulting
		// to "cloudrun" made failed requests look like Cloud Run decisions in metrics.
		if v, ok := attrs["dgem.backend"]; ok {
			logEntry["dgem_backend"] = v
		} else {
			logEntry["dgem_backend"] = "none"
		}
		if v, ok := attrs["dgem.backend_requested"]; ok {
			logEntry["dgem_backend_requested"] = v
		}
		if v, ok := attrs["dgem.template"]; ok {
			logEntry["dgem_template"] = v
		}
		if v, ok := attrs["dgem.user"]; ok {
			logEntry["dgem_user"] = v
		}
		if v, ok := attrs["dgem.multimodal"]; ok {
			logEntry["dgem_multimodal"] = v
		}
		if v, ok := attrs["dgem.max_entropy"]; ok {
			logEntry["dgem_max_entropy"] = v
		}
		if v, ok := attrs["dgem.gpu.reads"]; ok {
			logEntry["dgem_gpu_reads"] = v
		}
		if v, ok := attrs["dgem.total_wall_ms"]; ok {
			logEntry["dgem_total_wall_ms"] = v
		}
		if v, ok := attrs["dgem.gpu.forward_ms"]; ok {
			logEntry["dgem_gpu_forward_ms"] = v
		}
		if v, ok := attrs["dgem.gpu.prefill_ms"]; ok {
			logEntry["dgem_gpu_prefill_ms"] = v
		}
		if v, ok := attrs["dgem.gpu.denoise_ms"]; ok {
			logEntry["dgem_gpu_denoise_ms"] = v
		}
		if v, ok := attrs["dgem.gpu.network_ms"]; ok {
			logEntry["dgem_gpu_network_ms"] = v
		}
		if v, ok := attrs["dgem.gpu.cold_start_wait_ms"]; ok {
			logEntry["dgem_cold_start_wait_ms"] = v
		}
	} else if s.Name() == "dgem.gpu.warmup_lifecycle" {
		logEntry["event_type"] = "dgem.warmup.completed"
		if v, ok := attrs["dgem.warmup.trigger"]; ok {
			logEntry["dgem_warmup_trigger"] = v
		}
		if v, ok := attrs["dgem.warmup.total_ms"]; ok {
			logEntry["dgem_warmup_total_ms"] = v
		}
		if v, ok := attrs["dgem.warmup.stage1_activator_ms"]; ok {
			logEntry["dgem_warmup_stage1_ms"] = v
		}
		if v, ok := attrs["dgem.warmup.stage2_tmpfs_ms"]; ok {
			logEntry["dgem_warmup_stage2_ms"] = v
		}
		if v, ok := attrs["dgem.warmup.stage3_vllm_siglip_ms"]; ok {
			logEntry["dgem_warmup_stage3_ms"] = v
		}
		if v, ok := attrs["dgem.warmup.stage4_triton_jit_ms"]; ok {
			logEntry["dgem_warmup_stage4_ms"] = v
		}
		if v, ok := attrs["dgem.warmup.ewma_wake_seconds"]; ok {
			logEntry["dgem_warmup_ewma_seconds"] = v
		}
	}
	if p.projectID != "" && sc.TraceID().IsValid() {
		logEntry["logging.googleapis.com/trace"] = fmt.Sprintf("projects/%s/traces/%s", p.projectID, traceID)
		logEntry["logging.googleapis.com/spanId"] = spanID
		logEntry["logging.googleapis.com/trace_sampled"] = sc.IsSampled()
	}
	if sev, msg := spanSeverity(s, attrs); sev != "INFO" {
		logEntry["severity"] = sev
		if msg != "" {
			logEntry["error"] = msg
		}
	}
	if b, err := json.Marshal(logEntry); err == nil {
		fmt.Fprintln(spanLogWriter, string(b))
	}
}

// spanLogWriter receives the span log lines (stdout, which Cloud Run ships to Cloud Logging; tests swap it).
var spanLogWriter io.Writer = os.Stdout

// maxSpanErrorLen caps the error text in a span log line. Error messages can carry a caller's question id (e.g. the
// serving image's "question 'move_1': ...") but never request content.
const maxSpanErrorLen = 300

// spanSeverity is the log severity of a finished span and its error text: WARNING for a 4xx (the caller's request),
// ERROR for a 5xx or an error span without a status code, INFO otherwise. Before, every span logged at INFO with no
// reason, so a refused request left only its status code.
func spanSeverity(s sdktrace.ReadOnlySpan, attrs map[string]interface{}) (string, string) {
	status := 0
	if v, ok := attrs["http.status_code"]; ok {
		if n, ok := v.(int64); ok {
			status = int(n)
		}
	}
	msg := ""
	if s.Status().Code == codes.Error {
		msg = s.Status().Description
		if r := []rune(msg); len(r) > maxSpanErrorLen {
			msg = string(r[:maxSpanErrorLen]) + "..."
		}
	}
	switch {
	case status >= 500:
		return "ERROR", msg
	case status >= 400:
		return "WARNING", msg
	case s.Status().Code == codes.Error:
		return "ERROR", msg
	}
	return "INFO", ""
}

func (p *gatewaySpanProcessor) Shutdown(ctx context.Context) error   { return nil }
func (p *gatewaySpanProcessor) ForceFlush(ctx context.Context) error { return nil }

// initGatewayTracer initializes the OpenTelemetry TracerProvider with Google Cloud Trace export
// (when on GCP) and the in-memory + Cloud Logging span processor.
func initGatewayTracer(ctx context.Context) func(context.Context) error {
	gcpProjectID = detectGCPProjectID()

	res, _ := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName("dgemma-gateway"),
			semconv.ServiceVersion("1.0.0"),
			attribute.String("gcp.project_id", gcpProjectID),
		),
	)

	opts := []sdktrace.TracerProviderOption{
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(&gatewaySpanProcessor{projectID: gcpProjectID}),
	}

	if gcpProjectID != "" {
		if exporter, err := cloudtrace.New(cloudtrace.WithProjectID(gcpProjectID)); err == nil {
			opts = append(opts, sdktrace.WithBatcher(exporter))
		}
	}

	tp := sdktrace.NewTracerProvider(opts...)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))
	return tp.Shutdown
}

func gatewayTracer() trace.Tracer {
	return otel.Tracer(gatewayTracerName)
}

// extractTraceContextFromRequest extracts W3C traceparent or Google Cloud Run X-Cloud-Trace-Context
// so all gateway and dgemma model spans attach to the incoming Cloud Run request trace.
func extractTraceContextFromRequest(r *http.Request) context.Context {
	ctx := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
	if trace.SpanContextFromContext(ctx).IsValid() {
		return ctx
	}
	// Fallback: parse Google Cloud Run X-Cloud-Trace-Context: TRACE_ID/SPAN_ID_DEC;o=1
	xctc := strings.TrimSpace(r.Header.Get("X-Cloud-Trace-Context"))
	if xctc == "" {
		return ctx
	}
	slashIdx := strings.Index(xctc, "/")
	if slashIdx != 32 {
		return ctx
	}
	traceHex := xctc[:32]
	rest := xctc[33:]
	semiIdx := strings.Index(rest, ";")
	spanDecStr := rest
	if semiIdx >= 0 {
		spanDecStr = rest[:semiIdx]
	}
	tID, err := trace.TraceIDFromHex(traceHex)
	if err != nil {
		return ctx
	}
	var sID trace.SpanID
	if spanUint, err := strconv.ParseUint(spanDecStr, 10, 64); err == nil && spanUint > 0 {
		binary.BigEndian.PutUint64(sID[:], spanUint)
	} else {
		_, _ = hex.Decode(sID[:], []byte("0000000000000001"))
	}
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    tID,
		SpanID:     sID,
		TraceFlags: trace.FlagsSampled,
		Remote:     true,
	})
	return trace.ContextWithRemoteSpanContext(ctx, sc)
}

// injectTraceContextToRequest propagates W3C traceparent and X-Cloud-Trace-Context to upstream dgemma.
func injectTraceContextToRequest(ctx context.Context, req *http.Request) {
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))
	sc := trace.SpanContextFromContext(ctx)
	if sc.IsValid() {
		spanIDBytes := sc.SpanID()
		spanDec := binary.BigEndian.Uint64(spanIDBytes[:])
		req.Header.Set("X-Cloud-Trace-Context", fmt.Sprintf("%s/%d;o=1", sc.TraceID().String(), spanDec))
	}
}

// getTraceSpansByTraceID returns all recorded spans for a specific TraceID in top-down hierarchical
// and chronological waterfall order, enriched with Depth and OffsetMs relative to the trace root.
func getTraceSpansByTraceID(traceID string) []TraceSpanRecord {
	if traceID == "" {
		return nil
	}
	traceRingMu.Lock()
	var raw []TraceSpanRecord
	for _, s := range traceRing {
		if s.TraceID == traceID {
			raw = append(raw, s)
		}
	}
	traceRingMu.Unlock()

	if len(raw) == 0 {
		return nil
	}

	parsedStart := make(map[string]time.Time, len(raw))
	byID := make(map[string]TraceSpanRecord, len(raw))
	var minStart time.Time
	for i, s := range raw {
		t, err := time.Parse(time.RFC3339Nano, s.StartTime)
		if err != nil {
			t = time.Now()
		}
		parsedStart[s.SpanID] = t
		if i == 0 || t.Before(minStart) {
			minStart = t
		}
		byID[s.SpanID] = s
	}

	childrenMap := make(map[string][]TraceSpanRecord)
	var roots []TraceSpanRecord
	for _, s := range raw {
		if s.ParentSpanID != "" {
			if _, exists := byID[s.ParentSpanID]; exists {
				childrenMap[s.ParentSpanID] = append(childrenMap[s.ParentSpanID], s)
				continue
			}
		}
		roots = append(roots, s)
	}

	sortSpans := func(slice []TraceSpanRecord) {
		sort.SliceStable(slice, func(i, j int) bool {
			ti := parsedStart[slice[i].SpanID]
			tj := parsedStart[slice[j].SpanID]
			if !ti.Equal(tj) {
				return ti.Before(tj)
			}
			return slice[i].DurationMs > slice[j].DurationMs
		})
	}

	sortSpans(roots)
	for k := range childrenMap {
		sortSpans(childrenMap[k])
	}

	var ordered []TraceSpanRecord
	var walk func(sp TraceSpanRecord, depth int)
	walk = func(sp TraceSpanRecord, depth int) {
		sp.Depth = depth
		if st, ok := parsedStart[sp.SpanID]; ok && !minStart.IsZero() {
			off := float64(st.Sub(minStart).Microseconds()) / 1000.0
			if off < 0 {
				off = 0
			}
			sp.OffsetMs = off
		}
		ordered = append(ordered, sp)
		for _, child := range childrenMap[sp.SpanID] {
			walk(child, depth+1)
		}
	}

	for _, r := range roots {
		walk(r, 0)
	}
	return ordered
}

// getRecentTraces returns the most recent N spans from the ring buffer.
func getRecentTraces(limit int) []TraceSpanRecord {
	if limit <= 0 || limit > maxTraceItems {
		limit = 50
	}
	traceRingMu.Lock()
	defer traceRingMu.Unlock()
	if len(traceRing) <= limit {
		cp := make([]TraceSpanRecord, len(traceRing))
		copy(cp, traceRing)
		return cp
	}
	cp := make([]TraceSpanRecord, limit)
	copy(cp, traceRing[len(traceRing)-limit:])
	return cp
}
