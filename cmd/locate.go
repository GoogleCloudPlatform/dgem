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

// Guided locate: dgem first, Gemini only when needed.
//
//  1. dgem answers "is the target present?" and "which 3x3 cell holds it?" in one pass (~0.2 s).
//  2. Opt-in (skip_h > 0): if dgem says absent with normalized entropy below skip_h, return "absent" without calling
//     Gemini. It saves calls on photos at a small cost in missed positives, but misses small defects on
//     some image types, so it is off by default.
//  3. Gemini 3.x at LOW thinking returns the box (similar accuracy to default thinking, and faster).
//     Opt-in (hint): dgem's cell as a "may be wrong" hint; neutral on average in our tests.
//  4. Optionally, a mask from a SAM service at DGEM_SAM_URL prompted with that box (tighter masks than Gemini
//     polygons in our tests). Without DGEM_SAM_URL, masks are reported as unavailable.
//
// Used by `dgem locate` (CLI), POST /api/locate (gateway), the MCP `locate_object` tool and the Studio box canvas.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"google.golang.org/genai"
)

const (
	// recommendedLocateSkipH is the recommended skip threshold, used when the caller opts in.
	recommendedLocateSkipH = 0.16
	defaultLocateThinking  = "low"
)

// LocateRequest is the input shared by every surface.
type LocateRequest struct {
	Image       string  `json:"image" jsonschema:"Image URL or base64 data URI (the CLI and stdio MCP also accept a local path)."`
	Target      string  `json:"target" jsonschema:"Description of the object or element to locate (e.g. 'the checkout button', 'the table')."`
	Mask        bool    `json:"mask,omitempty" jsonschema:"Also return a mask from the SAM service at DGEM_SAM_URL (prompted with the box)."`
	GeminiModel string  `json:"gemini_model,omitempty" jsonschema:"Gemini 3.x model for the box (default gemini-3.8-flash; gemini-3.5-flash-lite also supported)."`
	Thinking    string  `json:"thinking,omitempty" jsonschema:"Gemini thinking level: low (default), medium, high or default."`
	SkipH       float64 `json:"skip_h,omitempty" jsonschema:"Opt-in: skip Gemini when dgem says absent with normalized entropy below this (0.16 recommended; 0 = never skip). Measured safe on photos and mobile UI, not on defect images."`
	Hint        bool    `json:"hint,omitempty" jsonschema:"Opt-in: pass dgem's 3x3 cell to Gemini as a 'may be wrong' hint (neutral on average)."`
	Backend     string  `json:"backend,omitempty" jsonschema:"dgem backend: vertex_first (default), vertex or cloudrun."`
	VertexURL   string  `json:"vertex_url,omitempty" jsonschema:"Optional Vertex AI endpoint ID or /invoke/* URL override."`
}

// LocateResult reports which path ran and the time spent in each stage.
type LocateResult struct {
	Target        string        `json:"target"`
	Present       bool          `json:"present"`
	Path          string        `json:"path"` // skipped_absent | gemini | gemini_absent
	Box           *BBoxCoords   `json:"box_1000,omitempty"`
	BoxPct        *[4]float64   `json:"box_pct,omitempty"` // [ymin, xmin, ymax, xmax] percent
	Mask          *LocateMask   `json:"mask,omitempty"`
	MaskNote      string        `json:"mask_note,omitempty"`
	Dgem          LocateDgem    `json:"dgem"`
	Gemini        *LocateGemini `json:"gemini,omitempty"`
	SamMs         float64       `json:"sam_ms,omitempty"`
	TotalMs       float64       `json:"total_ms"`
	BackendTarget string        `json:"backend_target,omitempty"`
}

type LocateDgem struct {
	Present   string  `json:"present"`
	PresentH  float64 `json:"present_h_norm"`
	GridCell  string  `json:"grid_cell"`
	GridCellH float64 `json:"grid_cell_h_norm"`
	Ms        float64 `json:"ms"`
}

type LocateGemini struct {
	Model         string  `json:"model"`
	Thinking      string  `json:"thinking"`
	Hint          string  `json:"hint,omitempty"`
	Ms            float64 `json:"ms"`
	ThoughtTokens int32   `json:"thought_tokens"`
}

// LocateMask is what the SAM service returns.
type LocateMask struct {
	PNGBase64  string       `json:"png_base64,omitempty"`
	PolygonPct [][2]float64 `json:"polygon_pct,omitempty"`
	Score      float64      `json:"score,omitempty"`
}

// locateDecideFunc runs one dgem decision (CLI: client.Decide; gateway/MCP: executeDecideWithWarmup).
type locateDecideFunc func(ctx context.Context, schema, state string, images []string) (*client.StructuredDecisionResponse, error)

// locateGeminiFunc returns a box in percent (nil when Gemini finds nothing). A package variable so tests can stub it.
var locateGeminiFunc = geminiLocateBox

// locateSAMFunc calls the SAM service. A package variable so tests can stub it.
var locateSAMFunc = samLocateMask

func locateSchema() string {
	cells := []string{"top_left", "top_center", "top_right", "middle_left", "middle_center", "middle_right",
		"bottom_left", "bottom_center", "bottom_right"}
	opts := make([]map[string]string, len(cells))
	for i, c := range cells {
		opts[i] = map[string]string{"name": c, "description": strings.ReplaceAll(c, "_", " ") + " of a 3x3 grid"}
	}
	s := map[string]interface{}{
		"instructions": "Look at the attached image and answer every question about the target described in the state.",
		"questions": []interface{}{
			map[string]interface{}{"id": "present", "type": "boolean", "instructions": "Is the described target visible in the image?"},
			map[string]interface{}{"id": "grid_cell", "type": "choice",
				"instructions": "Divide the image into a 3x3 grid. Which cell contains the centre of the target?", "options": opts},
		},
		"samples": 1,
	}
	b, _ := json.Marshal(s)
	return string(b)
}

func normalizedAnswer(qa client.QuestionAnswer, k int) (string, float64) {
	label, _, h, _ := dgemVisionAnswer(qa, k)
	return label, h
}

// runGuidedLocate executes the pipeline. decide is the dgem call for the surface.
func runGuidedLocate(ctx context.Context, req LocateRequest, decide locateDecideFunc) (*LocateResult, error) {
	start := time.Now()
	target := strings.TrimSpace(req.Target)
	if target == "" {
		return nil, fmt.Errorf("'target' is required")
	}
	if strings.TrimSpace(req.Image) == "" {
		return nil, fmt.Errorf("'image' is required")
	}
	skipH := req.SkipH
	thinking := strings.ToLower(strings.TrimSpace(req.Thinking))
	if thinking == "" {
		thinking = defaultLocateThinking
	}
	switch thinking {
	case "low", "medium", "high", "default":
	default:
		return nil, fmt.Errorf("thinking must be low, medium, high or default")
	}
	res := &LocateResult{Target: target}

	ctx, root := gatewayTracer().Start(ctx, "dgem.locate")
	defer root.End()

	// 1. dgem pass.
	dctx, dspan := gatewayTracer().Start(ctx, "dgem.locate.dgem")
	t0 := time.Now()
	state, _ := json.Marshal(map[string]string{"target": target})
	resp, err := decide(dctx, locateSchema(), string(state), []string{req.Image})
	res.Dgem.Ms = float64(time.Since(t0).Microseconds()) / 1000
	if err != nil {
		dspan.SetStatus(codes.Error, err.Error())
		dspan.End()
		root.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("dgem pass: %w", err)
	}
	res.Dgem.Present, res.Dgem.PresentH = normalizedAnswer(resp.Answers["present"], 2)
	res.Dgem.GridCell, res.Dgem.GridCellH = normalizedAnswer(resp.Answers["grid_cell"], 9)
	dspan.SetAttributes(attribute.String("dgem.locate.present", res.Dgem.Present),
		attribute.Float64("dgem.locate.present_h_norm", res.Dgem.PresentH),
		attribute.String("dgem.locate.grid_cell", res.Dgem.GridCell))
	dspan.End()

	// 2. Skip when dgem is confident the target is absent.
	if skipH > 0 && res.Dgem.Present == "no" && res.Dgem.PresentH < skipH {
		res.Path = "skipped_absent"
		res.TotalMs = float64(time.Since(start).Microseconds()) / 1000
		root.SetAttributes(attribute.String("dgem.locate.path", res.Path))
		return res, nil
	}

	// 3. Gemini box, hinted with dgem's cell.
	model := SanitizeCascadeModel(req.GeminiModel)
	g := &LocateGemini{Model: model, Thinking: thinking}
	if req.Hint && res.Dgem.GridCell != "" {
		g.Hint = fmt.Sprintf("Hint from a fast first-pass model (it may be wrong): the target is probably in the %s part of the image (3x3 grid).",
			strings.ReplaceAll(res.Dgem.GridCell, "_", " "))
	}
	gctx, gspan := gatewayTracer().Start(ctx, "dgem.locate.gemini")
	t1 := time.Now()
	box, thoughts, err := locateGeminiFunc(gctx, model, thinking, req.Image, target, g.Hint)
	g.Ms = float64(time.Since(t1).Microseconds()) / 1000
	g.ThoughtTokens = thoughts
	gspan.SetAttributes(attribute.String("dgem.locate.gemini_model", model), attribute.String("dgem.locate.thinking", thinking),
		attribute.Bool("dgem.locate.hinted", g.Hint != ""), attribute.Int("dgem.locate.thought_tokens", int(thoughts)))
	if err != nil {
		gspan.SetStatus(codes.Error, err.Error())
		gspan.End()
		root.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("gemini: %w", err)
	}
	gspan.End()
	res.Gemini = g
	if box == nil {
		res.Path = "gemini_absent"
		res.TotalMs = float64(time.Since(start).Microseconds()) / 1000
		root.SetAttributes(attribute.String("dgem.locate.path", res.Path))
		return res, nil
	}
	res.Path, res.Present, res.BoxPct = "gemini", true, box
	res.Box = &BBoxCoords{YMin: box[0] * 10, XMin: box[1] * 10, YMax: box[2] * 10, XMax: box[3] * 10}

	// 4. Optional SAM mask.
	if req.Mask {
		samURL := strings.TrimSpace(os.Getenv("DGEM_SAM_URL"))
		if samURL == "" {
			res.MaskNote = "mask unavailable: no SAM service configured (DGEM_SAM_URL)"
		} else {
			sctx, sspan := gatewayTracer().Start(ctx, "dgem.locate.sam")
			t2 := time.Now()
			m, err := locateSAMFunc(sctx, samURL, req.Image, *box)
			res.SamMs = float64(time.Since(t2).Microseconds()) / 1000
			if err != nil {
				sspan.SetStatus(codes.Error, err.Error())
				res.MaskNote = "mask unavailable: " + err.Error()
			} else {
				res.Mask = m
			}
			sspan.End()
		}
	}
	res.TotalMs = float64(time.Since(start).Microseconds()) / 1000
	root.SetAttributes(attribute.String("dgem.locate.path", res.Path))
	return res, nil
}

// geminiLocateBox asks Gemini for the target's box. The image may be a local path, data: URI or http(s) URL.
func geminiLocateBox(ctx context.Context, model, thinking, image, target, hint string) (*[4]float64, int32, error) {
	gc, _, err := getSharedGenaiClient(ctx)
	if err != nil {
		return nil, 0, err
	}
	parts, err := cascadeImageParts(ctx, []string{image})
	if err != nil {
		return nil, 0, err
	}
	parts = append(parts, genai.NewPartFromText(guidedBoxPrompt(target, hint)))
	cfg := &genai.GenerateContentConfig{ResponseMIMEType: "application/json", ResponseSchema: guidedBoxSchema()}
	if thinking != "" && thinking != "default" {
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevel(strings.ToUpper(thinking))}
	}
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		cctx, cancel := context.WithTimeout(ctx, 90*time.Second)
		resp, err := gc.Models.GenerateContent(cctx, model, []*genai.Content{{Role: "user", Parts: parts}}, cfg)
		cancel()
		if err == nil {
			var r guidedBoxReply
			if err = json.Unmarshal([]byte(strings.TrimSpace(resp.Text())), &r); err == nil {
				var thoughts int32
				if resp.UsageMetadata != nil {
					thoughts = resp.UsageMetadata.ThoughtsTokenCount
				}
				return clampBoxPct(boxFromReply(r)), thoughts, nil
			}
		}
		lastErr = err
		time.Sleep(time.Duration(1+2*attempt) * time.Second)
	}
	return nil, 0, lastErr
}

func clampBoxPct(b *[4]float64) *[4]float64 {
	if b == nil {
		return nil
	}
	for i := range b {
		b[i] = math.Max(0, math.Min(100, b[i]))
	}
	if b[2] <= b[0] || b[3] <= b[1] {
		return nil
	}
	return b
}

// samLocateMask calls a SAM service: POST <url> {"image": <data URI or URL>, "box_pct": [ymin, xmin, ymax, xmax]} ->
// {"png_base64": "...", "polygon_pct": [[y, x], ...], "score": 0.97}. Local paths are sent as data URIs.
func samLocateMask(ctx context.Context, samURL, image string, box [4]float64) (*LocateMask, error) {
	img := image
	if !strings.HasPrefix(img, "data:") && !strings.HasPrefix(img, "http://") && !strings.HasPrefix(img, "https://") {
		if !allowLocalImagePaths {
			return nil, fmt.Errorf("image must be a data: URI or an http(s) URL on this server")
		}
		b, err := os.ReadFile(img)
		if err != nil {
			return nil, err
		}
		img = "data:" + http.DetectContentType(b) + ";base64," + base64.StdEncoding.EncodeToString(b)
	}
	body, _ := json.Marshal(map[string]interface{}{"image": img, "box_pct": box})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, samURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok := strings.TrimSpace(os.Getenv("DGEM_SAM_TOKEN")); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	hr, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer hr.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(hr.Body, 32<<20))
	if hr.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("SAM service HTTP %d", hr.StatusCode)
	}
	var m LocateMask
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("SAM service reply: %w", err)
	}
	return &m, nil
}

// locateDecideVia is the dgem step for the gateway and MCP: executeDecideWithWarmup on the resolved backend, recording
// readout latency per backend (never MarkGPUWarm for Vertex).
func locateDecideVia(backendTarget, targetURL string) locateDecideFunc {
	return func(ctx context.Context, schema, state string, images []string) (*client.StructuredDecisionResponse, error) {
		t0 := time.Now()
		resp, _, _, err := executeDecideWithWarmup(ctx, schema, state, images, targetURL)
		if err == nil {
			recordBackendReadoutLatency(backendTarget, time.Since(t0).Milliseconds())
		}
		return resp, err
	}
}

// handleLocate serves POST /api/locate.
func handleLocate(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	writeErr := func(status int, msg string) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
	}
	if r.Method != http.MethodPost {
		writeErr(http.StatusMethodNotAllowed, "use POST")
		return
	}
	ctx, span := gatewayTracer().Start(extractTraceContextFromRequest(r), "dgem.gateway.locate")
	defer span.End()
	if id := span.SpanContext().TraceID().String(); id != "" {
		w.Header().Set("X-Dgem-Trace-Id", id)
	}
	var req struct {
		LocateRequest
		ImageURL string `json:"image_url,omitempty"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 40<<20)).Decode(&req); err != nil {
		writeErr(http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if req.Image == "" {
		req.Image = req.ImageURL
	}
	if prepared, err := prepareRequestImages(ctx, []string{req.Image}); err != nil {
		writeErr(imageErrorStatus(err), err.Error())
		return
	} else {
		req.Image = prepared[0] // the guided-locate dgem read, Gemini and SAM all get the image bytes, not the URL
	}
	backendTarget, targetURL, err := resolveBackendTarget(r, req.Backend, req.VertexURL)
	if err != nil {
		writeErr(http.StatusBadRequest, err.Error())
		return
	}
	if backendTarget == "local" {
		writeErr(http.StatusBadRequest, "locate needs a vision backend: vertex_first, vertex or cloudrun")
		return
	}
	span.SetAttributes(attribute.String("dgem.backend", backendTarget))
	res, err := runGuidedLocate(ctx, req.LocateRequest, locateDecideVia(backendTarget, targetURL))
	if err != nil {
		status := http.StatusBadGateway
		if strings.Contains(err.Error(), "is required") || strings.Contains(err.Error(), "thinking must be") {
			status = http.StatusBadRequest
		}
		writeErr(status, err.Error())
		return
	}
	res.BackendTarget = backendTarget
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-DGem-Backend-Used", backendTarget)
	_ = json.NewEncoder(w).Encode(res)
}
