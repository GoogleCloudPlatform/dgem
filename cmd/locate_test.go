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
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

const tinyPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

func fakeLocateDecide(present string, pPresent float64, cell string) locateDecideFunc {
	return func(ctx context.Context, schema, state string, images []string) (*client.StructuredDecisionResponse, error) {
		other := "no"
		if present == "no" {
			other = "yes"
		}
		return &client.StructuredDecisionResponse{Answers: map[string]client.QuestionAnswer{
			"present":   {Type: "noul", Label: present, Probabilities: map[string]float64{present: pPresent, other: 1 - pPresent}},
			"grid_cell": {Type: "choice", Choice: cell, Probabilities: map[string]float64{cell: 0.9, "middle_center": 0.1}},
		}}, nil
	}
}

func stubGemini(t *testing.T, box *[4]float64, gotHint *string, called *bool) func() {
	old := locateGeminiFunc
	locateGeminiFunc = func(ctx context.Context, model, thinking, image, target, hint string) (*[4]float64, int32, error) {
		if called != nil {
			*called = true
		}
		if gotHint != nil {
			*gotHint = hint
		}
		if thinking != "low" {
			t.Errorf("default thinking = %q, want low", thinking)
		}
		return box, 42, nil
	}
	return func() { locateGeminiFunc = old }
}

func TestLocateSkipsGeminiWhenConfidentlyAbsent(t *testing.T) {
	called := false
	defer stubGemini(t, &[4]float64{1, 2, 3, 4}, nil, &called)()
	res, err := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "the button", SkipH: recommendedLocateSkipH}, fakeLocateDecide("no", 0.999, "top_left"))
	if err != nil {
		t.Fatal(err)
	}
	if called || res.Path != "skipped_absent" || res.Present || res.Gemini != nil {
		t.Errorf("want skipped_absent without Gemini, got path=%s called=%v", res.Path, called)
	}
}

func TestLocateHintedGeminiWhenUnsureOrPresent(t *testing.T) {
	hint := ""
	defer stubGemini(t, &[4]float64{10, 20, 30, 40}, &hint, nil)()
	// Unsure "no" (p=0.6) must not skip.
	res, err := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "the button", SkipH: recommendedLocateSkipH, Hint: true}, fakeLocateDecide("no", 0.6, "bottom_right"))
	if err != nil {
		t.Fatal(err)
	}
	if res.Path != "gemini" || res.Box == nil || res.Box.XMax != 400 || res.Gemini.ThoughtTokens != 42 {
		t.Fatalf("unexpected result %+v", res)
	}
	if !strings.Contains(hint, "bottom right") || !strings.Contains(hint, "may be wrong") {
		t.Errorf("hint = %q", hint)
	}
	res, _ = runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "x"}, fakeLocateDecide("yes", 0.99, "top_left"))
	if hint != "" || res.Gemini.Hint != "" {
		t.Errorf("hint is opt-in but was sent: %q", hint)
	}
}

func TestLocateDoesNotSkipByDefault(t *testing.T) {
	called := false
	defer stubGemini(t, nil, nil, &called)()
	res, err := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "x"}, fakeLocateDecide("no", 0.999, "top_left"))
	if err != nil {
		t.Fatal(err)
	}
	if !called || res.Path != "gemini_absent" {
		t.Errorf("skipping is opt-in; path=%s called=%v", res.Path, called)
	}
}

func TestLocateMaskViaSAMService(t *testing.T) {
	defer stubGemini(t, &[4]float64{10, 20, 30, 40}, nil, nil)()
	var got map[string]interface{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_ = json.NewEncoder(w).Encode(LocateMask{PolygonPct: [][2]float64{{10, 20}, {10, 40}, {30, 40}}, Score: 0.9})
	}))
	defer srv.Close()

	t.Setenv("DGEM_SAM_URL", "")
	res, _ := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "x", Mask: true}, fakeLocateDecide("yes", 0.99, "top_left"))
	if res.Mask != nil || !strings.Contains(res.MaskNote, "DGEM_SAM_URL") {
		t.Errorf("without SAM URL: mask=%v note=%q", res.Mask, res.MaskNote)
	}
	t.Setenv("DGEM_SAM_URL", srv.URL)
	res, err := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "x", Mask: true}, fakeLocateDecide("yes", 0.99, "top_left"))
	if err != nil {
		t.Fatal(err)
	}
	if res.Mask == nil || len(res.Mask.PolygonPct) != 3 || got["image"] != tinyPNG {
		t.Errorf("SAM mask %+v, request %v", res.Mask, got)
	}
	if bp, _ := got["box_pct"].([]interface{}); len(bp) != 4 || bp[3].(float64) != 40 {
		t.Errorf("SAM box_pct = %v", got["box_pct"])
	}
}

func TestLocateValidatesInput(t *testing.T) {
	if _, err := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG}, fakeLocateDecide("yes", 1, "top_left")); err == nil {
		t.Error("missing target accepted")
	}
	if _, err := runGuidedLocate(context.Background(), LocateRequest{Image: tinyPNG, Target: "x", Thinking: "minimal"}, fakeLocateDecide("yes", 1, "top_left")); err == nil {
		t.Error("unsupported thinking level accepted")
	}
}

func TestLocateGatewayRejectsLocalAndPrivateImages(t *testing.T) {
	old := allowLocalImagePaths
	allowLocalImagePaths = false
	defer func() { allowLocalImagePaths = old }()
	for _, img := range []string{"/etc/passwd", "http://127.0.0.1/x.png", "http://169.254.169.254/computeMetadata/v1/"} {
		body, _ := json.Marshal(map[string]string{"image": img, "target": "x"})
		w := httptest.NewRecorder()
		handleLocate(w, httptest.NewRequest(http.MethodPost, "/api/locate", strings.NewReader(string(body))))
		if w.Code != http.StatusBadRequest {
			t.Errorf("image %q: status %d, want 400", img, w.Code)
		}
	}
	if _, err := samLocateMask(context.Background(), "http://127.0.0.1:9", "/etc/passwd", [4]float64{0, 0, 10, 10}); err == nil {
		t.Error("samLocateMask accepted local file path when allowLocalImagePaths is false")
	}
	w := httptest.NewRecorder()
	handleLocate(w, httptest.NewRequest(http.MethodGet, "/api/locate", nil))
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET status %d", w.Code)
	}
}

func TestClampBoxPct(t *testing.T) {
	if b := clampBoxPct(&[4]float64{-5, 10, 105, 20}); b == nil || b[0] != 0 || b[2] != 100 {
		t.Errorf("clamp = %v", b)
	}
	if clampBoxPct(&[4]float64{50, 50, 40, 60}) != nil {
		t.Error("inverted box accepted")
	}
}
