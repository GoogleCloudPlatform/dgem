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

func TestLocateSetOfMarksFastPathAndEscalation(t *testing.T) {
	marks := []LocateMarkInput{
		{ID: "M1", Label: "Search input", Box: []float64{5, 10, 12, 70}},
		{ID: "M2", Label: "Checkout button", Box: []float64{80, 60, 92, 95}},
	}

	// 1. Confident Set-of-Marks selection (H_norm < 0.16) returns som_mark directly without calling Gemini.
	geminiCalled := false
	defer stubGemini(t, &[4]float64{81, 61, 91, 94}, nil, &geminiCalled)()
	confidentSoM := func(ctx context.Context, schema, state string, images []string) (*client.StructuredDecisionResponse, error) {
		return &client.StructuredDecisionResponse{Answers: map[string]client.QuestionAnswer{
			"present":          {Type: "noul", Label: "yes", Probabilities: map[string]float64{"yes": 0.99, "no": 0.01}},
			"grid_cell":        {Type: "choice", Choice: "bottom_right", Probabilities: map[string]float64{"bottom_right": 0.95, "middle_center": 0.05}},
			"next_target_mark": {Type: "choice", Choice: "M2", Probabilities: map[string]float64{"M2": 0.99, "M1": 0.005, "none_visible": 0.005}},
			"element_state":    {Type: "choice", Choice: "interactive_ready", Probabilities: map[string]float64{"interactive_ready": 0.97, "disabled_grayed_out": 0.03}},
		}}, nil
	}
	res, err := runGuidedLocate(context.Background(), LocateRequest{
		Image:  tinyPNG,
		Target: "the checkout button",
		Marks:  marks,
	}, confidentSoM)
	if err != nil {
		t.Fatalf("runGuidedLocate confident SoM error: %v", err)
	}
	if geminiCalled {
		t.Errorf("geminiCalled = true, want false on confident som_mark fast path")
	}
	if res.Path != "som_mark" || !res.Present || res.Box == nil || res.Box.YMin != 800 || res.Box.XMax != 950 {
		t.Fatalf("got path=%q present=%v box=%+v, want path=\"som_mark\" present=true box=[800,600,920,950]", res.Path, res.Present, res.Box)
	}
	if res.SoM == nil || res.SoM.SelectedMark != "M2" || res.SoM.MatchedMark != "M2" || res.SoM.ElementState != "interactive_ready" {
		t.Fatalf("got SoM=%+v, want SelectedMark=M2 MatchedMark=M2 ElementState=interactive_ready", res.SoM)
	}

	// 2. Hesitant Set-of-Marks selection (H_norm >= 0.16) escalates to Gemini with cell-filtered mark hints and snaps MatchedMark.
	gotHint := ""
	defer stubGemini(t, &[4]float64{81, 61, 91, 94}, &gotHint, &geminiCalled)()
	hesitantSoM := func(ctx context.Context, schema, state string, images []string) (*client.StructuredDecisionResponse, error) {
		return &client.StructuredDecisionResponse{Answers: map[string]client.QuestionAnswer{
			"present":          {Type: "noul", Label: "yes", Probabilities: map[string]float64{"yes": 0.90, "no": 0.10}},
			"grid_cell":        {Type: "choice", Choice: "bottom_right", Probabilities: map[string]float64{"bottom_right": 0.85, "top_left": 0.15}},
			"next_target_mark": {Type: "choice", Choice: "M2", Probabilities: map[string]float64{"M2": 0.50, "M1": 0.30, "none_visible": 0.20}},
			"element_state":    {Type: "choice", Choice: "interactive_ready", Probabilities: map[string]float64{"interactive_ready": 0.80, "requires_scroll": 0.20}},
		}}, nil
	}
	resEsc, err := runGuidedLocate(context.Background(), LocateRequest{
		Image:  tinyPNG,
		Target: "the checkout button",
		Marks:  marks,
	}, hesitantSoM)
	if err != nil {
		t.Fatalf("runGuidedLocate hesitant SoM error: %v", err)
	}
	if !geminiCalled || resEsc.Path != "gemini" {
		t.Fatalf("got path=%q geminiCalled=%v, want path=\"gemini\" geminiCalled=true", resEsc.Path, geminiCalled)
	}
	if !strings.Contains(gotHint, "M2 (Checkout button)") || strings.Contains(gotHint, "M1 (Search input)") {
		t.Errorf("got hint=%q, want bottom_right filtered mark M2 without top_left M1", gotHint)
	}
	if resEsc.SoM == nil || resEsc.SoM.MatchedMark != "M2" {
		t.Fatalf("got SoM=%+v, want MatchedMark=M2", resEsc.SoM)
	}
}

func TestLocateGatewayRejectsInvalidMarks(t *testing.T) {
	parsed, err := loadLocateMarksArg(`[{"id":"M1","label":"CTA","box":[10,20,30,40]}]`)
	if err != nil || len(parsed) != 1 || parsed[0].ID != "M1" {
		t.Fatalf("loadLocateMarksArg = (%+v, %v), want 1 mark M1", parsed, err)
	}

	t.Setenv("DGEM_VERTEX_URL", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/123:rawPredict")
	badBodies := []string{
		`{"image":"` + tinyPNG + `","target":"btn","marks":[{"id":"M1","box":[10,20]}]}`,
		`{"image":"` + tinyPNG + `","target":"btn","marks":[{"id":"M1","box":[50,20,10,40]}]}`,
		`{"image":"` + tinyPNG + `","target":"btn","marks":[{"id":"none_visible","box":[10,20,30,40]}]}`,
	}
	for _, raw := range badBodies {
		w := httptest.NewRecorder()
		handleLocate(w, httptest.NewRequest(http.MethodPost, "/api/locate", strings.NewReader(raw)))
		if w.Code != http.StatusBadRequest {
			t.Errorf("body %s: got HTTP %d, want 400 (body=%s)", raw, w.Code, w.Body.String())
		}
	}
}
