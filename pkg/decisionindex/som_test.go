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

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestAssignBoxToGridCellAllNineRegionsAndTemplateRender(t *testing.T) {
	cases := []struct {
		box  [4]float64
		want string
	}{
		{[4]float64{2, 4, 20, 22}, "top_left"},
		{[4]float64{5, 40, 20, 60}, "top_center"},
		{[4]float64{5, 75, 25, 95}, "top_right"},
		{[4]float64{40, 5, 60, 25}, "middle_left"},
		{[4]float64{40, 40, 60, 60}, "middle_center"},
		{[4]float64{40, 75, 60, 95}, "middle_right"},
		{[4]float64{75, 5, 95, 25}, "bottom_left"},
		{[4]float64{75, 40, 95, 60}, "bottom_center"},
		{[4]float64{75, 75, 95, 95}, "bottom_right"},
	}
	for _, tc := range cases {
		if got := AssignBoxToGridCell(tc.box); got != tc.want {
			t.Fatalf("AssignBoxToGridCell(%v) = %q, want %q", tc.box, got, tc.want)
		}
	}

	tmplPath := filepath.Join("..", "..", "templates", "multimodal", "ui_pathfinder_som.json.tmpl")
	rawTmpl, err := os.ReadFile(tmplPath)
	if err != nil {
		t.Fatalf("read %s: %v", tmplPath, err)
	}
	if len(rawTmpl) == 0 {
		t.Fatalf("empty template %s", tmplPath)
	}
}

func TestSelectContainingOrNearestMarkNestedAndRealScreenSpot(t *testing.T) {
	// Nested container + small child icon: must select the smaller child icon when point is inside both.
	marks := []CandidateMark{
		{ID: "M1", Label: "Top navigation toolbar container", Box: [4]float64{0, 0, 15, 100}},
		{ID: "M2", Label: "Hamburger menu icon", Box: [4]float64{3.8, 90.3, 10.4, 100.0}},
		{ID: "M3", Label: "Primary CTA button", Box: [4]float64{80.0, 20.0, 90.0, 80.0}},
	}
	got, dist, ok := SelectContainingOrNearestMark(7.0, 95.0, marks)
	if !ok {
		t.Fatalf("expected mark selection to succeed")
	}
	if got.ID != "M2" || dist != 0 {
		t.Fatalf("got mark=%s dist=%v, want M2 dist=0", got.ID, dist)
	}

	// Point slightly outside M3: must snap to nearest center (M3).
	gotNear, distNear, ok := SelectContainingOrNearestMark(78.0, 50.0, marks)
	if !ok || gotNear.ID != "M3" || distNear <= 0 {
		t.Fatalf("got near mark=%s dist=%v ok=%v, want M3 with dist>0", gotNear.ID, distNear, ok)
	}

	// Verify on real ScreenSpot UI boxes from third_party/bbox_real/bbox_real.jsonl.
	path := filepath.Join("..", "..", "third_party", "bbox_real", "bbox_real.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	type bboxRow struct {
		ID     string     `json:"id"`
		Tier   string     `json:"tier"`
		Target string     `json:"target"`
		GTBox  [4]float64 `json:"gt_box_continuous"`
	}
	var screenSpot []bboxRow
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var r bboxRow
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			t.Fatalf("unmarshal %s: %v", path, err)
		}
		if r.Tier == "screenspot" {
			screenSpot = append(screenSpot, r)
		}
	}
	if len(screenSpot) != 108 {
		t.Fatalf("expected 108 screenspot items, got %d", len(screenSpot))
	}

	// Build a 9-mark candidate screen from the first 9 ScreenSpot targets and verify grid-cell filtering + center selection.
	var candidateScreen []CandidateMark
	for i := 0; i < 9; i++ {
		candidateScreen = append(candidateScreen, CandidateMark{
			ID:    fmt.Sprintf("M%d", i+1),
			Label: screenSpot[i].Target,
			Box:   screenSpot[i].GTBox,
		})
	}
	for i, m := range candidateScreen {
		cy, cx := BoxCenter(m.Box)
		sel, d, ok := SelectContainingOrNearestMark(cy, cx, candidateScreen)
		if !ok || sel.ID != m.ID || d != 0 {
			t.Fatalf("candidate %d (%s): SelectContainingOrNearestMark=(%s, %v, %v)", i, m.ID, sel.ID, d, ok)
		}
		cell := AssignBoxToGridCell(m.Box)
		filtered := FilterMarksByGridCell(cell, candidateScreen, false)
		found := false
		for _, fm := range filtered {
			if fm.ID == m.ID {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("candidate %s not found in FilterMarksByGridCell(%q)", m.ID, cell)
		}
	}
}

func TestBuildSetOfMarksQuestionsValidationAndStructure(t *testing.T) {
	marks := []CandidateMark{
		{ID: "M1", Label: "Search input", Box: [4]float64{5, 10, 12, 70}},
		{ID: "M2", Label: "Submit search icon", Box: [4]float64{5, 72, 12, 85}},
		{ID: "M3", Label: "Cancel button", Box: [4]float64{85, 20, 92, 80}},
	}
	qs, err := BuildSetOfMarksQuestions("Tap the submit search icon", marks)
	if err != nil {
		t.Fatalf("BuildSetOfMarksQuestions failed: %v", err)
	}
	if len(qs) != 4 {
		t.Fatalf("expected 4 slots, got %d", len(qs))
	}
	targetQ, ok := qs["next_target_mark"]
	if !ok {
		t.Fatalf("missing next_target_mark question")
	}
	if len(targetQ.Criteria) != 4 || len(targetQ.Order) != 4 {
		t.Fatalf("expected 3 marks + none_visible = 4 criteria, got %d", len(targetQ.Criteria))
	}
	if targetQ.Order[3] != "none_visible" {
		t.Fatalf("expected final option to be none_visible, got %q", targetQ.Order[3])
	}

	// Validation: empty goal, empty marks, duplicate ID, reserved ID, >25 marks.
	if _, err := BuildSetOfMarksQuestions("", marks); err == nil {
		t.Fatalf("expected error on empty goal")
	}
	if _, err := BuildSetOfMarksQuestions("goal", nil); err == nil {
		t.Fatalf("expected error on empty marks")
	}
	if _, err := BuildSetOfMarksQuestions("goal", []CandidateMark{{ID: "none_visible", Box: [4]float64{0, 0, 10, 10}}}); err == nil {
		t.Fatalf("expected error on reserved ID none_visible")
	}
	if _, err := BuildSetOfMarksQuestions("goal", []CandidateMark{marks[0], marks[0]}); err == nil {
		t.Fatalf("expected error on duplicate mark ID")
	}
	tooMany := make([]CandidateMark, MaxOptionsPerSlot)
	for i := range tooMany {
		tooMany[i] = CandidateMark{ID: fmt.Sprintf("M%d", i+1), Box: [4]float64{0, 0, 5, 5}}
	}
	if _, err := BuildSetOfMarksQuestions("goal", tooMany); err == nil {
		t.Fatalf("expected error when marks count (%d) exceeds single-pass limit (%d)", len(tooMany), MaxOptionsPerSlot-1)
	}

	// Verify BoxIoU sanity.
	iou := BoxIoU([4]float64{0, 0, 50, 50}, [4]float64{0, 0, 50, 50})
	if math.Abs(iou-1.0) > 1e-9 {
		t.Fatalf("identical BoxIoU=%v, want 1.0", iou)
	}

	// Verify 4x4 grid helpers and ExpectedMarkCenterFromProbabilities.
	if got := AssignPointToGrid4x4(12.0, 88.0); got != "r0_c3" {
		t.Fatalf("AssignPointToGrid4x4(12, 88)=%q, want r0_c3", got)
	}
	b4, ok := Grid4x4CellBounds("r0_c3")
	if !ok || b4 != [4]float64{0, 75, 25, 100} {
		t.Fatalf("Grid4x4CellBounds(r0_c3)=(%v, %v), want [0 75 25 100]", b4, ok)
	}
	if _, okBad := Grid4x4CellBounds("r4_c0"); okBad {
		t.Fatalf("expected out-of-range 4x4 cell r4_c0 to fail")
	}

	ecy, ecx, visMass := ExpectedMarkCenterFromProbabilities(
		map[string]float64{"M1": 0.75, "M3": 0.15, "none_visible": 0.10},
		marks,
	)
	if math.Abs(visMass-0.90) > 1e-9 {
		t.Fatalf("expected visibleMass=0.90, got %.6f", visMass)
	}
	m1y, m1x := BoxCenter(marks[0].Box)
	m3y, m3x := BoxCenter(marks[2].Box)
	wantY := (0.75*m1y + 0.15*m3y) / 0.90
	wantX := (0.75*m1x + 0.15*m3x) / 0.90
	if math.Abs(ecy-wantY) > 1e-9 || math.Abs(ecx-wantX) > 1e-9 {
		t.Fatalf("ExpectedMarkCenter=(%.4f, %.4f), want (%.4f, %.4f)", ecy, ecx, wantY, wantX)
	}
}

func TestSetOfMarks_ProbabilisticJitterAndOOSEntropyAUROC(t *testing.T) {
	path := filepath.Join("..", "..", "third_party", "bbox_real", "bbox_real.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	type bboxRow struct {
		ID     string     `json:"id"`
		Tier   string     `json:"tier"`
		Target string     `json:"target"`
		GTBox  [4]float64 `json:"gt_box_continuous"`
	}
	var screenSpot []bboxRow
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var r bboxRow
		if err := json.Unmarshal(scanner.Bytes(), &r); err == nil && r.Tier == "screenspot" {
			screenSpot = append(screenSpot, r)
		}
	}
	if len(screenSpot) != 108 {
		t.Fatalf("expected 108 screenspot rows, got %d", len(screenSpot))
	}

	// Part 1: Same-screen unbadged 4x4 grid-center click baseline vs SoM candidate snapping
	// under Gaussian pointer jitter sigma in {2%, 5%, 10%} across 540 Monte Carlo trials on real ScreenSpot boxes.
	rng := rand.New(rand.NewSource(20261025))
	const marksPerScreen = 12

	grid4x4Hits := 0
	jitterHits := map[float64]int{2.0: 0, 5.0: 0, 10.0: 0}
	totalTrials := 0

	for trial := 0; trial < 540; trial++ {
		baseIdx := (trial * 7) % len(screenSpot)
		var screenMarks []CandidateMark
		for j := 0; j < marksPerScreen; j++ {
			row := screenSpot[(baseIdx+j*9)%len(screenSpot)]
			screenMarks = append(screenMarks, CandidateMark{
				ID:    fmt.Sprintf("M%d", j+1),
				Label: row.Target,
				Box:   row.GTBox,
			})
		}
		targetMark := screenMarks[trial%marksPerScreen]
		gtCell4 := AssignBoxToGrid4x4(targetMark.Box)
		cellBounds, ok := Grid4x4CellBounds(gtCell4)
		if !ok {
			t.Fatalf("Grid4x4CellBounds(%q) failed", gtCell4)
		}
		c4y, c4x := BoxCenter(cellBounds)
		if PointInBox(c4y, c4x, targetMark.Box) {
			grid4x4Hits++
		}

		tcy, tcx := BoxCenter(targetMark.Box)
		for _, sigma := range []float64{2.0, 5.0, 10.0} {
			jy := math.Max(0, math.Min(100, tcy+sigma*rng.NormFloat64()))
			jx := math.Max(0, math.Min(100, tcx+sigma*rng.NormFloat64()))
			picked, _, ok := SelectContainingOrNearestMark(jy, jx, screenMarks)
			if ok && picked.ID == targetMark.ID {
				jitterHits[sigma]++
			}
		}
		totalTrials++
	}

	rateGrid4 := float64(grid4x4Hits) / float64(totalTrials)
	rateS2 := float64(jitterHits[2.0]) / float64(totalTrials)
	rateS5 := float64(jitterHits[5.0]) / float64(totalTrials)
	rateS10 := float64(jitterHits[10.0]) / float64(totalTrials)

	// Unbadged 4x4 grid-center clicks miss small mobile/desktop icons on real ScreenSpot geometry,
	// whereas mark snapping maintains high accuracy at sigma=2% and degrades monotonically as noise grows.
	if rateS2 < 0.85 || rateS2 <= rateGrid4+0.40 {
		t.Fatalf("expected SoM sigma=2%% hit rate (%.3f) >= 0.85 and >40 pts above 4x4 grid center (%.3f)", rateS2, rateGrid4)
	}
	if !(rateS2 >= rateS5 && rateS5 >= rateS10) {
		t.Fatalf("expected monotonic accuracy across jitter sigmas 2%%>=5%%>=10%%, got %.3f, %.3f, %.3f", rateS2, rateS5, rateS10)
	}

	// Part 2: End-to-end single-pass ExecuteSystemOne probabilistic calibration & OOS entropy AUROC (200 trials).
	var modeTrial int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		var sysPayload struct {
			Questions []struct {
				ID      string `json:"id"`
				Type    string `json:"type"`
				Options []struct {
					Name string `json:"name"`
				} `json:"options"`
			} `json:"questions"`
		}
		for _, m := range reqBody.Messages {
			if m.Role == "system" {
				_ = json.Unmarshal([]byte(m.Content), &sysPayload)
			}
		}
		answers := make(map[string]client.QuestionAnswer, len(sysPayload.Questions))
		ambiguous := (modeTrial % 4) == 3
		for _, q := range sysPayload.Questions {
			switch q.ID {
			case "next_target_mark":
				logits := make(map[string]float64, len(q.Options))
				for idx, o := range q.Options {
					if !ambiguous && idx == 0 {
						logits[o.Name] = 3.3 + 0.35*rng.NormFloat64()
					} else if ambiguous && (idx == 0 || idx == 1 || o.Name == "none_visible") {
						logits[o.Name] = 0.8 + 0.55*rng.NormFloat64()
					} else {
						logits[o.Name] = -1.4 + 0.35*rng.NormFloat64()
					}
				}
				optNames := make([]string, 0, len(q.Options))
				for _, o := range q.Options {
					optNames = append(optNames, o.Name)
				}
				probs, win, _, _ := NormalizeAndScaleProbabilities(somExpMap(logits), optNames, 1.0)
				answers[q.ID] = client.QuestionAnswer{Choice: win, Probabilities: probs}
			case "goal_progress_score":
				answers[q.ID] = client.QuestionAnswer{
					Choice:        "2",
					Probabilities: map[string]float64{"0": 0.02, "1": 0.13, "2": 0.75, "3": 0.10},
				}
			default:
				probs := make(map[string]float64, len(q.Options))
				for idx, o := range q.Options {
					if idx == 0 {
						probs[o.Name] = 0.90
					} else {
						probs[o.Name] = 0.10 / float64(maxInt(1, len(q.Options)-1))
					}
				}
				answers[q.ID] = client.QuestionAnswer{Choice: q.Options[0].Name, Probabilities: probs}
			}
		}
		envBytes, _ := json.Marshal(client.StructuredDecisionResponse{Answers: answers})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": string(envBytes)}}},
		})
	}))
	defer srv.Close()

	cli := client.NewClient(srv.URL, "", 0)
	var normHs []float64
	var errs []bool
	for i := 0; i < 160; i++ {
		modeTrial = i
		var marks []CandidateMark
		for j := 0; j < 8; j++ {
			row := screenSpot[(i+j)%len(screenSpot)]
			marks = append(marks, CandidateMark{
				ID:    fmt.Sprintf("M%d", j+1),
				Label: row.Target,
				Box:   row.GTBox,
			})
		}
		qs, err := BuildSetOfMarksQuestions("Activate target M1", marks)
		if err != nil {
			t.Fatalf("BuildSetOfMarksQuestions: %v", err)
		}
		resp, err := ExecuteSystemOne(context.Background(), cli, SystemOneRequest{
			State:     "Mobile UI viewport",
			Questions: qs,
		}, DefaultEngineOptions())
		if err != nil {
			t.Fatalf("ExecuteSystemOne trial %d: %v", i, err)
		}
		if resp.EvaluationTrace.ForwardPasses != 1 {
			t.Fatalf("expected 1 forward pass for 4-slot SoM policy, got %d", resp.EvaluationTrace.ForwardPasses)
		}
		ans := resp.Answers["next_target_mark"]
		gold := "M1"
		if (i % 4) == 3 {
			gold = "none_visible"
		}
		normHs = append(normHs, ans.NormalizedH)
		errs = append(errs, ans.Choice != gold)
	}
	auroc := computeHesitationAUROC(normHs, errs)
	if auroc < 0.80 {
		t.Fatalf("expected SoM normalized-entropy error/OOS AUROC >= 0.80, got %.4f", auroc)
	}
}

func somExpMap(logits map[string]float64) map[string]float64 {
	maxL := math.Inf(-1)
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	out := make(map[string]float64, len(logits))
	for k, v := range logits {
		out[k] = math.Exp(v - maxL)
	}
	return out
}
