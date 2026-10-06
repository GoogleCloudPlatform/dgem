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
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

const testLocalizationSchema = `{"instructions":"... from 00 (0%, top/left edge) ... in 5% increments.","questions":[
 {"id":"object_present","type":"boolean"},
 {"id":"ymin","type":"choice","options":[{"name":"00"},{"name":"05"},{"name":"10"},{"name":"100"}]},
 {"id":"xmin","type":"choice","options":[{"name":"00"},{"name":"05"},{"name":"10"},{"name":"100"}]},
 {"id":"ymax","type":"choice","options":[{"name":"00"},{"name":"05"},{"name":"10"},{"name":"100"}]},
 {"id":"xmax","type":"choice","options":[{"name":"00"},{"name":"05"},{"name":"10"},{"name":"100"}]}]}`

func near(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestTransformBoxForVariant(t *testing.T) {
	b := [4]float64{10, 20, 40, 60}
	cases := map[string][4]float64{
		variantOriginal:  b,
		variantBlank:     b,
		variantHFlip:     {10, 40, 40, 80},
		variantVFlip:     {60, 20, 90, 60},
		variantPadRight:  {10, 20.0 / 1.5, 40, 60.0 / 1.5},
		variantPadLeft:   {10, 100 - 100/1.5 + 20/1.5, 40, 100 - 100/1.5 + 60/1.5},
		variantPadBottom: {10 / 1.5, 20, 40 / 1.5, 60},
	}
	for v, want := range cases {
		got := transformBoxForVariant(b, v)
		for i := range got {
			if !near(got[i], want[i]) {
				t.Errorf("%s: got %v want %v", v, got, want)
				break
			}
		}
	}
}

func TestTransformImageMatchesBoxTransform(t *testing.T) {
	// A 10x10 white image with one black pixel at (x=2, y=1).
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			img.Set(x, y, color.White)
		}
	}
	img.Set(2, 1, color.Black)
	isBlack := func(im image.Image, x, y int) bool {
		r, _, _, _ := im.At(x, y).RGBA()
		return r == 0
	}
	if !isBlack(transformImage(img, variantHFlip), 7, 1) {
		t.Error("hflip: pixel not mirrored horizontally")
	}
	if !isBlack(transformImage(img, variantVFlip), 2, 8) {
		t.Error("vflip: pixel not mirrored vertically")
	}
	pl := transformImage(img, variantPadLeft)
	if pl.Bounds().Dx() != 15 || !isBlack(pl, 7, 1) || isBlack(pl, 0, 0) {
		t.Error("pad_left: wrong size, offset or fill")
	}
	pb := transformImage(img, variantPadBottom)
	if pb.Bounds().Dy() != 15 || !isBlack(pb, 2, 1) {
		t.Error("pad_bottom: wrong size or offset")
	}
	if isBlack(transformImage(img, variantBlank), 2, 1) {
		t.Error("blank: content survived")
	}
}

func TestApplySchemaVariant(t *testing.T) {
	rev, err := applySchemaVariant(testLocalizationSchema, variantReversed)
	if err != nil {
		t.Fatal(err)
	}
	cq, _ := coordQuestionsFromSchema(rev)
	if cq["ymin"].K != 4 || cq["ymin"].Values["100"] != 100 {
		t.Fatalf("reversed lost labels: %+v", cq["ymin"])
	}
	var m map[string]interface{}
	_ = json.Unmarshal([]byte(rev), &m)
	first := m["questions"].([]interface{})[1].(map[string]interface{})["options"].([]interface{})[0].(map[string]interface{})["name"]
	if first != "100" {
		t.Errorf("reversed: first option = %v, want 100", first)
	}

	d9, err := applySchemaVariant(testLocalizationSchema, variantDigits9Reversed)
	if err != nil {
		t.Fatal(err)
	}
	cq, _ = coordQuestionsFromSchema(d9)
	if cq["xmax"].K != 9 || !near(cq["xmax"].BinWidth, 12.5) || cq["xmax"].Max != 100 {
		t.Errorf("digits9: %+v", cq["xmax"])
	}
	_ = json.Unmarshal([]byte(d9), &m)
	q := m["questions"].([]interface{})[1].(map[string]interface{})
	if q["type"] != "score" || q["levels"].([]interface{})[0] != "100" {
		t.Errorf("digits9_reversed: %v", q)
	}
	if !schemaHasQuestion(d9, "object_present") {
		t.Error("non-coordinate question dropped")
	}

	band := `{"questions":[{"id":"obj1_ymin","type":"choice","options":[{"name":"00","description":"0-10%"},{"name":"10","description":"10-20%"}]}]}`
	if _, err := applySchemaVariant(band, variantDigits9); err != errVariantNotApplicable {
		t.Errorf("digits9 on band template: err=%v, want not applicable", err)
	}
	cq, _ = coordQuestionsFromSchema(band)
	if cq["obj1_ymin"].Values["10"] != 15 {
		t.Errorf("band label should map to its centre, got %v", cq["obj1_ymin"].Values)
	}
}

func TestScoreMissingSlotIsFailureNotGroundTruth(t *testing.T) {
	cqs, _ := coordQuestionsFromSchema(testLocalizationSchema)
	item := BboxSuiteItem{ID: "x", ObjectPresent: true, GTBoxContinuous: [4]float64{0, 0, 100, 100}}
	answers := map[string]client.QuestionAnswer{
		"object_present": {Type: "noul", Label: "yes"},
		"ymin":           {Type: "choice", Probabilities: map[string]float64{"00": 1}},
		"xmin":           {Type: "choice", Probabilities: map[string]float64{"00": 1}},
		"ymax":           {Type: "choice", Probabilities: map[string]float64{"100": 1}},
		// xmax missing: the old harness filled it with the ground-truth coordinate (IoU 1).
	}
	r := scoreBboxAnswers(item, variantOriginal, answers, cqs, true)
	if r.ExpectationIoU != 0 || len(r.MissingSlots) != 1 || r.MissingSlots[0] != "xmax" {
		t.Fatalf("missing slot must score 0 and be recorded: iou=%v missing=%v", r.ExpectationIoU, r.MissingSlots)
	}

	delete(answers, "object_present")
	answers["xmax"] = client.QuestionAnswer{Type: "choice", Probabilities: map[string]float64{"100": 1}}
	r = scoreBboxAnswers(item, variantOriginal, answers, cqs, true)
	if r.ObjectPresentPred || r.ExpectationIoU != 0 {
		t.Errorf("missing object_present must not fall back to ground truth: %+v", r)
	}
}

func TestScoreExpectationAndEntropy(t *testing.T) {
	cqs, _ := coordQuestionsFromSchema(testLocalizationSchema)
	item := BboxSuiteItem{ID: "x", ObjectPresent: true, GTBoxContinuous: [4]float64{5, 5, 100, 100}}
	answers := map[string]client.QuestionAnswer{
		"object_present": {Type: "noul", Label: "yes"},
		"ymin":           {Type: "choice", Probabilities: map[string]float64{"00": 0.5, "10": 0.5}},
		"xmin":           {Type: "choice", Probabilities: map[string]float64{"05": 1}},
		"ymax":           {Type: "choice", Probabilities: map[string]float64{"100": 1}},
		"xmax":           {Type: "choice", Probabilities: map[string]float64{"100": 1}},
	}
	r := scoreBboxAnswers(item, variantOriginal, answers, cqs, true)
	e := r.Edges["ymin"]
	if !near(e.ExpectedCoord, 5) || e.ArgmaxBin != "00" || !near(e.NormalizedEntropy, math.Log(2)/math.Log(4)) {
		t.Errorf("ymin telemetry: %+v", e)
	}
	if !near(r.ExpectationIoU, 1) || !e.HasGT || e.GTCoord != 5 {
		t.Errorf("expectation IoU %v, edge %+v", r.ExpectationIoU, e)
	}
}

func TestScoreDETRMatchesObjectsOrderFree(t *testing.T) {
	schema := `{"questions":[
	 {"id":"obj1_class","type":"choice","options":[{"name":"a"},{"name":"b"}]},
	 {"id":"obj1_ymin","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj1_xmin","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj1_ymax","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj1_xmax","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj2_class","type":"choice","options":[{"name":"a"},{"name":"b"}]},
	 {"id":"obj2_ymin","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj2_xmin","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj2_ymax","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]},
	 {"id":"obj2_xmax","type":"choice","options":[{"name":"00"},{"name":"50"},{"name":"100"}]}]}`
	cqs, _ := coordQuestionsFromSchema(schema)
	left := [4]float64{0, 0, 50, 50}
	right := [4]float64{50, 50, 100, 100}
	item := BboxSuiteItem{ID: "d", ObjectPresent: true, GTBoxContinuous: left, SecondaryBoxContinuous: &right,
		ExpectedDiscreteSlots: map[string]interface{}{"obj1_class": "a", "obj2_class": "b"}}
	p := func(v string) client.QuestionAnswer {
		return client.QuestionAnswer{Type: "choice", Probabilities: map[string]float64{v: 1}}
	}
	// The model puts the right object in query 1 and the left one in query 2.
	answers := map[string]client.QuestionAnswer{
		"obj1_class": {Type: "choice", Choice: "b"}, "obj2_class": {Type: "choice", Choice: "a"},
		"obj1_ymin": p("50"), "obj1_xmin": p("50"), "obj1_ymax": p("100"), "obj1_xmax": p("100"),
		"obj2_ymin": p("00"), "obj2_xmin": p("00"), "obj2_ymax": p("50"), "obj2_xmax": p("50"),
	}
	r := scoreBboxAnswers(item, variantOriginal, answers, cqs, false)
	if !r.ObjectsSwapped || !near(r.ExpectationIoU, 1) || r.ClassHits != 2 || r.ClassTotal != 2 {
		t.Errorf("DETR matching: swapped=%v iou=%v class=%d/%d", r.ObjectsSwapped, r.ExpectationIoU, r.ClassHits, r.ClassTotal)
	}
	if r.Edges["obj1_ymin"].GTCoord != 50 {
		t.Errorf("obj1 edges should take the matched ground truth, got %+v", r.Edges["obj1_ymin"])
	}
}

func TestStatsHelpers(t *testing.T) {
	if a := aurocScore([]float64{0.9, 0.8}, []float64{0.1, 0.2}); a != 1 {
		t.Errorf("auroc separable = %v", a)
	}
	if a := aurocScore([]float64{0.5}, []float64{0.5}); a != 0.5 {
		t.Errorf("auroc tie = %v", a)
	}
	if s := spearman([]float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}); !near(s, 1) {
		t.Errorf("spearman = %v", s)
	}
	st := bootstrapMean([]float64{1, 1, 1}, 1)
	if st.Mean != 1 || st.Lo != 1 || st.Hi != 1 {
		t.Errorf("bootstrap constant = %+v", st)
	}
	e := BboxEdgeTelemetry{BinWidth: 5, LabelMin: 0, LabelMax: 100}
	if d := nearestLabelDistance(17.5, e); !near(d, 2.5) {
		t.Errorf("quantization distance = %v", d)
	}
	band := BboxEdgeTelemetry{BinWidth: 10, LabelMin: 5, LabelMax: 95}
	if v := nearestLabel(78, band); v != 75 {
		t.Errorf("band nearest = %v", v)
	}
}

func TestBaselinesAndLift(t *testing.T) {
	gt := [4]float64{25, 25, 75, 75}
	cases := []BboxCaseResult{
		{ID: "a", Variant: variantOriginal, ObjectPresentGT: true, GTBox: gt, ArgmaxIoU: 1, ExpectationIoU: 1},
		{ID: "b", Variant: variantOriginal, ObjectPresentGT: true, GTBox: gt, ArgmaxIoU: 0.5, ExpectationIoU: 0.5},
	}
	bs := computeBboxBaselines(cases, nil)
	if len(bs) != 2 || bs[0].Name != "constant_center" || bs[0].MeanIoU.Mean != 1 {
		t.Fatalf("baselines: %+v", bs)
	}
	if !near(bs[0].LiftExpectation.Mean, -0.25) {
		t.Errorf("lift = %v, want -0.25", bs[0].LiftExpectation.Mean)
	}
}

func TestParseBboxVariantsAlwaysIncludesOriginal(t *testing.T) {
	v, err := parseBboxVariants("hflip,reversed")
	if err != nil || v[0] != variantOriginal || len(v) != 3 {
		t.Errorf("got %v, %v", v, err)
	}
	if _, err := parseBboxVariants("nope"); err == nil {
		t.Error("unknown variant accepted")
	}
	if v, _ := parseBboxVariants("all"); len(v) != len(allBboxVariants) {
		t.Errorf("all = %v", v)
	}
}

func TestTruthEdgeVerdicts(t *testing.T) {
	gt := [4]float64{20, 20, 60, 60}
	_, v := truthEdgeVerdicts([4]float64{25, 15, 61, 50}, gt, 2.5)
	want := [4]string{verdictIn, verdictOut, verdictCorrect, verdictIn}
	if v != want {
		t.Errorf("got %v want %v", v, want)
	}
}

func TestCalibrationBoxes(t *testing.T) {
	gt := [4]float64{0, 20, 30, 60}
	items := calibrationBoxes(gt, []float64{5, 20})
	// ymin "out" is against the border for both sizes, so 1 + 4*2*2 - 2.
	if len(items) != 15 {
		t.Fatalf("got %d items", len(items))
	}
	for _, it := range items {
		if !boxValid(it.Box) {
			t.Errorf("%s produced invalid box %v", it.Perturbation, it.Box)
		}
		if it.Perturbation == "exact" {
			continue
		}
		_, v := truthEdgeVerdicts(it.Box, gt, 2.5)
		dir := it.Perturbation[strings.LastIndex(it.Perturbation, "_")+1:]
		nWrong := 0
		for _, x := range v {
			if x != verdictCorrect {
				nWrong++
				if (dir == "in") != (x == verdictIn) {
					t.Errorf("%s labelled %s", it.Perturbation, x)
				}
			}
		}
		if nWrong != 1 {
			t.Errorf("%s: %d edges off, want 1", it.Perturbation, nWrong)
		}
	}
	if k := cohenKappa([]string{"a", "b", "a", "b"}, []string{"a", "b", "a", "b"}); k != 1 {
		t.Errorf("kappa = %v", k)
	}
}

func TestRenderBoxOverlay(t *testing.T) {
	dir := t.TempDir()
	src := dir + "/in.png"
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	f, _ := os.Create(src)
	_ = png.Encode(f, img)
	f.Close()
	out := dir + "/out.png"
	if err := renderBoxOverlay(src, [4]float64{10, 20, 50, 80}, out); err != nil {
		t.Fatal(err)
	}
	g, _ := os.Open(out)
	defer g.Close()
	o, _, err := image.Decode(g)
	if err != nil {
		t.Fatal(err)
	}
	r, _, b, _ := o.At(20, 30).RGBA() // left edge
	if r>>8 != 255 || b>>8 != 255 {
		t.Error("left edge not drawn")
	}
	if r, _, _, _ := o.At(50, 30).RGBA(); r != 0 {
		t.Error("interior painted")
	}
}

func TestAbsentAnswerNeedsNoCoordinates(t *testing.T) {
	cqs, _ := coordQuestionsFromSchema(testLocalizationSchema)
	item := BboxSuiteItem{ID: "a", ObjectPresent: false}
	r := scoreBboxAnswers(item, variantOriginal, map[string]client.QuestionAnswer{"object_present": {Type: "noul", Label: "no"}}, cqs, true)
	if len(r.MissingSlots) != 0 || r.ObjectPresentPred {
		t.Errorf("absent answer without coordinates: missing=%v", r.MissingSlots)
	}
}

func TestFilterSchemaQuestions(t *testing.T) {
	out, err := filterSchemaQuestions(testLocalizationSchema, map[string]string{"ymin": "05", "object_present": "yes"})
	if err != nil {
		t.Fatal(err)
	}
	if !schemaHasQuestion(out, "ymin") || schemaHasQuestion(out, "xmax") || !schemaHasQuestion(out, "object_present") {
		t.Errorf("filtered schema: %s", out)
	}
	if _, err := filterSchemaQuestions(testLocalizationSchema, map[string]string{"nope": "x"}); err == nil {
		t.Error("empty filter accepted")
	}
}

func TestCellCropPct(t *testing.T) {
	b, ok := cellCropPct("top_left")
	if !ok || b[0] != 0 || b[1] != 0 || !near(b[2], 50) || !near(b[3], 50) {
		t.Errorf("top_left crop = %v", b)
	}
	b, _ = cellCropPct("middle_center")
	if !near(b[0], 100.0/6) || !near(b[3], 100-100.0/6) {
		t.Errorf("middle_center crop = %v", b)
	}
	if _, ok := cellCropPct("nowhere"); ok {
		t.Error("bad cell accepted")
	}
}
