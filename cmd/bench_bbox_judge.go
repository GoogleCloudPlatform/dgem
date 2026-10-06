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

// Gemini 3.x as a visual judge of box overlays. The judge sees the image with a
// proposed box drawn on it and grades each edge. Before its verdicts are used on images without
// ground truth, it is measured on (a) ground-truth boxes with edges shifted by known amounts and
// (b) dgem's own boxes, both scored against ground truth.

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"
	"google.golang.org/genai"
)

var (
	judgeDataset   string
	judgeReceipt   string
	judgeModel     string
	judgeOutput    string
	judgeWorkers   int
	judgeCalibrate bool
	judgeShifts    string
	judgeTol       float64
	judgeOffThresh float64
	judgeImageDir  string
	judgeLimit     int
	judgeRescore   string
)

var benchBboxJudgeCmd = &cobra.Command{
	Use:     "bench-bbox-judge",
	GroupID: "eval",
	Short:   "Validate Gemini 3.x as a visual judge of bounding boxes",
	Long: `bench-bbox-judge draws a proposed box on each image and asks a Gemini 3.x model to grade every edge
(correct / too_far_in / too_far_out) and the box overall. Verdicts are scored against ground truth:
  --calibrate   ground-truth boxes with one edge shifted in or out by known amounts (--shifts), plus exact boxes
  --receipt     the boxes in a bench-bbox receipt (original variant, every repeat)
An edge is "correct" when it is within --tol points of the truth; edges between --tol and --off are ambiguous and
left out of the binary metrics. Two-object and absent-target cases are skipped.`,
	Example: `  dgem bench-bbox-judge --calibrate --receipt bbox.json \
    --judge-model gemini-3.8-flash -o judge.json`,
	RunE: runBenchBboxJudge,
}

func init() {
	f := benchBboxJudgeCmd.Flags()
	f.StringVarP(&judgeDataset, "dataset", "d", "benchmarks/bbox_suite.jsonl", "Bounding-box JSONL suite (ground truth)")
	f.StringVar(&judgeReceipt, "receipt", "", "bench-bbox receipt whose boxes are judged")
	f.StringVar(&judgeModel, "judge-model", DefaultCascadeGeminiModel, "Gemini 3.x judge model")
	f.StringVarP(&judgeOutput, "output", "o", "", "Write the judge receipt to this file")
	f.IntVarP(&judgeWorkers, "workers", "w", 8, "Concurrent Gemini calls")
	f.BoolVar(&judgeCalibrate, "calibrate", false, "Judge ground-truth boxes with controlled edge shifts")
	f.StringVar(&judgeShifts, "shifts", "5,10,20", "Edge shift sizes in points for --calibrate")
	f.Float64Var(&judgeTol, "tol", 2.5, "An edge within this many points of the truth counts as correct")
	f.Float64Var(&judgeOffThresh, "off", 5, "An edge more than this many points off counts as wrong (binary metrics)")
	f.StringVar(&judgeImageDir, "image-dir", "", "Where overlay images are written (default: a temp directory)")
	f.IntVarP(&judgeLimit, "limit", "n", 0, "Limit the number of judged items (0 = all)")
	f.StringVar(&judgeRescore, "rescore", "", "Recompute the summaries of an existing judge receipt (no Gemini calls)")
	RootCmd.AddCommand(benchBboxJudgeCmd)
}

var judgeEdgeNames = [4]string{"top", "left", "bottom", "right"}

const (
	verdictCorrect = "correct"
	verdictIn      = "too_far_in"
	verdictOut     = "too_far_out"
)

// JudgeItem is one judged box.
type JudgeItem struct {
	Source       string     `json:"source"` // calibration | receipt
	CaseID       string     `json:"case_id"`
	Perturbation string     `json:"perturbation,omitempty"`
	Shift        float64    `json:"shift,omitempty"`
	Repeat       int        `json:"repeat,omitempty"`
	GTBox        [4]float64 `json:"gt_box"`
	Box          [4]float64 `json:"box"`
	IoU          float64    `json:"iou"`
	EdgeErrors   [4]float64 `json:"edge_errors"` // signed, + = toward bottom/right
	TruthEdges   [4]string  `json:"truth_edges"`
	JudgeEdges   [4]string  `json:"judge_edges,omitempty"`
	JudgeOK      bool       `json:"judge_acceptable"`
	LatencyMs    float64    `json:"latency_ms,omitempty"`
	Error        string     `json:"error,omitempty"`
}

// JudgeSummary scores a set of judged items against ground truth.
type JudgeSummary struct {
	Source string `json:"source"`
	Items  int    `json:"items"`
	Errors int    `json:"errors"`
	// Three-class edge verdicts (correct / too_far_in / too_far_out) against the truth at --tol.
	EdgeAccuracy3 float64        `json:"edge_accuracy_3class"`
	EdgeKappa3    float64        `json:"edge_kappa_3class"`
	Confusion     map[string]int `json:"confusion_truth_to_judge"`
	// Binary: judge says not-correct vs edge more than --off points wrong (ambiguous band excluded).
	EdgeBinaryN      int     `json:"edge_binary_n"`
	EdgeRecallWrong  float64 `json:"edge_recall_wrong"`
	EdgeSpecificity  float64 `json:"edge_specificity"`
	EdgeBinaryKappa  float64 `json:"edge_binary_kappa"`
	DirectionCorrect float64 `json:"direction_accuracy_on_flagged_wrong"`
	// Box level. The judge's "acceptable" means "tight": compare it with "all four edges within
	// --tol" (primary) and, for reference, with IoU >= 0.75.
	BoxAgreementTight float64 `json:"box_agreement_all_edges_within_tol"`
	BoxKappaTight     float64 `json:"box_kappa_all_edges_within_tol"`
	ActualTightRate   float64 `json:"actual_all_edges_within_tol_rate"`
	BoxAgreement      float64 `json:"box_agreement_iou75"`
	BoxKappa          float64 `json:"box_kappa_iou75"`
	JudgePassRate     float64 `json:"judge_pass_rate"`
	ActualPassRate75  float64 `json:"actual_pass_rate_iou75"`
	// FlagRateByError: share of edges the judge called not-correct, by absolute edge error (points).
	// The error at which it crosses 0.5 is the judge's practical resolution.
	FlagRateByError  map[string]float64 `json:"flag_rate_by_abs_error"`
	FlagCountByError map[string]int     `json:"edges_by_abs_error"`
	// Calibration only: share of shifted edges the judge called correct, by shift size.
	CorrectRateByShift map[string]float64 `json:"judged_correct_rate_by_shift,omitempty"`
	MeanLatencyMs      float64            `json:"mean_latency_ms"`
}

type JudgeReport struct {
	Timestamp  string         `json:"timestamp"`
	JudgeModel string         `json:"judge_model"`
	Dataset    string         `json:"dataset"`
	Receipt    string         `json:"receipt,omitempty"`
	ReceiptOf  string         `json:"receipt_target_model,omitempty"`
	Tol        float64        `json:"tol"`
	Off        float64        `json:"off"`
	GitCommit  string         `json:"git_commit,omitempty"`
	Summaries  []JudgeSummary `json:"summaries"`
	Items      []JudgeItem    `json:"items"`
}

// truthEdgeVerdicts labels each edge of box against gt: correct within tol, else in/out.
func truthEdgeVerdicts(box, gt [4]float64, tol float64) ([4]float64, [4]string) {
	var errs [4]float64
	var v [4]string
	for i := range box {
		e := box[i] - gt[i]
		errs[i] = e
		switch {
		case math.Abs(e) <= tol:
			v[i] = verdictCorrect
		case (i < 2 && e > 0) || (i >= 2 && e < 0):
			v[i] = verdictIn // min edge moved toward the centre, or max edge moved toward the centre
		default:
			v[i] = verdictOut
		}
	}
	return errs, v
}

func clampBox(b [4]float64) [4]float64 {
	for i := range b {
		b[i] = math.Max(0, math.Min(100, b[i]))
	}
	return b
}

// calibrationBoxes returns the exact box plus every edge shifted in and out by each size. An inward
// shift is capped so the box keeps at least 2 points of extent.
func calibrationBoxes(gt [4]float64, shifts []float64) []JudgeItem {
	items := []JudgeItem{{Perturbation: "exact", Box: gt}}
	for ei := range gt {
		for _, d := range shifts {
			for _, dir := range []string{"in", "out"} {
				b := gt
				sign := 1.0 // + moves toward bottom/right
				if (ei < 2 && dir == "out") || (ei >= 2 && dir == "in") {
					sign = -1
				}
				b[ei] += sign * d
				if dir == "in" {
					if ei < 2 {
						b[ei] = math.Min(b[ei], gt[ei+2]-2)
					} else {
						b[ei] = math.Max(b[ei], gt[ei-2]+2)
					}
				}
				b = clampBox(b)
				if b == gt {
					continue // out-shift against the image border
				}
				items = append(items, JudgeItem{Perturbation: fmt.Sprintf("%s_%s", judgeEdgeNames[ei], dir), Shift: d, Box: b})
			}
		}
	}
	return items
}

// renderBoxOverlay draws box (percent) as a magenta outline on the image and writes a PNG.
func renderBoxOverlay(src string, box [4]float64, out string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return err
	}
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	w, h := float64(b.Dx()), float64(b.Dy())
	x0, y0 := int(box[1]/100*w), int(box[0]/100*h)
	x1, y1 := int(box[3]/100*w), int(box[2]/100*h)
	t := int(math.Max(2, math.Round(math.Min(w, h)/250)))
	col := &image.Uniform{color.RGBA{255, 0, 255, 255}}
	rect := func(r image.Rectangle) { draw.Draw(dst, r.Intersect(dst.Bounds()), col, image.Point{}, draw.Src) }
	// Lines are centred on the edge coordinate.
	rect(image.Rect(x0-t/2, y0-t/2, x1+t/2+1, y0+t/2+1))
	rect(image.Rect(x0-t/2, y1-t/2, x1+t/2+1, y1+t/2+1))
	rect(image.Rect(x0-t/2, y0-t/2, x0+t/2+1, y1+t/2+1))
	rect(image.Rect(x1-t/2, y0-t/2, x1+t/2+1, y1+t/2+1))
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	wf, err := os.Create(out)
	if err != nil {
		return err
	}
	defer wf.Close()
	return png.Encode(wf, dst)
}

type judgeReply struct {
	Top        string `json:"top"`
	Left       string `json:"left"`
	Bottom     string `json:"bottom"`
	Right      string `json:"right"`
	Acceptable bool   `json:"acceptable"`
}

func judgeOne(ctx context.Context, model, overlay, target string) (judgeReply, float64, error) {
	verdict := &genai.Schema{Type: genai.TypeString, Enum: []string{verdictCorrect, verdictIn, verdictOut}}
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"top": verdict, "left": verdict, "bottom": verdict, "right": verdict,
			"acceptable": {Type: genai.TypeBoolean},
		},
		Required:         []string{"top", "left", "bottom", "right", "acceptable"},
		PropertyOrdering: []string{"top", "left", "bottom", "right", "acceptable"},
	}
	prompt := fmt.Sprintf(`The magenta rectangle drawn on this image is a proposed bounding box for the target "%s".
Grade each edge of the rectangle against the target's true extent:
- "correct": the edge lies on the target's true edge (within about 2%% of the image size)
- "too_far_in": the edge cuts into the target (part of the target lies outside the rectangle on that side)
- "too_far_out": the edge leaves a visible gap beyond the target on that side
Set "acceptable" to true only if the rectangle tightly encloses the target.`, strings.ReplaceAll(target, "_", " "))
	var r judgeReply
	ms, err := callGeminiJSON(ctx, model, overlay, prompt, schema, &r)
	return r, ms, err
}

func cohenKappa(truth, pred []string) float64 {
	n := float64(len(truth))
	if n == 0 {
		return 0
	}
	agree := 0.0
	ct, cp := map[string]float64{}, map[string]float64{}
	for i := range truth {
		if truth[i] == pred[i] {
			agree++
		}
		ct[truth[i]]++
		cp[pred[i]]++
	}
	po := agree / n
	pe := 0.0
	for k, v := range ct {
		pe += (v / n) * (cp[k] / n)
	}
	if pe >= 1 {
		return 1
	}
	return (po - pe) / (1 - pe)
}

func summarizeJudge(source string, items []JudgeItem, tol, off float64) JudgeSummary {
	s := JudgeSummary{Source: source, Confusion: map[string]int{}}
	var t3, p3, tb, pb, boxT, boxP, tightT []string
	flagN, flagHit := map[string]int{}, map[string]int{}
	var dirHit, dirN float64
	var lat []float64
	byShiftHit, byShiftN := map[string]float64{}, map[string]float64{}
	for _, it := range items {
		if it.Source != source {
			continue
		}
		if it.Error != "" {
			s.Errors++
			continue
		}
		s.Items++
		lat = append(lat, it.LatencyMs)
		for i := 0; i < 4; i++ {
			t3 = append(t3, it.TruthEdges[i])
			p3 = append(p3, it.JudgeEdges[i])
			s.Confusion[it.TruthEdges[i]+"->"+it.JudgeEdges[i]]++
			ae := math.Abs(it.EdgeErrors[i])
			bin := errorBin(ae)
			flagN[bin]++
			if it.JudgeEdges[i] != verdictCorrect {
				flagHit[bin]++
			}
			if ae <= tol || ae > off {
				truthWrong := ae > off
				judgeWrong := it.JudgeEdges[i] != verdictCorrect
				tb = append(tb, fmt.Sprint(truthWrong))
				pb = append(pb, fmt.Sprint(judgeWrong))
				if truthWrong && judgeWrong {
					dirN++
					if it.JudgeEdges[i] == it.TruthEdges[i] {
						dirHit++
					}
				}
			}
		}
		if source == "calibration" {
			key := "exact"
			if it.Shift > 0 {
				key = fmt.Sprintf("%g", it.Shift)
				ei := -1
				for i, n := range judgeEdgeNames {
					if strings.HasPrefix(it.Perturbation, n+"_") {
						ei = i
					}
				}
				if ei >= 0 {
					byShiftN[key]++
					if it.JudgeEdges[ei] == verdictCorrect {
						byShiftHit[key]++
					}
				}
			} else {
				for i := 0; i < 4; i++ {
					byShiftN[key]++
					if it.JudgeEdges[i] == verdictCorrect {
						byShiftHit[key]++
					}
				}
			}
		}
		tight := true
		for _, v := range it.TruthEdges {
			if v != verdictCorrect {
				tight = false
			}
		}
		tightT = append(tightT, fmt.Sprint(tight))
		boxT = append(boxT, fmt.Sprint(it.IoU >= 0.75))
		boxP = append(boxP, fmt.Sprint(it.JudgeOK))
	}
	agreeRate := func(a, b []string) float64 {
		if len(a) == 0 {
			return 0
		}
		n := 0
		for i := range a {
			if a[i] == b[i] {
				n++
			}
		}
		return float64(n) / float64(len(a))
	}
	s.EdgeAccuracy3, s.EdgeKappa3 = agreeRate(t3, p3), cohenKappa(t3, p3)
	s.EdgeBinaryN, s.EdgeBinaryKappa = len(tb), cohenKappa(tb, pb)
	var tp, fn, tn, fp float64
	for i := range tb {
		switch {
		case tb[i] == "true" && pb[i] == "true":
			tp++
		case tb[i] == "true":
			fn++
		case pb[i] == "false":
			tn++
		default:
			fp++
		}
	}
	if tp+fn > 0 {
		s.EdgeRecallWrong = tp / (tp + fn)
	}
	if tn+fp > 0 {
		s.EdgeSpecificity = tn / (tn + fp)
	}
	if dirN > 0 {
		s.DirectionCorrect = dirHit / dirN
	}
	s.BoxAgreement, s.BoxKappa = agreeRate(boxT, boxP), cohenKappa(boxT, boxP)
	s.BoxAgreementTight, s.BoxKappaTight = agreeRate(tightT, boxP), cohenKappa(tightT, boxP)
	for _, v := range tightT {
		if v == "true" {
			s.ActualTightRate++
		}
	}
	if len(tightT) > 0 {
		s.ActualTightRate /= float64(len(tightT))
	}
	s.FlagRateByError, s.FlagCountByError = map[string]float64{}, flagN
	for k, n := range flagN {
		s.FlagRateByError[k] = float64(flagHit[k]) / float64(n)
	}
	for i := range boxT {
		if boxP[i] == "true" {
			s.JudgePassRate++
		}
		if boxT[i] == "true" {
			s.ActualPassRate75++
		}
	}
	if len(boxT) > 0 {
		s.JudgePassRate /= float64(len(boxT))
		s.ActualPassRate75 /= float64(len(boxT))
	}
	if len(byShiftN) > 0 {
		s.CorrectRateByShift = map[string]float64{}
		for k, n := range byShiftN {
			s.CorrectRateByShift[k] = byShiftHit[k] / n
		}
	}
	s.MeanLatencyMs = mean(lat)
	return s
}

var errorBins = []struct {
	hi   float64
	name string
}{{0.5, "0-0.5"}, {1.5, "0.5-1.5"}, {2.5, "1.5-2.5"}, {5, "2.5-5"}, {10, "5-10"}, {math.Inf(1), "10+"}}

func errorBin(ae float64) string {
	for _, b := range errorBins {
		if ae <= b.hi {
			return b.name
		}
	}
	return "10+"
}

func parseShifts(raw string) ([]float64, error) {
	var out []float64
	for _, p := range strings.Split(raw, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v float64
		if _, err := fmt.Sscanf(p, "%g", &v); err != nil || v <= 0 {
			return nil, fmt.Errorf("bad shift %q", p)
		}
		out = append(out, v)
	}
	return out, nil
}

func runBenchBboxJudge(cmd *cobra.Command, args []string) error {
	if judgeRescore != "" {
		b, err := os.ReadFile(judgeRescore)
		if err != nil {
			return err
		}
		var rep JudgeReport
		if err := json.Unmarshal(b, &rep); err != nil {
			return err
		}
		for i := range rep.Items {
			rep.Items[i].EdgeErrors, rep.Items[i].TruthEdges = truthEdgeVerdicts(rep.Items[i].Box, rep.Items[i].GTBox, judgeTol)
		}
		rep.Tol, rep.Off, rep.Summaries = judgeTol, judgeOffThresh, nil
		for _, src := range []string{"calibration", "receipt"} {
			if s := summarizeJudge(src, rep.Items, judgeTol, judgeOffThresh); s.Items+s.Errors > 0 {
				rep.Summaries = append(rep.Summaries, s)
			}
		}
		return emitJudgeReport(rep)
	}
	if !judgeCalibrate && judgeReceipt == "" {
		return fmt.Errorf("pass --calibrate and/or --receipt")
	}
	bboxDataset = judgeDataset
	suite, err := loadBboxSuite()
	if err != nil {
		return err
	}
	byID := map[string]BboxSuiteItem{}
	var single []BboxSuiteItem
	for _, it := range suite {
		byID[it.ID] = it
		if it.ObjectPresent && it.SecondaryBoxContinuous == nil {
			single = append(single, it)
		}
	}
	shifts, err := parseShifts(judgeShifts)
	if err != nil {
		return err
	}
	model := SanitizeCascadeModel(judgeModel)
	report := JudgeReport{Timestamp: time.Now().UTC().Format(time.RFC3339), JudgeModel: model, Dataset: judgeDataset,
		Receipt: judgeReceipt, Tol: judgeTol, Off: judgeOffThresh, GitCommit: gitShortCommit()}

	var items []JudgeItem
	if judgeCalibrate {
		for _, it := range single {
			for _, c := range calibrationBoxes(it.GTBoxContinuous, shifts) {
				c.Source, c.CaseID, c.GTBox = "calibration", it.ID, it.GTBoxContinuous
				items = append(items, c)
			}
		}
	}
	if judgeReceipt != "" {
		b, err := os.ReadFile(judgeReceipt)
		if err != nil {
			return err
		}
		var rec BboxBenchmarkReport
		if err := json.Unmarshal(b, &rec); err != nil {
			return err
		}
		report.ReceiptOf = rec.TargetModel
		for _, r := range rec.Cases {
			if (r.Variant != "" && r.Variant != variantOriginal) || r.Error != "" || r.Skipped != "" {
				continue
			}
			it, ok := byID[r.ID]
			if !ok || !it.ObjectPresent || it.SecondaryBoxContinuous != nil || !r.ObjectPresentPred || !boxValid(r.ExpectationBox) {
				continue
			}
			items = append(items, JudgeItem{Source: "receipt", CaseID: r.ID, Repeat: r.Repeat, GTBox: it.GTBoxContinuous, Box: r.ExpectationBox})
		}
	}
	if judgeLimit > 0 && judgeLimit < len(items) {
		items = items[:judgeLimit]
	}
	for i := range items {
		items[i].IoU = compute2DIoU(items[i].GTBox, items[i].Box)
		items[i].EdgeErrors, items[i].TruthEdges = truthEdgeVerdicts(items[i].Box, items[i].GTBox, judgeTol)
	}

	dir := judgeImageDir
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "dgem-bbox-judge")
	}
	ctx := context.Background()
	ch := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for w := 0; w < max(1, judgeWorkers); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				it := &items[i]
				src := byID[it.CaseID]
				overlay := filepath.Join(dir, fmt.Sprintf("%s__%s_%d_%03d.png", it.CaseID, it.Source, it.Repeat, i))
				if err := renderBoxOverlay(src.ImagePath, it.Box, overlay); err != nil {
					it.Error = err.Error()
				} else if r, ms, err := judgeOne(ctx, model, overlay, src.Target); err != nil {
					it.Error = err.Error()
				} else {
					it.JudgeEdges = [4]string{r.Top, r.Left, r.Bottom, r.Right}
					it.JudgeOK, it.LatencyMs = r.Acceptable, ms
				}
				mu.Lock()
				done++
				if done%25 == 0 || done == len(items) {
					fmt.Fprintf(os.Stderr, "  bench-bbox-judge: %d/%d\n", done, len(items))
				}
				mu.Unlock()
			}
		}()
	}
	for i := range items {
		ch <- i
	}
	close(ch)
	wg.Wait()

	report.Items = items
	for _, src := range []string{"calibration", "receipt"} {
		s := summarizeJudge(src, items, judgeTol, judgeOffThresh)
		if s.Items+s.Errors > 0 {
			report.Summaries = append(report.Summaries, s)
		}
	}
	return emitJudgeReport(report)
}

func emitJudgeReport(report JudgeReport) error {
	model := report.JudgeModel
	if judgeOutput != "" {
		b, _ := json.MarshalIndent(report, "", "  ")
		if err := os.WriteFile(judgeOutput, b, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("\nBounding-box judge validation  judge=%s  tol=%.1f  off=%.1f\n", model, report.Tol, report.Off)
	fmt.Println(strings.Repeat("=", 110))
	for _, s := range report.Summaries {
		fmt.Printf("[%s] items=%d errors=%d latency=%.0f ms\n", s.Source, s.Items, s.Errors, s.MeanLatencyMs)
		fmt.Printf("  edges, 3-class: accuracy %.3f, kappa %.3f\n", s.EdgeAccuracy3, s.EdgeKappa3)
		fmt.Printf("  edges, wrong (>%.0f pts) vs right (<=%.1f): n=%d recall %.3f specificity %.3f kappa %.3f; direction right on %.3f of caught errors\n",
			judgeOffThresh, judgeTol, s.EdgeBinaryN, s.EdgeRecallWrong, s.EdgeSpecificity, s.EdgeBinaryKappa, s.DirectionCorrect)
		fmt.Printf("  box acceptable vs all edges within tol: agreement %.3f kappa %.3f (judge pass %.3f, actual %.3f); vs IoU>=0.75: agreement %.3f (actual %.3f)\n",
			s.BoxAgreementTight, s.BoxKappaTight, s.JudgePassRate, s.ActualTightRate, s.BoxAgreement, s.ActualPassRate75)
		line := "  flagged not-correct by |edge error|:"
		for _, b := range errorBins {
			if n := s.FlagCountByError[b.name]; n > 0 {
				line += fmt.Sprintf(" %s=%.2f (n=%d)", b.name, s.FlagRateByError[b.name], n)
			}
		}
		fmt.Println(line)
		if len(s.CorrectRateByShift) > 0 {
			keys := make([]string, 0, len(s.CorrectRateByShift))
			for k := range s.CorrectRateByShift {
				keys = append(keys, k)
			}
			sort.Slice(keys, func(i, j int) bool {
				if keys[i] == "exact" {
					return true
				}
				if keys[j] == "exact" {
					return false
				}
				var a, b float64
				fmt.Sscanf(keys[i], "%g", &a)
				fmt.Sscanf(keys[j], "%g", &b)
				return a < b
			})
			line := "  judged \"correct\" by shift:"
			for _, k := range keys {
				line += fmt.Sprintf(" %s=%.2f", k, s.CorrectRateByShift[k])
			}
			fmt.Println(line)
		}
		keys := make([]string, 0, len(s.Confusion))
		for k := range s.Confusion {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		line = "  confusion (truth->judge):"
		for _, k := range keys {
			line += fmt.Sprintf(" %s=%d", k, s.Confusion[k])
		}
		fmt.Println(line)
	}
	fmt.Println()
	return nil
}
