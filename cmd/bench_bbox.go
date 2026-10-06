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
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
)

var (
	bboxDataset     string
	bboxDir         string
	bboxTarget      string
	bboxOutput      string
	bboxLimit       int
	bboxSimulate    bool
	bboxAnnotate    bool
	bboxJSON        bool
	bboxRepeat      int
	bboxVariants    string
	bboxWorkers     int
	bboxSamples     int
	bboxVariantDir  string
	bboxFromReceipt string
	bboxEngine      string
	bboxGeminiModel string
	bboxPredFile    string
	bboxPredModel   string
	bboxPredThresh  float64
)

var benchBboxCmd = &cobra.Command{
	Use:     "bench-bbox",
	GroupID: "eval",
	Short:   "Run the bounding-box localization suite with image-free baselines, probe variants and confidence intervals",
	Long: `bench-bbox evaluates DiffusionGemma on 2D spatial grounding (benchmarks/bbox_suite.jsonl or a custom directory via --dir).
For every image it reads [ymin, xmin, ymax, xmax] in one forward pass and reports:
  1. Argmax and softmax-expectation boxes, mIoU and Acc@0.5/0.75 with case-level bootstrap 95% intervals.
  2. Image-free baselines (a fixed centre box, the leave-one-out mean ground-truth box and, with the
     blank variant, the model's own box on a blank image) and the model's paired lift over each.
  3. Edge error split into quantization (distance to the nearest label) and localization, signed bias
     per edge, and whether per-edge entropy predicts edge error (AUROC, Spearman).
  4. Occlusion entropy compared within each occluded / unoccluded twin pair.
  5. Probe variants (--variants): option order (reversed), digit labels (digits9, digits9_reversed),
     image transforms (hflip, vflip, pad_right, pad_left, pad_bottom) with transform consistency, and blank.
A slot the model does not return is scored as a failure (never filled from ground truth).`,
	Example: `  # Re-baseline on the current endpoint: 3 repeats, every probe variant
  dgem bench-bbox --vertex-url <ENDPOINT_ID> --repeat 3 --variants all -o receipt.json

  # Re-analyze an existing receipt (adds baselines, intervals and the error split)
  dgem bench-bbox --from-receipt bbox.json

  # Offline harness check with simulated slot distributions
  dgem bench-bbox --simulate -o bbox_simulated.json

  # Custom folder of images with box overlays
  dgem bench-bbox --dir ./my-images --target "primary object" --annotate -o ./my-images/results.json`,
	RunE: runBenchBbox,
}

func init() {
	benchBboxCmd.Flags().StringVarP(&bboxDataset, "dataset", "d", "benchmarks/bbox_suite.jsonl", "Path to bounding-box JSONL evaluation suite")
	benchBboxCmd.Flags().StringVar(&bboxDir, "dir", "", "Directory of custom PNG/JPG images to localize")
	benchBboxCmd.Flags().StringVar(&bboxTarget, "target", "primary_foreground_object", "Default target description when using --dir")
	benchBboxCmd.Flags().StringVarP(&bboxOutput, "output", "o", "", "Export structured JSON benchmark receipt to file")
	benchBboxCmd.Flags().IntVarP(&bboxLimit, "limit", "n", 0, "Limit number of cases to evaluate (0 = all)")
	benchBboxCmd.Flags().BoolVar(&bboxSimulate, "simulate", false, "Run with reference Gaussian/occlusion slot distributions (no live GPU required)")
	benchBboxCmd.Flags().BoolVar(&bboxAnnotate, "annotate", false, "Write annotated SVG box overlays alongside evaluated images (original variant, first repeat)")
	benchBboxCmd.Flags().BoolVar(&bboxJSON, "json", false, "Emit machine-readable JSON summary to stdout")
	benchBboxCmd.Flags().IntVar(&bboxRepeat, "repeat", 1, "Run every case N times to measure run-to-run stability")
	benchBboxCmd.Flags().StringVar(&bboxVariants, "variants", "original", "Comma-separated probe variants, or 'all': "+strings.Join(allBboxVariants, ","))
	benchBboxCmd.Flags().IntVarP(&bboxWorkers, "workers", "w", 4, "Concurrent requests")
	benchBboxCmd.Flags().IntVar(&bboxSamples, "samples", 1, "Noise draws per request (template variable 'samples')")
	benchBboxCmd.Flags().StringVar(&bboxVariantDir, "variant-dir", "", "Where transformed images are written (default: a temp directory)")
	benchBboxCmd.Flags().StringVar(&bboxFromReceipt, "from-receipt", "", "Re-analyze an existing receipt instead of calling a server")
	benchBboxCmd.Flags().StringVar(&bboxEngine, "engine", "dgem", "Localizer: dgem (the decision server) or gemini (Vertex Gemini 3.x reference, same scoring)")
	benchBboxCmd.Flags().StringVar(&bboxPredFile, "predictions", "", "For --engine predictions: JSONL of {id, model, box_pct [ymin,xmin,ymax,xmax], score} (e.g. from scripts/detectors/run_detectors.py)")
	benchBboxCmd.Flags().StringVar(&bboxPredModel, "pred-model", "", "For --engine predictions: which model's rows to score")
	benchBboxCmd.Flags().Float64Var(&bboxPredThresh, "pred-threshold", 0, "For --engine predictions: a box scoring below this counts as \"absent\"")
	benchBboxCmd.Flags().StringVar(&bboxGeminiModel, "gemini-model", DefaultCascadeGeminiModel, "Gemini 3.x model for --engine gemini (gemini-3.8-flash, gemini-3.5-flash-lite)")

	RootCmd.AddCommand(benchBboxCmd)
}

// BboxSuiteItem represents a row in benchmarks/bbox_suite.jsonl.
type BboxSuiteItem struct {
	ID                     string                 `json:"id"`
	Tier                   string                 `json:"tier"`
	PairID                 string                 `json:"pair_id,omitempty"`
	Template               string                 `json:"template"`
	ImagePath              string                 `json:"image_path"`
	SVGPath                string                 `json:"svg_path"`
	Target                 string                 `json:"target"`
	ObjectPresent          bool                   `json:"object_present"`
	GTBoxContinuous        [4]float64             `json:"gt_box_continuous"`
	SecondaryBoxContinuous *[4]float64            `json:"secondary_box_continuous,omitempty"`
	OccludedEdge           string                 `json:"occluded_edge"`
	ExpectedDiscreteSlots  map[string]interface{} `json:"expected_discrete_slots"`
	Notes                  string                 `json:"notes"`
	// Tags label the factors of a generated sweep (scripts/generate_bbox_sweep.py); the report
	// breaks results down by every tag value.
	Tags map[string]string `json:"tags,omitempty"`
	// VisibleBoxContinuous is the unoccluded part of the target when it differs from the full box.
	VisibleBoxContinuous *[4]float64 `json:"visible_box_continuous,omitempty"`
}

// BboxEdgeTelemetry stores one coordinate slot's prediction and normalized entropy.
type BboxEdgeTelemetry struct {
	QuestionID        string  `json:"question_id,omitempty"`
	Edge              string  `json:"edge,omitempty"`
	ArgmaxBin         string  `json:"argmax_bin"`
	ArgmaxCoord       float64 `json:"argmax_coord"`
	ExpectedCoord     float64 `json:"expected_coord"`
	Confidence        float64 `json:"confidence"`
	RawEntropyNats    float64 `json:"raw_entropy_nats"`
	NormalizedEntropy float64 `json:"normalized_entropy"`
	// K is the number of labels the slot offers; LabelsReturned how many came back with a
	// probability (fewer than K means the entropy is over a truncated distribution).
	K              int     `json:"k,omitempty"`
	LabelsReturned int     `json:"labels_returned,omitempty"`
	BinWidth       float64 `json:"bin_width,omitempty"`
	LabelMin       float64 `json:"label_min,omitempty"`
	LabelMax       float64 `json:"label_max,omitempty"`
	GTCoord        float64 `json:"gt_coord,omitempty"`
	HasGT          bool    `json:"has_gt,omitempty"`
	Missing        bool    `json:"missing,omitempty"`
	NoDistribution bool    `json:"no_distribution,omitempty"`
}

// BboxCaseResult captures the outcome for a single fixture, variant and repeat.
type BboxCaseResult struct {
	ID                string                       `json:"id"`
	Variant           string                       `json:"variant,omitempty"`
	Repeat            int                          `json:"repeat"`
	Tier              string                       `json:"tier"`
	PairID            string                       `json:"pair_id,omitempty"`
	Target            string                       `json:"target"`
	Tags              map[string]string            `json:"tags,omitempty"`
	ObjectPresentGT   bool                         `json:"object_present_gt"`
	ObjectPresentPred bool                         `json:"object_present_pred"`
	PresenceAsked     bool                         `json:"presence_asked,omitempty"`
	OccludedEdge      string                       `json:"occluded_edge"`
	GTBox             [4]float64                   `json:"gt_box"`
	GTBox2            *[4]float64                  `json:"gt_box_2,omitempty"`
	ArgmaxBox         [4]float64                   `json:"argmax_box"`
	ExpectationBox    [4]float64                   `json:"expectation_box"`
	ArgmaxBox2        *[4]float64                  `json:"argmax_box_2,omitempty"`
	ExpectationBox2   *[4]float64                  `json:"expectation_box_2,omitempty"`
	ObjectsSwapped    bool                         `json:"objects_swapped,omitempty"`
	ClassHits         int                          `json:"class_hits,omitempty"`
	ClassTotal        int                          `json:"class_total,omitempty"`
	ArgmaxIoU         float64                      `json:"argmax_iou"`
	ExpectationIoU    float64                      `json:"expectation_iou"`
	VisibleIoU        *float64                     `json:"visible_iou,omitempty"`
	IoUGain           float64                      `json:"iou_gain"`
	PassAcc50         bool                         `json:"pass_acc_50"`
	PassAcc75         bool                         `json:"pass_acc_75"`
	InvalidBox        bool                         `json:"invalid_box,omitempty"`
	MissingSlots      []string                     `json:"missing_slots,omitempty"`
	MaxEdgeEntropy    float64                      `json:"max_edge_normalized_entropy"`
	HighestEntropyDir string                       `json:"highest_entropy_edge"`
	Edges             map[string]BboxEdgeTelemetry `json:"edges,omitempty"`
	Skipped           string                       `json:"skipped,omitempty"`
	Error             string                       `json:"error,omitempty"`
	WallTimeMs        float64                      `json:"wall_time_ms"`
}

// BboxBenchmarkReport summarizes a full run. The top-level metric fields describe the
// original variant (averaged over repeats) and keep the pre-2026-10 receipt shape.
type BboxBenchmarkReport struct {
	Timestamp             string               `json:"timestamp"`
	Mode                  string               `json:"mode"`
	TargetURL             string               `json:"target_url"`
	TargetModel           string               `json:"target_model"`
	ServerVersion         string               `json:"server_version,omitempty"`
	ServerRevision        string               `json:"server_revision,omitempty"`
	GitCommit             string               `json:"git_commit,omitempty"`
	Dataset               string               `json:"dataset,omitempty"`
	Repeats               int                  `json:"repeats,omitempty"`
	Samples               int                  `json:"samples,omitempty"`
	VariantsRequested     []string             `json:"variants_requested,omitempty"`
	SourceReceipt         string               `json:"source_receipt,omitempty"`
	TotalCases            int                  `json:"total_cases"`
	MeanArgmaxIoU         float64              `json:"mean_argmax_iou"`
	MeanExpectationIoU    float64              `json:"mean_expectation_iou"`
	SubBinIoUGainPct      float64              `json:"subbin_iou_gain_pct"`
	AccAt50ArgmaxPct      float64              `json:"acc_at_50_argmax_pct"`
	AccAt50ExpectationPct float64              `json:"acc_at_50_expectation_pct"`
	AccAt75ArgmaxPct      float64              `json:"acc_at_75_argmax_pct"`
	AccAt75ExpectationPct float64              `json:"acc_at_75_expectation_pct"`
	VisibleEdgeMeanHNorm  float64              `json:"visible_edge_mean_h_norm"`
	OccludedEdgeMeanHNorm float64              `json:"occluded_edge_mean_h_norm"`
	OcclusionEntropyRatio float64              `json:"occlusion_entropy_multiplier"`
	Baselines             []BboxBaseline       `json:"baselines,omitempty"`
	Breakdowns            []BboxTagBreakdown   `json:"breakdowns,omitempty"`
	Variants              []BboxVariantSummary `json:"variants,omitempty"`
	Cases                 []BboxCaseResult     `json:"cases"`
}

var bins5Pct = []string{
	"00", "05", "10", "15", "20", "25", "30", "35", "40", "45",
	"50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100",
}

func compute2DIoU(a, b [4]float64) float64 {
	interYMin := math.Max(a[0], b[0])
	interXMin := math.Max(a[1], b[1])
	interYMax := math.Min(a[2], b[2])
	interXMax := math.Min(a[3], b[3])

	interH := math.Max(0.0, interYMax-interYMin)
	interW := math.Max(0.0, interXMax-interXMin)
	interArea := interH * interW

	areaA := math.Max(0.0, a[2]-a[0]) * math.Max(0.0, a[3]-a[1])
	areaB := math.Max(0.0, b[2]-b[0]) * math.Max(0.0, b[3]-b[1])
	union := areaA + areaB - interArea
	if union <= 0 {
		return 0.0
	}
	return interArea / union
}

// evaluateCoordinateDistribution turns one coordinate slot's label distribution into an
// argmax coordinate, an expected coordinate and entropy normalized by the slot's label count.
func evaluateCoordinateDistribution(probs map[string]float64, fallbackChoice string, cq coordQuestion) BboxEdgeTelemetry {
	k := cq.K
	if k < 2 {
		k = len(bins5Pct)
	}
	value := func(label string) (float64, bool) {
		if v, ok := cq.Values[label]; ok {
			return v, true
		}
		v, err := strconv.ParseFloat(label, 64)
		return v, err == nil
	}
	tel := BboxEdgeTelemetry{QuestionID: cq.ID, K: k, BinWidth: cq.BinWidth, LabelMin: cq.Min, LabelMax: cq.Max}
	if e, ok := coordEdgeOf(cq.ID); ok {
		tel.Edge = e
	}
	if len(probs) == 0 {
		v, _ := value(fallbackChoice)
		tel.ArgmaxBin, tel.ArgmaxCoord, tel.ExpectedCoord, tel.Confidence = fallbackChoice, v, v, 1
		tel.NoDistribution = true
		return tel
	}

	totalP := 0.0
	for label, p := range probs {
		if _, ok := value(label); ok {
			totalP += p
		}
	}
	if totalP <= 0 {
		totalP = 1.0
	}
	// Deterministic argmax tie-break: iterate labels in sorted order.
	labels := make([]string, 0, len(probs))
	for label := range probs {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	bestP := -1.0
	for _, label := range labels {
		v, ok := value(label)
		if !ok {
			continue
		}
		p := probs[label] / totalP
		tel.LabelsReturned++
		if p > bestP {
			bestP, tel.ArgmaxBin, tel.ArgmaxCoord = p, label, v
		}
		tel.ExpectedCoord += v * p
		if p > 1e-12 {
			tel.RawEntropyNats -= p * math.Log(p)
		}
	}
	tel.Confidence = bestP
	tel.NormalizedEntropy = tel.RawEntropyNats / math.Log(float64(k))
	return tel
}

// simulateEdgeProbs returns a reference distribution over a slot's labels: a tight Gaussian
// around the ground truth for a visible edge, or mass spread over three label widths either
// side for an occluded one.
func simulateEdgeProbs(gtCoord float64, isOccluded bool, cq coordQuestion) map[string]float64 {
	w := cq.BinWidth
	if w <= 0 {
		w = 5
	}
	probs := make(map[string]float64, len(cq.Values))
	nearest, best := "", math.Inf(1)
	for name, v := range cq.Values {
		if d := math.Abs(v - gtCoord); d < best || (d == best && name < nearest) {
			nearest, best = name, d
		}
	}
	sigma := 0.52 * w
	for name, v := range cq.Values {
		d := v - gtCoord
		if isOccluded {
			if math.Abs(d) <= 3*w {
				probs[name] = 0.155
			} else {
				probs[name] = 0.003
			}
			continue
		}
		p := math.Exp(-(d * d) / (2 * sigma * sigma))
		if name == nearest {
			p *= 1.08
		}
		probs[name] = p
	}
	return probs
}

// defaultCoordQuestion is the 21-label 5% grid of the single-object template.
func defaultCoordQuestion(id string) coordQuestion {
	cq := coordQuestion{ID: id, Values: map[string]float64{}, K: len(bins5Pct), BinWidth: 5, Min: 0, Max: 100}
	for _, b := range bins5Pct {
		v, _ := strconv.ParseFloat(b, 64)
		cq.Values[b] = v
	}
	return cq
}

func bboxPrefixes(cqs map[string]coordQuestion) []string {
	if _, ok := cqs["obj1_ymin"]; ok {
		if _, ok2 := cqs["obj2_ymin"]; ok2 {
			return []string{"obj1_", "obj2_"}
		}
		return []string{"obj1_"}
	}
	return []string{""}
}

func boxValid(b [4]float64) bool { return b[2] > b[0] && b[3] > b[1] }

// scoreBboxAnswers scores one response. gt boxes are already in the variant's frame. A slot
// the model did not return makes its box invalid (IoU 0); nothing is filled from ground truth.
func scoreBboxAnswers(item BboxSuiteItem, variant string, answers map[string]client.QuestionAnswer, cqs map[string]coordQuestion, hasPresence bool) BboxCaseResult {
	gt1 := transformBoxForVariant(item.GTBoxContinuous, variant)
	var gt2 *[4]float64
	if item.SecondaryBoxContinuous != nil {
		b := transformBoxForVariant(*item.SecondaryBoxContinuous, variant)
		gt2 = &b
	}
	r := BboxCaseResult{
		ID: item.ID, Variant: variant, Tier: item.Tier, PairID: item.PairID, Target: item.Target, Tags: item.Tags,
		ObjectPresentGT: item.ObjectPresent, OccludedEdge: item.OccludedEdge, GTBox: gt1, GTBox2: gt2,
		PresenceAsked: hasPresence, Edges: map[string]BboxEdgeTelemetry{},
	}
	if r.OccludedEdge == "" {
		r.OccludedEdge = "none"
	}

	r.ObjectPresentPred = true
	if hasPresence {
		if ans, ok := answers["object_present"]; ok {
			v := strings.ToLower(ans.DisplayValue())
			r.ObjectPresentPred = v == "yes" || v == "true"
		} else {
			r.ObjectPresentPred = false
			r.MissingSlots = append(r.MissingSlots, "object_present")
		}
	}

	prefixes := bboxPrefixes(cqs)
	argBoxes := make([][4]float64, len(prefixes))
	expBoxes := make([][4]float64, len(prefixes))
	complete := make([]bool, len(prefixes))
	for pi, p := range prefixes {
		complete[pi] = true
		for ei, e := range bboxEdges {
			qid := p + e
			cq, ok := cqs[qid]
			if !ok {
				cq = defaultCoordQuestion(qid)
			}
			ans, ok := answers[qid]
			if !ok {
				r.MissingSlots = append(r.MissingSlots, qid)
				r.Edges[qid] = BboxEdgeTelemetry{QuestionID: qid, Edge: e, Missing: true}
				complete[pi] = false
				continue
			}
			tel := evaluateCoordinateDistribution(ans.Probabilities, ans.DisplayValue(), cq)
			r.Edges[qid] = tel
			argBoxes[pi][ei] = tel.ArgmaxCoord
			expBoxes[pi][ei] = tel.ExpectedCoord
		}
		if !complete[pi] {
			argBoxes[pi], expBoxes[pi] = [4]float64{}, [4]float64{}
		}
	}

	// Match predicted objects to ground-truth objects (order-free, by expectation IoU).
	gts := [][4]float64{gt1}
	if gt2 != nil && len(prefixes) > 1 {
		gts = append(gts, *gt2)
	}
	assign := []int{0, 1}[:len(prefixes)]
	if len(prefixes) == 2 && len(gts) == 2 {
		straight := compute2DIoU(expBoxes[0], gts[0]) + compute2DIoU(expBoxes[1], gts[1])
		swapped := compute2DIoU(expBoxes[0], gts[1]) + compute2DIoU(expBoxes[1], gts[0])
		if swapped > straight {
			assign = []int{1, 0}
			r.ObjectsSwapped = true
		}
	}

	var argIoUs, expIoUs []float64
	for pi, p := range prefixes {
		gi := assign[pi]
		if gi >= len(gts) {
			continue
		}
		for ei, e := range bboxEdges {
			tel := r.Edges[p+e]
			tel.GTCoord, tel.HasGT = gts[gi][ei], item.ObjectPresent
			r.Edges[p+e] = tel
		}
		if !boxValid(argBoxes[pi]) || !complete[pi] {
			r.InvalidBox = true
		}
		a, x := 0.0, 0.0
		if r.ObjectPresentPred && complete[pi] {
			a, x = compute2DIoU(gts[gi], argBoxes[pi]), compute2DIoU(gts[gi], expBoxes[pi])
		}
		argIoUs = append(argIoUs, a)
		expIoUs = append(expIoUs, x)
		if want, ok := item.ExpectedDiscreteSlots[fmt.Sprintf("obj%d_class", gi+1)].(string); ok {
			if ans, ok := answers[p+"class"]; ok {
				r.ClassTotal++
				if ans.DisplayValue() == want {
					r.ClassHits++
				}
			}
		}
	}
	// Coordinates are not required once the model answered "absent" (Gemini returns none then).
	if hasPresence && !r.ObjectPresentPred {
		kept := r.MissingSlots[:0]
		for _, m := range r.MissingSlots {
			if _, isCoord := coordEdgeOf(m); !isCoord {
				kept = append(kept, m)
			}
		}
		r.MissingSlots = kept
	}
	// Boxes are the raw coordinate readout, kept even when the model answers "absent" (the
	// blank-image prior baseline needs them); IoU is 0 whenever presence is answered "no".
	r.ArgmaxBox, r.ExpectationBox = argBoxes[0], expBoxes[0]
	if len(prefixes) > 1 {
		a, x := argBoxes[1], expBoxes[1]
		r.ArgmaxBox2, r.ExpectationBox2 = &a, &x
	}
	if item.ObjectPresent {
		r.ArgmaxIoU, r.ExpectationIoU = mean(argIoUs), mean(expIoUs)
		r.IoUGain = r.ExpectationIoU - r.ArgmaxIoU
		r.PassAcc50, r.PassAcc75 = r.ExpectationIoU >= 0.5, r.ExpectationIoU >= 0.75
		if item.VisibleBoxContinuous != nil && len(prefixes) == 1 {
			v := 0.0
			if r.ObjectPresentPred && complete[0] {
				v = compute2DIoU(transformBoxForVariant(*item.VisibleBoxContinuous, variant), expBoxes[0])
			}
			r.VisibleIoU = &v
		}
	} else {
		r.InvalidBox = false
	}
	r.HighestEntropyDir = "none"
	for _, e := range bboxEdges {
		if tel, ok := r.Edges[prefixes[0]+e]; ok && !tel.Missing && tel.NormalizedEntropy > r.MaxEdgeEntropy {
			r.MaxEdgeEntropy, r.HighestEntropyDir = tel.NormalizedEntropy, e
		}
	}
	return r
}

// simulatedBboxAnswers builds reference answers for --simulate, run through the same scorer.
func simulatedBboxAnswers(item BboxSuiteItem, variant string, cqs map[string]coordQuestion, hasPresence bool) map[string]client.QuestionAnswer {
	answers := map[string]client.QuestionAnswer{}
	if hasPresence {
		lbl := "no"
		if item.ObjectPresent && variant != variantBlank {
			lbl = "yes"
		}
		answers["object_present"] = client.QuestionAnswer{Type: "noul", Label: lbl}
	}
	gt1 := transformBoxForVariant(item.GTBoxContinuous, variant)
	gt2 := gt1
	if item.SecondaryBoxContinuous != nil {
		gt2 = transformBoxForVariant(*item.SecondaryBoxContinuous, variant)
	}
	for qid, cq := range cqs {
		e, _ := coordEdgeOf(qid)
		ei := 0
		for i, name := range bboxEdges {
			if name == e {
				ei = i
			}
		}
		gt := gt1
		if strings.HasPrefix(qid, "obj2_") {
			gt = gt2
		}
		var probs map[string]float64
		if variant == variantBlank {
			probs = map[string]float64{}
			for name := range cq.Values {
				probs[name] = 1
			}
		} else {
			occluded := item.OccludedEdge == e && !strings.HasPrefix(qid, "obj2_")
			if variantSwapsObjectOrder(variant) && item.OccludedEdge != "none" && item.OccludedEdge != "" {
				occluded = transformEdgeName(item.OccludedEdge, variant) == e && !strings.HasPrefix(qid, "obj2_")
			}
			probs = simulateEdgeProbs(gt[ei], occluded, cq)
		}
		answers[qid] = client.QuestionAnswer{Type: "choice", Probabilities: probs}
	}
	return answers
}

// transformEdgeName maps an edge name through a flip (hflip swaps xmin/xmax, vflip ymin/ymax).
func transformEdgeName(edge, variant string) string {
	swap := map[string]map[string]string{
		variantHFlip: {"xmin": "xmax", "xmax": "xmin"},
		variantVFlip: {"ymin": "ymax", "ymax": "ymin"},
	}
	if m, ok := swap[variant]; ok {
		if v, ok := m[edge]; ok {
			return v
		}
	}
	return edge
}

func schemaHasQuestion(schemaJSON, id string) bool {
	var m struct {
		Questions []struct {
			ID string `json:"id"`
		} `json:"questions"`
	}
	if json.Unmarshal([]byte(schemaJSON), &m) != nil {
		return false
	}
	for _, q := range m.Questions {
		if q.ID == id {
			return true
		}
	}
	return false
}

func fetchBboxServerInfo(c *client.Client) (string, string) {
	// Base URLs are ".../v1" (Cloud Run, local) or ".../invoke/v1/chat/completions" (Vertex).
	u := strings.TrimSuffix(strings.TrimRight(c.BaseURL, "/"), "/chat/completions")
	u = strings.TrimSuffix(u, "/v1") + "/health"
	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return "", ""
	}
	if c.AuthToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}
	hc := &http.Client{Timeout: 15 * time.Second}
	resp, err := hc.Do(req)
	if err != nil {
		return "", ""
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var h map[string]interface{}
	if json.Unmarshal(b, &h) != nil {
		return "", ""
	}
	str := func(k string) string {
		if v, ok := h[k].(string); ok {
			return v
		}
		return ""
	}
	return str("version"), str("revision")
}

func gitShortCommit() string {
	out, err := exec.Command("git", "rev-parse", "--short=10", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func loadBboxSuite() ([]BboxSuiteItem, error) {
	var suite []BboxSuiteItem
	if bboxDir != "" {
		return loadCustomDirSuite(bboxDir, bboxTarget)
	}
	f, err := os.Open(bboxDataset)
	if err != nil {
		return nil, fmt.Errorf("failed to open bbox suite %s (hint: run `python3 scripts/generate_bbox_fixtures.py` first): %w", bboxDataset, err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var item BboxSuiteItem
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return nil, fmt.Errorf("failed to parse JSONL line: %w", err)
		}
		suite = append(suite, item)
	}
	return suite, scanner.Err()
}

type bboxJob struct {
	idx     int
	item    BboxSuiteItem
	variant string
	repeat  int
}

func runBboxJob(ctx context.Context, c *client.Client, engine *template.Engine, job bboxJob, variantDir string) BboxCaseResult {
	item := job.item
	base := BboxCaseResult{ID: item.ID, Variant: job.variant, Repeat: job.repeat, Tier: item.Tier, PairID: item.PairID,
		Target: item.Target, Tags: item.Tags, ObjectPresentGT: item.ObjectPresent, OccludedEdge: item.OccludedEdge,
		GTBox: transformBoxForVariant(item.GTBoxContinuous, job.variant)}
	start := time.Now()
	vars := map[string]interface{}{"target": item.Target, "samples": bboxSamples}
	rendered, err := engine.RenderFile(item.Template, vars)
	if err != nil {
		base.Error = "render: " + err.Error()
		return base
	}
	schemaContent, stateContent, err := template.ParseStructuredPayload(rendered, vars)
	if err != nil {
		base.Error = "parse: " + err.Error()
		return base
	}
	schemaContent, err = applySchemaVariant(schemaContent, job.variant)
	if err == errVariantNotApplicable {
		base.Skipped = "variant not applicable to template"
		return base
	} else if err != nil {
		base.Error = "variant: " + err.Error()
		return base
	}
	cqs, err := coordQuestionsFromSchema(schemaContent)
	if err != nil {
		base.Error = "schema: " + err.Error()
		return base
	}
	hasPresence := schemaHasQuestion(schemaContent, "object_present")

	var answers map[string]client.QuestionAnswer
	if bboxEngine == "predictions" {
		if isSchemaVariant(job.variant) || job.variant != variantOriginal {
			base.Skipped = "predictions engine scores the original variant only"
			return base
		}
		answers = predictionAnswers(item, hasPresence)
		if answers == nil {
			base.Error = "no prediction for case"
			return base
		}
	} else if bboxEngine == "gemini" {
		if isSchemaVariant(job.variant) {
			base.Skipped = "schema variant not applicable to gemini"
			return base
		}
		img, err := materializeImageVariant(item.ImagePath, job.variant, variantDir)
		if err != nil {
			base.Error = "image: " + err.Error()
			return base
		}
		answers, _, err = geminiBboxAnswers(ctx, SanitizeCascadeModel(bboxGeminiModel), item, img, schemaContent, cqs, hasPresence)
		if err != nil {
			base.Error = err.Error()
			return base
		}
	} else if bboxSimulate {
		answers = simulatedBboxAnswers(item, job.variant, cqs, hasPresence)
	} else {
		img, err := materializeImageVariant(item.ImagePath, job.variant, variantDir)
		if err != nil {
			base.Error = "image: " + err.Error()
			return base
		}
		resp, _, err := c.Decide(ctx, schemaContent, stateContent, img)
		if err != nil {
			base.Error = err.Error()
			return base
		}
		answers = resp.Answers
	}
	r := scoreBboxAnswers(item, job.variant, answers, cqs, hasPresence)
	r.Repeat = job.repeat
	r.WallTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	return r
}

func runBenchBbox(cmd *cobra.Command, args []string) error {
	suite, err := loadBboxSuite()
	if err != nil {
		return err
	}
	if bboxLimit > 0 && bboxLimit < len(suite) {
		suite = suite[:bboxLimit]
	}
	if bboxFromReceipt != "" {
		return reanalyzeBboxReceipt(bboxFromReceipt, suite)
	}
	variants, err := parseBboxVariants(bboxVariants)
	if err != nil {
		return err
	}
	if bboxRepeat < 1 {
		bboxRepeat = 1
	}
	variantDir := bboxVariantDir
	if variantDir == "" {
		variantDir = filepath.Join(os.TempDir(), "dgem-bbox-variants")
	}

	if bboxEngine == "predictions" {
		if err := loadPredictions(bboxPredFile, bboxPredModel); err != nil {
			return err
		}
	}
	c := GetClient()
	if c.MaxRetries < 2 {
		c.MaxRetries = 2
	}
	engine := template.NewEngine()
	ctx := context.Background()

	var jobs []bboxJob
	for _, v := range variants {
		for rep := 0; rep < bboxRepeat; rep++ {
			for _, item := range suite {
				jobs = append(jobs, bboxJob{idx: len(jobs), item: item, variant: v, repeat: rep})
			}
		}
	}
	results := make([]BboxCaseResult, len(jobs))
	workers := bboxWorkers
	if workers < 1 {
		workers = 1
	}
	ch := make(chan bboxJob)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range ch {
				res := runBboxJob(ctx, c, engine, job, variantDir)
				results[job.idx] = res
				mu.Lock()
				done++
				if !bboxJSON && (done%25 == 0 || done == len(jobs)) {
					fmt.Fprintf(os.Stderr, "  bench-bbox: %d/%d requests\n", done, len(jobs))
				}
				mu.Unlock()
			}
		}()
	}
	for _, j := range jobs {
		ch <- j
	}
	close(ch)
	wg.Wait()

	errCount := 0
	for i, r := range results {
		if r.Error != "" {
			errCount++
			if errCount <= 3 {
				fmt.Fprintf(os.Stderr, "  error on %s/%s#%d: %s\n", r.ID, r.Variant, r.Repeat, r.Error)
			}
		}
		if bboxAnnotate && r.Variant == variantOriginal && r.Repeat == 0 && r.Error == "" && jobs[i].item.ImagePath != "" {
			_, _ = writeAnnotatedOverlay(jobs[i].item.ImagePath, r)
		}
	}
	if errCount == len(results) {
		return fmt.Errorf("all %d requests failed (hint: pass --simulate for offline verification or check -u / --vertex-url): %s", errCount, results[0].Error)
	}

	mode := "live"
	if bboxSimulate {
		mode = "simulated_reference"
	}
	model, targetURL := c.Model, c.BaseURL
	if bboxEngine == "gemini" {
		mode, model, targetURL = "gemini_reference", SanitizeCascadeModel(bboxGeminiModel), "vertex-gemini"
	}
	if bboxEngine == "predictions" {
		mode, model, targetURL = "offline_predictions", bboxPredModel, bboxPredFile
	}
	report := BboxBenchmarkReport{
		Timestamp: time.Now().UTC().Format(time.RFC3339), Mode: mode, TargetURL: targetURL, TargetModel: model,
		GitCommit: gitShortCommit(), Dataset: bboxDataset, Repeats: bboxRepeat, Samples: bboxSamples,
		VariantsRequested: variants, Cases: results,
	}
	if bboxDir != "" {
		report.Dataset = bboxDir
	}
	if !bboxSimulate && bboxEngine == "dgem" {
		report.ServerVersion, report.ServerRevision = fetchBboxServerInfo(c)
	}
	finalizeBboxReport(&report, variants)
	return emitBboxReport(report)
}

// finalizeBboxReport fills the variant summaries, baselines and legacy top-level fields.
func finalizeBboxReport(report *BboxBenchmarkReport, variants []string) {
	original := map[string]BboxCaseResult{}
	for _, r := range report.Cases {
		if r.Variant == variantOriginal && r.Error == "" && r.Skipped == "" {
			original[bboxRunKey(r.ID, r.Repeat)] = r
		}
	}
	report.Variants = nil
	var blank *BboxVariantSummary
	for _, v := range variants {
		s := summarizeBboxVariant(v, report.Cases, original)
		report.Variants = append(report.Variants, s)
		if v == variantBlank {
			sc := s
			blank = &sc
		}
	}
	report.Baselines = computeBboxBaselines(report.Cases, blank)
	report.Breakdowns = computeBboxBreakdowns(report.Cases)

	if len(report.Variants) > 0 && report.Variants[0].Variant == variantOriginal {
		o := report.Variants[0]
		report.TotalCases = o.Cases
		report.MeanArgmaxIoU = o.MeanArgmaxIoU.Mean
		report.MeanExpectationIoU = o.MeanExpectationIoU.Mean
		report.SubBinIoUGainPct = (o.MeanExpectationIoU.Mean - o.MeanArgmaxIoU.Mean) * 100
		report.AccAt50ArgmaxPct, report.AccAt50ExpectationPct = o.AccAt50Argmax, o.AccAt50Expectation
		report.AccAt75ArgmaxPct, report.AccAt75ExpectationPct = o.AccAt75Argmax, o.AccAt75Expectation
	}
	var vis, occ []float64
	for _, r := range original {
		if !r.ObjectPresentGT || r.PairID == "" {
			continue
		}
		for _, e := range bboxEdges {
			tel, ok := r.Edges[e]
			if !ok || tel.Missing {
				continue
			}
			if r.OccludedEdge == e {
				occ = append(occ, tel.NormalizedEntropy)
			} else {
				vis = append(vis, tel.NormalizedEntropy)
			}
		}
	}
	report.VisibleEdgeMeanHNorm, report.OccludedEdgeMeanHNorm = mean(vis), mean(occ)
	report.OcclusionEntropyRatio = 0
	if report.VisibleEdgeMeanHNorm > 0 && len(occ) > 0 {
		report.OcclusionEntropyRatio = report.OccludedEdgeMeanHNorm / report.VisibleEdgeMeanHNorm
	}
}

// reanalyzeBboxReceipt recomputes summaries for an existing receipt. Receipts written before
// 2026-10 carry only original-variant cases with edges keyed by edge name; the ground-truth
// coordinate and label grid are restored from the suite.
func reanalyzeBboxReceipt(path string, suite []BboxSuiteItem) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var report BboxBenchmarkReport
	if err := json.Unmarshal(b, &report); err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	byID := map[string]BboxSuiteItem{}
	for _, it := range suite {
		byID[it.ID] = it
	}
	seen := map[string]bool{}
	var variants []string
	for i := range report.Cases {
		r := &report.Cases[i]
		if r.Variant == "" {
			r.Variant = variantOriginal
		}
		if !seen[r.Variant] {
			seen[r.Variant] = true
			variants = append(variants, r.Variant)
		}
		it, ok := byID[r.ID]
		if ok && r.PairID == "" {
			r.PairID = it.PairID
		}
		if !ok || len(r.Edges) == 0 {
			continue
		}
		band := strings.Contains(it.Template, "detr")
		for k, tel := range r.Edges {
			if tel.HasGT || tel.Missing {
				continue
			}
			e, isEdge := coordEdgeOf(k)
			if !isEdge {
				continue
			}
			ei := 0
			for j, name := range bboxEdges {
				if name == e {
					ei = j
				}
			}
			tel.Edge, tel.QuestionID = e, k
			tel.GTCoord, tel.HasGT = r.GTBox[ei], r.ObjectPresentGT
			if tel.BinWidth == 0 {
				// Legacy receipts parsed band labels as band starts on the DETR template.
				if band {
					tel.BinWidth, tel.LabelMin, tel.LabelMax, tel.K = 10, 0, 90, 10
				} else {
					tel.BinWidth, tel.LabelMin, tel.LabelMax, tel.K = 5, 0, 100, 21
				}
			}
			r.Edges[k] = tel
		}
		// Legacy absent cases were scored IoU 1; absent cases are excluded from box metrics.
	}
	sort.SliceStable(variants, func(i, j int) bool { return variants[i] == variantOriginal && variants[j] != variantOriginal })
	report.SourceReceipt = path
	finalizeBboxReport(&report, variants)
	return emitBboxReport(report)
}

func fmtStat(s BboxStat) string {
	if s.N == 0 {
		return "—"
	}
	return fmt.Sprintf("%.3f [%.3f, %.3f]", s.Mean, s.Lo, s.Hi)
}

func emitBboxReport(report BboxBenchmarkReport) error {
	if bboxOutput != "" {
		b, _ := json.MarshalIndent(report, "", "  ")
		if err := os.WriteFile(bboxOutput, b, 0644); err != nil {
			return fmt.Errorf("failed to write output receipt %s: %w", bboxOutput, err)
		}
	}
	if bboxJSON {
		b, _ := json.MarshalIndent(report, "", "  ")
		fmt.Println(string(b))
		return nil
	}
	printBboxReport(report)
	return nil
}

func printBboxReport(report BboxBenchmarkReport) {
	fmt.Println()
	fmt.Printf("Spatial Grounding Benchmark (%s)  target=%s  server=%s %s  repeats=%d\n",
		report.Mode, report.TargetModel, report.ServerVersion, report.ServerRevision, report.Repeats)
	fmt.Println(strings.Repeat("=", 128))

	fmt.Printf("%-34s | %-28s | %-10s | %-10s | %-9s | %-10s | %-12s\n",
		"CASE ID (original, repeat 0)", "E[BOX] [ymin,xmin,ymax,xmax]", "ARGMAX IoU", "EXPECT IoU", "IoU GAIN", "MAX H_NORM", "PEAK EDGE")
	fmt.Println(strings.Repeat("-", 128))
	for _, r := range report.Cases {
		if r.Variant != variantOriginal || r.Repeat != 0 {
			continue
		}
		if r.Error != "" || r.Skipped != "" {
			fmt.Printf("%-34s | %s%s\n", r.ID, r.Error, r.Skipped)
			continue
		}
		peak := r.HighestEntropyDir
		if r.OccludedEdge != "none" && r.HighestEntropyDir == r.OccludedEdge {
			peak += " (OCC)"
		}
		box := fmt.Sprintf("[%4.1f,%4.1f,%4.1f,%4.1f]", r.ExpectationBox[0], r.ExpectationBox[1], r.ExpectationBox[2], r.ExpectationBox[3])
		if !r.ObjectPresentGT {
			fmt.Printf("%-34s | %-28s | absent target: predicted present=%v\n", r.ID, box, r.ObjectPresentPred)
			continue
		}
		flag := ""
		if len(r.MissingSlots) > 0 {
			flag = " missing=" + strings.Join(r.MissingSlots, ",")
		}
		fmt.Printf("%-34s | %-28s | %-10.4f | %-10.4f | %+8.1f%% | %-10.4f | %-12s%s\n",
			r.ID, box, r.ArgmaxIoU, r.ExpectationIoU, r.IoUGain*100, r.MaxEdgeEntropy, peak, flag)
	}

	fmt.Println()
	fmt.Printf("%-17s | %-5s | %-24s | %-24s | %-6s | %-6s | %-6s | %-24s | %-7s | %-6s\n",
		"VARIANT", "CASES", "mIoU ARGMAX [95% CI]", "mIoU EXPECT [95% CI]", "@0.5", "@0.75", "CENTER", "CONSISTENCY vs ORIGINAL", "H AUROC", "MISS")
	fmt.Println(strings.Repeat("-", 128))
	for _, v := range report.Variants {
		cons := "—"
		if v.ConsistencyIoU != nil {
			cons = fmtStat(*v.ConsistencyIoU)
		}
		fmt.Printf("%-17s | %-5d | %-24s | %-24s | %5.1f%% | %5.1f%% | %5.1f%% | %-24s | %-7.3f | %-6d\n",
			v.Variant, v.BoxCases, fmtStat(v.MeanArgmaxIoU), fmtStat(v.MeanExpectationIoU),
			v.AccAt50Expectation, v.AccAt75Expectation, v.CenterHitPct, cons, v.EntropyErrorAUROC, v.MissingSlotRuns)
	}

	fmt.Println()
	fmt.Println("Signed bias of the expectation box by variant (pts; + = toward bottom/right), and how often presence was answered yes")
	for _, v := range report.Variants {
		line := fmt.Sprintf("  %-17s", v.Variant)
		for _, e := range bboxEdges {
			if st, ok := v.Edges[e]; ok {
				line += fmt.Sprintf(" %s %+6.2f", e, st.SignedBiasExpectation)
			}
		}
		if v.PredictedPresentRate != nil {
			line += fmt.Sprintf(" | present %.0f%%", *v.PredictedPresentRate*100)
		}
		fmt.Println(line)
	}

	if len(report.Baselines) > 0 {
		fmt.Println()
		fmt.Println("Image-free baselines (original variant; lift = model IoU − baseline IoU, paired by case)")
		for _, b := range report.Baselines {
			fmt.Printf("  %-18s mIoU %-24s Acc@0.5 %5.1f%% center %5.1f%% | lift expect %-24s lift argmax %s\n",
				b.Name, fmtStat(b.MeanIoU), b.AccAt50Pct, b.CenterHitPct, fmtStat(b.LiftExpectation), fmtStat(b.LiftArgmax))
		}
	}

	if len(report.Breakdowns) > 0 {
		fmt.Println()
		fmt.Println("Breakdown by tag (original variant, means over repeats; occ-edge H~ = entropy of the tagged edge)")
		fmt.Printf("  %-12s %-24s %5s  %-24s %6s %7s %8s %7s %8s\n", "TAG", "VALUE", "N", "mIoU EXPECT [95% CI]", "@0.5", "CENTER", "VIS IoU", "PRESENT", "EDGE H~")
		for _, b := range report.Breakdowns {
			vis, edgeH := "—", "—"
			if b.MeanVisibleIoU != nil {
				vis = fmt.Sprintf("%.3f", *b.MeanVisibleIoU)
			}
			if b.TaggedEdgeHNorm != nil {
				edgeH = fmt.Sprintf("%.3f", *b.TaggedEdgeHNorm)
			}
			miou := "—"
			if b.BoxCases > 0 {
				miou = fmtStat(b.MeanExpectationIoU)
			}
			fmt.Printf("  %-12s %-24s %5d  %-24s %5.1f%% %6.1f%% %8s %6.0f%% %8s\n", b.Tag, b.Value, b.Cases, miou, b.AccAt50Pct, b.CenterHitPct, vis, b.PresentRate*100, edgeH)
		}
	}

	if len(report.Variants) > 0 {
		o := report.Variants[0]
		fmt.Println()
		fmt.Printf("Edge error (%s): quantization floor %.2f pts; on-grid edges (n=%d) MAE argmax %.2f / expect %.2f; off-grid (n=%d) argmax %.2f / expect %.2f\n",
			o.Variant, o.Grid.QuantizationFloorMAE, o.Grid.OnGridN, o.Grid.OnGridMAEArgmax, o.Grid.OnGridMAEExpectation,
			o.Grid.OffGridN, o.Grid.OffGridMAEArgmax, o.Grid.OffGridMAEExpect)
		fmt.Printf("Expectation gain: on-grid cases %s | off-grid cases %s\n", fmtStat(o.ExpectationGainOnGrid), fmtStat(o.ExpectationGainOffGrid))
		for _, e := range bboxEdges {
			if st, ok := o.Edges[e]; ok {
				fmt.Printf("  %-5s signed bias expect %+6.2f argmax %+6.2f | MAE argmax %5.2f expect %5.2f | mean H~ %.3f (n=%d)\n",
					e, st.SignedBiasExpectation, st.SignedBiasArgmax, st.MAEArgmax, st.MAEExpectation, st.MeanHNorm, st.N)
			}
		}
		fmt.Printf("Entropy → edge error: AUROC %.3f (%d wrong / %d right edges), Spearman %.3f\n",
			o.EntropyErrorAUROC, o.EntropyErrorPositive, o.EntropyErrorNegative, o.EntropyErrorSpearman)
		for _, p := range o.PairedOcclusion {
			fmt.Printf("Occlusion %-22s edge %-4s H~ %.3f vs twin %.3f (Δ %+.3f; other edges Δ %+.3f; rank %d/4)\n",
				p.PairID, p.OccludedEdge, p.HOccluded, p.HTwin, p.Delta, p.OtherEdgesDelta, p.RankInCase)
		}
		if o.EdgeArgmaxAgreement != nil {
			fmt.Printf("Repeat stability: edge argmax agreement %.3f, pairwise expectation IoU %.3f\n", *o.EdgeArgmaxAgreement, *o.RepeatPairwiseIoU)
		}
		if o.AbsentRuns > 0 {
			fmt.Printf("Absent target: %d false positives in %d runs\n", o.FalsePositives, o.AbsentRuns)
		}
	}
	fmt.Println()
}

func loadCustomDirSuite(dirPath, defaultTarget string) ([]BboxSuiteItem, error) {
	manifestPath := filepath.Join(dirPath, "manifest.jsonl")
	if st, err := os.Stat(manifestPath); err == nil && !st.IsDir() {
		f, err := os.Open(manifestPath)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		var items []BboxSuiteItem
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			var it BboxSuiteItem
			if err := json.Unmarshal([]byte(line), &it); err != nil {
				return nil, err
			}
			if it.Template == "" {
				it.Template = "templates/multimodal/bbox_localization.json.tmpl"
			}
			if !filepath.IsAbs(it.ImagePath) && !strings.HasPrefix(it.ImagePath, dirPath) {
				it.ImagePath = filepath.Join(dirPath, it.ImagePath)
			}
			items = append(items, it)
		}
		return items, nil
	}

	indexTxtPath := filepath.Join(dirPath, "index.txt")
	if st, err := os.Stat(indexTxtPath); err == nil && !st.IsDir() {
		f, err := os.Open(indexTxtPath)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		var items []BboxSuiteItem
		sc := bufio.NewScanner(f)
		idx := 0
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			var imgName, question string
			if strings.Contains(line, "|") {
				parts := strings.SplitN(line, "|", 2)
				imgName, question = strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
			} else if strings.Contains(line, "\t") {
				parts := strings.SplitN(line, "\t", 2)
				imgName, question = strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
			} else if strings.Contains(line, ":") && !strings.HasPrefix(line, "http") {
				parts := strings.SplitN(line, ":", 2)
				imgName, question = strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
			} else {
				fields := strings.Fields(line)
				imgName = fields[0]
				if len(fields) > 1 {
					question = strings.TrimSpace(strings.TrimPrefix(line, imgName))
				}
			}
			if question == "" {
				question = defaultTarget
			}
			idx++
			base := strings.TrimSuffix(filepath.Base(imgName), filepath.Ext(imgName))
			imgPath := imgName
			if !filepath.IsAbs(imgPath) && !strings.HasPrefix(imgPath, dirPath) {
				imgPath = filepath.Join(dirPath, imgName)
			}
			items = append(items, BboxSuiteItem{
				ID:              fmt.Sprintf("%02d-%s", idx, base),
				Tier:            "custom_index_txt",
				Template:        "templates/multimodal/bbox_localization.json.tmpl",
				ImagePath:       imgPath,
				Target:          question,
				ObjectPresent:   true,
				GTBoxContinuous: [4]float64{25.0, 25.0, 75.0, 75.0},
				OccludedEdge:    "none",
			})
		}
		if len(items) > 0 {
			return items, nil
		}
	}

	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read custom image directory %s: %w", dirPath, err)
	}
	var imgFiles []string
	for _, e := range entries {
		if e.IsDir() || strings.HasPrefix(e.Name(), "annotated_") {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" {
			imgFiles = append(imgFiles, filepath.Join(dirPath, e.Name()))
		}
	}
	sort.Strings(imgFiles)
	if len(imgFiles) == 0 {
		return nil, fmt.Errorf("no .png/.jpg/.webp images found in %s", dirPath)
	}

	var suite []BboxSuiteItem
	for _, imgPath := range imgFiles {
		base := strings.TrimSuffix(filepath.Base(imgPath), filepath.Ext(imgPath))
		item := BboxSuiteItem{
			ID:              base,
			Tier:            "custom_dir",
			Template:        "templates/multimodal/bbox_localization.json.tmpl",
			ImagePath:       imgPath,
			Target:          defaultTarget,
			ObjectPresent:   true,
			GTBoxContinuous: [4]float64{25.0, 25.0, 75.0, 75.0},
			OccludedEdge:    "none",
		}
		// Optional per-image sidecar JSON: <image_stem>.json
		sidecarPath := filepath.Join(dirPath, base+".json")
		if b, err := os.ReadFile(sidecarPath); err == nil {
			_ = json.Unmarshal(b, &item)
		}
		suite = append(suite, item)
	}
	return suite, nil
}

func writeAnnotatedOverlay(imagePath string, res BboxCaseResult) (string, error) {
	imgBytes, err := os.ReadFile(imagePath)
	if err != nil {
		return "", err
	}
	b64 := base64.StdEncoding.EncodeToString(imgBytes)
	ext := strings.ToLower(filepath.Ext(imagePath))
	mime := "image/png"
	if ext == ".jpg" || ext == ".jpeg" {
		mime = "image/jpeg"
	} else if ext == ".webp" {
		mime = "image/webp"
	}

	ay1, ax1, ay2, ax2 := res.ArgmaxBox[0]*10, res.ArgmaxBox[1]*10, res.ArgmaxBox[2]*10, res.ArgmaxBox[3]*10
	ey1, ex1, ey2, ex2 := res.ExpectationBox[0]*10, res.ExpectationBox[1]*10, res.ExpectationBox[2]*10, res.ExpectationBox[3]*10

	svg := fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
  <image href="data:%s;base64,%s" x="0" y="0" width="1000" height="1000" preserveAspectRatio="none"/>
  <rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="#0ea5e9" stroke-width="4" stroke-dasharray="12,8"/>
  <rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="#10b981" stroke-width="5"/>
  <rect x="%.1f" y="%.1f" width="460" height="34" fill="#0f172a" opacity="0.85" rx="4"/>
  <text x="%.1f" y="%.1f" font-family="monospace" font-size="16" font-weight="bold" fill="#10b981">E[box]=[%.1f, %.1f, %.1f, %.1f] H_max=%.3f (%s)</text>
</svg>
`,
		mime, b64,
		ax1, ay1, math.Max(4, ax2-ax1), math.Max(4, ay2-ay1),
		ex1, ey1, math.Max(4, ex2-ex1), math.Max(4, ey2-ey1),
		ex1, math.Max(4, ey1-38),
		ex1+10, math.Max(26, ey1-16),
		res.ExpectationBox[0], res.ExpectationBox[1], res.ExpectationBox[2], res.ExpectationBox[3],
		res.MaxEdgeEntropy, res.HighestEntropyDir,
	)

	outDir := filepath.Dir(imagePath)
	base := strings.TrimSuffix(filepath.Base(imagePath), filepath.Ext(imagePath))
	outPath := filepath.Join(outDir, "annotated_"+base+".svg")
	return outPath, os.WriteFile(outPath, []byte(svg), 0644)
}

// detectorPrediction is one row of an offline predictions file (one box per case and model).
type detectorPrediction struct {
	ID        string     `json:"id"`
	Model     string     `json:"model"`
	BoxPct    [4]float64 `json:"box_pct"`
	Score     float64    `json:"score"`
	LatencyMs float64    `json:"latency_ms"`
	Found     bool       `json:"found"`
}

var loadedPredictions map[string]detectorPrediction

func loadPredictions(path, model string) error {
	if path == "" || model == "" {
		return fmt.Errorf("--engine predictions needs --predictions and --pred-model")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	loadedPredictions = map[string]detectorPrediction{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var p detectorPrediction
		if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
			return err
		}
		if p.Model == model {
			loadedPredictions[p.ID] = p
		}
	}
	if len(loadedPredictions) == 0 {
		return fmt.Errorf("no rows for model %q in %s", model, path)
	}
	return sc.Err()
}

func predictionAnswers(item BboxSuiteItem, hasPresence bool) map[string]client.QuestionAnswer {
	p, ok := loadedPredictions[item.ID]
	if !ok {
		return nil
	}
	present := p.Found && p.Score >= bboxPredThresh
	answers := map[string]client.QuestionAnswer{}
	if hasPresence {
		lbl := "no"
		if present {
			lbl = "yes"
		}
		answers["object_present"] = client.QuestionAnswer{Type: "noul", Label: lbl}
	}
	if p.Found {
		for i, e := range bboxEdges {
			answers[e] = client.QuestionAnswer{Type: "choice", Probabilities: map[string]float64{strconv.FormatFloat(p.BoxPct[i], 'f', 2, 64): 1}}
		}
	}
	return answers
}
