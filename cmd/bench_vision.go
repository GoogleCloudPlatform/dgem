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

// multi-aspect image decisions. One template asks several typed questions about
// an image in one pass (presence, 3x3 grid cell, element count, spatial relation, image quality,
// occlusion); every item's manifest row carries ground-truth answers for the aspects that apply.
// Each aspect is scored against a majority-class baseline, with hesitation-to-error AUROC, and the
// same questions can be put to a Gemini 3.x reference (--engine gemini).

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
	"google.golang.org/genai"
)

var (
	visionDataset  string
	visionTemplate string
	visionOutput   string
	visionEngine   string
	visionGemini   string
	visionRepeat   int
	visionWorkers  int
	visionLimit    int
	visionVariant  string
	visionCascadeT float64
	visionCascadeM string
	visionCascadeP string
	visionOnlyGT   bool
)

var benchVisionCmd = &cobra.Command{
	Use:     "bench-vision",
	GroupID: "eval",
	Short:   "Score multi-aspect image decisions against ground truth, a majority baseline and Gemini",
	Long: `bench-vision asks every question of a multi-aspect template about each image in one pass and scores each aspect
that has a ground-truth answer in the manifest row ("aspects": {question_id: label}). Per aspect it reports accuracy
with a case-level bootstrap interval, the majority-class baseline and the paired lift over it, how often each label
is predicted, and whether hesitation (normalized entropy) ranks wrong answers above right ones (AUROC).
--engine gemini puts the same questions to a Gemini 3.x model with the options as an enum.
--variant blank sends a blank image of the same size: the model's prompt prior for every aspect.`,
	Example: `  dgem bench-vision --vertex-url <ENDPOINT_ID> -d benchmarks/bbox_sweep.jsonl --repeat 2 -o vision.json
  dgem bench-vision --engine gemini --gemini-model gemini-3.8-flash -d benchmarks/bbox_sweep.jsonl -o vision_gemini.json`,
	RunE: runBenchVision,
}

func init() {
	f := benchVisionCmd.Flags()
	f.StringVarP(&visionDataset, "dataset", "d", "benchmarks/bbox_sweep.jsonl", "Manifest with per-item \"aspects\" ground truth")
	f.StringVarP(&visionTemplate, "template", "t", "templates/multimodal/vision_aspects.json.tmpl", "Multi-aspect template")
	f.StringVarP(&visionOutput, "output", "o", "", "Write the receipt to this file")
	f.StringVar(&visionEngine, "engine", "dgem", "dgem or gemini")
	f.StringVar(&visionGemini, "gemini-model", DefaultCascadeGeminiModel, "Gemini 3.x model for --engine gemini")
	f.IntVar(&visionRepeat, "repeat", 1, "Runs per item")
	f.IntVarP(&visionWorkers, "workers", "w", 8, "Concurrent requests")
	f.IntVarP(&visionLimit, "limit", "n", 0, "Limit items (0 = all)")
	f.StringVar(&visionVariant, "variant", "original", "original or blank (a blank image of the same size)")
	f.Float64Var(&visionCascadeT, "cascade-threshold", 0, "Live Stage-2 cascade (engine dgem): send slots with entropy >= this many nats, with the image, to Gemini (0 = off)")
	f.BoolVar(&visionOnlyGT, "only-scored", false, "Ask only the questions the item has ground truth for (a caller asking just what it needs)")
	f.StringVar(&visionCascadeM, "cascade-model", DefaultCascadeGeminiModel, "Gemini 3.x model for --cascade-threshold")
	f.StringVar(&visionCascadeP, "cascade-prior", "soft", "How Stage 1's answer is shown to Gemini: full, soft or none (issue #75)")
	RootCmd.AddCommand(benchVisionCmd)
}

type visionItem struct {
	ID        string            `json:"id"`
	ImagePath string            `json:"image_path"`
	Target    string            `json:"target"`
	Reference string            `json:"reference,omitempty"`
	Aspects   map[string]string `json:"aspects"`
	Tags      map[string]string `json:"tags,omitempty"`
}

// VisionAnswer is one scored aspect of one run.
type VisionAnswer struct {
	Aspect    string `json:"aspect"`
	Expected  string `json:"expected"`
	Predicted string `json:"predicted"`
	Correct   bool   `json:"correct"`
	// Live cascade (--cascade-threshold): dgem's own answer and whether Gemini replaced it.
	Stage1     string  `json:"stage1,omitempty"`
	Stage1OK   *bool   `json:"stage1_correct,omitempty"`
	Escalated  bool    `json:"escalated,omitempty"`
	Confidence float64 `json:"confidence,omitempty"`
	HNorm      float64 `json:"h_norm,omitempty"`
	HasDist    bool    `json:"has_distribution,omitempty"`
	// Probabilities over the aspect's labels (dgem only), for Brier/ECE in the regression matrix.
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
}

// VisionItemRow is one (case, aspect, repeat) answer in the flat shape the regression matrix reads
// (scripts/matrix/metrics.py rows()).
type VisionItemRow struct {
	ID               string             `json:"id"`
	Expected         string             `json:"expected"`
	Actual           string             `json:"actual"`
	Accurate         bool               `json:"accurate"`
	Confidence       float64            `json:"confidence"`
	TopProbabilities map[string]float64 `json:"top_probabilities,omitempty"`
	WallTimeMs       float64            `json:"wall_time_ms"`
}

type VisionCaseResult struct {
	ID         string            `json:"id"`
	Repeat     int               `json:"repeat"`
	Tags       map[string]string `json:"tags,omitempty"`
	Answers    []VisionAnswer    `json:"answers"`
	Error      string            `json:"error,omitempty"`
	WallTimeMs float64           `json:"wall_time_ms"`
	// Live cascade timing: dgem's part, Gemini's part (0 when nothing escalated) and slots escalated.
	Stage1Ms       float64 `json:"stage1_ms,omitempty"`
	CascadeMs      float64 `json:"cascade_ms,omitempty"`
	EscalatedSlots int     `json:"escalated_slots,omitempty"`
	TotalSlots     int     `json:"total_slots,omitempty"`
	CascadeError   string  `json:"cascade_error,omitempty"`
}

type VisionAspectSummary struct {
	Aspect           string             `json:"aspect"`
	Items            int                `json:"items"`
	Options          int                `json:"options"`
	Accuracy         BboxStat           `json:"accuracy"`
	MajorityLabel    string             `json:"majority_label"`
	MajorityAccuracy float64            `json:"majority_accuracy"`
	Lift             BboxStat           `json:"lift_over_majority"`
	PredictedDist    map[string]float64 `json:"predicted_label_share"`
	HesitationAUROC  float64            `json:"hesitation_error_auroc"` // -1 when undefined
	Wrong            int                `json:"wrong_answers"`
	// BySet: accuracy per value of the "set" tag.
	BySet map[string]float64 `json:"accuracy_by_set,omitempty"`
}

type VisionReport struct {
	Timestamp     string                `json:"timestamp"`
	Engine        string                `json:"engine"`
	TargetModel   string                `json:"target_model"`
	ServerVersion string                `json:"server_version,omitempty"`
	Dataset       string                `json:"dataset"`
	Template      string                `json:"template"`
	Variant       string                `json:"variant"`
	Repeats       int                   `json:"repeats"`
	GitCommit     string                `json:"git_commit,omitempty"`
	Aspects       []VisionAspectSummary `json:"aspects"`
	Cases         []VisionCaseResult    `json:"cases"`
	// ReceiptKind and Items let generic tooling (the regression matrix) score the receipt per answer.
	ReceiptKind string          `json:"receipt_kind"`
	Items       []VisionItemRow `json:"items"`
}

type visionQuestion struct {
	ID           string
	Type         string
	Instructions string
	Options      []string
}

func visionQuestions(schemaJSON string) ([]visionQuestion, string) {
	var m struct {
		Instructions string `json:"instructions"`
		Questions    []struct {
			ID           string `json:"id"`
			Type         string `json:"type"`
			Instructions string `json:"instructions"`
			Options      []struct {
				Name string `json:"name"`
			} `json:"options"`
		} `json:"questions"`
	}
	_ = json.Unmarshal([]byte(schemaJSON), &m)
	var out []visionQuestion
	for _, q := range m.Questions {
		vq := visionQuestion{ID: q.ID, Type: q.Type, Instructions: q.Instructions}
		if q.Type == "boolean" || q.Type == "noul" || q.Type == "bool" {
			vq.Options = []string{"yes", "no"}
		}
		for _, o := range q.Options {
			vq.Options = append(vq.Options, o.Name)
		}
		out = append(out, vq)
	}
	return out, m.Instructions
}

func normalizeYesNo(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "true", "yes":
		return "yes"
	case "false", "no":
		return "no"
	}
	return v
}

// dgemVisionAnswer reads one answer's label, top probability and normalized entropy.
func dgemVisionAnswer(qa client.QuestionAnswer, nOptions int) (string, float64, float64, bool) {
	label := normalizeYesNo(qa.DisplayValue())
	if len(qa.Probabilities) == 0 || nOptions < 2 {
		return label, qa.Confidence, 0, false
	}
	tot, h, best := 0.0, 0.0, 0.0
	for _, p := range qa.Probabilities {
		tot += p
	}
	for _, p := range qa.Probabilities {
		p /= tot
		if p > best {
			best = p
		}
		if p > 1e-12 {
			h -= p * math.Log(p)
		}
	}
	return label, best, h / math.Log(float64(nOptions)), true
}

func geminiVisionAnswers(ctx context.Context, model, img string, qs []visionQuestion, preamble, stateJSON string) (map[string]string, error) {
	props := map[string]*genai.Schema{}
	var req, order []string
	var sb strings.Builder
	sb.WriteString(preamble + "\nState: " + stateJSON + "\nAnswer every question with one of its allowed values:\n")
	for _, q := range qs {
		props[q.ID] = &genai.Schema{Type: genai.TypeString, Enum: q.Options}
		req = append(req, q.ID)
		order = append(order, q.ID)
		fmt.Fprintf(&sb, "- %s: %s (one of: %s)\n", q.ID, q.Instructions, strings.Join(q.Options, ", "))
	}
	schema := &genai.Schema{Type: genai.TypeObject, Properties: props, Required: req, PropertyOrdering: order}
	out := map[string]string{}
	_, err := callGeminiJSON(ctx, model, img, sb.String(), schema, &out)
	return out, err
}

func runBenchVision(cmd *cobra.Command, args []string) error {
	f, err := os.Open(visionDataset)
	if err != nil {
		return err
	}
	var items []visionItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}
		var it visionItem
		if err := json.Unmarshal(sc.Bytes(), &it); err != nil {
			f.Close()
			return err
		}
		if len(it.Aspects) > 0 {
			items = append(items, it)
		}
	}
	f.Close()
	if visionLimit > 0 && visionLimit < len(items) {
		items = items[:visionLimit]
	}
	if len(items) == 0 {
		return fmt.Errorf("no items with \"aspects\" in %s", visionDataset)
	}
	if visionVariant != variantOriginal && visionVariant != variantBlank {
		return fmt.Errorf("--variant must be original or blank")
	}

	c := GetClient()
	if c.MaxRetries < 2 {
		c.MaxRetries = 2
	}
	engine := template.NewEngine()
	ctx := context.Background()
	model := c.Model
	if visionEngine == "gemini" {
		model = SanitizeCascadeModel(visionGemini)
	}
	variantDir := os.TempDir() + "/dgem-vision-variants"

	type job struct{ idx, item, rep int }
	var jobs []job
	for rep := 0; rep < max(1, visionRepeat); rep++ {
		for i := range items {
			jobs = append(jobs, job{len(jobs), i, rep})
		}
	}
	results := make([]VisionCaseResult, len(jobs))
	ch := make(chan job)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for w := 0; w < max(1, visionWorkers); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				it := items[j.item]
				res := VisionCaseResult{ID: it.ID, Repeat: j.rep, Tags: it.Tags}
				start := time.Now()
				ref := strings.ReplaceAll(it.Reference, "_", " ")
				if ref == "" {
					ref = "other element"
				}
				vars := map[string]interface{}{"target": strings.ReplaceAll(it.Target, "_", " "), "reference": ref, "samples": 1}
				rendered, err := engine.RenderFile(visionTemplate, vars)
				var schemaJSON, stateJSON string
				if err == nil {
					schemaJSON, stateJSON, err = template.ParseStructuredPayload(rendered, vars)
				}
				if err == nil && visionOnlyGT {
					schemaJSON, err = filterSchemaQuestions(schemaJSON, it.Aspects)
				}
				img := it.ImagePath
				if err == nil {
					img, err = materializeImageVariant(it.ImagePath, visionVariant, variantDir)
				}
				if err != nil {
					res.Error = err.Error()
				} else {
					qs, pre := visionQuestions(schemaJSON)
					nOpt := map[string]int{}
					for _, q := range qs {
						nOpt[q.ID] = len(q.Options)
					}
					if visionEngine == "gemini" {
						ans, gerr := geminiVisionAnswers(ctx, model, img, qs, pre, stateJSON)
						if gerr != nil {
							res.Error = gerr.Error()
						}
						for a, want := range it.Aspects {
							if got, ok := ans[a]; ok {
								res.Answers = append(res.Answers, VisionAnswer{Aspect: a, Expected: want, Predicted: got, Correct: got == want})
							}
						}
					} else {
						resp, _, derr := c.Decide(ctx, schemaJSON, stateJSON, img)
						var stage1 map[string]client.QuestionAnswer
						var cas *CascadeExecutionSummary
						if derr == nil && visionCascadeT > 0 {
							res.Stage1Ms = float64(time.Since(start).Microseconds()) / 1000
							stage1 = map[string]client.QuestionAnswer{}
							for k, v := range resp.Answers {
								stage1[k] = v
							}
							t1 := time.Now()
							cas = ExecuteStage2GeminiCascadeWithImages(WithStage2Prior(ctx, visionCascadeP), "entropy", visionCascadeT, visionCascadeM, nil,
								schemaJSON, stateJSON, resp, []string{img})
							if cas != nil {
								res.EscalatedSlots, res.TotalSlots, res.CascadeError = cas.EscalatedCount, cas.TotalSlots, cas.Error
								if cas.Triggered {
									res.CascadeMs = float64(time.Since(t1).Microseconds()) / 1000
								}
							}
						}
						if derr != nil {
							res.Error = derr.Error()
						} else {
							for a, want := range it.Aspects {
								qa, ok := resp.Answers[a]
								if !ok {
									res.Answers = append(res.Answers, VisionAnswer{Aspect: a, Expected: want, Predicted: "<missing>"})
									continue
								}
								label, conf, h, has := dgemVisionAnswer(qa, nOpt[a])
								var probs map[string]float64
								if has {
									probs = map[string]float64{}
									for k, v := range qa.Probabilities {
										probs[normalizeYesNo(k)] = v
									}
								}
								va := VisionAnswer{Aspect: a, Expected: want, Predicted: label,
									Correct: label == want, Confidence: conf, HNorm: h, HasDist: has, Probabilities: probs}
								if stage1 != nil {
									s1, _, h1, _ := dgemVisionAnswer(stage1[a], nOpt[a])
									ok1 := s1 == want
									va.Stage1, va.Stage1OK, va.HNorm = s1, &ok1, h1
									if cas != nil {
										va.Escalated = cas.Slots[a].Escalated && cas.Slots[a].Stage2Value != ""
									}
								}
								res.Answers = append(res.Answers, va)
							}
						}
					}
					sort.Slice(res.Answers, func(x, y int) bool { return res.Answers[x].Aspect < res.Answers[y].Aspect })
				}
				res.WallTimeMs = float64(time.Since(start).Microseconds()) / 1000
				results[j.idx] = res
				mu.Lock()
				done++
				if done%25 == 0 || done == len(jobs) {
					fmt.Fprintf(os.Stderr, "  bench-vision: %d/%d\n", done, len(jobs))
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

	rep := VisionReport{Timestamp: time.Now().UTC().Format(time.RFC3339), Engine: visionEngine, TargetModel: model,
		Dataset: visionDataset, Template: visionTemplate, Variant: visionVariant, Repeats: max(1, visionRepeat),
		GitCommit: gitShortCommit(), Cases: results}
	if visionEngine == "dgem" {
		rep.ServerVersion, _ = fetchBboxServerInfo(c)
	}
	qs := map[string]int{}
	if rendered, err := engine.RenderFile(visionTemplate, map[string]interface{}{}); err == nil {
		if sj, _, err := template.ParseStructuredPayload(rendered, nil); err == nil {
			vq, _ := visionQuestions(sj)
			for _, q := range vq {
				qs[q.ID] = len(q.Options)
			}
		}
	}
	rep.Aspects = summarizeVision(results, qs)
	rep.ReceiptKind = "bench-vision"
	for _, r := range results {
		if r.Error != "" {
			continue
		}
		for _, a := range r.Answers {
			conf := a.Confidence
			if conf == 0 {
				conf = 1 // engines without distributions
			}
			rep.Items = append(rep.Items, VisionItemRow{ID: r.ID + "#" + a.Aspect, Expected: a.Expected, Actual: a.Predicted,
				Accurate: a.Correct, Confidence: conf, TopProbabilities: a.Probabilities, WallTimeMs: r.WallTimeMs})
		}
	}
	if visionOutput != "" {
		b, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(visionOutput, b, 0o644); err != nil {
			return err
		}
	}
	nerr := 0
	for _, r := range results {
		if r.Error != "" {
			nerr++
		}
	}
	fmt.Printf("\nMulti-aspect image decisions  engine=%s model=%s server=%s variant=%s repeats=%d errors=%d\n",
		rep.Engine, rep.TargetModel, rep.ServerVersion, rep.Variant, rep.Repeats, nerr)
	fmt.Println(strings.Repeat("=", 120))
	fmt.Printf("%-14s %5s %3s  %-24s %-26s %-24s %-8s %s\n", "ASPECT", "ITEMS", "K", "ACCURACY [95% CI]", "MAJORITY (label)", "LIFT [95% CI]", "H AUROC", "TOP PREDICTIONS")
	for _, a := range rep.Aspects {
		type kv struct {
			k string
			v float64
		}
		var top []kv
		for k, v := range a.PredictedDist {
			top = append(top, kv{k, v})
		}
		sort.Slice(top, func(i, j int) bool { return top[i].v > top[j].v })
		var ts []string
		for i := 0; i < len(top) && i < 3; i++ {
			ts = append(ts, fmt.Sprintf("%s %.0f%%", top[i].k, 100*top[i].v))
		}
		fmt.Printf("%-14s %5d %3d  %-24s %-26s %-24s %-8.3f %s\n", a.Aspect, a.Items, a.Options, fmtStat(a.Accuracy),
			fmt.Sprintf("%.3f (%s)", a.MajorityAccuracy, a.MajorityLabel), fmtStat(a.Lift), a.HesitationAUROC, strings.Join(ts, ", "))
	}
	if visionCascadeT > 0 {
		printVisionCascade(results, visionCascadeT, visionCascadeM)
	}
	fmt.Println()
	return nil
}

// printVisionCascade reports the live cascade: per aspect, dgem-only vs cascaded accuracy and the share escalated;
// overall, the share of requests that called Gemini and the latency distribution.
func printVisionCascade(results []VisionCaseResult, thr float64, model string) {
	type agg struct{ n, s1, fin, esc int }
	per := map[string]*agg{}
	var total, withGemini []float64
	var s1ms []float64
	reqs, called, errs, slots, escSlots := 0, 0, 0, 0, 0
	for _, r := range results {
		if r.Error != "" {
			continue
		}
		reqs++
		slots += r.TotalSlots
		escSlots += r.EscalatedSlots
		if r.CascadeError != "" {
			errs++
		}
		total = append(total, r.WallTimeMs)
		s1ms = append(s1ms, r.Stage1Ms)
		if r.CascadeMs > 0 {
			called++
			withGemini = append(withGemini, r.WallTimeMs)
		}
		for _, a := range r.Answers {
			g := per[a.Aspect]
			if g == nil {
				g = &agg{}
				per[a.Aspect] = g
			}
			g.n++
			if a.Stage1OK != nil && *a.Stage1OK {
				g.s1++
			}
			if a.Correct {
				g.fin++
			}
			if a.Escalated {
				g.esc++
			}
		}
	}
	pct := func(v []float64, q float64) float64 {
		if len(v) == 0 {
			return 0
		}
		c := append([]float64(nil), v...)
		sort.Float64s(c)
		return c[int(q*float64(len(c)-1))]
	}
	fmt.Printf("\nLive cascade: entropy >= %.2f nats -> %s (with the image)\n", thr, model)
	fmt.Printf("%-14s %6s %10s %10s %10s\n", "ASPECT", "N", "DGEM ONLY", "CASCADED", "ESCALATED")
	names := make([]string, 0, len(per))
	for k := range per {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		g := per[k]
		fmt.Printf("%-14s %6d %10.3f %10.3f %9.0f%%\n", k, g.n, float64(g.s1)/float64(g.n), float64(g.fin)/float64(g.n), 100*float64(g.esc)/float64(g.n))
	}
	fmt.Printf("requests %d: Gemini called on %d (%.0f%%), %d cascade errors; slots escalated %d/%d (%.0f%%)\n",
		reqs, called, 100*float64(called)/float64(max(1, reqs)), errs, escSlots, slots, 100*float64(escSlots)/float64(max(1, slots)))
	fmt.Printf("latency ms: dgem p50 %.0f | end-to-end p50 %.0f p90 %.0f | requests that called Gemini p50 %.0f p90 %.0f\n",
		pct(s1ms, 0.5), pct(total, 0.5), pct(total, 0.9), pct(withGemini, 0.5), pct(withGemini, 0.9))
}

func summarizeVision(results []VisionCaseResult, nOptions map[string]int) []VisionAspectSummary {
	type agg struct {
		perItem          map[string][]float64
		expected         map[string]string
		pred             map[string]int
		posH, negH       []float64
		hasDist          bool
		total, wrong     int
		setHit, setTotal map[string]int
		itemSet          map[string]string
	}
	aspects := map[string]*agg{}
	for _, r := range results {
		if r.Error != "" {
			continue
		}
		for _, a := range r.Answers {
			g := aspects[a.Aspect]
			if g == nil {
				g = &agg{perItem: map[string][]float64{}, expected: map[string]string{}, pred: map[string]int{},
					setHit: map[string]int{}, setTotal: map[string]int{}, itemSet: map[string]string{}}
				aspects[a.Aspect] = g
			}
			v := 0.0
			if a.Correct {
				v = 1
			} else {
				g.wrong++
			}
			g.perItem[r.ID] = append(g.perItem[r.ID], v)
			g.expected[r.ID] = a.Expected
			g.pred[a.Predicted]++
			g.total++
			set := r.Tags["set"]
			g.itemSet[r.ID] = set
			g.setTotal[set]++
			if a.Correct {
				g.setHit[set]++
			}
			if a.HasDist {
				g.hasDist = true
				if a.Correct {
					g.negH = append(g.negH, a.HNorm)
				} else {
					g.posH = append(g.posH, a.HNorm)
				}
			}
		}
	}
	names := make([]string, 0, len(aspects))
	for k := range aspects {
		names = append(names, k)
	}
	sort.Strings(names)
	var out []VisionAspectSummary
	for i, name := range names {
		g := aspects[name]
		counts := map[string]int{}
		for _, e := range g.expected {
			counts[e]++
		}
		maj, mc := "", -1
		for k, n := range counts {
			if n > mc || (n == mc && k < maj) {
				maj, mc = k, n
			}
		}
		ids := make([]string, 0, len(g.perItem))
		for id := range g.perItem {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		var acc, lift []float64
		for _, id := range ids {
			m := mean(g.perItem[id])
			acc = append(acc, m)
			b := 0.0
			if g.expected[id] == maj {
				b = 1
			}
			lift = append(lift, m-b)
		}
		s := VisionAspectSummary{Aspect: name, Items: len(ids), Options: nOptions[name], MajorityLabel: maj,
			MajorityAccuracy: float64(mc) / float64(len(ids)), Accuracy: bootstrapMean(acc, int64(200+i)),
			Lift: bootstrapMean(lift, int64(300+i)), PredictedDist: map[string]float64{}, Wrong: g.wrong,
			HesitationAUROC: -1, BySet: map[string]float64{}}
		for k, n := range g.pred {
			s.PredictedDist[k] = float64(n) / float64(g.total)
		}
		if g.hasDist {
			s.HesitationAUROC = finiteOr(aurocScore(g.posH, g.negH), -1)
		}
		for set, n := range g.setTotal {
			if set != "" {
				s.BySet[set] = float64(g.setHit[set]) / float64(n)
			}
		}
		out = append(out, s)
	}
	return out
}

// filterSchemaQuestions keeps only the questions whose id is a key of keep.
func filterSchemaQuestions(schemaJSON string, keep map[string]string) (string, error) {
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(schemaJSON), &m); err != nil {
		return "", err
	}
	qs, _ := m["questions"].([]interface{})
	var out []interface{}
	for _, q := range qs {
		if qm, ok := q.(map[string]interface{}); ok {
			if id, _ := qm["id"].(string); keep[id] != "" {
				out = append(out, q)
			}
		}
	}
	if len(out) == 0 {
		return "", fmt.Errorf("no scored questions in the schema")
	}
	m["questions"] = out
	b, err := json.Marshal(m)
	return string(b), err
}
