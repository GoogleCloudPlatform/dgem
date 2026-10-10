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

// dgem-guided Gemini localization. For every item dgem first answers "is the target present?" and
// "which 3x3 cell holds it?" in one pass; Gemini 3.x then returns a box (or a polygon mask) under one of
// several strategies and thinking levels:
//
//	full   the whole image, no guidance (Gemini alone)
//	hint   the whole image plus dgem's cell as a text hint ("may be wrong")
//	crop   the image cropped to dgem's cell plus half a cell of margin when dgem is confident (normalized
//	       entropy < 0.16), the box mapped back; falls back to the full image if Gemini finds nothing in the
//	       crop or the box touches a crop edge that is not the image border
//	poly   the whole image, asking for a polygon outline (native mask) instead of a box
//
// The "skip absent" strategy is scored offline from the dgem answers and the full-image results.
// Per call it records the box, latency and Gemini token usage; scripts/analyze_guided.py scores it.

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
	"google.golang.org/genai"
)

var (
	guidedDataset    string
	guidedTemplate   string
	guidedOutput     string
	guidedModel      string
	guidedConditions string
	guidedNegConds   string
	guidedWorkers    int
	guidedLimit      int
	guidedConfident  float64
	guidedPolySet    string
)

var benchGuidedCmd = &cobra.Command{
	Use:     "bench-guided",
	GroupID: "eval",
	Short:   "Gemini 3.x boxes and masks guided by a fast dgem pass (skip, crop, hint; thinking levels)",
	Long: `bench-guided runs dgem once per item (presence + 3x3 grid cell) and then Gemini 3.x under each condition
"<strategy>@<thinking>" in --conditions, where strategy is full | hint | crop | poly and thinking is default | medium |
low | high. Positive items run every condition; negative items (object_present false) run --neg-conditions, used to
score skipping Gemini when dgem confidently says the target is absent. Output: a JSON receipt with one record per
(item, condition); score it with scripts/analyze_guided.py.`,
	Example: `  dgem bench-guided --vertex-url <ENDPOINT_ID> --gcp-auth -d third_party/bbox_real/bbox_real_aspects.jsonl -o guided.json`,
	RunE:    runBenchGuided,
}

func init() {
	f := benchGuidedCmd.Flags()
	f.StringVarP(&guidedDataset, "dataset", "d", "third_party/bbox_real/bbox_real_aspects.jsonl", "Items with gt_box_continuous (positives) and aspects.present")
	f.StringVarP(&guidedTemplate, "template", "t", "templates/multimodal/vision_aspects_generic.json.tmpl", "dgem template (present + grid_cell are asked)")
	f.StringVarP(&guidedOutput, "output", "o", "", "Receipt path")
	f.StringVar(&guidedModel, "gemini-model", DefaultCascadeGeminiModel, "Gemini 3.x model")
	f.StringVar(&guidedConditions, "conditions", "full@default,full@medium,full@low,hint@default,hint@low,crop@default,crop@low,poly@low", "Conditions for positive items")
	f.StringVar(&guidedNegConds, "neg-conditions", "full@default,full@low", "Conditions for negative items")
	f.IntVarP(&guidedWorkers, "workers", "w", 8, "Concurrent items")
	f.IntVarP(&guidedLimit, "limit", "n", 0, "Limit items (0 = all)")
	f.Float64Var(&guidedConfident, "confident", 0.16, "dgem normalized entropy below which its grid cell is trusted for cropping")
	f.StringVar(&guidedPolySet, "poly-set", "refcoco", "Run poly conditions only on items whose tags.set equals this (empty = all)")
	RootCmd.AddCommand(benchGuidedCmd)
}

type guidedItem struct {
	ID        string            `json:"id"`
	ImagePath string            `json:"image_path"`
	Target    string            `json:"target"`
	GTBox     *[4]float64       `json:"gt_box_continuous,omitempty"`
	Aspects   map[string]string `json:"aspects"`
	Tags      map[string]string `json:"tags"`
}

// GuidedDgem is dgem's first pass for an item.
type GuidedDgem struct {
	Present     string  `json:"present"`
	PresentH    float64 `json:"present_h_norm"`
	GridCell    string  `json:"grid_cell"`
	GridH       float64 `json:"grid_h_norm"`
	Ms          float64 `json:"ms"`
	Error       string  `json:"error,omitempty"`
	CellCorrect *bool   `json:"cell_correct,omitempty"`
}

// GuidedCall is one Gemini call (or chain, for crop with fallback) under one condition.
type GuidedCall struct {
	ItemID       string       `json:"item_id"`
	Positive     bool         `json:"positive"`
	Condition    string       `json:"condition"`
	Strategy     string       `json:"strategy"`
	Thinking     string       `json:"thinking"`
	Present      bool         `json:"present"`
	BoxPct       *[4]float64  `json:"box_pct,omitempty"`
	Polygon      [][2]float64 `json:"polygon_pct,omitempty"` // [y, x] percent
	Cropped      bool         `json:"cropped,omitempty"`
	CropPct      *[4]float64  `json:"crop_pct,omitempty"`
	FellBack     bool         `json:"fell_back,omitempty"`
	GeminiMs     float64      `json:"gemini_ms"`
	PromptTokens int32        `json:"prompt_tokens"`
	ImageTokens  int32        `json:"image_tokens"`
	ThoughtToks  int32        `json:"thought_tokens"`
	OutputTokens int32        `json:"output_tokens"`
	Calls        int          `json:"calls"`
	Error        string       `json:"error,omitempty"`
}

type GuidedItemResult struct {
	ID       string            `json:"id"`
	Target   string            `json:"target"`
	Positive bool              `json:"positive"`
	GTBox    *[4]float64       `json:"gt_box,omitempty"`
	Tags     map[string]string `json:"tags,omitempty"`
	Dgem     GuidedDgem        `json:"dgem"`
	Calls    []GuidedCall      `json:"calls"`
}

type GuidedReport struct {
	Timestamp     string             `json:"timestamp"`
	GeminiModel   string             `json:"gemini_model"`
	DgemModel     string             `json:"dgem_model"`
	ServerVersion string             `json:"server_version,omitempty"`
	Dataset       string             `json:"dataset"`
	Conditions    []string           `json:"conditions"`
	NegConditions []string           `json:"neg_conditions"`
	Confident     float64            `json:"confident_h_norm"`
	GitCommit     string             `json:"git_commit,omitempty"`
	Items         []GuidedItemResult `json:"items"`
}

var gridRows = map[string]int{"top": 0, "middle": 1, "bottom": 2}
var gridCols = map[string]int{"left": 0, "center": 1, "right": 2}

// cellCropPct is the crop window [ymin, xmin, ymax, xmax] (percent) for a 3x3 cell plus half a cell of margin.
func cellCropPct(cell string) (*[4]float64, bool) {
	parts := strings.SplitN(cell, "_", 2)
	if len(parts) != 2 {
		return nil, false
	}
	r, ok1 := gridRows[parts[0]]
	c, ok2 := gridCols[parts[1]]
	if !ok1 || !ok2 {
		return nil, false
	}
	third := 100.0 / 3
	b := [4]float64{float64(r)*third - third/2, float64(c)*third - third/2, float64(r+1)*third + third/2, float64(c+1)*third + third/2}
	for i := range b {
		b[i] = math.Max(0, math.Min(100, b[i]))
	}
	return &b, true
}

func cropImageToPct(src string, crop [4]float64, dir string) (string, error) {
	f, err := os.Open(src)
	if err != nil {
		return "", err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return "", err
	}
	b := img.Bounds()
	w, h := float64(b.Dx()), float64(b.Dy())
	r := image.Rect(int(crop[1]/100*w), int(crop[0]/100*h), int(math.Ceil(crop[3]/100*w)), int(math.Ceil(crop[2]/100*h))).Add(b.Min)
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(dst, dst.Bounds(), img, r.Min, draw.Src)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	out := filepath.Join(dir, fmt.Sprintf("%s__crop_%.0f_%.0f_%.0f_%.0f.png", strings.TrimSuffix(filepath.Base(src), filepath.Ext(src)), crop[0], crop[1], crop[2], crop[3]))
	wf, err := os.Create(out)
	if err != nil {
		return "", err
	}
	defer wf.Close()
	return out, png.Encode(wf, dst)
}

type geminiUsage struct {
	prompt, image, thoughts, output int32
	ms                              float64
}

func callGeminiGuided(ctx context.Context, model, thinking, imgPath, prompt string, schema *genai.Schema, out interface{}) (geminiUsage, error) {
	gc, _, err := getSharedGenaiClient(ctx)
	if err != nil {
		return geminiUsage{}, err
	}
	data, err := os.ReadFile(imgPath)
	if err != nil {
		return geminiUsage{}, err
	}
	contents := []*genai.Content{{Role: "user", Parts: []*genai.Part{genai.NewPartFromBytes(data, imageMime(imgPath)), genai.NewPartFromText(prompt)}}}
	cfg := &genai.GenerateContentConfig{ResponseMIMEType: "application/json", ResponseSchema: schema}
	if thinking != "" && thinking != "default" {
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevel(strings.ToUpper(thinking))}
	}
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		cctx, cancel := context.WithTimeout(ctx, 180*time.Second)
		start := time.Now()
		resp, err := gc.Models.GenerateContent(cctx, model, contents, cfg)
		ms := float64(time.Since(start).Microseconds()) / 1000
		cancel()
		if err == nil {
			u := geminiUsage{ms: ms}
			if m := resp.UsageMetadata; m != nil {
				u.prompt, u.thoughts, u.output = m.PromptTokenCount, m.ThoughtsTokenCount, m.CandidatesTokenCount
				for _, d := range m.PromptTokensDetails {
					if d.Modality == genai.MediaModalityImage {
						u.image += d.TokenCount
					}
				}
			}
			txt := strings.TrimSpace(resp.Text())
			if err = json.Unmarshal([]byte(txt), out); err == nil {
				return u, nil
			}
			err = fmt.Errorf("decode %q: %w", truncate(txt, 160), err)
		}
		lastErr = err
		time.Sleep(time.Duration(2+3*attempt*attempt) * time.Second)
	}
	return geminiUsage{}, lastErr
}

type guidedBoxReply struct {
	Present bool      `json:"present"`
	Box2D   []float64 `json:"box_2d"`
}

type guidedPolyReply struct {
	Present bool        `json:"present"`
	Points  [][]float64 `json:"points"`
}

func guidedBoxSchema() *genai.Schema {
	return &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{
		"present": {Type: genai.TypeBoolean},
		"box_2d":  {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeNumber}},
	}, Required: []string{"present", "box_2d"}, PropertyOrdering: []string{"present", "box_2d"}}
}

func guidedBoxPrompt(target, hint string) string {
	p := fmt.Sprintf("Find \"%s\" in this image. If it is visible, set present to true and return its bounding box as "+
		"box_2d [ymin, xmin, ymax, xmax] normalized to 0-1000 over this image. If it is not visible, set present to false "+
		"and return an empty box_2d.", target)
	if hint != "" {
		p += " " + hint
	}
	return p
}

func addUsage(c *GuidedCall, u geminiUsage) {
	c.GeminiMs += u.ms
	c.PromptTokens += u.prompt
	c.ImageTokens += u.image
	c.ThoughtToks += u.thoughts
	c.OutputTokens += u.output
	c.Calls++
}

func boxFromReply(r guidedBoxReply) *[4]float64 {
	if !r.Present || len(r.Box2D) != 4 {
		return nil
	}
	b := [4]float64{r.Box2D[0] / 10, r.Box2D[1] / 10, r.Box2D[2] / 10, r.Box2D[3] / 10}
	return &b
}

func runGuidedCondition(ctx context.Context, model string, it guidedItem, dg GuidedDgem, cond, cropDir string) GuidedCall {
	parts := strings.SplitN(cond, "@", 2)
	strategy, thinking := parts[0], "default"
	if len(parts) == 2 {
		thinking = parts[1]
	}
	call := GuidedCall{ItemID: it.ID, Positive: it.Aspects["present"] == "yes", Condition: cond, Strategy: strategy, Thinking: thinking}
	target := strings.ReplaceAll(it.Target, "_", " ")
	full := func(hint string) {
		var r guidedBoxReply
		u, err := callGeminiGuided(ctx, model, thinking, it.ImagePath, guidedBoxPrompt(target, hint), guidedBoxSchema(), &r)
		addUsage(&call, u)
		if err != nil {
			call.Error = err.Error()
			return
		}
		call.BoxPct = boxFromReply(r)
		call.Present = call.BoxPct != nil
	}
	switch strategy {
	case "full":
		full("")
	case "hint":
		hint := ""
		if dg.Error == "" && dg.GridCell != "" {
			hint = fmt.Sprintf("Hint from a fast first-pass model (it may be wrong): the target is probably in the %s part of the image (3x3 grid).",
				strings.ReplaceAll(dg.GridCell, "_", " "))
		}
		full(hint)
	case "crop":
		crop, ok := cellCropPct(dg.GridCell)
		if dg.Error != "" || !ok || dg.GridH >= guidedConfident {
			full("")
			return call
		}
		path, err := cropImageToPct(it.ImagePath, *crop, cropDir)
		if err != nil {
			call.Error = err.Error()
			return call
		}
		call.Cropped, call.CropPct = true, crop
		var r guidedBoxReply
		u, err := callGeminiGuided(ctx, model, thinking, path, guidedBoxPrompt(target, ""), guidedBoxSchema(), &r)
		addUsage(&call, u)
		if err != nil {
			call.Error = err.Error()
			return call
		}
		if b := boxFromReply(r); b != nil {
			// Map from crop percent to image percent.
			hgt, wid := crop[2]-crop[0], crop[3]-crop[1]
			m := [4]float64{crop[0] + b[0]*hgt/100, crop[1] + b[1]*wid/100, crop[0] + b[2]*hgt/100, crop[1] + b[3]*wid/100}
			// A box touching a crop edge that is not the image border may be truncated.
			const eps = 1.0
			truncated := (b[0] < eps && crop[0] > 0) || (b[1] < eps && crop[1] > 0) || (b[2] > 100-eps && crop[2] < 100) || (b[3] > 100-eps && crop[3] < 100)
			if !truncated {
				call.BoxPct, call.Present = &m, true
				return call
			}
		}
		call.FellBack = true
		full("")
	case "poly":
		var r guidedPolyReply
		schema := &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{
			"present": {Type: genai.TypeBoolean},
			"points":  {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeNumber}}},
		}, Required: []string{"present", "points"}, PropertyOrdering: []string{"present", "points"}}
		prompt := fmt.Sprintf("Outline \"%s\" in this image. If it is visible, set present to true and return a polygon that "+
			"traces its outline as 8 to 40 points [y, x], each normalized to 0-1000, in order around the object. If it is "+
			"not visible, set present to false and return no points.", target)
		u, err := callGeminiGuided(ctx, model, thinking, it.ImagePath, prompt, schema, &r)
		addUsage(&call, u)
		if err != nil {
			call.Error = err.Error()
			return call
		}
		if r.Present {
			for _, p := range r.Points {
				if len(p) == 2 {
					call.Polygon = append(call.Polygon, [2]float64{p[0] / 10, p[1] / 10})
				}
			}
			call.Present = len(call.Polygon) >= 3
		}
	default:
		call.Error = "unknown strategy " + strategy
	}
	return call
}

func runBenchGuided(cmd *cobra.Command, args []string) error {
	f, err := os.Open(guidedDataset)
	if err != nil {
		return err
	}
	var items []guidedItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}
		var it guidedItem
		if err := json.Unmarshal(sc.Bytes(), &it); err != nil {
			f.Close()
			return err
		}
		items = append(items, it)
	}
	f.Close()
	if guidedLimit > 0 && guidedLimit < len(items) {
		items = items[:guidedLimit]
	}
	conds := strings.Split(guidedConditions, ",")
	negConds := []string{}
	if strings.TrimSpace(guidedNegConds) != "" {
		negConds = strings.Split(guidedNegConds, ",")
	}
	model := SanitizeCascadeModel(guidedModel)
	c := GetClient()
	if c.MaxRetries < 2 {
		c.MaxRetries = 2
	}
	engine := template.NewEngine()
	ctx := context.Background()
	cropDir := filepath.Join(os.TempDir(), "dgem-guided-crops")

	results := make([]GuidedItemResult, len(items))
	ch := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for w := 0; w < max(1, guidedWorkers); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				it := items[i]
				res := GuidedItemResult{ID: it.ID, Target: it.Target, Positive: it.Aspects["present"] == "yes", GTBox: it.GTBox, Tags: it.Tags}
				res.Dgem = guidedDgemPass(ctx, c, engine, it)
				list := negConds
				if res.Positive {
					list = conds
				}
				for _, cond := range list {
					if strings.HasPrefix(cond, "poly") && guidedPolySet != "" && it.Tags["set"] != guidedPolySet {
						continue
					}
					res.Calls = append(res.Calls, runGuidedCondition(ctx, model, it, res.Dgem, strings.TrimSpace(cond), cropDir))
				}
				results[i] = res
				mu.Lock()
				done++
				if done%10 == 0 || done == len(items) {
					fmt.Fprintf(os.Stderr, "  bench-guided: %d/%d items\n", done, len(items))
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

	rep := GuidedReport{Timestamp: time.Now().UTC().Format(time.RFC3339), GeminiModel: model, DgemModel: c.Model,
		Dataset: guidedDataset, Conditions: conds, NegConditions: negConds, Confident: guidedConfident,
		GitCommit: gitShortCommit(), Items: results}
	rep.ServerVersion, _ = fetchBboxServerInfo(c)
	if guidedOutput != "" {
		b, _ := json.MarshalIndent(rep, "", " ")
		if err := os.WriteFile(guidedOutput, b, 0o644); err != nil {
			return err
		}
	}
	nerr := 0
	for _, r := range results {
		for _, cl := range r.Calls {
			if cl.Error != "" {
				nerr++
			}
		}
	}
	fmt.Printf("bench-guided: %d items, %d Gemini condition errors; score with scripts/analyze_guided.py %s\n", len(results), nerr, guidedOutput)
	return nil
}

func guidedDgemPass(ctx context.Context, c *client.Client, engine *template.Engine, it guidedItem) GuidedDgem {
	var g GuidedDgem
	start := time.Now()
	vars := map[string]interface{}{"target": strings.ReplaceAll(it.Target, "_", " "), "samples": 1}
	rendered, err := engine.RenderFile(guidedTemplate, vars)
	var schemaJSON, stateJSON string
	if err == nil {
		schemaJSON, stateJSON, err = template.ParseStructuredPayload(rendered, vars)
	}
	if err == nil {
		schemaJSON, err = filterSchemaQuestions(schemaJSON, map[string]string{"present": "x", "grid_cell": "x"})
	}
	if err != nil {
		g.Error = err.Error()
		return g
	}
	resp, _, err := c.Decide(ctx, schemaJSON, stateJSON, it.ImagePath)
	g.Ms = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		g.Error = err.Error()
		return g
	}
	if qa, ok := resp.Answers["present"]; ok {
		g.Present, _, g.PresentH, _ = dgemVisionAnswer(qa, 2)
	}
	if qa, ok := resp.Answers["grid_cell"]; ok {
		g.GridCell, _, g.GridH, _ = dgemVisionAnswer(qa, 9)
	}
	if want := it.Aspects["grid_cell"]; want != "" {
		ok := want == g.GridCell
		g.CellCorrect = &ok
	}
	return g
}
