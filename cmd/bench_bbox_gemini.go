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

// Gemini 3.x as a reference localizer. `bench-bbox --engine gemini`
// asks Gemini for boxes on the same cases and variants and scores them with the same code,
// so dgem and Gemini receipts are directly comparable.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"google.golang.org/genai"
)

// geminiBoxObject is one object in Gemini's reply; box_2d is [ymin, xmin, ymax, xmax] in 0..1000.
type geminiBoxObject struct {
	Label string    `json:"label"`
	Box2D []float64 `json:"box_2d"`
}

type geminiBoxReply struct {
	Present bool              `json:"present"`
	Objects []geminiBoxObject `json:"objects"`
}

// geminiClassOptions lists the class names of a schema's obj1_class slot, if any.
func geminiClassOptions(schemaJSON string) []string {
	var m struct {
		Questions []struct {
			ID      string `json:"id"`
			Options []struct {
				Name string `json:"name"`
			} `json:"options"`
		} `json:"questions"`
	}
	if json.Unmarshal([]byte(schemaJSON), &m) != nil {
		return nil
	}
	for _, q := range m.Questions {
		if q.ID == "obj1_class" {
			var out []string
			for _, o := range q.Options {
				if o.Name != "empty_background" {
					out = append(out, o.Name)
				}
			}
			return out
		}
	}
	return nil
}

func imageMime(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	}
	return "image/png"
}

// callGeminiJSON sends one image and a prompt and decodes the JSON reply into out.
func callGeminiJSON(ctx context.Context, model, imgPath, prompt string, schema *genai.Schema, out interface{}) (float64, error) {
	gc, _, err := getSharedGenaiClient(ctx)
	if err != nil {
		return 0, err
	}
	data, err := os.ReadFile(imgPath)
	if err != nil {
		return 0, err
	}
	contents := []*genai.Content{{Role: "user", Parts: []*genai.Part{
		genai.NewPartFromBytes(data, imageMime(imgPath)),
		genai.NewPartFromText(prompt),
	}}}
	cfg := &genai.GenerateContentConfig{ResponseMIMEType: "application/json", ResponseSchema: schema}
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		cctx, cancel := context.WithTimeout(ctx, 300*time.Second)
		start := time.Now()
		resp, err := gc.Models.GenerateContent(cctx, model, contents, cfg)
		ms := float64(time.Since(start).Microseconds()) / 1000
		cancel()
		if err == nil {
			txt := strings.TrimSpace(resp.Text())
			if err = json.Unmarshal([]byte(txt), out); err == nil {
				return ms, nil
			}
			err = fmt.Errorf("decode gemini reply %q: %w", truncate(txt, 200), err)
		}
		lastErr = err
		time.Sleep(time.Duration(1+attempt*attempt) * 2 * time.Second)
	}
	return 0, lastErr
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// geminiBboxAnswers asks Gemini for the case's box(es) and returns them as one-hot answers keyed
// like the dgem schema (ymin.. or obj1_ymin..), so scoreBboxAnswers applies unchanged.
func geminiBboxAnswers(ctx context.Context, model string, item BboxSuiteItem, imgPath, schemaJSON string, cqs map[string]coordQuestion, hasPresence bool) (map[string]client.QuestionAnswer, float64, error) {
	prefixes := bboxPrefixes(cqs)
	classes := geminiClassOptions(schemaJSON)
	target := strings.ReplaceAll(item.Target, "_", " ")
	var prompt string
	if len(prefixes) > 1 {
		prompt = fmt.Sprintf("Detect the %d primary interactive elements in this image. For each, give its label (one of: %s) "+
			"and its bounding box as box_2d [ymin, xmin, ymax, xmax] normalized to 0-1000, where (0,0) is the top-left corner "+
			"of the full image. Set present to false and return no objects if none are visible.",
			len(prefixes), strings.Join(classes, ", "))
	} else {
		prompt = fmt.Sprintf("Is the target \"%s\" visible in this image? If it is, return its bounding box as box_2d "+
			"[ymin, xmin, ymax, xmax] normalized to 0-1000, where (0,0) is the top-left corner of the full image. "+
			"If it is not visible, set present to false and return no objects.", target)
	}
	labelSchema := &genai.Schema{Type: genai.TypeString}
	if len(classes) > 0 {
		labelSchema.Enum = classes
	}
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"present": {Type: genai.TypeBoolean},
			"objects": {Type: genai.TypeArray, Items: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"label":  labelSchema,
					"box_2d": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeNumber}},
				},
				Required: []string{"label", "box_2d"},
			}},
		},
		Required: []string{"present", "objects"},
	}
	var reply geminiBoxReply
	ms, err := callGeminiJSON(ctx, model, imgPath, prompt, schema, &reply)
	if err != nil {
		return nil, 0, err
	}
	answers := map[string]client.QuestionAnswer{}
	if hasPresence {
		lbl := "no"
		if reply.Present && len(reply.Objects) > 0 {
			lbl = "yes"
		}
		answers["object_present"] = client.QuestionAnswer{Type: "noul", Label: lbl}
	}
	for pi, p := range prefixes {
		if pi >= len(reply.Objects) || len(reply.Objects[pi].Box2D) != 4 {
			continue // missing object: scored as missing slots
		}
		o := reply.Objects[pi]
		for ei, e := range bboxEdges {
			v := o.Box2D[ei] / 10 // 0..1000 -> percent
			answers[p+e] = client.QuestionAnswer{Type: "choice", Probabilities: map[string]float64{strconv.FormatFloat(v, 'f', 2, 64): 1}}
		}
		if len(prefixes) > 1 {
			answers[p+"class"] = client.QuestionAnswer{Type: "choice", Choice: o.Label}
		}
	}
	return answers, ms, nil
}
