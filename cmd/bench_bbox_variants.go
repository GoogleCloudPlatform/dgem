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

// probe variants. Each variant changes either the schema (option order, label
// style) or the image (flip, padding, blank), and transformBoxForVariant maps a
// ground-truth box into the variant's coordinate frame so the same metrics apply.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Variant names accepted by --variants.
const (
	variantOriginal        = "original"
	variantBlank           = "blank"
	variantHFlip           = "hflip"
	variantVFlip           = "vflip"
	variantPadRight        = "pad_right"
	variantPadLeft         = "pad_left"
	variantPadBottom       = "pad_bottom"
	variantReversed        = "reversed"
	variantDigits9         = "digits9"
	variantDigits9Reversed = "digits9_reversed"
)

// padFactor is how much a pad_* variant extends the canvas along one axis.
const padFactor = 1.5

var allBboxVariants = []string{
	variantOriginal, variantBlank, variantHFlip, variantVFlip, variantPadRight,
	variantPadLeft, variantPadBottom, variantReversed, variantDigits9, variantDigits9Reversed,
}

// errVariantNotApplicable marks a variant that does not apply to a case (e.g. digits9 on
// the decile-band DETR template); the case is recorded as skipped, not failed.
var errVariantNotApplicable = errors.New("variant not applicable")

var digits9Levels = []string{"0", "12.5", "25", "37.5", "50", "62.5", "75", "87.5", "100"}

func isImageVariant(v string) bool {
	switch v {
	case variantBlank, variantHFlip, variantVFlip, variantPadRight, variantPadLeft, variantPadBottom:
		return true
	}
	return false
}

func isSchemaVariant(v string) bool {
	switch v {
	case variantReversed, variantDigits9, variantDigits9Reversed:
		return true
	}
	return false
}

func parseBboxVariants(raw string) ([]string, error) {
	if strings.TrimSpace(raw) == "" || raw == variantOriginal {
		return []string{variantOriginal}, nil
	}
	if raw == "all" {
		return append([]string(nil), allBboxVariants...), nil
	}
	known := map[string]bool{}
	for _, v := range allBboxVariants {
		known[v] = true
	}
	seen := map[string]bool{}
	out := []string{}
	for _, v := range strings.Split(raw, ",") {
		v = strings.TrimSpace(v)
		if v == "" || seen[v] {
			continue
		}
		if !known[v] {
			return nil, fmt.Errorf("unknown variant %q (known: %s, or all)", v, strings.Join(allBboxVariants, ", "))
		}
		seen[v] = true
		out = append(out, v)
	}
	// Transform consistency is measured against the original, so always run it.
	if !seen[variantOriginal] {
		out = append([]string{variantOriginal}, out...)
	}
	return out, nil
}

// bboxEdges are the four coordinate slot suffixes, in [ymin, xmin, ymax, xmax] order.
var bboxEdges = [4]string{"ymin", "xmin", "ymax", "xmax"}

// coordEdgeOf returns the box edge a question id refers to ("ymin", "obj2_xmax" -> "xmax").
func coordEdgeOf(id string) (string, bool) {
	for _, e := range bboxEdges {
		if id == e || strings.HasSuffix(id, "_"+e) {
			return e, true
		}
	}
	return "", false
}

// coordQuestion describes the coordinate value of every label of one coordinate slot.
type coordQuestion struct {
	ID       string             `json:"id"`
	Values   map[string]float64 `json:"values"`
	K        int                `json:"k"`
	BinWidth float64            `json:"bin_width"`
	Min      float64            `json:"min"`
	Max      float64            `json:"max"`
}

var bandDescRe = regexp.MustCompile(`^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%`)

// coordQuestionsFromSchema maps each coordinate slot's labels to coordinates. A label
// whose description is a band ("60-70%") maps to the band centre; otherwise the label
// itself is the coordinate.
func coordQuestionsFromSchema(schemaJSON string) (map[string]coordQuestion, error) {
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(schemaJSON), &m); err != nil {
		return nil, err
	}
	qs, _ := m["questions"].([]interface{})
	out := map[string]coordQuestion{}
	for _, raw := range qs {
		q, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		id, _ := q["id"].(string)
		if _, ok := coordEdgeOf(id); !ok {
			continue
		}
		cq := coordQuestion{ID: id, Values: map[string]float64{}}
		var order []float64
		if opts, ok := q["options"].([]interface{}); ok {
			for _, o := range opts {
				name, desc := "", ""
				switch ov := o.(type) {
				case map[string]interface{}:
					name, _ = ov["name"].(string)
					desc, _ = ov["description"].(string)
				case string:
					name = ov
				}
				v, err := strconv.ParseFloat(name, 64)
				if mm := bandDescRe.FindStringSubmatch(desc); mm != nil {
					lo, _ := strconv.ParseFloat(mm[1], 64)
					hi, _ := strconv.ParseFloat(mm[2], 64)
					v, err = (lo+hi)/2, nil
				}
				if err != nil {
					continue
				}
				cq.Values[name] = v
				order = append(order, v)
			}
		}
		if levels, ok := q["levels"].([]interface{}); ok {
			for _, l := range levels {
				name := fmt.Sprint(l)
				if v, err := strconv.ParseFloat(name, 64); err == nil {
					cq.Values[name] = v
					order = append(order, v)
				}
			}
		}
		cq.K = len(cq.Values)
		cq.BinWidth = minSpacing(order)
		for i, v := range order {
			if i == 0 || v < cq.Min {
				cq.Min = v
			}
			if i == 0 || v > cq.Max {
				cq.Max = v
			}
		}
		out[id] = cq
	}
	return out, nil
}

func minSpacing(vals []float64) float64 {
	best := 0.0
	for i := 0; i < len(vals); i++ {
		for j := i + 1; j < len(vals); j++ {
			d := vals[i] - vals[j]
			if d < 0 {
				d = -d
			}
			if d > 0 && (best == 0 || d < best) {
				best = d
			}
		}
	}
	return best
}

// applySchemaVariant rewrites the coordinate slots of a rendered schema.
//
//	reversed          coordinate options in reverse order, so letter A means 100
//	digits9           coordinates as a 9-level score (labels 1..9, 12.5% steps)
//	digits9_reversed  the same with the levels reversed (label 1 means 100)
//
// Non-coordinate slots are unchanged. digits9* is not applicable to band templates.
func applySchemaVariant(schemaJSON, variant string) (string, error) {
	if !isSchemaVariant(variant) {
		return schemaJSON, nil
	}
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(schemaJSON), &m); err != nil {
		return "", err
	}
	qs, _ := m["questions"].([]interface{})
	for _, raw := range qs {
		q, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		id, _ := q["id"].(string)
		if _, ok := coordEdgeOf(id); !ok {
			continue
		}
		switch variant {
		case variantReversed:
			if opts, ok := q["options"].([]interface{}); ok {
				q["options"] = reversedSlice(opts)
			}
			if levels, ok := q["levels"].([]interface{}); ok {
				q["levels"] = reversedSlice(levels)
			}
		case variantDigits9, variantDigits9Reversed:
			if opts, ok := q["options"].([]interface{}); ok {
				for _, o := range opts {
					if ov, ok := o.(map[string]interface{}); ok {
						if d, _ := ov["description"].(string); bandDescRe.MatchString(d) {
							return "", errVariantNotApplicable
						}
					}
				}
			}
			levels := make([]interface{}, len(digits9Levels))
			for i, l := range digits9Levels {
				levels[i] = l
			}
			if variant == variantDigits9Reversed {
				levels = reversedSlice(levels)
			}
			delete(q, "options")
			q["type"] = "score"
			q["levels"] = levels
		}
	}
	if strings.HasPrefix(variant, variantDigits9) {
		if ins, ok := m["instructions"].(string); ok {
			ins = strings.ReplaceAll(ins, "from 00 (0%", "from 0 (0%")
			ins = strings.ReplaceAll(ins, "in 5% increments", "in 12.5% steps")
			m["instructions"] = ins
		}
	}
	b, err := json.Marshal(m)
	return string(b), err
}

func reversedSlice(in []interface{}) []interface{} {
	out := make([]interface{}, len(in))
	for i := range in {
		out[len(in)-1-i] = in[i]
	}
	return out
}

// transformBoxForVariant maps a [ymin, xmin, ymax, xmax] box (percent) into the frame of an
// image variant. Schema variants and blank leave it unchanged.
func transformBoxForVariant(b [4]float64, variant string) [4]float64 {
	s := 100.0 / padFactor // original extent after padding, in percent of the new axis
	switch variant {
	case variantHFlip:
		return [4]float64{b[0], 100 - b[3], b[2], 100 - b[1]}
	case variantVFlip:
		return [4]float64{100 - b[2], b[1], 100 - b[0], b[3]}
	case variantPadRight:
		return [4]float64{b[0], b[1] * s / 100, b[2], b[3] * s / 100}
	case variantPadLeft:
		off := 100 - s
		return [4]float64{b[0], off + b[1]*s/100, b[2], off + b[3]*s/100}
	case variantPadBottom:
		return [4]float64{b[0] * s / 100, b[1], b[2] * s / 100, b[3]}
	}
	return b
}

// variantSwapsObjectOrder reports whether a variant mirrors the left/top object ordering a
// DETR template asks for; matching is order-free, so this is informational only.
func variantSwapsObjectOrder(v string) bool { return v == variantHFlip || v == variantVFlip }

// materializeImageVariant writes the image variant of src into dir (cached by content hash)
// and returns its path. Padding and blank use the colour of the top-left pixel.
func materializeImageVariant(src, variant, dir string) (string, error) {
	if !isImageVariant(variant) {
		return src, nil
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	base := strings.TrimSuffix(filepath.Base(src), filepath.Ext(src))
	out := filepath.Join(dir, fmt.Sprintf("%s__%s__%s.png", base, variant, hex.EncodeToString(sum[:4])))
	if st, err := os.Stat(out); err == nil && st.Size() > 0 {
		return out, nil
	}
	f, err := os.Open(src)
	if err != nil {
		return "", err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return "", fmt.Errorf("decode %s: %w", src, err)
	}
	res := transformImage(img, variant)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	w, err := os.Create(out)
	if err != nil {
		return "", err
	}
	defer w.Close()
	return out, png.Encode(w, res)
}

func transformImage(img image.Image, variant string) image.Image {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	bg := color.RGBAModel.Convert(img.At(b.Min.X, b.Min.Y))
	fill := func(dst *image.RGBA) { draw.Draw(dst, dst.Bounds(), &image.Uniform{bg}, image.Point{}, draw.Src) }
	switch variant {
	case variantBlank:
		dst := image.NewRGBA(image.Rect(0, 0, w, h))
		fill(dst)
		return dst
	case variantHFlip, variantVFlip:
		dst := image.NewRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				sx, sy := x, y
				if variant == variantHFlip {
					sx = w - 1 - x
				} else {
					sy = h - 1 - y
				}
				dst.Set(x, y, img.At(b.Min.X+sx, b.Min.Y+sy))
			}
		}
		return dst
	case variantPadRight, variantPadLeft, variantPadBottom:
		nw, nh := w, h
		off := image.Point{}
		switch variant {
		case variantPadRight:
			nw = int(float64(w) * padFactor)
		case variantPadLeft:
			nw = int(float64(w) * padFactor)
			off.X = nw - w
		case variantPadBottom:
			nh = int(float64(h) * padFactor)
		}
		dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
		fill(dst)
		draw.Draw(dst, image.Rectangle{Min: off, Max: off.Add(image.Pt(w, h))}, img, b.Min, draw.Src)
		return dst
	}
	return img
}
