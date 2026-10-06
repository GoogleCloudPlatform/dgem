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
	"fmt"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/spf13/cobra"
)

var (
	locateImage    string
	locateTarget   string
	locateMask     bool
	locateModel    string
	locateThinking string
	locateSkipH    float64
	locateHint     bool
	locateFormat   string
)

var locateCmd = &cobra.Command{
	Use:     "locate",
	GroupID: "core",
	Short:   "Locate an object in an image: dgem first, Gemini 3.x box only when needed, optional SAM mask",
	Long: `locate runs the guided pipeline measured in 1. dgem answers "is the target present?" and "which 3x3 cell holds it?" in one pass (~0.2 s).
  2. With --skip-h > 0 (0.16 recommended on photos and mobile UI; not on defect images), it returns absent without
     calling Gemini when dgem is confident the target is absent.
  3. Gemini 3.x (default LOW thinking) returns the box; --hint adds dgem's cell as a "may be wrong" hint.
  4. With --mask and DGEM_SAM_URL set, a SAM service turns the box into a mask
     (POST {"image", "box_pct"} -> {"png_base64", "polygon_pct", "score"}).
Gemini needs Application Default Credentials and GOOGLE_CLOUD_PROJECT (or GCP_PROJECT).`,
	Example: `  dgem locate --vertex-url <ENDPOINT_ID> -I screenshot.png --target "the checkout button"
  dgem locate -u "<CLOUD_RUN_URL>/v1" --gcp-auth -I photo.jpg --target "the dog on the left" -f json`,
	RunE: func(cmd *cobra.Command, args []string) error {
		c := GetClient()
		decide := func(ctx context.Context, schema, state string, images []string) (*client.StructuredDecisionResponse, error) {
			resp, _, err := c.Decide(ctx, schema, state, images...)
			return resp, err
		}
		res, err := runGuidedLocate(context.Background(), LocateRequest{Image: locateImage, Target: locateTarget,
			Mask: locateMask, GeminiModel: locateModel, Thinking: locateThinking, SkipH: locateSkipH, Hint: locateHint}, decide)
		if err != nil {
			return err
		}
		if locateFormat == "json" {
			b, _ := json.MarshalIndent(res, "", "  ")
			fmt.Println(string(b))
			return nil
		}
		fmt.Printf("target:   %s\n", res.Target)
		fmt.Printf("dgem:     present=%s (H~ %.2f), cell=%s (H~ %.2f), %.0f ms\n", res.Dgem.Present, res.Dgem.PresentH,
			res.Dgem.GridCell, res.Dgem.GridCellH, res.Dgem.Ms)
		switch res.Path {
		case "skipped_absent":
			fmt.Println("result:   absent (dgem confident; Gemini skipped)")
		case "gemini_absent":
			fmt.Printf("result:   absent (Gemini %s found nothing, %.0f ms)\n", res.Gemini.Model, res.Gemini.Ms)
		default:
			b := res.BoxPct
			fmt.Printf("box:      [ymin %.1f, xmin %.1f, ymax %.1f, xmax %.1f] %% (Gemini %s @ %s, %.0f ms)\n",
				b[0], b[1], b[2], b[3], res.Gemini.Model, res.Gemini.Thinking, res.Gemini.Ms)
		}
		if res.Mask != nil {
			fmt.Printf("mask:     %d polygon points, score %.2f (%.0f ms)\n", len(res.Mask.PolygonPct), res.Mask.Score, res.SamMs)
		} else if res.MaskNote != "" {
			fmt.Println("mask:    ", res.MaskNote)
		}
		fmt.Printf("total:    %.0f ms\n", res.TotalMs)
		return nil
	},
}

func init() {
	f := locateCmd.Flags()
	f.StringVarP(&locateImage, "image", "I", "", "Image path, URL or data URI")
	f.StringVar(&locateTarget, "target", "", "What to locate")
	f.BoolVar(&locateMask, "mask", false, "Also request a SAM mask (needs DGEM_SAM_URL)")
	f.StringVar(&locateModel, "gemini-model", DefaultCascadeGeminiModel, "Gemini 3.x model (gemini-3.8-flash, gemini-3.5-flash-lite)")
	f.StringVar(&locateThinking, "thinking", defaultLocateThinking, "Gemini thinking level: low, medium, high, default")
	f.Float64Var(&locateSkipH, "skip-h", 0, "Skip Gemini when dgem says absent below this normalized entropy (0 = never; 0.16 recommended on photos/mobile UI)")
	f.BoolVar(&locateHint, "hint", false, "Pass dgem's grid cell to Gemini as a hint")
	f.StringVarP(&locateFormat, "format", "f", "table", "Output: table or json")
	RootCmd.AddCommand(locateCmd)
}
