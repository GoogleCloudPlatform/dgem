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
	"os"
	"sort"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/permutation"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	decideTemplate          string
	decideVars              []string
	decideDataFile          string
	decideFormat            string
	decideSchema            string
	decideState             string
	decideImages            []string
	decideDualMirror        bool
	decideNullPriorDebias   bool
	decidePriorAlpha        float64
	decideSuggestExpansions bool
	decideLayout            string
	decideTemperature       float64
	decideExpansionEntropy  float64
)

var decideCmd = &cobra.Command{
	Use:     "decide",
	GroupID: "core",
	Short:   "Execute a discrete diffusion slot readout decision using single-forward evaluation",
	Long: `decide evaluates propositions, categorical choices, and ordered scales
via discrete diffusion slot readout in a single forward pass without autoregressive
text generation overhead (vLLM structured generation mode).
You can specify a template definition file (-t), key-value pairs (-v key=val),
or raw schema/state payloads. Pass --dual-mirror to evaluate forward and
reversed option orderings simultaneously on the same O(1) diffusion canvas,
--null-prior-debias to divide out content-free positional 'A'-bias, or
--suggest-expansions to dynamically detect unclassified/high-entropy items and propose
new {"name", "description"} choice options.`,
	RunE: runDecide,
}

func init() {
	decideCmd.Flags().StringVarP(&decideTemplate, "template", "t", "", "Path to Go template definition file (.json.tmpl)")
	decideCmd.Flags().StringArrayVarP(&decideVars, "var", "v", nil, "Template variables in key=value format (can be specified multiple times)")
	decideCmd.Flags().StringVarP(&decideDataFile, "data", "d", "", "JSON file containing template variables")
	decideCmd.Flags().StringVarP(&decideFormat, "format", "f", "table", "Output format: 'table' or 'json'")
	decideCmd.Flags().StringVar(&decideSchema, "schema", "", "Raw JSON schema file path (skips template engine)")
	decideCmd.Flags().StringVar(&decideState, "state", "", "Raw JSON state string or file path")
	decideCmd.Flags().StringArrayVarP(&decideImages, "image", "I", nil, "Attach local image file path or remote image URL (can be specified multiple times)")
	decideCmd.Flags().BoolVar(&decideDualMirror, "dual-mirror", false, "Evaluate forward + reversed option slots simultaneously in 1 diffusion canvas pass (0ms overhead)")
	decideCmd.Flags().BoolVar(&decideNullPriorDebias, "null-prior-debias", false, "Divide out calibrated content-free positional 'A'-bias in logit space")
	decideCmd.Flags().Float64Var(&decidePriorAlpha, "prior-alpha", 0.50, "Damping exponent alpha in [0, 1] for content-free null-prior de-biasing")
	decideCmd.Flags().StringVar(&decideLayout, "layout", "", "Prompt layout: document_first (the state first, then the questions; server default from v0.2.0) or schema_first (the questions as the system prompt). Empty: the template's \"layout\" or the server default. See docs/policies/prompt-layout.md")
	decideCmd.Flags().Float64Var(&decideTemperature, "temperature", 0, "Post-hoc temperature for every answer (T > 1 softens overconfidence); 0 = the template's \"temperature\" key, else none. Fit it on labelled examples of your own (docs/confidence/calibrate-your-policy.md)")
	decideCmd.Flags().BoolVar(&decideSuggestExpansions, "suggest-expansions", false, "Dynamically inject an 'other_unclassified' catch-all (if absent) and propose new {"+"\"name\", \"description\""+"} options when unclassified or high-entropy")
	decideCmd.Flags().Float64Var(&decideExpansionEntropy, "expansion-entropy", 0.35, "Shannon entropy threshold (in nats) on choice slots to trigger taxonomy expansion suggestions")

	RootCmd.AddCommand(decideCmd)
}

func parseVariables() (map[string]interface{}, error) {
	vars := make(map[string]interface{})

	// Load from data file if provided
	if decideDataFile != "" {
		dataBytes, err := os.ReadFile(decideDataFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read data file %s: %w", decideDataFile, err)
		}
		if err := json.Unmarshal(dataBytes, &vars); err != nil {
			return nil, fmt.Errorf("failed to parse JSON from %s: %w", decideDataFile, err)
		}
	}

	// Override or populate with CLI -v key=val pairs
	for _, kv := range decideVars {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid variable format %q (expected key=value)", kv)
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])

		// Auto-parse booleans, numbers, or JSON arrays/objects if applicable
		if val == "true" {
			vars[key] = true
		} else if val == "false" {
			vars[key] = false
		} else if (strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]")) || (strings.HasPrefix(val, "{") && strings.HasSuffix(val, "}")) {
			var parsedJSON interface{}
			if err := json.Unmarshal([]byte(val), &parsedJSON); err == nil {
				vars[key] = parsedJSON
			} else {
				vars[key] = val
			}
		} else {
			vars[key] = val
		}
	}

	return vars, nil
}

func runDecide(cmd *cobra.Command, args []string) error {
	ctx := context.Background()
	c := GetClient()

	var schemaContent, stateContent string
	var err error

	vars, err := parseVariables()
	if err != nil {
		return err
	}

	if decideTemplate != "" {
		engine := template.NewEngine()
		rendered, err := engine.RenderFile(decideTemplate, vars)
		if err != nil {
			return fmt.Errorf("template render failed: %w", err)
		}

		schemaContent, stateContent, err = template.ParseStructuredPayload(rendered, vars)
		if err != nil {
			return fmt.Errorf("failed to process rendered template: %w", err)
		}
	} else if decideSchema != "" {
		sb, err := os.ReadFile(decideSchema)
		if err != nil {
			return fmt.Errorf("failed to read schema file: %w", err)
		}
		schemaContent = string(sb)

		if decideState != "" {
			if strings.HasPrefix(strings.TrimSpace(decideState), "{") {
				stateContent = decideState
			} else {
				stb, err := os.ReadFile(decideState)
				if err != nil {
					return fmt.Errorf("failed to read state file: %w", err)
				}
				stateContent = string(stb)
			}
		} else {
			stb, _ := json.Marshal(vars)
			stateContent = string(stb)
		}
	} else {
		return fmt.Errorf("must specify either --template (-t) or --schema")
	}

	var injectedSlots map[string]bool
	var existingOptions map[string][]client.ProposedOption
	if schemaContent, err = ApplyPromptLayout(schemaContent, decideLayout); err != nil {
		return err
	}
	if decideSuggestExpansions {
		schemaContent, injectedSlots, existingOptions = InjectUnclassifiedCatchAll(schemaContent)
	} else {
		_, _, existingOptions = InjectUnclassifiedCatchAll(schemaContent)
	}

	var slotOpts map[string][]permutation.OptionItem
	if decideDualMirror {
		schemaContent, slotOpts, _ = permutation.InjectDualMirrorSchema(schemaContent)
	} else if decideNullPriorDebias {
		slotOpts = permutation.ExtractSchemaSlotOptions(schemaContent)
	}

	resp, stats, err := c.Decide(ctx, schemaContent, stateContent, decideImages...)
	if err != nil {
		return fmt.Errorf("decision query failed: %w", err)
	}

	if decideDualMirror || decideNullPriorDebias {
		permutation.PostProcessDecisionResponse(resp, slotOpts, decideDualMirror, decideNullPriorDebias, decidePriorAlpha)
	}

	// Policy temperature last, on the final distributions (template "temperature" key, or --temperature).
	var reqT *client.Temperature
	if decideTemperature > 0 {
		reqT = &client.Temperature{All: decideTemperature}
	}
	if _, _, err := applyPolicyTemperature(resp, schemaContent, reqT, nil); err != nil {
		return err
	}

	if decideSuggestExpansions || (resp.Diagnostics.Thought != nil && strings.Contains(resp.Diagnostics.Thought.Text, "SUGGESTED_")) {
		resp.SuggestedExpansions = SynthesizeTaxonomyExpansions(
			ctx,
			c,
			decideTemplate,
			stateContent,
			resp,
			injectedSlots,
			existingOptions,
			decideExpansionEntropy,
		)
	}

	if decideFormat == "json" {
		pretty, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(pretty))
	} else {
		printDecisionTable(resp)
	}

	if viper.GetBool("stats") {
		PrintStats(stats)
	}

	return nil
}

func printDecisionTable(resp *client.StructuredDecisionResponse) {
	fmt.Println()
	fmt.Printf("%-16s | %-10s | %-20s | %-10s | %-10s | %-10s\n",
		"QUESTION", "TYPE", "VALUE / CHOICE", "CONFIDENCE", "STDERR", "AGREEMENT")
	fmt.Println(strings.Repeat("-", 88))

	keys := make([]string, 0, len(resp.Answers))
	for k := range resp.Answers {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		ans := resp.Answers[k]
		qType := ans.DisplayType()
		val := ans.DisplayValue()
		confStr := fmt.Sprintf("%.1f%%", ans.Confidence*100)
		stderrStr := fmt.Sprintf("±%.4f", ans.Stderr)
		agreeStr := fmt.Sprintf("%.2f", ans.Agreement)

		fmt.Printf("%-16s | %-10s | %-20s | %-10s | %-10s | %-10s\n",
			k, qType, val, confStr, stderrStr, agreeStr)
	}
	if resp.Diagnostics.Thought != nil && strings.TrimSpace(resp.Diagnostics.Thought.Text) != "" {
		fmt.Println(strings.Repeat("-", 88))
		fmt.Printf("THOUGHT (%d tok, %.0fms): %s\n",
			resp.Diagnostics.Thought.Tokens,
			resp.Diagnostics.Thought.Ms,
			strings.TrimSpace(resp.Diagnostics.Thought.Text))
	}
	if len(resp.SuggestedExpansions) > 0 {
		fmt.Println(strings.Repeat("-", 88))
		fmt.Println("SUGGESTED TAXONOMY EXPANSIONS:")
		for _, sug := range resp.SuggestedExpansions {
			optJSON, _ := json.Marshal(sug.SuggestedOption)
			fmt.Printf("  • Slot %q (%s)\n", sug.QuestionID, sug.TriggerReason)
			fmt.Printf("    Proposed Option : %s\n", string(optJSON))
			if sug.TemplatePatchHint != "" {
				fmt.Printf("    Actionable Hint : %s\n", sug.TemplatePatchHint)
			}
		}
	}
}
