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
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	askTemplate  string
	askVars      []string
	askMaxTokens int
	askThink     bool
	askImages    []string
)

var askCmd = &cobra.Command{
	Use:     "ask [PROMPT]",
	GroupID: "core",
	Short:   "Send a generative prompt or rendered template to DiffusionGemma",
	Long: `ask sends a natural-language prompt or template to DiffusionGemma
for standard generative completion. You can enable or disable thinking mode
and inspect generation statistics.`,
	RunE: runAsk,
}

func init() {
	askCmd.Flags().StringVarP(&askTemplate, "template", "t", "", "Path to Go prompt template file (.txt.tmpl)")
	askCmd.Flags().StringArrayVarP(&askVars, "var", "v", nil, "Template variables in key=value format")
	askCmd.Flags().IntVar(&askMaxTokens, "max-tokens", 256, "Maximum tokens to generate")
	askCmd.Flags().BoolVar(&askThink, "think", false, "Enable thinking mode (<|think|>)")
	askCmd.Flags().StringArrayVarP(&askImages, "image", "I", nil, "Attach local image file path or remote image URL (can be specified multiple times)")

	RootCmd.AddCommand(askCmd)
}

func runAsk(cmd *cobra.Command, args []string) error {
	ctx := context.Background()
	c := GetClient()

	var promptText string

	if askTemplate != "" {
		vars := make(map[string]interface{})
		for _, kv := range askVars {
			parts := strings.SplitN(kv, "=", 2)
			if len(parts) == 2 {
				vars[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
			}
		}

		engine := template.NewEngine()
		rendered, err := engine.RenderFile(askTemplate, vars)
		if err != nil {
			return fmt.Errorf("failed to render prompt template: %w", err)
		}
		promptText = strings.TrimSpace(rendered)
	} else if len(args) > 0 {
		promptText = strings.Join(args, " ")
	} else {
		return fmt.Errorf("must provide either a prompt as arguments or a --template (-t)")
	}

	// Model target selection for thinking mode
	targetModel := c.Model
	if !askThink && !strings.Contains(targetModel, ":think=false") {
		targetModel = targetModel + ":think=false"
	} else if askThink && !strings.Contains(targetModel, ":think") {
		targetModel = targetModel + ":think"
	}

	userContent, err := client.BuildMultimodalContent(promptText, askImages)
	if err != nil {
		return fmt.Errorf("failed to process multimodal content: %w", err)
	}

	req := client.ChatCompletionRequest{
		Model: targetModel,
		Messages: []client.ChatMessage{
			{Role: "user", Content: userContent},
		},
		MaxTokens: askMaxTokens,
	}

	resp, stats, err := c.Complete(ctx, req)
	if err != nil {
		return fmt.Errorf("completion request failed: %w", err)
	}

	if len(resp.Choices) > 0 {
		fmt.Println(resp.Choices[0].Message.Content)
	}

	if viper.GetBool("stats") {
		PrintStats(stats)
	}

	return nil
}
