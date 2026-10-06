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
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/spf13/cobra"
)

var (
	renderTemplate string
	renderVars     []string
	renderDataFile string
)

var templateCmd = &cobra.Command{
	Use:     "template",
	GroupID: "core",
	Short:   "Inspect, list, and render Go template definition files",
	Long:    `template provides utilities to manage and preview prompt and schema templates.`,
}

var templateListCmd = &cobra.Command{
	Use:   "list [DIRECTORY]",
	Short: "List available template files in a directory",
	RunE: func(cmd *cobra.Command, args []string) error {
		dir := "templates"
		if len(args) > 0 {
			dir = args[0]
		}

		fmt.Printf("\nAvailable Templates in %s/:\n", dir)
		fmt.Println(strings.Repeat("-", 65))
		err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(d.Name(), ".tmpl") {
				fmt.Printf("  • %-45s\n", path)
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("failed to walk directory %s: %w", dir, err)
		}
		fmt.Println()
		return nil
	},
}

var templateRenderCmd = &cobra.Command{
	Use:   "render",
	Short: "Render a template definition with provided variables to inspect output",
	RunE: func(cmd *cobra.Command, args []string) error {
		if renderTemplate == "" {
			return fmt.Errorf("must specify --template (-t)")
		}

		vars := make(map[string]interface{})
		for _, kv := range renderVars {
			parts := strings.SplitN(kv, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				val := strings.TrimSpace(parts[1])
				if (strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]")) || (strings.HasPrefix(val, "{") && strings.HasSuffix(val, "}")) {
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
		}

		engine := template.NewEngine()
		rendered, err := engine.RenderFile(renderTemplate, vars)
		if err != nil {
			return fmt.Errorf("render failed: %w", err)
		}

		fmt.Println("───────────────────────── RENDERED TEMPLATE ─────────────────────────")
		fmt.Println(rendered)
		fmt.Println("───────────────────────────────────────────────────────────────────────")
		return nil
	},
}

func init() {
	templateRenderCmd.Flags().StringVarP(&renderTemplate, "template", "t", "", "Path to template file (.tmpl)")
	templateRenderCmd.Flags().StringArrayVarP(&renderVars, "var", "v", nil, "Template variables in key=value format")
	templateRenderCmd.Flags().StringVarP(&renderDataFile, "data", "d", "", "JSON data file")

	templateCmd.AddCommand(templateListCmd)
	templateCmd.AddCommand(templateRenderCmd)
	RootCmd.AddCommand(templateCmd)
}
