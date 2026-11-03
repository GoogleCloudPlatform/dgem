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
	"net/http"
	"os"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/decisionindex"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	soPort         int
	soHost         string
	soUpstreamURL  string
	soTemperature  float64
	soMaxSlots     int
	soBracketSize  int
	soCatchAll     string
	soNoulMode     string
	soOptionOrder  string
	soLabels       string
	soPromptLayout string
	soAPIKey       string
	soNullPrior    bool
	soDualMirror   bool
	soPriorAlpha   float64
	soNaiveLimits  bool
)

var systemoneCmd = &cobra.Command{
	Use:   "systemone",
	Short: "Standalone SystemOne wire-protocol server and evaluation adapter",
	Long: `systemone provides a dedicated, lightweight HTTP adapter serving POST /v1/systemone
compatible with external decision benchmarks that speak the /v1/systemone protocol, such as
apolinario/decision-index and JevBench.`,
}

var systemoneServeCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start dedicated POST /v1/systemone HTTP adapter server",
	Long: `serve launches a standalone HTTP adapter on the specified port exposing POST /v1/systemone.
It intercepts and decomposes wide choices (>26 options) into 2-stage bracket tournaments and
multi-slot questions (>8 questions) into canvas batches, routing sub-requests to an upstream
DiffusionGemma server via /v1/chat/completions.`,
	Example: `  # 1. Start SystemOne adapter in front of local diffgemma (or vLLM on :8081)
  dgem systemone serve --port 8080 --upstream http://127.0.0.1:8081/v1

  # 2. Point upstream decision-index runner directly at the adapter
  python -m decision_index run --engine http --option base_url=http://127.0.0.1:8080 --option model=dgem`,
	RunE: runSystemOneServe,
}

func init() {
	systemoneServeCmd.Flags().IntVarP(&soPort, "port", "p", 8080, "Port to listen on (overrides PORT or SYSTEMONE_PORT env var)")
	systemoneServeCmd.Flags().StringVar(&soHost, "host", defaultBindHost(), "Host interface to bind (default 127.0.0.1; 0.0.0.0 on Cloud Run or Vertex, where K_SERVICE or AIP_HTTP_PORT is set)")
	systemoneServeCmd.Flags().StringVar(&soUpstreamURL, "upstream", "", "Upstream DiffusionGemma /v1 endpoint URL (defaults to -u / --url or http://127.0.0.1:8081/v1)")
	systemoneServeCmd.Flags().Float64Var(&soTemperature, "temperature", 1.0, "Post-hoc slot logit temperature scaling factor T* (default 1.0 unscaled)")
	systemoneServeCmd.Flags().IntVar(&soMaxSlots, "max-slots", decisionindex.MaxSlotsPerPass, "Maximum simultaneous questions per forward pass (default 8)")
	systemoneServeCmd.Flags().IntVar(&soBracketSize, "bracket-size", decisionindex.BracketSize, "Maximum options per Round-1 tournament bracket (default 20)")
	systemoneServeCmd.Flags().StringVar(&soCatchAll, "catch-all", "off", "Wide-option catch-all handling ('none of the listed', 'out of scope', ...): off, final (skip Round 1, compete in the final), both (every bracket and the final), verify (final over real options, then pick vs catch-all)")
	systemoneServeCmd.Flags().StringVar(&soPromptLayout, "prompt-layout", "document_first", "Server prompt layout: document_first (the state first, then the questions; servers without the \"layout\" field ignore it), or schema_first (questions as the system prompt, the server default)")
	systemoneServeCmd.Flags().StringVar(&soOptionOrder, "option-order", decisionindex.OptionOrderAlpha, "How each question's options are labelled A, B, C…: alpha (sorted keys, the default) or source (the order the request lists them)")
	systemoneServeCmd.Flags().StringVar(&soLabels, "labels", decisionindex.LabelsAZ, "Server label set for choice options: az (A-Z, brackets above 26, the default), az_aa or az_lower (52 labels: up to 52 options in one read, 52-wide brackets above; needs a serving image with label sets). Experimental")
	systemoneServeCmd.Flags().StringVar(&soNoulMode, "noul-mode", "noul", "How yes/no questions are read: noul, or choice (2-option yes/no choice with the true/false criteria as descriptions)")
	systemoneServeCmd.Flags().StringVar(&soAPIKey, "api-key", "", "Optional secret key to enforce Authorization: Bearer <key> (env: SYSTEMONE_API_KEY or API_KEY)")
	systemoneServeCmd.Flags().BoolVar(&soNullPrior, "null-prior-debias", false, "Divide out positional 'A'-bias prior")
	systemoneServeCmd.Flags().BoolVar(&soDualMirror, "dual-mirror", false, "Evaluate dual-mirror reversed option ordering")
	systemoneServeCmd.Flags().Float64Var(&soPriorAlpha, "prior-alpha", 0.50, "Prior alpha exponent for null-prior de-biasing")
	systemoneServeCmd.Flags().BoolVar(&soNaiveLimits, "naive-limits", false, "Enforce naive 26-option and 10-slot capacity rejections (for benchmark ablation)")

	systemoneCmd.AddCommand(systemoneServeCmd)
	RootCmd.AddCommand(systemoneCmd)
}

// systemOneHealth answers GET /health for readiness probes. It is registered outside authWrap, so it reports liveness
// and version only; the upstream URL and adapter settings are in the startup log.
func systemOneHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":   "ok",
		"service":  "dgem-systemone-adapter",
		"version":  Version,
		"revision": buildRevision(),
	})
}

func runSystemOneServe(cmd *cobra.Command, args []string) error {
	switch soCatchAll {
	case "off", "final", "both", "verify":
	default:
		return fmt.Errorf("--catch-all must be off, final, both or verify (got %q)", soCatchAll)
	}
	if soPromptLayout != "schema_first" && soPromptLayout != "document_first" {
		return fmt.Errorf("--prompt-layout must be schema_first or document_first (got %q)", soPromptLayout)
	}
	if soOptionOrder != decisionindex.OptionOrderAlpha && soOptionOrder != decisionindex.OptionOrderSource {
		return fmt.Errorf("--option-order must be alpha or source (got %q)", soOptionOrder)
	}
	switch soLabels {
	case decisionindex.LabelsAZ, decisionindex.LabelsAZAA, decisionindex.LabelsAZLower:
	default:
		return fmt.Errorf("--labels must be az, az_aa or az_lower (got %q)", soLabels)
	}
	if soNoulMode != "noul" && soNoulMode != "choice" {
		return fmt.Errorf("--noul-mode must be noul or choice (got %q)", soNoulMode)
	}
	defaultRetriesForLongRunning()

	if envPort := os.Getenv("PORT"); envPort != "" && !cmd.Flags().Changed("port") {
		var p int
		if _, err := fmt.Sscanf(envPort, "%d", &p); err == nil && p > 0 {
			soPort = p
		}
	} else if envSOPort := os.Getenv("SYSTEMONE_PORT"); envSOPort != "" && !cmd.Flags().Changed("port") {
		var p int
		if _, err := fmt.Sscanf(envSOPort, "%d", &p); err == nil && p > 0 {
			soPort = p
		}
	}

	upstream := soUpstreamURL
	if upstream == "" {
		if envUp := os.Getenv("SYSTEMONE_UPSTREAM_URL"); envUp != "" {
			upstream = envUp
		} else if envUp2 := os.Getenv("UPSTREAM_DGEMMA_URL"); envUp2 != "" {
			upstream = envUp2
		} else if v := viper.GetString("url"); v != "" && v != "http://127.0.0.1:8080/v1" {
			upstream = v
		} else {
			upstream = "http://127.0.0.1:8081/v1"
		}
	}

	apiKey := soAPIKey
	if apiKey == "" {
		if k := os.Getenv("SYSTEMONE_API_KEY"); k != "" {
			apiKey = k
		} else if k2 := os.Getenv("API_KEY"); k2 != "" {
			apiKey = k2
		}
	}

	opts := decisionindex.EngineOptions{
		MaxSlotsPerPass:   soMaxSlots,
		MaxOptionsPerSlot: decisionindex.MaxOptionsPerSlot,
		NaiveLimits:       soNaiveLimits,
		TemperatureScale:  soTemperature,
		MaxConcurrency:    4,
		DualMirror:        soDualMirror,
		NullPriorDebias:   soNullPrior,
		PriorAlpha:        soPriorAlpha,
		BracketSize:       soBracketSize,
		CatchAll:          soCatchAll,
		NoulMode:          soNoulMode,
		OptionOrder:       soOptionOrder,
		Labels:            soLabels,
		PromptLayout:      soPromptLayout,
	}

	// A network server: image URLs in requests are fetched by pkg/client (and sent upstream as data: URIs) with a client
	// that only connects to public addresses, at connect time and on every redirect.
	allowLocalImagePaths = false
	client.ImageHTTPClient = imageFetchClient
	cli := GetClientForURL(upstream)

	mux := http.NewServeMux()

	// Authentication middleware
	authWrap := func(handler http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if apiKey != "" {
				authHeader := r.Header.Get("Authorization")
				expected := "Bearer " + apiKey
				if authHeader != expected {
					w.Header().Set("WWW-Authenticate", `Bearer realm="dgem-systemone"`)
					http.Error(w, `{"error":"unauthorized: invalid or missing Bearer token"}`, http.StatusUnauthorized)
					return
				}
			}
			handler(w, r)
		}
	}

	soHandler := decisionindex.NewSystemOneHTTPHandler(cli, opts)
	mux.HandleFunc("/v1/systemone", authWrap(soHandler))

	mux.HandleFunc("/health", systemOneHealth)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"service":     "dgem-systemone-adapter",
			"description": "Standalone HTTP adapter serving POST /v1/systemone for apolinario/decision-index",
			"routes": []string{
				"POST /v1/systemone",
				"GET  /health",
			},
		})
	})

	addr := fmt.Sprintf("%s:%d", soHost, soPort)
	fmt.Fprintf(os.Stderr, "🚀 dgem SystemOne adapter listening on http://%s -> upstream %s (T=%.2f, max_slots=%d, bracket=%d, auth=%v)\n",
		addr, upstream, soTemperature, soMaxSlots, soBracketSize, apiKey != "")

	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
	}
	return srv.ListenAndServe()
}
