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
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile     string
	serverURL   string
	modelName   string
	timeout     time.Duration
	showStats   bool
	authToken   string
	gcpAuth     bool
	iapClientID string

	tokenCacheMu sync.Mutex
	tokenCache   = map[string]cachedToken{}
)

type cachedToken struct {
	token   string
	expires time.Time
}

// RootCmd is the base command for dgem.
var RootCmd = &cobra.Command{
	Use:   "dgem",
	Short: "dgem is a CLI assistant and evaluation tool for DiffusionGemma",
	Long: `dgem interacts with local or remote DiffusionGemma inference engines.
It provides high-performance discrete diffusion slot readout decisions (slot-evaluated
in a single forward pass without conversational text overhead),
generative prompt execution, Go template rendering, and comprehensive request/response telemetry.`,
}

// Execute runs the root command.
func Execute() {
	if err := RootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	RootCmd.AddGroup(
		&cobra.Group{ID: "core", Title: "Core Inference & Template Commands:"},
		&cobra.Group{ID: "eval", Title: "Benchmark & Calibration Suites:"},
	)

	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.config/dgem/config.yaml or .dgem.yaml)")
	RootCmd.PersistentFlags().StringVarP(&serverURL, "url", "u", "http://127.0.0.1:8080/v1", "Base URL for the DiffusionGemma server")
	RootCmd.PersistentFlags().StringVarP(&modelName, "model", "m", "diffgemma-26b-a4b-it-q4", "Model ID to target")
	RootCmd.PersistentFlags().DurationVar(&timeout, "timeout", 120*time.Second, "Client timeout")
	RootCmd.PersistentFlags().BoolVarP(&showStats, "stats", "s", false, "Display execution timing, token breakdown, and inference telemetry")
	RootCmd.PersistentFlags().StringVarP(&authToken, "token", "k", "", "Authorization Bearer token / API key")
	RootCmd.PersistentFlags().BoolVar(&gcpAuth, "gcp-auth", false, "Automatically obtain GCP IAM identity token via gcloud auth print-identity-token or Cloud Run metadata server")
	RootCmd.PersistentFlags().StringVar(&iapClientID, "iap-client-id", "", "OAuth 2.0 Client ID / Audience for Identity-Aware Proxy (IAP) protected endpoints (env: DGEM_IAP_CLIENT_ID)")
	RootCmd.PersistentFlags().StringVar(&serveVertexURL, "vertex-url", "", "Vertex AI Dedicated Endpoint ID or /invoke/* URL (env: DGEM_VERTEX_URL)")

	viper.BindPFlag("url", RootCmd.PersistentFlags().Lookup("url"))
	viper.BindPFlag("model", RootCmd.PersistentFlags().Lookup("model"))
	viper.BindPFlag("timeout", RootCmd.PersistentFlags().Lookup("timeout"))
	viper.BindPFlag("stats", RootCmd.PersistentFlags().Lookup("stats"))
	viper.BindPFlag("token", RootCmd.PersistentFlags().Lookup("token"))
	viper.BindPFlag("gcp_auth", RootCmd.PersistentFlags().Lookup("gcp-auth"))
	viper.BindPFlag("iap_client_id", RootCmd.PersistentFlags().Lookup("iap-client-id"))
	viper.BindPFlag("vertex_url", RootCmd.PersistentFlags().Lookup("vertex-url"))
	RootCmd.PersistentFlags().Int("http-retries", 0, "Retry HTTP 429/503 responses up to N times with backoff (default 0; serve and mcp default to 3)")
	viper.BindPFlag("http_retries", RootCmd.PersistentFlags().Lookup("http-retries"))
}

// defaultRetriesForLongRunning enables HTTP 429/503 retries for serve/mcp unless the user set --http-retries.
func defaultRetriesForLongRunning() {
	if f := RootCmd.PersistentFlags().Lookup("http-retries"); f != nil && !f.Changed {
		viper.Set("http_retries", 3)
	}
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		if err == nil {
			viper.AddConfigPath(home + "/.config/dgem")
		}
		viper.AddConfigPath(".")
		viper.SetConfigType("yaml")
		viper.SetConfigName(".dgem")
	}

	viper.SetEnvPrefix("DGEM")
	viper.AutomaticEnv()
	_ = viper.ReadInConfig()
}

// fetchADCTokens reads Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS
// or ~/.config/gcloud/application_default_credentials.json) and exchanges the authorized_user
// refresh_token directly with https://oauth2.googleapis.com/token for both an OAuth2 access_token
// and an OIDC id_token (accepted by Cloud Run IAP via programmaticClients).
func fetchADCTokens() (accessToken string, idToken string) {
	adcPath := strings.TrimSpace(os.Getenv("GOOGLE_APPLICATION_CREDENTIALS"))
	if adcPath == "" {
		if home, err := os.UserHomeDir(); err == nil {
			adcPath = home + "/.config/gcloud/application_default_credentials.json"
		}
	}
	if adcPath == "" {
		return "", ""
	}
	raw, err := os.ReadFile(adcPath)
	if err != nil {
		return "", ""
	}
	var cred struct {
		ClientID     string `json:"client_id"`
		ClientSecret string `json:"client_secret"`
		RefreshToken string `json:"refresh_token"`
		Type         string `json:"type"`
	}
	if err := json.Unmarshal(raw, &cred); err != nil || cred.RefreshToken == "" || cred.ClientID == "" {
		return "", ""
	}
	form := url.Values{}
	form.Set("client_id", cred.ClientID)
	form.Set("client_secret", cred.ClientSecret)
	form.Set("refresh_token", cred.RefreshToken)
	form.Set("grant_type", "refresh_token")

	httpClient := &http.Client{Timeout: 4 * time.Second}
	resp, err := httpClient.PostForm("https://oauth2.googleapis.com/token", form)
	if err != nil {
		return "", ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", ""
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", ""
	}
	var tokResp struct {
		AccessToken string `json:"access_token"`
		IDToken     string `json:"id_token"`
	}
	if err := json.Unmarshal(body, &tokResp); err != nil {
		return "", ""
	}
	return strings.TrimSpace(tokResp.AccessToken), strings.TrimSpace(tokResp.IDToken)
}

// isRemoteGCPURL returns true if u targets Cloud Run (*.run.app), Vertex AI, or custom gateway hosts in DGEM_GATEWAY_HOSTS.
func isRemoteGCPURL(u string) bool {
	lower := strings.ToLower(strings.TrimSpace(u))
	if strings.Contains(lower, ".run.app") || client.IsVertexEndpointURL(u) {
		return true
	}
	if extra := os.Getenv("DGEM_GATEWAY_HOSTS"); extra != "" {
		for _, h := range strings.Split(extra, ",") {
			if h = strings.TrimSpace(strings.ToLower(h)); h != "" && strings.Contains(lower, h) {
				return true
			}
		}
	}
	return false
}

func isLoopbackURL(u string) bool {
	lower := strings.ToLower(strings.TrimSpace(u))
	return strings.Contains(lower, "127.0.0.1") ||
		strings.Contains(lower, "localhost") ||
		strings.Contains(lower, "::1") ||
		strings.Contains(lower, "0.0.0.0")
}

// FetchGCPAccessToken mints an OAuth2 access token (cloud-platform scope) for Vertex AI Endpoints (:rawPredict).
// It checks (1) GOOGLE_OAUTH_ACCESS_TOKEN, (2) GCP Metadata Server, (3) pure-Go ADC (application_default_credentials.json), and (4) gcloud CLI.
func FetchGCPAccessToken() string {
	const cacheKey = "oauth2_access_token|cloud-platform"
	tokenCacheMu.Lock()
	if cached, ok := tokenCache[cacheKey]; ok && time.Now().Before(cached.expires) {
		tokenCacheMu.Unlock()
		return cached.token
	}
	tokenCacheMu.Unlock()

	if envTok := strings.TrimSpace(os.Getenv("GOOGLE_OAUTH_ACCESS_TOKEN")); envTok != "" {
		return envTok
	}

	var token string
	metaURL := "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
	if req, err := http.NewRequest("GET", metaURL, nil); err == nil {
		req.Header.Set("Metadata-Flavor", "Google")
		httpClient := &http.Client{Timeout: 800 * time.Millisecond}
		if resp, err := httpClient.Do(req); err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				if body, err := io.ReadAll(resp.Body); err == nil {
					var parsed struct {
						AccessToken string `json:"access_token"`
					}
					if json.Unmarshal(body, &parsed) == nil && parsed.AccessToken != "" {
						token = strings.TrimSpace(parsed.AccessToken)
					}
				}
			}
		}
	}

	// Pure-Go ADC fast path (~80ms, caches both access_token and id_token)
	if token == "" {
		accTok, idTok := fetchADCTokens()
		if accTok != "" {
			token = accTok
		}
		if idTok != "" {
			tokenCacheMu.Lock()
			tokenCache["adc_id_token"] = cachedToken{
				token:   idTok,
				expires: time.Now().Add(45 * time.Minute),
			}
			tokenCacheMu.Unlock()
		}
	}

	if token == "" {
		if out, err := exec.Command("gcloud", "auth", "application-default", "print-access-token").Output(); err == nil {
			token = strings.TrimSpace(string(out))
		}
	}
	if token == "" {
		if out, err := exec.Command("gcloud", "auth", "print-access-token").Output(); err == nil {
			token = strings.TrimSpace(string(out))
		}
	}

	if token != "" {
		tokenCacheMu.Lock()
		tokenCache[cacheKey] = cachedToken{
			token:   token,
			expires: time.Now().Add(45 * time.Minute),
		}
		tokenCacheMu.Unlock()
	}
	return token
}

// FetchGCPIdentityToken mints an OIDC identity token for Cloud Run IAM / IAP,
// or automatically mints an OAuth2 access token when targetURL is a Vertex AI Endpoint (:rawPredict).
func FetchGCPIdentityToken(explicitAudience, targetURL string) string {
	if client.IsVertexEndpointURL(targetURL) && explicitAudience == "" {
		return FetchGCPAccessToken()
	}

	cacheKey := explicitAudience + "|" + targetURL
	tokenCacheMu.Lock()
	if cached, ok := tokenCache[cacheKey]; ok && time.Now().Before(cached.expires) {
		tokenCacheMu.Unlock()
		return cached.token
	}
	if explicitAudience == "" {
		if cached, ok := tokenCache["adc_id_token"]; ok && time.Now().Before(cached.expires) {
			tokenCacheMu.Unlock()
			return cached.token
		}
	}
	tokenCacheMu.Unlock()

	aud := strings.TrimSpace(explicitAudience)
	if aud == "" && targetURL != "" {
		if parsed, err := url.Parse(targetURL); err == nil && parsed.Scheme != "" && parsed.Host != "" {
			aud = parsed.Scheme + "://" + parsed.Host
		}
	}

	var token string
	// 1. Pure-Go Application Default Credentials (ADC) fast path on workstations
	if os.Getenv("K_SERVICE") == "" && explicitAudience == "" {
		accTok, idTok := fetchADCTokens()
		if idTok != "" {
			token = idTok
			tokenCacheMu.Lock()
			tokenCache["adc_id_token"] = cachedToken{
				token:   idTok,
				expires: time.Now().Add(45 * time.Minute),
			}
			if accTok != "" {
				tokenCache["oauth2_access_token|cloud-platform"] = cachedToken{
					token:   accTok,
					expires: time.Now().Add(45 * time.Minute),
				}
			}
			tokenCacheMu.Unlock()
		}
	}

	// 2. Try GCP Metadata Server inside Cloud Run / GCE containers
	if token == "" && aud != "" {
		metaURL := "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=" + url.QueryEscape(aud)
		req, err := http.NewRequest("GET", metaURL, nil)
		if err == nil {
			req.Header.Set("Metadata-Flavor", "Google")
			httpClient := &http.Client{Timeout: 800 * time.Millisecond}
			if resp, err := httpClient.Do(req); err == nil {
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					if body, err := io.ReadAll(resp.Body); err == nil {
						token = strings.TrimSpace(string(body))
					}
				}
			}
		}
	}

	// 3. Fall back to gcloud CLI on developer workstations
	if token == "" {
		args := []string{"auth", "print-identity-token"}
		if explicitAudience != "" {
			args = append(args, "--audiences="+explicitAudience)
		}
		if out, err := exec.Command("gcloud", args...).Output(); err == nil {
			token = strings.TrimSpace(string(out))
		}
	}

	if token != "" {
		tokenCacheMu.Lock()
		tokenCache[cacheKey] = cachedToken{
			token:   token,
			expires: time.Now().Add(45 * time.Minute),
		}
		tokenCacheMu.Unlock()
	}
	return token
}

// GetClient returns a configured API client using Viper values.
func GetClient() *client.Client {
	if f := RootCmd.PersistentFlags().Lookup("vertex-url"); f != nil && f.Changed {
		norm, err := expandAndValidateVertexURL(f.Value.String())
		if err != nil {
			// Never fall back silently to --url (default localhost): benchmarks would record
			// connection-refused errors as a 0% run against the wrong target.
			fmt.Fprintf(os.Stderr, "Error: --vertex-url: %v\n", err)
			os.Exit(2)
		}
		if norm != "" {
			return GetClientForURL(norm)
		}
	}
	return GetClientForURL(viper.GetString("url"))
}

// GetClientForURL returns a configured API client targeting an explicit upstream URL
// (either a Cloud Run /v1 URL, IAP Gateway URL, or Vertex AI :rawPredict endpoint URL).
func GetClientForURL(targetURL string) *client.Client {
	if strings.TrimSpace(targetURL) == "" {
		targetURL = viper.GetString("url")
	}
	model := viper.GetString("model")
	to := viper.GetDuration("timeout")
	token := viper.GetString("token")
	iapAud := viper.GetString("iap_client_id")

	if token == "" && !isLoopbackURL(targetURL) && (viper.GetBool("gcp_auth") || iapAud != "" || isRemoteGCPURL(targetURL)) {
		aud := iapAud
		if client.IsVertexEndpointURL(targetURL) {
			aud = ""
		}
		token = FetchGCPIdentityToken(aud, targetURL)
	}

	c := client.NewClient(targetURL, model, to)
	c.MaxRetries = viper.GetInt("http_retries")
	if token != "" {
		c.WithAuthToken(token)
	}
	return c
}

// PrintStats prints formatted telemetry from the request and response.
func PrintStats(stats *client.RequestStats) {
	if stats == nil {
		return
	}

	fmt.Println()
	fmt.Println("──────────────────────────────── STATS ────────────────────────────────")
	fmt.Printf("  Model:             %s\n", stats.Model)
	fmt.Printf("  Endpoint:          %s\n", stats.Endpoint)

	fmt.Println("\n  Timing:")
	fmt.Printf("    • Total Wall Time:     %v\n", stats.WallTime.Round(time.Millisecond))
	if stats.PrefillMs > 0 {
		fmt.Printf("    • Server Prefill:      %.0f ms\n", stats.PrefillMs)
	}
	if stats.DenoiseMs > 0 {
		fmt.Printf("    • Server Denoise:      %.0f ms\n", stats.DenoiseMs)
	}

	fmt.Println("\n  Token Breakdown:")
	fmt.Printf("    • Prompt Tokens:       %d tokens\n", stats.PromptTokens)
	if stats.ReusedTokens > 0 {
		hitPct := float64(stats.ReusedTokens) / float64(stats.PromptTokens) * 100
		fmt.Printf("    • KV Cache Reused:     %d tokens (%.1f%% cache hit rate)\n", stats.ReusedTokens, hitPct)
	}
	fmt.Printf("    • Completion Tokens:   %d tokens\n", stats.OutputTokens)
	fmt.Printf("    • Total Tokens:        %d tokens\n", stats.TotalTokens)

	if stats.Diagnostics != nil {
		fmt.Println("\n  Inference Mechanics:")
		if stats.DenoiseSteps > 0 {
			fmt.Printf("    • Denoise Steps:       %d step(s)\n", stats.DenoiseSteps)
		}
		if stats.SamplesN > 0 {
			policy := stats.Diagnostics.Samples.Policy
			fmt.Printf("    • Noise Samples (N):   %d sample(s) (policy: %s, threshold: %.2f)\n",
				stats.SamplesN, policy.Mode, policy.Threshold)
			fmt.Printf("    • Multi-Read Extended: %v (first-read max entropy: %.4f nats)\n",
				stats.Extended, policy.FirstReadMaxEntropy)
		}

		if len(stats.Diagnostics.Questions) > 0 {
			fmt.Println("\n  Question Diagnostics:")
			for qID, diag := range stats.Diagnostics.Questions {
				fmt.Printf("    • %-12s: argmax='%s' (entropy=%.4f nats, label_mass=%.1f%%)\n",
					qID, strings.TrimSpace(diag.ArgmaxToken), diag.Entropy, diag.LabelMass*100)
			}
		}
	}
	fmt.Println("───────────────────────────────────────────────────────────────────────")
}
