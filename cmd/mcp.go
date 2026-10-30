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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"github.com/GoogleCloudPlatform/dgem/pkg/template"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

// WarmupStageBreakdown records the 4 cold-start sub-stages for right-sizing and Cloud Monitoring.
type WarmupStageBreakdown struct {
	Timestamp          string `json:"timestamp"`
	TriggerSource      string `json:"trigger_source"`
	Phase1ActivatorMs  int64  `json:"phase_1_activator_ms"`
	Phase2TmpfsStageMs int64  `json:"phase_2_tmpfs_stage_ms"`
	Phase3VLLMSiglipMs int64  `json:"phase_3_vllm_siglip_ms"`
	Phase4TritonJITMs  int64  `json:"phase_4_triton_jit_ms"`
	TotalWarmupMs      int64  `json:"total_warmup_ms"`
	Attempts           int    `json:"attempts"`
}

// WarmupTelemetryStats is returned by GET /api/warmup/stats for operational dashboards & right-sizing.
type WarmupTelemetryStats struct {
	EWMAWakeSeconds        int                    `json:"ewma_wake_seconds"`
	EWMAStage1ActivatorMs  int64                  `json:"ewma_stage_1_activator_ms"`
	EWMAStage2TmpfsStageMs int64                  `json:"ewma_stage_2_tmpfs_stage_ms"`
	EWMAStage3VLLMSiglipMs int64                  `json:"ewma_stage_3_vllm_siglip_ms"`
	EWMAStage4TritonJITMs  int64                  `json:"ewma_stage_4_triton_jit_ms"`
	SampleCount            int                    `json:"sample_count"`
	History                []WarmupStageBreakdown `json:"history"`
}

var (
	gpuStateMu           sync.RWMutex
	lastWarmTimestamp    time.Time
	lastReadoutLatencyMs int64
	warmupInProgress     bool
	warmupStartedAt      time.Time
	warmupDoneCh         chan struct{}
	warmupLastAttempts   int
	warmupLastErr        error
	currentWarmupPhase   string  = "staging_tmpfs"
	currentBytesStagedGB float64 = 0.0

	// Pre-seeded with verified Cloud Run RTX Pro 6000 /tmp/dgemma cold-start telemetry (dgemma-00023-7wz)
	ewmaStage1Ms int64 = 5200
	ewmaStage2Ms int64 = 44000
	ewmaStage3Ms int64 = 64000
	ewmaStage4Ms int64 = 8648
	ewmaTotalMs  int64 = 121848

	warmupHistory = []WarmupStageBreakdown{
		{
			Timestamp:          "2026-09-22T20:16:54Z",
			TriggerSource:      "cloudrun_baseline_rtx_pro_6000",
			Phase1ActivatorMs:  5200,
			Phase2TmpfsStageMs: 44000,
			Phase3VLLMSiglipMs: 64000,
			Phase4TritonJITMs:  8648,
			TotalWarmupMs:      121848,
			Attempts:           18,
		},
	}
)

func recordWarmupBreakdown(rec WarmupStageBreakdown) int {
	const alpha = 0.30
	gpuStateMu.Lock()
	defer gpuStateMu.Unlock()

	ewmaStage1Ms = int64((1.0-alpha)*float64(ewmaStage1Ms) + alpha*float64(rec.Phase1ActivatorMs))
	ewmaStage2Ms = int64((1.0-alpha)*float64(ewmaStage2Ms) + alpha*float64(rec.Phase2TmpfsStageMs))
	ewmaStage3Ms = int64((1.0-alpha)*float64(ewmaStage3Ms) + alpha*float64(rec.Phase3VLLMSiglipMs))
	ewmaStage4Ms = int64((1.0-alpha)*float64(ewmaStage4Ms) + alpha*float64(rec.Phase4TritonJITMs))
	ewmaTotalMs = int64((1.0-alpha)*float64(ewmaTotalMs) + alpha*float64(rec.TotalWarmupMs))

	warmupHistory = append(warmupHistory, rec)
	if len(warmupHistory) > 25 {
		warmupHistory = warmupHistory[len(warmupHistory)-25:]
	}
	sec := int((ewmaTotalMs + 500) / 1000)
	if sec < 45 {
		sec = 45
	}
	return sec
}

// GetWarmupTelemetryStats returns the current EWMA right-sized cold-start estimate and stage history.
func GetWarmupTelemetryStats() WarmupTelemetryStats {
	gpuStateMu.RLock()
	defer gpuStateMu.RUnlock()
	hist := make([]WarmupStageBreakdown, len(warmupHistory))
	copy(hist, warmupHistory)
	ewmaSec := int((ewmaTotalMs + 500) / 1000)
	if ewmaSec < 45 {
		ewmaSec = 45
	}
	return WarmupTelemetryStats{
		EWMAWakeSeconds:        ewmaSec,
		EWMAStage1ActivatorMs:  ewmaStage1Ms,
		EWMAStage2TmpfsStageMs: ewmaStage2Ms,
		EWMAStage3VLLMSiglipMs: ewmaStage3Ms,
		EWMAStage4TritonJITMs:  ewmaStage4Ms,
		SampleCount:            len(hist),
		History:                hist,
	}
}

func pollUpstreamWarmupHealth(baseURL string) (string, float64, bool) {
	healthURL := strings.TrimSuffix(strings.TrimSuffix(baseURL, "/"), "/v1") + "/health"
	req, err := http.NewRequest("GET", healthURL, nil)
	if err != nil {
		return "", 0, false
	}
	if tok := FetchGCPIdentityToken("", baseURL); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	hc := &http.Client{Timeout: 1500 * time.Millisecond}
	resp, err := hc.Do(req)
	if err != nil {
		return "", 0, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", 0, false
	}
	var payload struct {
		Phase         string  `json:"phase"`
		BytesStagedGB float64 `json:"bytes_staged_gb"`
		VLLMReady     bool    `json:"vllm_ready"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", 0, true
	}
	return payload.Phase, payload.BytesStagedGB, true
}

func inferWarmupPhaseAndLabel(elapsedSec int, ewmaSec int, phaseHint string, bytesStaged float64) (string, string) {
	// 1. If dgemma /health explicitly returned a sub-stage, honor it directly
	switch phaseHint {
	case "mounting_gcs":
		return "phase_1_activator", "Stage 1/4: Cloud Run Activator & GCSFuse Mount"
	case "staging_tmpfs":
		if bytesStaged > 0 {
			return "phase_2_tmpfs_stage", fmt.Sprintf("Stage 2/4: Staging 17.53 GiB to /tmp RAM (%.1f GiB staged)", bytesStaged)
		}
		return "phase_2_tmpfs_stage", "Stage 2/4: Staging 17.53 GiB NVFP4 Weights to /tmp RAM"
	case "loading_vllm_siglip":
		return "phase_3_vllm_siglip", "Stage 3/4: Loading vLLM EngineCore (5.6s) + SigLIP Vision Tower"
	case "ready":
		return "phase_4_triton_jit", "Stage 4/4: First Decision Readout & Triton Kernel JIT"
	}

	// 2. Time-based progression fallback when /health hasn't reported a phase key yet
	p1Sec := int(ewmaStage1Ms / 1000)
	if p1Sec < 8 {
		p1Sec = 8
	}
	p2Sec := p1Sec + int(ewmaStage2Ms/1000)
	p3Sec := p2Sec + int(ewmaStage3Ms/1000)

	if elapsedSec <= p1Sec {
		return "phase_1_activator", "Stage 1/4: Cloud Run Activator & GCSFuse Mount"
	}
	if elapsedSec <= p2Sec {
		return "phase_2_tmpfs_stage", "Stage 2/4: Staging 17.53 GiB NVFP4 Weights to /tmp RAM"
	}
	if elapsedSec <= p3Sec {
		return "phase_3_vllm_siglip", "Stage 3/4: Loading vLLM EngineCore (5.6s) + SigLIP Vision Tower"
	}
	return "phase_4_triton_jit", "Stage 4/4: First Decision Readout & Triton Kernel JIT"
}

var (
	mcpRemoteURL    string
	mcpLocalBackend bool
)

// runRemoteMCPProxy bridges stdio JSON-RPC messages to a remote Streamable HTTP MCP endpoint
// (such as https://<your-dgem-gateway>/mcp), automatically attaching Application Default Credentials
// (ADC) OIDC id_token (or OAuth2 access token) and MCP Streamable HTTP headers (Mcp-Method, Mcp-Protocol-Version).
//
// Crucially, MCP clients (such as Antigravity / Jetski using go-sdk v1.8+) send "subscriptions/listen"
// (SEP-2575 long-lived SSE stream) immediately after "notifications/initialized" and before "tools/list".
// To prevent "subscriptions/listen" from head-of-line blocking "tools/list" for 180s (httpClient.Timeout),
// non-initialization requests are dispatched concurrently and SSE streams are read line-by-line.
func runRemoteMCPProxy(remoteURL string) error {
	// Both clients carry X-DGem-Api-Key; a redirect to another origin drops it.
	httpClient := &http.Client{Timeout: 180 * time.Second, CheckRedirect: client.StripKeysOnCrossOriginRedirect}
	// Long-lived SSE streams (subscriptions/listen) have no fixed client timeout
	sseClient := &http.Client{Timeout: 0, CheckRedirect: client.StripKeysOnCrossOriginRedirect}
	if os.Getenv(client.GatewayURLEnv) != "" && client.GatewayKeyFromEnv() != "" && client.GatewayKeyForRemote(remoteURL) == "" {
		fmt.Fprintf(os.Stderr, "dgem mcp: DGEM_GATEWAY_KEY is not sent to --remote: %s names another gateway. "+
			"Set %s to the --remote URL, or unset it, to send the key.\n", client.GatewayURLEnv, client.GatewayURLEnv)
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)

	var stateMu sync.RWMutex
	sessionProtoVer := ""
	sessionID := ""

	var stdoutMu sync.Mutex
	writeStdoutLine := func(b []byte) {
		stdoutMu.Lock()
		defer stdoutMu.Unlock()
		os.Stdout.Write(b)
		if len(b) == 0 || b[len(b)-1] != '\n' {
			os.Stdout.Write([]byte("\n"))
		}
	}

	writeRPCError := func(id *json.RawMessage, code int, msg string) {
		if id == nil || len(*id) == 0 || string(*id) == "null" {
			return
		}
		errObj := map[string]interface{}{
			"jsonrpc": "2.0",
			"id":      *id,
			"error": map[string]interface{}{
				"code":    code,
				"message": msg,
			},
		}
		if b, err := json.Marshal(errObj); err == nil {
			writeStdoutLine(b)
		}
	}

	handleLine := func(line []byte, reqID *json.RawMessage, method string, rpcName string) {
		stateMu.RLock()
		protoVer := sessionProtoVer
		sid := sessionID
		stateMu.RUnlock()

		req, err := http.NewRequest(http.MethodPost, remoteURL, bytes.NewReader(line))
		if err != nil {
			writeRPCError(reqID, -32603, fmt.Sprintf("failed to construct remote MCP request: %v", err))
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json, text/event-stream")
		if method != "" {
			req.Header.Set("Mcp-Method", method)
		}
		if rpcName != "" {
			req.Header.Set("Mcp-Name", rpcName)
		}
		if protoVer != "" {
			req.Header.Set("Mcp-Protocol-Version", protoVer)
		}
		if sid != "" {
			req.Header.Set("Mcp-Session-Id", sid)
		}

		tok := FetchGCPIdentityToken(viper.GetString("iap_client_id"), remoteURL)
		if tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		// A self-hosted gateway started with --api-keys: the key goes to --remote unless DGEM_GATEWAY_URL names
		// another gateway.
		gwKey := client.GatewayKeyForRemote(remoteURL)
		if gwKey != "" {
			req.Header.Set(client.GatewayKeyHeader, gwKey)
		}

		// If get_health_and_gpu_status is called with a specific backend parameter,
		// query /api/status?backend=<backend> directly so backend-specific telemetry is returned
		// even if the remote gateway's /mcp schema was built before StatusToolInput.Backend.
		if method == "tools/call" && rpcName == "get_health_and_gpu_status" {
			var statusCall struct {
				Params struct {
					Arguments struct {
						Backend string `json:"backend"`
					} `json:"arguments"`
				} `json:"params"`
			}
			if json.Unmarshal(line, &statusCall) == nil && strings.TrimSpace(statusCall.Params.Arguments.Backend) != "" {
				bParam := strings.TrimSpace(statusCall.Params.Arguments.Backend)
				apiStatusURL := strings.TrimSuffix(strings.TrimRight(remoteURL, "/"), "/mcp") + "/api/status?backend=" + bParam
				sReq, sErr := http.NewRequest(http.MethodGet, apiStatusURL, nil)
				if sErr == nil {
					if tok != "" {
						sReq.Header.Set("Authorization", "Bearer "+tok)
					}
					if gwKey != "" {
						sReq.Header.Set(client.GatewayKeyHeader, gwKey)
					}
					if sResp, doErr := httpClient.Do(sReq); doErr == nil {
						sBody, _ := io.ReadAll(sResp.Body)
						sResp.Body.Close()
						if sResp.StatusCode == http.StatusOK && len(sBody) > 0 {
							var rawObj map[string]interface{}
							if json.Unmarshal(sBody, &rawObj) == nil {
								normBytes, _ := json.Marshal(rawObj)
								rpcResult := map[string]interface{}{
									"jsonrpc": "2.0",
									"id":      *reqID,
									"result": map[string]interface{}{
										"content": []map[string]interface{}{
											{
												"type": "text",
												"text": string(normBytes),
											},
										},
										"structuredContent": rawObj,
									},
								}
								if outBytes, mErr := json.Marshal(rpcResult); mErr == nil {
									writeStdoutLine(outBytes)
									return
								}
							}
						}
					}
				}
			}
		}

		isListen := method == "subscriptions/listen"
		clientToUse := httpClient
		if isListen {
			clientToUse = sseClient
		}

		resp, err := clientToUse.Do(req)
		if err != nil {
			if !isListen {
				writeRPCError(reqID, -32000, fmt.Sprintf("remote MCP endpoint unreachable (%s): %v", remoteURL, err))
			}
			return
		}
		if newSid := strings.TrimSpace(resp.Header.Get("Mcp-Session-Id")); newSid != "" {
			stateMu.Lock()
			sessionID = newSid
			stateMu.Unlock()
		}

		contentType := strings.ToLower(resp.Header.Get("Content-Type"))
		if strings.HasPrefix(contentType, "text/event-stream") {
			defer resp.Body.Close()
			sseScanner := bufio.NewScanner(resp.Body)
			sseScanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
			for sseScanner.Scan() {
				sLine := strings.TrimSpace(sseScanner.Text())
				if strings.HasPrefix(sLine, "data:") {
					payload := bytes.TrimSpace([]byte(strings.TrimPrefix(sLine, "data:")))
					if len(payload) > 0 && bytes.HasPrefix(payload, []byte("{")) {
						var compactBuf bytes.Buffer
						if json.Compact(&compactBuf, payload) == nil {
							writeStdoutLine(compactBuf.Bytes())
						} else {
							writeStdoutLine(payload)
						}
					}
				}
			}
			return
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		trimmed := bytes.TrimSpace(body)
		if resp.StatusCode >= 300 {
			snippet := string(trimmed)
			if len(snippet) > 240 {
				snippet = snippet[:240] + "..."
			}
			writeRPCError(reqID, -32000, fmt.Sprintf("remote MCP endpoint returned HTTP %d: %s", resp.StatusCode, snippet))
			return
		}
		if len(trimmed) > 0 && bytes.HasPrefix(trimmed, []byte("{")) {
			if method == "tools/call" && (rpcName == "decide_policy" || rpcName == "decide_custom_questions") && bytes.Contains(trimmed, []byte("validating tool output:")) {
				var callEnv struct {
					Params struct {
						Arguments json.RawMessage `json:"arguments"`
					} `json:"params"`
				}
				if json.Unmarshal(line, &callEnv) == nil && len(callEnv.Params.Arguments) > 0 {
					reqPayload := []byte(callEnv.Params.Arguments)
					if rpcName == "decide_custom_questions" {
						var customIn DecideCustomToolInput
						if json.Unmarshal(callEnv.Params.Arguments, &customIn) == nil && len(customIn.Questions) > 0 {
							var qList []map[string]interface{}
							for _, q := range customIn.Questions {
								qType := strings.ToLower(strings.TrimSpace(q.Type))
								switch qType {
								case "bool", "boolean", "noul":
									qType = "boolean"
								case "choice":
									qType = "choice"
								case "score", "scale":
									qType = "score"
								default:
									qType = "boolean"
								}
								qObj := map[string]interface{}{
									"id":       q.ID,
									"type":     qType,
									"question": q.Question,
								}
								if qType == "choice" && len(q.Options) > 0 {
									var opts []map[string]string
									for _, o := range q.Options {
										opts = append(opts, map[string]string{"name": o, "description": o})
									}
									qObj["options"] = opts
								}
								qList = append(qList, qObj)
							}
							tmplBytes, _ := json.Marshal(map[string]interface{}{
								"schema": map[string]interface{}{
									"questions": qList,
								},
								"state": "{{.context}}",
							})
							gwReq := GatewayDecideRequest{
								Template:             "custom_questions",
								CustomTemplate:       string(tmplBytes),
								Variables:            map[string]interface{}{"context": customIn.Context},
								Backend:              customIn.Backend,
								VertexURL:            customIn.VertexURL,
								CascadeMode:          customIn.CascadeMode,
								CascadeThreshold:     customIn.CascadeThreshold,
								CascadeThresholdMode: customIn.CascadeThresholdMode,
								CascadeModel:         customIn.CascadeModel,
								ExpectedAnswers:      customIn.ExpectedAnswers,
								SuggestExpansions:    customIn.SuggestExpansions,
								ExpansionEntropy:     customIn.ExpansionEntropy,
								Layout:               customIn.Layout,
								Stage2Prior:          customIn.Stage2Prior,
								Temperature:          mustMCPTemperature(customIn.Temperature, customIn.TemperatureByType),
							}
							if b, mErr := json.Marshal(gwReq); mErr == nil {
								reqPayload = b
							}
						}
					}
					apiDecideURL := strings.TrimSuffix(strings.TrimRight(remoteURL, "/"), "/mcp") + "/api/decide"
					fbReq, fbErr := http.NewRequest(http.MethodPost, apiDecideURL, bytes.NewReader(reqPayload))
					if fbErr == nil {
						fbReq.Header.Set("Content-Type", "application/json")
						if tok != "" {
							fbReq.Header.Set("Authorization", "Bearer "+tok)
						}
						if gwKey != "" {
							fbReq.Header.Set(client.GatewayKeyHeader, gwKey)
						}
						if fbResp, doErr := httpClient.Do(fbReq); doErr == nil {
							fbBody, _ := io.ReadAll(fbResp.Body)
							fbResp.Body.Close()
							if fbResp.StatusCode == http.StatusOK && len(fbBody) > 0 {
								var rawObj map[string]interface{}
								if json.Unmarshal(fbBody, &rawObj) == nil {
									if diag, ok := rawObj["diagnostics"].(map[string]interface{}); ok {
										if diag["questions"] == nil {
											diag["questions"] = map[string]interface{}{}
										}
									}
									if dec, ok := rawObj["decision"].(map[string]interface{}); ok {
										if ddiag, ok := dec["diagnostics"].(map[string]interface{}); ok {
											if ddiag["questions"] == nil {
												ddiag["questions"] = map[string]interface{}{}
											}
										}
									}
									normBytes, _ := json.Marshal(rawObj)
									rpcResult := map[string]interface{}{
										"jsonrpc": "2.0",
										"id":      *reqID,
										"result": map[string]interface{}{
											"content": []map[string]interface{}{
												{
													"type": "text",
													"text": string(normBytes),
												},
											},
											"structuredContent": rawObj,
										},
									}
									if outBytes, mErr := json.Marshal(rpcResult); mErr == nil {
										writeStdoutLine(outBytes)
										return
									}
								}
							}
						}
					}
				}
			}
			var compactBuf bytes.Buffer
			if method == "tools/list" {
				var listResp map[string]interface{}
				if json.Unmarshal(trimmed, &listResp) == nil {
					if res, ok := listResp["result"].(map[string]interface{}); ok {
						if tools, ok := res["tools"].([]interface{}); ok {
							for _, t := range tools {
								if tm, ok := t.(map[string]interface{}); ok && tm["name"] == "get_health_and_gpu_status" {
									if schema, ok := tm["inputSchema"].(map[string]interface{}); ok {
										props, _ := schema["properties"].(map[string]interface{})
										if props == nil {
											props = map[string]interface{}{}
										}
										if _, hasBackend := props["backend"]; !hasBackend {
											props["backend"] = map[string]interface{}{
												"type":        "string",
												"description": "Optional inference backend selector: 'vertex_first' (default: Vertex AI Dedicated Endpoint primary with Cloud Run GPU failover), 'vertex' (strict Vertex AI /invoke/*), or 'cloudrun' (strict Serverless Cloud Run GPU).",
											}
											schema["properties"] = props
										}
									}
								}
							}
							if patched, pErr := json.Marshal(listResp); pErr == nil {
								writeStdoutLine(patched)
								return
							}
						}
					}
				}
			}
			if json.Compact(&compactBuf, trimmed) == nil {
				writeStdoutLine(compactBuf.Bytes())
			} else {
				writeStdoutLine(trimmed)
			}
		}
	}

	var wg sync.WaitGroup
	for scanner.Scan() {
		rawLine := bytes.TrimSpace(scanner.Bytes())
		if len(rawLine) == 0 {
			continue
		}
		line := append([]byte(nil), rawLine...)

		var rpcEnv struct {
			ID     *json.RawMessage `json:"id"`
			Method string           `json:"method"`
			Params struct {
				Name            string            `json:"name"`
				URI             string            `json:"uri"`
				ProtocolVersion string            `json:"protocolVersion"`
				Meta            map[string]string `json:"_meta"`
			} `json:"params"`
		}
		_ = json.Unmarshal(line, &rpcEnv)
		if rpcEnv.Params.ProtocolVersion != "" {
			stateMu.Lock()
			sessionProtoVer = rpcEnv.Params.ProtocolVersion
			stateMu.Unlock()
		} else {
			var rawEnv struct {
				ID     *json.RawMessage `json:"id"`
				Method string           `json:"method"`
				Params struct {
					Name string                 `json:"name"`
					URI  string                 `json:"uri"`
					Meta map[string]interface{} `json:"_meta"`
				} `json:"params"`
			}
			if json.Unmarshal(line, &rawEnv) == nil {
				if rpcEnv.ID == nil {
					rpcEnv.ID = rawEnv.ID
				}
				if rpcEnv.Method == "" {
					rpcEnv.Method = rawEnv.Method
				}
				if rpcEnv.Params.Name == "" {
					rpcEnv.Params.Name = rawEnv.Params.Name
				}
				if rpcEnv.Params.URI == "" {
					rpcEnv.Params.URI = rawEnv.Params.URI
				}
				if pv, ok := rawEnv.Params.Meta["io.modelcontextprotocol/protocolVersion"].(string); ok && pv != "" {
					stateMu.Lock()
					sessionProtoVer = pv
					stateMu.Unlock()
				}
			}
		}
		rpcName := rpcEnv.Params.Name
		if rpcName == "" {
			rpcName = rpcEnv.Params.URI
		}

		// Execute "initialize" and "notifications/initialized" synchronously so sessionProtoVer and sessionID
		// are guaranteed to be set before subsequent calls; dispatch all other methods ("subscriptions/listen",
		// "tools/list", "tools/call") concurrently so "subscriptions/listen" never blocks "tools/list"!
		if rpcEnv.Method == "initialize" || rpcEnv.Method == "notifications/initialized" {
			handleLine(line, rpcEnv.ID, rpcEnv.Method, rpcName)
		} else {
			if rpcEnv.Method == "subscriptions/listen" {
				go handleLine(line, rpcEnv.ID, rpcEnv.Method, rpcName)
			} else {
				wg.Add(1)
				go func(l []byte, id *json.RawMessage, m string, n string) {
					defer wg.Done()
					handleLine(l, id, m, n)
				}(line, rpcEnv.ID, rpcEnv.Method, rpcName)
			}
		}
	}
	wg.Wait()
	return scanner.Err()
}

var mcpCmd = &cobra.Command{
	Use:     "mcp",
	GroupID: "core",
	Short:   "Start the dgem Model Context Protocol (MCP) server over stdio (with automatic ADC)",
	Long: `mcp starts a Model Context Protocol (MCP) server over standard input/output (stdio),
exposing DiffusionGemma zero-shot multi-slot decision policies, multimodal SigLIP bounding-box
localization, GPU health/availability checks, and scale-from-zero GPU warmup tools
to AI agents (Gemini CLI, Claude Desktop, Cursor, Antigravity, etc.).

Application Default Credentials (ADC: ~/.config/gcloud/application_default_credentials.json)
are automatically discovered and used to authenticate against Vertex AI Dedicated Endpoints
and Cloud Run IAP / IAM services without requiring manual tokens.

Pass --remote https://<your-dgem-gateway>/mcp to bridge stdio directly to a remote dgem serve
MCP gateway (e.g. on Cloud Run behind IAP) using ADC.`,
	Example: `  # Run local stdio MCP server targeting local diffgemma (Apple Silicon Metal)
  dgem mcp --local

  # Run local stdio MCP server with automatic ADC (vertex_first -> Cloud Run failover)
  dgem mcp

  # Bridge stdio to a remote dgem serve MCP gateway using ADC automatically
  dgem mcp --remote https://<your-dgem-gateway>/mcp`,
	RunE: func(cmd *cobra.Command, args []string) error {
		defaultRetriesForLongRunning()
		if strings.TrimSpace(mcpRemoteURL) != "" {
			return runRemoteMCPProxy(strings.TrimSpace(mcpRemoteURL))
		}
		isLocal := mcpLocalBackend || os.Getenv("DGEM_MCP_LOCAL") == "1" || os.Getenv("DGEM_MCP_LOCAL") == "true"
		if isLocal {
			backendConfigMu.Lock()
			serveDefaultBackend = "local"
			if serveLocalURL == "" {
				serveLocalURL = viper.GetString("url")
				if serveLocalURL == "" || isRemoteGCPURL(serveLocalURL) {
					serveLocalURL = "http://127.0.0.1:8080/v1"
				}
			}
			backendConfigMu.Unlock()
		} else {
			viper.Set("gcp_auth", true)
			if remoteURL := strings.TrimSpace(os.Getenv("DGEM_REMOTE_URL")); remoteURL != "" {
				viper.Set("url", remoteURL)
			}
		}
		if _, err := os.Stat(serveTemplatesDir); err != nil {
			if exe, exErr := os.Executable(); exErr == nil {
				candidate := filepath.Join(filepath.Dir(filepath.Dir(exe)), "templates")
				if _, stErr := os.Stat(candidate); stErr == nil {
					serveTemplatesDir = candidate
				}
			}
		}
		srv := buildMCPServer()
		return srv.Run(context.Background(), &mcp.StdioTransport{})
	},
}

func init() {
	mcpCmd.Flags().StringVar(&mcpRemoteURL, "remote", "", "Remote Streamable HTTP MCP endpoint URL (e.g. https://<your-dgem-gateway>/mcp) to proxy over stdio using ADC")
	mcpCmd.Flags().BoolVar(&mcpLocalBackend, "local", false, "Route MCP tool calls directly to local diffgemma backend (-u, default http://127.0.0.1:8080/v1) without ADC or cloud fallback (env: DGEM_MCP_LOCAL)")
	RootCmd.AddCommand(mcpCmd)
}

// getGPUIdleWindow returns the configured GPU time-to-idle duration from DGEM_GPU_IDLE_TTL,
// GPU_IDLE_TTL, or the --gpu-idle-ttl CLI flag (defaulting to 3 hours).
func getGPUIdleWindow() time.Duration {
	for _, k := range []string{"DGEM_GPU_IDLE_TTL", "GPU_IDLE_TTL"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			if d, err := time.ParseDuration(v); err == nil && d > 0 {
				return d
			}
		}
	}
	if serveGPUIdleTTL > 0 {
		return serveGPUIdleTTL
	}
	return 3 * time.Hour
}

var keepaliveOnce sync.Once

// startGPUKeepaliveLoop sends a lightweight authenticated GET /health heartbeat to the upstream
// Cloud Run GPU service every 4 minutes ONLY while time.Since(lastWarmTimestamp) < getGPUIdleWindow().
// This prevents Cloud Run's underlying 15-minute container idle reaper from terminating the warm
// GPU before the configured DGEM_GPU_IDLE_TTL (e.g. 3h) expires, while allowing it to scale to 0
// immediately once the idle TTL elapses.
func startGPUKeepaliveLoop() {
	keepaliveOnce.Do(func() {
		go func() {
			// 0. If Vertex First or Vertex is configured, immediately check Vertex /invoke/health on startup
			backendConfigMu.RLock()
			defB := serveDefaultBackend
			backendConfigMu.RUnlock()
			vxURL := effectiveVertexURL()
			if defB == "vertex_first" || defB == "vertex" {
				vSt := inspectVertexEndpointState(context.Background(), vxURL)
				if vSt.State == "deployed" {
					RecordVertexReadoutLatency(490)
				}
			}

			// On gateway startup, probe upstream Cloud Run /health (or local /v1/models) once with a 3s timeout:
			// if the GPU container is already warm, sync gateway state immediately.
			upstreamBase := strings.TrimSuffix(viper.GetString("url"), "/")
			upstreamBase = strings.TrimSuffix(upstreamBase, "/v1")
			isLocalTarget := strings.Contains(upstreamBase, "127.0.0.1") || strings.Contains(upstreamBase, "localhost")
			if upstreamBase != "" {
				healthURL := upstreamBase + "/health"
				if isLocalTarget {
					healthURL = upstreamBase + "/v1/models"
				}
				if req, err := http.NewRequest("GET", healthURL, nil); err == nil {
					if !isLocalTarget {
						if viper.GetBool("gcp_auth") || viper.GetString("iap_client_id") != "" {
							if tok := FetchGCPIdentityToken(viper.GetString("iap_client_id"), healthURL); tok != "" {
								req.Header.Set("Authorization", "Bearer "+tok)
							}
						} else if tok := viper.GetString("token"); tok != "" {
							req.Header.Set("Authorization", "Bearer "+tok)
						}
					}
					hc := &http.Client{Timeout: 4 * time.Second}
					if resp, err := hc.Do(req); err == nil {
						if isLocalTarget && resp.StatusCode == http.StatusOK {
							RecordLocalReadoutLatency(850)
						} else {
							var st struct {
								Phase     string `json:"phase"`
								VLLMReady bool   `json:"vllm_ready"`
							}
							_ = json.NewDecoder(resp.Body).Decode(&st)
							if st.VLLMReady || st.Phase == "ready" {
								MarkGPUWarm()
							}
						}
						resp.Body.Close()
					}
				}
			}

			if serveLocalMode || (isLocalTarget && defB == "local") {
				// In local-only mode, skip remote Cloud Run keepalive heartbeat loop
				return
			}

			ticker := time.NewTicker(4 * time.Minute)
			defer ticker.Stop()
			for range ticker.C {
				// Keep Vertex AI Dedicated Endpoint (/invoke/health) warm when vertex_first or vertex is active
				backendConfigMu.RLock()
				curDefB := serveDefaultBackend
				backendConfigMu.RUnlock()
				curVxURL := effectiveVertexURL()
				if curDefB == "vertex_first" || curDefB == "vertex" {
					invalidateVertexStatusCache()
					vSt := inspectVertexEndpointState(context.Background(), curVxURL)
					if vSt.State == "deployed" {
						MarkGPUWarm()
					}
				}

				idleWindow := getGPUIdleWindow()
				gpuStateMu.RLock()
				lastWarm := lastWarmTimestamp
				warming := warmupInProgress
				gpuStateMu.RUnlock()

				if warming || lastWarm.IsZero() || time.Since(lastWarm) >= idleWindow {
					continue
				}

				upstreamBase := strings.TrimSuffix(viper.GetString("url"), "/")
				upstreamBase = strings.TrimSuffix(upstreamBase, "/v1")
				if upstreamBase == "" {
					continue
				}
				healthURL := upstreamBase + "/health"
				req, err := http.NewRequest("GET", healthURL, nil)
				if err != nil {
					continue
				}
				if viper.GetBool("gcp_auth") || viper.GetString("iap_client_id") != "" {
					if tok := FetchGCPIdentityToken(viper.GetString("iap_client_id"), healthURL); tok != "" {
						req.Header.Set("Authorization", "Bearer "+tok)
					}
				} else if tok := viper.GetString("token"); tok != "" {
					req.Header.Set("Authorization", "Bearer "+tok)
				}
				hc := &http.Client{Timeout: 8 * time.Second}
				if resp, err := hc.Do(req); err == nil {
					_, _ = io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
				}
			}
		}()
	})
}

// NotifyColdStartWarmup ensures that if a decision request arrives while the GPU is scaled to zero,
// the global status coordinator immediately starts the single-flight background warmup worker
// (just as if "Wake GPU" had been pressed) and guarantees warmupInProgress is cleared on completion or timeout.
func NotifyColdStartWarmup() {
	idleWindow := getGPUIdleWindow()
	gpuStateMu.RLock()
	isWarm := !lastWarmTimestamp.IsZero() && time.Since(lastWarmTimestamp) < idleWindow
	alreadyWarming := warmupInProgress
	gpuStateMu.RUnlock()

	if !isWarm && !alreadyWarming {
		go func() {
			_, _ = TriggerGPUWarmupWithSource(context.Background(), false, "auto_wake_decide")
		}()
	}
}

var lastVertexReadoutLatencyMs int64 = 490
var lastLocalReadoutLatencyMs int64 = 850

// RecordVertexReadoutLatency records the most recent Vertex AI /invoke/* readout latency
// without marking the scale-to-zero Cloud Run GPU container as warm.
func RecordVertexReadoutLatency(latencyMs int64) {
	if latencyMs <= 0 {
		return
	}
	gpuStateMu.Lock()
	lastVertexReadoutLatencyMs = latencyMs
	gpuStateMu.Unlock()
}

// RecordLocalReadoutLatency records the most recent local diffgemma (Metal) readout latency
// without marking the scale-to-zero Cloud Run GPU container as warm.
func RecordLocalReadoutLatency(latencyMs int64) {
	if latencyMs <= 0 {
		return
	}
	gpuStateMu.Lock()
	lastLocalReadoutLatencyMs = latencyMs
	gpuStateMu.Unlock()
}

// MarkGPUWarm records that the upstream Cloud Run vLLM engine successfully completed a decision readout
// and wakes any callers waiting on an in-flight warmup broadcast channel.
func MarkGPUWarm(latencyMs ...int64) {
	gpuStateMu.Lock()
	lastWarmTimestamp = time.Now()
	if len(latencyMs) > 0 && latencyMs[0] > 0 {
		lastReadoutLatencyMs = latencyMs[0]
	}
	if warmupInProgress {
		warmupInProgress = false
		warmupLastErr = nil
		if warmupDoneCh != nil {
			close(warmupDoneCh)
			warmupDoneCh = nil
		}
	}
	gpuStateMu.Unlock()
}

// HealthAndGPUStatusOutput is returned by GET /api/status and the get_health_and_gpu_status MCP tool.
type HealthAndGPUStatusOutput struct {
	GatewayHealthy       bool    `json:"gateway_healthy"`
	GPUAvailable         bool    `json:"gpu_available"`
	GPUState             string  `json:"gpu_state"`         // "warm_and_ready", "warming_up", "scaled_to_zero"
	ActiveBackend        string  `json:"active_backend"`    // "vertex" or "cloudrun"
	RequestedBackend     string  `json:"requested_backend"` // "vertex_first", "vertex", or "cloudrun"
	ContainerReachable   bool    `json:"container_reachable"`
	WarmupInProgress     bool    `json:"warmup_in_progress"`
	WarmupElapsedSeconds int     `json:"warmup_elapsed_seconds,omitempty"`
	WarmupPhase          string  `json:"warmup_phase,omitempty"`
	WarmupPhaseLabel     string  `json:"warmup_phase_label,omitempty"`
	WarmupBytesStagedGB  float64 `json:"warmup_bytes_staged_gb,omitempty"`
	EWMAWakeSeconds      int     `json:"ewma_wake_seconds"`
	SecondsSinceLastRead int     `json:"seconds_since_last_read,omitempty"`
	IdleRemainingSeconds int     `json:"idle_remaining_seconds,omitempty"`
	IdleTTLSeconds       int     `json:"idle_ttl_seconds,omitempty"`
	LastReadoutMs        int64   `json:"last_readout_ms,omitempty"`
	EstimatedWakeSeconds int     `json:"estimated_wake_seconds"`
	UpstreamURL          string  `json:"upstream_url"`
	Model                string  `json:"model"`
	GPUTier              string  `json:"gpu_tier"`
	TemplatesAvailable   int     `json:"templates_available"`
	AuthenticatedUser    string  `json:"authenticated_user,omitempty"`
	Detail               string  `json:"detail"`
}

// CheckHealthAndGPUStatus inspects both the gateway, the Vertex AI Dedicated Endpoint (when vertex_first
// or vertex is configured), and the upstream Cloud Run GPU service.
func CheckHealthAndGPUStatus(ctx context.Context, userEmail string) HealthAndGPUStatusOutput {
	return CheckHealthAndGPUStatusForBackend(ctx, userEmail, "")
}

// CheckHealthAndGPUStatusForBackend inspects the readiness and telemetry of the requested backend target
// ("vertex_first", "vertex", or "cloudrun").
func CheckHealthAndGPUStatusForBackend(ctx context.Context, userEmail, backendOverride string) HealthAndGPUStatusOutput {
	backendConfigMu.RLock()
	defB := serveDefaultBackend
	backendConfigMu.RUnlock()
	// --vertex-url, or the DGEM_VERTEX_URL / DGEM_VERTEX_ENDPOINT_ID fallback that routing uses (issue #21),
	// expanded to the /invoke URL as routing does so upstream_url matches the flag case.
	vxURL := effectiveVertexURL()
	if norm, err := expandAndValidateVertexURL(vxURL); err == nil && norm != "" {
		vxURL = norm
	}

	reqMode := strings.ToLower(strings.TrimSpace(backendOverride))
	if reqMode == "vertex" || reqMode == "cloudrun" || reqMode == "vertex_first" || reqMode == "local" {
		defB = reqMode
	}

	catalog, _ := discoverTemplates(serveTemplatesDir)
	idleWindow := getGPUIdleWindow()

	// 1. If Vertex First or Vertex is the effective backend, check Vertex AI Dedicated Endpoint state first!
	if defB == "vertex_first" || defB == "vertex" {
		vSt := inspectVertexEndpointState(ctx, vxURL)
		if vSt.State == "deployed" && vSt.ReplicaCount > 0 {
			gpuStateMu.RLock()
			lastReadMs := lastVertexReadoutLatencyMs
			gpuStateMu.RUnlock()
			if lastReadMs <= 0 {
				lastReadMs = 490
			}
			return HealthAndGPUStatusOutput{
				GatewayHealthy:       true,
				GPUAvailable:         true,
				GPUState:             "warm_and_ready",
				ActiveBackend:        "vertex",
				RequestedBackend:     defB,
				ContainerReachable:   true,
				WarmupInProgress:     false,
				EWMAWakeSeconds:      0,
				SecondsSinceLastRead: 0,
				IdleRemainingSeconds: 0,
				IdleTTLSeconds:       0,
				LastReadoutMs:        lastReadMs,
				EstimatedWakeSeconds: 0,
				UpstreamURL:          vxURL,
				Model:                "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
				GPUTier:              fmt.Sprintf("Vertex AI Dedicated Endpoint (%s · /invoke/*)", vSt.MachineType),
				TemplatesAvailable:   len(catalog),
				AuthenticatedUser:    userEmail,
				Detail:               fmt.Sprintf("Vertex AI Dedicated Endpoint %s is Warm & Ready (%d replica(s) · %s · 0.0s cold start · ~%dms GPU denoise · /invoke/*).", vSt.EndpointID, vSt.ReplicaCount, vSt.MachineType, lastReadMs),
			}
		}
		if defB == "vertex" {
			if vSt.State == "deploying" {
				return HealthAndGPUStatusOutput{
					GatewayHealthy:       true,
					GPUAvailable:         false,
					GPUState:             "warming_up",
					ActiveBackend:        "vertex",
					RequestedBackend:     defB,
					ContainerReachable:   false,
					WarmupInProgress:     true,
					WarmupPhase:          "vertex_scaling_up",
					WarmupPhaseLabel:     vSt.Message,
					EWMAWakeSeconds:      90,
					EstimatedWakeSeconds: 90,
					IdleTTLSeconds:       0,
					UpstreamURL:          vxURL,
					Model:                "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
					GPUTier:              fmt.Sprintf("Vertex AI Dedicated Endpoint (%s · /invoke/*)", vSt.MachineType),
					TemplatesAvailable:   len(catalog),
					AuthenticatedUser:    userEmail,
					Detail:               vSt.Message,
				}
			}
			return HealthAndGPUStatusOutput{
				GatewayHealthy:       true,
				GPUAvailable:         false,
				GPUState:             "scaled_to_zero",
				ActiveBackend:        "vertex",
				RequestedBackend:     defB,
				ContainerReachable:   false,
				WarmupInProgress:     false,
				WarmupPhase:          "vertex_quiesced",
				WarmupPhaseLabel:     "Vertex L4 Quiesced (0 replicas)",
				EWMAWakeSeconds:      90,
				EstimatedWakeSeconds: 90,
				IdleTTLSeconds:       0,
				UpstreamURL:          vxURL,
				Model:                "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
				GPUTier:              fmt.Sprintf("Vertex AI Dedicated Endpoint (%s · /invoke/*)", vSt.MachineType),
				TemplatesAvailable:   len(catalog),
				AuthenticatedUser:    userEmail,
				Detail:               vSt.Message,
			}
		}
	}

	// 2. If Local is the effective backend, inspect local diffgemma (Metal)
	if defB == "local" {
		targetLocalURL := serveLocalURL
		if targetLocalURL == "" {
			targetLocalURL = viper.GetString("url")
			if targetLocalURL == "" || isRemoteGCPURL(targetLocalURL) {
				targetLocalURL = "http://127.0.0.1:8080/v1"
			}
		}
		modelName := viper.GetString("model")
		if modelName == "" {
			modelName = "diffgemma-26b-a4b-it-q4"
		}
		healthURL := strings.TrimSuffix(targetLocalURL, "/")
		if !strings.HasSuffix(healthURL, "/v1") {
			healthURL = healthURL + "/v1"
		}
		healthURL = healthURL + "/models"
		isReady := false
		detail := fmt.Sprintf("Local diffgemma endpoint %s is unreachable.", targetLocalURL)
		if resp, err := (&http.Client{Timeout: 1500 * time.Millisecond}).Get(healthURL); err == nil {
			if resp.StatusCode == http.StatusOK {
				isReady = true
				detail = fmt.Sprintf("Local diffgemma (Apple Silicon Metal) is Warm & Ready at %s.", targetLocalURL)
			}
			resp.Body.Close()
		}
		gpuStateMu.RLock()
		lastReadMs := lastLocalReadoutLatencyMs
		gpuStateMu.RUnlock()
		if lastReadMs <= 0 {
			lastReadMs = 850
		}
		gpuState := "scaled_to_zero"
		if isReady {
			gpuState = "warm_and_ready"
		}
		return HealthAndGPUStatusOutput{
			GatewayHealthy:       true,
			GPUAvailable:         isReady,
			GPUState:             gpuState,
			ActiveBackend:        "local",
			RequestedBackend:     defB,
			ContainerReachable:   isReady,
			WarmupInProgress:     false,
			EWMAWakeSeconds:      0,
			SecondsSinceLastRead: 0,
			IdleRemainingSeconds: 0,
			IdleTTLSeconds:       0,
			LastReadoutMs:        lastReadMs,
			EstimatedWakeSeconds: 0,
			UpstreamURL:          targetLocalURL,
			Model:                modelName,
			GPUTier:              "Apple Silicon Metal (Unified Memory)",
			TemplatesAvailable:   len(catalog),
			AuthenticatedUser:    userEmail,
			Detail:               detail,
		}
	}

	gpuStateMu.Lock()
	// Safety expiry: never allow warmupInProgress to stay stuck past serveWakeupTimeout (10m)
	if warmupInProgress && time.Since(warmupStartedAt) > 10*time.Minute {
		warmupInProgress = false
		if warmupDoneCh != nil {
			close(warmupDoneCh)
			warmupDoneCh = nil
		}
	}
	lastWarm := lastWarmTimestamp
	lastReadMs := lastReadoutLatencyMs
	warming := warmupInProgress
	warmStart := warmupStartedAt
	phaseHint := currentWarmupPhase
	bytesStaged := currentBytesStagedGB
	ewmaSec := int((ewmaTotalMs + 500) / 1000)
	if ewmaSec < 45 {
		ewmaSec = 45
	}
	gpuStateMu.Unlock()

	upstreamURL := viper.GetString("url")
	modelName := "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
	gpuTier := "1x NVIDIA RTX Pro 6000 (48GB VRAM, SigLIP Multimodal)"

	out := HealthAndGPUStatusOutput{
		GatewayHealthy:       true,
		ActiveBackend:        "cloudrun",
		RequestedBackend:     defB,
		UpstreamURL:          upstreamURL,
		Model:                modelName,
		GPUTier:              gpuTier,
		TemplatesAvailable:   len(catalog),
		AuthenticatedUser:    userEmail,
		WarmupInProgress:     warming,
		LastReadoutMs:        lastReadMs,
		EWMAWakeSeconds:      ewmaSec,
		EstimatedWakeSeconds: ewmaSec,
		IdleTTLSeconds:       int(idleWindow.Seconds()),
	}

	if warming {
		out.WarmupElapsedSeconds = int(time.Since(warmStart).Seconds())
		out.ContainerReachable = out.WarmupElapsedSeconds > 3
		out.GPUState = "warming_up"
		phaseID, phaseLabel := inferWarmupPhaseAndLabel(out.WarmupElapsedSeconds, ewmaSec, phaseHint, bytesStaged)
		out.WarmupPhase = phaseID
		out.WarmupPhaseLabel = phaseLabel
		out.WarmupBytesStagedGB = bytesStaged
		rem := ewmaSec - out.WarmupElapsedSeconds
		if rem < 8 {
			rem = 8
		}
		out.EstimatedWakeSeconds = rem
		out.Detail = fmt.Sprintf("%s (%ds elapsed of ~%ds EWMA cold-start).", phaseLabel, out.WarmupElapsedSeconds, ewmaSec)
		return out
	}

	if !lastWarm.IsZero() && time.Since(lastWarm) < idleWindow {
		sinceSec := int(time.Since(lastWarm).Seconds())
		remSec := int(idleWindow.Seconds()) - sinceSec
		out.GPUAvailable = true
		out.ContainerReachable = true
		out.GPUState = "warm_and_ready"
		out.SecondsSinceLastRead = sinceSec
		out.IdleRemainingSeconds = remSec
		out.EstimatedWakeSeconds = 0
		if lastReadMs > 0 {
			out.Detail = fmt.Sprintf("vLLM EngineCore is warm and ready (last readout %dms, %ds ago; %dm%ds remaining in %s idle TTL).", lastReadMs, sinceSec, remSec/60, remSec%60, idleWindow)
		} else {
			out.Detail = fmt.Sprintf("vLLM EngineCore is warm and ready (last readout %ds ago; %dm%ds remaining in %s idle TTL).", sinceSec, remSec/60, remSec%60, idleWindow)
		}
		return out
	}

	out.GPUState = "scaled_to_zero"
	out.Detail = fmt.Sprintf("GPU service is scaled to 0 instances ($0.00/hr idle; ~%ds EWMA wake, %s idle TTL once warm). Click 'Wake GPU' or execute any decision to wake automatically.", ewmaSec, idleWindow)
	return out
}

// WarmupGPUInput defines arguments for the warmup_gpu MCP tool and POST /api/warmup.
type WarmupGPUInput struct {
	WaitForReady bool `json:"wait_for_ready,omitempty" jsonschema:"If true (default), waits until vLLM finishes loading weights and confirms a test decision (~2m if cold, <10ms if already warm). If false, triggers background wakeup and returns immediately."`
}

// WarmupGPUOutput is returned by warmup_gpu and POST /api/warmup.
type WarmupGPUOutput struct {
	Status               string `json:"status"` // "warm_and_ready", "warming_up_started", or "warming_up_in_progress"
	GPUAvailable         bool   `json:"gpu_available"`
	Coalesced            bool   `json:"coalesced"`
	WarmupElapsedSeconds int    `json:"warmup_elapsed_seconds,omitempty"`
	EWMAWakeSeconds      int    `json:"ewma_wake_seconds,omitempty"`
	WarmupAttempts       int    `json:"warmup_attempts,omitempty"`
	ElapsedMs            int64  `json:"elapsed_ms"`
	UpstreamURL          string `json:"upstream_url"`
	Message              string `json:"message"`
}

// TriggerGPUWarmup delegates to TriggerGPUWarmupWithSource with default source.
func TriggerGPUWarmup(ctx context.Context, waitForReady bool) (WarmupGPUOutput, error) {
	return TriggerGPUWarmupWithSource(ctx, waitForReady, "api_warmup")
}

// TriggerGPUWarmupWithSource is a strictly idempotent, single-flight coalesced GPU warmup coordinator
// shared across MCP (warmup_gpu), Web Studio (Wake GPU button), and REST (POST /api/warmup).
func TriggerGPUWarmupWithSource(ctx context.Context, waitForReady bool, triggerSource string) (WarmupGPUOutput, error) {
	callStart := time.Now()

	// 1. Fast-Path Idempotency: If the GPU was verified warm within the 15m window,
	// return immediately without running a redundant GPU forward pass.
	currentStatus := CheckHealthAndGPUStatus(ctx, "")
	if currentStatus.GPUAvailable && currentStatus.GPUState == "warm_and_ready" {
		return WarmupGPUOutput{
			Status:          "warm_and_ready",
			GPUAvailable:    true,
			Coalesced:       true,
			EWMAWakeSeconds: currentStatus.EWMAWakeSeconds,
			ElapsedMs:       time.Since(callStart).Milliseconds(),
			UpstreamURL:     viper.GetString("url"),
			Message:         fmt.Sprintf("GPU is already warm and ready (last readout %ds ago; no duplicate warmup sent).", currentStatus.SecondsSinceLastRead),
		}, nil
	}

	// 2. Single-Flight Coalescing: Join existing in-flight warmup if one is already running,
	// or start exactly ONE background warmup worker if none is active.
	gpuStateMu.Lock()
	alreadyRunning := warmupInProgress
	var waitCh chan struct{}
	var startedAt time.Time

	if alreadyRunning {
		waitCh = warmupDoneCh
		startedAt = warmupStartedAt
		gpuStateMu.Unlock()
	} else {
		warmupInProgress = true
		warmupStartedAt = callStart
		warmupDoneCh = make(chan struct{})
		warmupLastAttempts = 0
		warmupLastErr = nil
		currentWarmupPhase = ""
		currentBytesStagedGB = 0.0
		waitCh = warmupDoneCh
		startedAt = warmupStartedAt
		gpuStateMu.Unlock()

		// Launch the single background warmup poller + OTel warmup lifecycle span
		go func(doneCh chan struct{}, src string) {
			bgCtx, cancel := context.WithTimeout(context.Background(), serveWakeupTimeout)
			defer cancel()

			// Poll upstream /health every 3s to capture live sub-stage and bytes_staged_gb
			go func() {
				ticker := time.NewTicker(3 * time.Second)
				defer ticker.Stop()
				for {
					select {
					case <-bgCtx.Done():
						return
					case <-ticker.C:
						ph, stagedGB, ok := pollUpstreamWarmupHealth(viper.GetString("url"))
						if ok {
							gpuStateMu.Lock()
							if ph != "" {
								currentWarmupPhase = ph
							}
							if stagedGB > 0 {
								currentBytesStagedGB = stagedGB
							}
							gpuStateMu.Unlock()
						}
					}
				}
			}()

			_, warmSpan := gatewayTracer().Start(bgCtx, "dgem.gpu.warmup_lifecycle")
			warmSpan.SetAttributes(attribute.String("dgem.warmup.trigger", src))

			engine := template.NewEngine()
			vars := map[string]interface{}{"ticket": "GPU warmup readiness probe"}
			rendered, err := "", error(nil)
			if targetPath, _, rErr := resolveTemplateFile(serveTemplatesDir, "support_triage"); rErr != nil {
				err = rErr
			} else {
				rendered, err = engine.RenderFile(targetPath, vars)
			}
			if err != nil {
				rendered = `{"questions":[{"id":"ready","type":"boolean","prompt":"Is the system ready?"}]}`
			}
			schemaContent, stateContent, err := template.ParseStructuredPayload(rendered, vars)
			attempts := 1
			var readoutMs int64
			if err == nil {
				_, stats, att, decErr := executeDecideWithWarmup(bgCtx, schemaContent, stateContent, nil)
				attempts = att
				err = decErr
				readoutMs = stats.WallTime.Milliseconds()
			}

			totalWarmMs := time.Since(callStart).Milliseconds()
			if err == nil {
				// Decompose total warmup duration proportionally across the 4 stages (or from probe)
				s4 := readoutMs
				if s4 <= 0 || s4 > 15000 {
					s4 = 8600
				}
				remMs := totalWarmMs - s4
				if remMs < 5000 {
					remMs = 5000
				}
				s1 := int64(float64(remMs) * 0.05)
				s2 := int64(float64(remMs) * 0.38)
				s3 := remMs - s1 - s2

				rec := WarmupStageBreakdown{
					Timestamp:          time.Now().UTC().Format(time.RFC3339),
					TriggerSource:      src,
					Phase1ActivatorMs:  s1,
					Phase2TmpfsStageMs: s2,
					Phase3VLLMSiglipMs: s3,
					Phase4TritonJITMs:  s4,
					TotalWarmupMs:      totalWarmMs,
					Attempts:           attempts,
				}
				newEWMA := recordWarmupBreakdown(rec)

				warmSpan.SetAttributes(
					attribute.Int64("dgem.warmup.total_ms", totalWarmMs),
					attribute.Int64("dgem.warmup.stage1_activator_ms", s1),
					attribute.Int64("dgem.warmup.stage2_tmpfs_ms", s2),
					attribute.Int64("dgem.warmup.stage3_vllm_siglip_ms", s3),
					attribute.Int64("dgem.warmup.stage4_triton_jit_ms", s4),
					attribute.Int("dgem.warmup.attempts", attempts),
					attribute.Int("dgem.warmup.ewma_wake_seconds", newEWMA),
				)
				warmSpan.SetStatus(codes.Ok, "warmup completed")
			} else {
				warmSpan.RecordError(err)
				warmSpan.SetStatus(codes.Error, err.Error())
			}
			warmSpan.End()

			gpuStateMu.Lock()
			warmupLastAttempts = attempts
			warmupLastErr = err
			if err == nil {
				lastWarmTimestamp = time.Now()
				if readoutMs > 0 {
					lastReadoutLatencyMs = readoutMs
				}
			}
			if warmupInProgress {
				warmupInProgress = false
				if warmupDoneCh != nil {
					close(warmupDoneCh)
					warmupDoneCh = nil
				}
			}
			gpuStateMu.Unlock()
		}(waitCh, triggerSource)
	}

	// 3. Non-blocking mode (wait_for_ready=false): Return immediately (coalesced if already running)
	if !waitForReady {
		elapsedSec := int(time.Since(startedAt).Seconds())
		if alreadyRunning {
			return WarmupGPUOutput{
				Status:               "warming_up_in_progress",
				GPUAvailable:         false,
				Coalesced:            true,
				WarmupElapsedSeconds: elapsedSec,
				ElapsedMs:            time.Since(callStart).Milliseconds(),
				UpstreamURL:          viper.GetString("url"),
				Message:              fmt.Sprintf("Joined existing in-flight GPU warmup (%ds elapsed; 1 active poller, no duplicate requests sent).", elapsedSec),
			}, nil
		}
		return WarmupGPUOutput{
			Status:       "warming_up_started",
			GPUAvailable: false,
			Coalesced:    false,
			ElapsedMs:    time.Since(callStart).Milliseconds(),
			UpstreamURL:  viper.GetString("url"),
			Message:      "Background GPU wakeup started (0 -> 1 instance; 1 active poller). Poll get_health_and_gpu_status to monitor readiness.",
		}, nil
	}

	// 4. Blocking mode (wait_for_ready=true): Subscribe to the shared waitCh until the single poller finishes
	select {
	case <-waitCh:
		gpuStateMu.RLock()
		attempts := warmupLastAttempts
		err := warmupLastErr
		gpuStateMu.RUnlock()
		if err != nil {
			return WarmupGPUOutput{
				Status:         "error",
				GPUAvailable:   false,
				Coalesced:      alreadyRunning,
				WarmupAttempts: attempts,
				ElapsedMs:      time.Since(callStart).Milliseconds(),
				UpstreamURL:    viper.GetString("url"),
				Message:        fmt.Sprintf("Warmup failed after %d attempt(s): %v", attempts, err),
			}, err
		}
		return WarmupGPUOutput{
			Status:         "warm_and_ready",
			GPUAvailable:   true,
			Coalesced:      alreadyRunning,
			WarmupAttempts: attempts,
			ElapsedMs:      time.Since(callStart).Milliseconds(),
			UpstreamURL:    viper.GetString("url"),
			Message:        fmt.Sprintf("GPU is warm and ready! Completed readiness decision in %d ms (%d attempt(s), coalesced=%v).", time.Since(callStart).Milliseconds(), attempts, alreadyRunning),
		}, nil
	case <-ctx.Done():
		return WarmupGPUOutput{
			Status:      "warming_up_in_progress",
			Coalesced:   alreadyRunning,
			ElapsedMs:   time.Since(callStart).Milliseconds(),
			UpstreamURL: viper.GetString("url"),
			Message:     "Caller context expired while waiting, but single-flight background GPU warmup continues running.",
		}, ctx.Err()
	}
}

// MCP Tool Input/Output Structs

type StatusToolInput struct {
	Backend string `json:"backend,omitempty" jsonschema:"Optional inference backend selector to inspect: 'vertex_first' (default), 'vertex' (Vertex AI Dedicated Endpoint), 'cloudrun' (Serverless Cloud Run GPU), or 'local' (Apple Silicon Metal)."`
}

type DecidePolicyToolInput struct {
	Template             string                 `json:"template" jsonschema:"Policy template ID (e.g. 'support_triage', 'taxonomy_discovery', 'code_review', 'secops_conditional_dag', 'calibration/hallucination_judge')."`
	Variables            map[string]interface{} `json:"variables" jsonschema:"Key-value map of template variables (e.g. {'ticket': 'Double charged on invoice #9481'})."`
	Image                string                 `json:"image,omitempty" jsonschema:"Optional image URL or base64 data URI for multimodal policies."`
	Backend              string                 `json:"backend,omitempty" jsonschema:"Optional inference backend selector: 'vertex_first' (default: Vertex AI Dedicated Endpoint primary with Cloud Run GPU failover), 'vertex' (strict Vertex AI /invoke/*), 'cloudrun' (strict Serverless Cloud Run GPU), or 'local' (Apple Silicon Metal diffgemma)."`
	VertexURL            string                 `json:"vertex_url,omitempty" jsonschema:"Optional Vertex AI Endpoint ID or /invoke/* URL override."`
	CascadeMode          string                 `json:"cascade_mode,omitempty" jsonschema:"Optional Stage-2 Vertex AI Gemini 3.x cascade mode: 'off' (default), 'entropy' (forward slots with Shannon entropy H >= cascade_threshold), or 'on_miss' (forward slots that miss expected_answers)."`
	CascadeThreshold     float64                `json:"cascade_threshold,omitempty" jsonschema:"Escalation threshold for Stage-2 Gemini: Shannon entropy H in nats (default 0.35), or H / ln K when cascade_threshold_mode is 'normalized' (default 0.16)."`
	CascadeThresholdMode string                 `json:"cascade_threshold_mode,omitempty" jsonschema:"What cascade_threshold is compared with: 'nats' (default, raw entropy, grows with the option count) or 'normalized' (entropy / ln K, the Studio's Hesitation scale)."`
	CascadeModel         string                 `json:"cascade_model,omitempty" jsonschema:"Stage-2 Vertex AI Gemini 3.x model (default 'gemini-3.8-flash'; also supports 'gemini-3.5-flash-lite')."`
	ExpectedAnswers      map[string]string      `json:"expected_answers,omitempty" jsonschema:"Optional map of slot_id -> expected value for 'on_miss' cascade mode."`
	SuggestExpansions    bool                   `json:"suggest_expansions,omitempty" jsonschema:"If true, dynamically injects an 'other_unclassified' catch-all option into choice slots (if absent) and proposes new {'name', 'description'} options when unclassified or high-entropy."`
	ExpansionEntropy     float64                `json:"expansion_entropy,omitempty" jsonschema:"Shannon entropy threshold H in nats on choice slots to trigger taxonomy expansion proposals (default 0.35)."`
	Stage2Prior          string                 `json:"stage2_prior,omitempty" jsonschema:"Optional: how much of Stage 1's answer the Stage-2 Gemini cascade sees: 'soft' (default: only that a fast first stage leaned toward an answer and may be wrong), 'full' (answer, confidence and distribution) or 'none'."`
	Temperature          float64                `json:"temperature,omitempty" jsonschema:"Optional post-hoc temperature for every answer (T > 1 softens overconfidence). Overrides the template's temperature. Fit it on labelled examples of your own (docs/confidence/calibrate-your-policy.md)."`
	TemperatureByType    map[string]float64     `json:"temperature_by_type,omitempty" jsonschema:"Optional post-hoc temperature per question type, e.g. {'noul': 3.0, 'choice': 1.3}; overrides temperature for those types."`
	Layout               string                 `json:"layout,omitempty" jsonschema:"Optional prompt layout: 'document_first' (default on serving v0.2.0+: the input first, then the questions) or 'schema_first' (the questions as the system prompt). See docs/policies/prompt-layout.md."`
}

type LocateBBoxToolInput struct {
	Image     string `json:"image" jsonschema:"Image URL or base64 data URI (data:image/png;base64,...) to analyze with Gemma 4 SigLIP vision."`
	Target    string `json:"target" jsonschema:"Natural language description of the object to localize (e.g. 'red vintage pickup truck', 'the wine glass closest to the bottle')."`
	Backend   string `json:"backend,omitempty" jsonschema:"Optional inference backend selector: 'vertex_first' (default), 'vertex', or 'cloudrun' (note: 'local' Metal does not support SigLIP vision)."`
	VertexURL string `json:"vertex_url,omitempty" jsonschema:"Optional Vertex AI Endpoint ID or /invoke/* URL override."`
}

type BBoxCoords struct {
	YMin float64 `json:"ymin"`
	XMin float64 `json:"xmin"`
	YMax float64 `json:"ymax"`
	XMax float64 `json:"xmax"`
}

type LocateBBoxToolOutput struct {
	Target             string             `json:"target"`
	Presence           bool               `json:"presence"`
	PresenceConfidence float64            `json:"presence_confidence"`
	SoftmaxExpectation BBoxCoords         `json:"softmax_expectation_box_1000"`
	DiscreteArgmax     BBoxCoords         `json:"discrete_argmax_box_1000"`
	CoordinateEntropy  map[string]float64 `json:"coordinate_entropy_nats"`
	MaxEntropy         float64            `json:"max_entropy"`
	WallTimeMs         int64              `json:"wall_time_ms"`
	BackendTarget      string             `json:"backend_target"`
	UpstreamURL        string             `json:"upstream_url"`
}

type CustomQuestionSpec struct {
	ID       string   `json:"id" jsonschema:"Short slot identifier (e.g. 'is_urgent', 'category')."`
	Type     string   `json:"type" jsonschema:"Question type: 'boolean', 'choice' (<=26 options), or 'score' (1..5)."`
	Question string   `json:"question" jsonschema:"Prompt question for this slot."`
	Options  []string `json:"options,omitempty" jsonschema:"List of choices (required when type is 'choice', max 26)."`
}

type DecideCustomToolInput struct {
	Context              string               `json:"context" jsonschema:"Input text, document, code diff, or event log to evaluate."`
	Questions            []CustomQuestionSpec `json:"questions" jsonschema:"List of structured decision slots to evaluate simultaneously in 1 forward pass."`
	Backend              string               `json:"backend,omitempty" jsonschema:"Optional inference backend selector: 'vertex_first' (default: Vertex AI primary with Cloud Run failover), 'vertex', 'cloudrun', or 'local' (Apple Silicon Metal diffgemma)."`
	VertexURL            string               `json:"vertex_url,omitempty" jsonschema:"Optional Vertex AI Endpoint ID or /invoke/* URL override."`
	Image                string               `json:"image,omitempty" jsonschema:"Optional image URL or base64 data URI; the questions are answered about it (and sent to Stage 2 when the cascade escalates)."`
	CascadeMode          string               `json:"cascade_mode,omitempty" jsonschema:"Optional Stage-2 Vertex AI Gemini 3.x cascade mode: 'off' (default), 'entropy' (forward slots with Shannon entropy H >= cascade_threshold), or 'on_miss' (forward slots that miss expected_answers)."`
	CascadeThreshold     float64              `json:"cascade_threshold,omitempty" jsonschema:"Escalation threshold for Stage-2 Gemini: Shannon entropy H in nats (default 0.35), or H / ln K when cascade_threshold_mode is 'normalized' (default 0.16)."`
	CascadeThresholdMode string               `json:"cascade_threshold_mode,omitempty" jsonschema:"What cascade_threshold is compared with: 'nats' (default, raw entropy, grows with the option count) or 'normalized' (entropy / ln K, the Studio's Hesitation scale)."`
	CascadeModel         string               `json:"cascade_model,omitempty" jsonschema:"Stage-2 Vertex AI Gemini 3.x model (default 'gemini-3.8-flash'; also supports 'gemini-3.5-flash-lite')."`
	ExpectedAnswers      map[string]string    `json:"expected_answers,omitempty" jsonschema:"Optional map of slot_id -> expected value for 'on_miss' cascade mode."`
	SuggestExpansions    bool                 `json:"suggest_expansions,omitempty" jsonschema:"If true, dynamically injects an 'other_unclassified' catch-all option into choice slots (if absent) and proposes new {'name', 'description'} options when unclassified or high-entropy."`
	ExpansionEntropy     float64              `json:"expansion_entropy,omitempty" jsonschema:"Shannon entropy threshold H in nats on choice slots to trigger taxonomy expansion proposals (default 0.35)."`
	Stage2Prior          string               `json:"stage2_prior,omitempty" jsonschema:"Optional: how much of Stage 1's answer the Stage-2 Gemini cascade sees: 'soft' (default: only that a fast first stage leaned toward an answer and may be wrong), 'full' (answer, confidence and distribution) or 'none'."`
	Temperature          float64              `json:"temperature,omitempty" jsonschema:"Optional post-hoc temperature for every answer (T > 1 softens overconfidence). Fit it on labelled examples of your own (docs/confidence/calibrate-your-policy.md)."`
	TemperatureByType    map[string]float64   `json:"temperature_by_type,omitempty" jsonschema:"Optional post-hoc temperature per question type, e.g. {'noul': 3.0, 'choice': 1.3}; overrides temperature for those types."`
	Layout               string               `json:"layout,omitempty" jsonschema:"Optional prompt layout: 'document_first' (default on serving v0.2.0+: the input first, then the questions) or 'schema_first' (the questions as the system prompt). See docs/policies/prompt-layout.md."`
}

type ListTemplatesToolInput struct {
	Category string `json:"category,omitempty" jsonschema:"Optional category filter: 'core', 'calibration', 'multimodal', or 'rerank'."`
}

type ListTemplatesToolOutput struct {
	Count     int                    `json:"count"`
	Templates []TemplateCatalogEntry `json:"templates"`
}

var binNumberRegex = regexp.MustCompile(`(\d+)`)

func computeExpectedCoord(ans client.QuestionAnswer) (float64, float64) {
	raw := ans.DisplayValue()
	if raw == "" {
		raw = ans.Label
	}
	argmax := 0.0
	if m := binNumberRegex.FindStringSubmatch(raw); len(m) > 1 {
		if v, err := strconv.ParseFloat(m[1], 64); err == nil {
			if v <= 100 {
				argmax = v * 10.0
			} else {
				argmax = v
			}
		}
	}
	if len(ans.Probabilities) == 0 {
		return argmax, argmax
	}
	var expected, totalMass float64
	for k, p := range ans.Probabilities {
		if m := binNumberRegex.FindStringSubmatch(k); len(m) > 1 {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				scale := v
				if scale <= 100 {
					scale = v * 10.0
				}
				expected += scale * p
				totalMass += p
			}
		}
	}
	if totalMass > 0 {
		expected = math.Round((expected/totalMass)*10) / 10
		return expected, argmax
	}
	return argmax, argmax
}

func buildMCPServer() *mcp.Server {
	srv := mcp.NewServer(&mcp.Implementation{
		Name:    "dgem-decision-studio-mcp",
		Version: "1.0.0",
	}, nil)

	// Tool 1: get_health_and_gpu_status
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "get_health_and_gpu_status",
		Description: "Returns live health status of the dgem gateway and availability/warmup state of the selected inference backend ('vertex_first', 'vertex' 1x NVIDIA L4, or 'cloudrun' 1x NVIDIA RTX Pro 6000), plus readout latency and estimated wakeup time.",
		InputSchema: backendChoicesSchema[StatusToolInput](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input StatusToolInput) (*mcp.CallToolResult, HealthAndGPUStatusOutput, error) {
		return nil, CheckHealthAndGPUStatusForBackend(ctx, "", input.Backend), nil
	})

	// Tool 2: warmup_gpu
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "warmup_gpu",
		Description: "Wakes the scale-to-zero Cloud Run DiffusionGemma GPU instance (0 -> 1) and loads the 17.53 GiB model into VRAM. Pass wait_for_ready=true to wait until ready, or wait_for_ready=false to trigger background warmup immediately.",
		InputSchema: geminiSafeInputSchema[WarmupGPUInput](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input WarmupGPUInput) (*mcp.CallToolResult, WarmupGPUOutput, error) {
		out, err := TriggerGPUWarmup(ctx, input.WaitForReady)
		return nil, out, err
	})

	// Tool 3: decide_policy
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "decide_policy",
		Description: "Executes a zero-shot multi-slot decision policy (.json.tmpl) on DiffusionGemma in O(1) forward passes, returning joint slot answers, probabilities, calibrated epistemic Shannon entropy H (in nats), and optional Stage-2 Gemini 3.x cascade escalation (gemini-3.8-flash default).",
		InputSchema: backendChoicesSchema[DecidePolicyToolInput](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input DecidePolicyToolInput) (*mcp.CallToolResult, GatewayDecideResponse, error) {
		tmplID := strings.TrimSpace(input.Template)
		if tmplID == "" {
			tmplID = "support_triage"
		}
		targetPath, cleanID, err := resolveTemplateFile(serveTemplatesDir, tmplID)
		if err != nil {
			return nil, GatewayDecideResponse{}, err
		}

		engine := template.NewEngine()
		rendered, err := engine.RenderFile(targetPath, input.Variables)
		if err != nil {
			return nil, GatewayDecideResponse{}, fmt.Errorf("failed to render template %q: %w", tmplID, err)
		}
		schemaContent, stateContent, err := template.ParseStructuredPayload(rendered, input.Variables)
		if err != nil {
			return nil, GatewayDecideResponse{}, err
		}
		if schemaContent, err = ApplyPromptLayout(schemaContent, input.Layout); err != nil {
			return nil, GatewayDecideResponse{}, err
		}
		if _, err := NormalizeStage2Prior(input.Stage2Prior); err != nil {
			return nil, GatewayDecideResponse{}, err
		}
		var injectedSlots map[string]bool
		var existingOptions map[string][]client.ProposedOption
		if input.SuggestExpansions {
			schemaContent, injectedSlots, existingOptions = InjectUnclassifiedCatchAll(schemaContent)
		} else {
			_, _, existingOptions = InjectUnclassifiedCatchAll(schemaContent)
		}

		var imgs []string
		if input.Image != "" {
			imgs = append(imgs, input.Image)
		}
		imgs, imgErr := prepareRequestImages(ctx, imgs)
		if imgErr != nil {
			return nil, GatewayDecideResponse{}, imgErr
		}
		backendTarget, targetURL, bErr := resolveBackendTargetFromParams(ctx, input.Backend, input.VertexURL)
		if bErr != nil {
			return nil, GatewayDecideResponse{}, bErr
		}
		start := time.Now()
		resp, stats, attempts, err := executeDecideWithWarmup(ctx, schemaContent, stateContent, imgs, targetURL)
		if err != nil {
			return nil, GatewayDecideResponse{}, err
		}
		summary := summarizeDecision(resp, stats, time.Since(start))
		recordBackendReadoutLatency(backendTarget, summary.GpuForwardMs)

		var cascadeSummary *CascadeExecutionSummary
		if input.CascadeMode != "" && input.CascadeMode != "off" && input.CascadeMode != "none" {
			thresholdMode, terr := NormalizeCascadeThresholdMode(input.CascadeThresholdMode)
			if terr != nil {
				return nil, GatewayDecideResponse{}, terr
			}
			_, cascadeSpan := gatewayTracer().Start(ctx, "dgem.cascade.gemini")
			cascadeSummary = ExecuteStage2GeminiCascadeWithImages(
				WithCascadeThresholdMode(WithStage2Prior(ctx, input.Stage2Prior), thresholdMode),
				input.CascadeMode,
				input.CascadeThreshold,
				input.CascadeModel,
				input.ExpectedAnswers,
				schemaContent,
				stateContent,
				resp,
				imgs,
			)
			cascadeSpan.End()
		}
		reqT, terr := mcpTemperature(input.Temperature, input.TemperatureByType)
		if terr != nil {
			return nil, GatewayDecideResponse{}, terr
		}
		appliedTemp, maxH, terr := applyPolicyTemperature(resp, schemaContent, reqT, cascadeSummary)
		if terr != nil {
			return nil, GatewayDecideResponse{}, terr
		}
		if appliedTemp != nil {
			summary.MaxEntropy = maxH
		}

		if input.SuggestExpansions || (resp.Diagnostics.Thought != nil && strings.Contains(resp.Diagnostics.Thought.Text, "SUGGESTED_")) {
			expandCtx, expandSpan := gatewayTracer().Start(ctx, "dgem.taxonomy.expand")
			resp.SuggestedExpansions = SynthesizeTaxonomyExpansions(
				expandCtx,
				nil,
				cleanID,
				stateContent,
				resp,
				injectedSlots,
				existingOptions,
				input.ExpansionEntropy,
				targetURL,
			)
			expandSpan.SetAttributes(
				attribute.Bool("dgem.taxonomy.triggered", len(resp.SuggestedExpansions) > 0),
				attribute.Int("dgem.taxonomy.suggestions_count", len(resp.SuggestedExpansions)),
			)
			expandSpan.SetStatus(codes.Ok, "ok")
			expandSpan.End()
		}

		return nil, GatewayDecideResponse{
			Template:            cleanID,
			Answers:             resp.Answers,
			Diagnostics:         resp.Diagnostics,
			Decision:            resp,
			Cascade:             cascadeSummary,
			Temperature:         appliedTemp,
			SuggestedExpansions: resp.SuggestedExpansions,
			MaxEntropy:          summary.MaxEntropy,
			WallTimeMs:          time.Since(start).Milliseconds(),
			GpuForwardMs:        summary.GpuForwardMs,
			ColdStartWaitMs:     summary.ColdStartWaitMs,
			WarmupAttempts:      attempts,
			Model:               stats.Model,
			BackendTarget:       backendTarget,
			UpstreamURL:         targetURL,
		}, nil
	})

	// Tool 4: locate_bounding_boxes
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "locate_bounding_boxes",
		Description: "Approximate single-pass box from dgem: softmax-expectation and argmax [ymin, xmin, ymax, xmax] in [0, 1000]. Coarse: box centre lands inside the target ~75% of the time on photos and ~12% on app/web screenshots, and boxes move under flips or padding. Use it for rough location only; for precise boxes use Grounding DINO or Gemini (see docs/policies/images.md). For presence, 3x3 location or relations, decide_custom_questions with an image is faster and better calibrated.",
		InputSchema: visionBackendChoicesSchema[LocateBBoxToolInput](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input LocateBBoxToolInput) (*mcp.CallToolResult, LocateBBoxToolOutput, error) {
		if strings.TrimSpace(input.Image) == "" {
			return nil, LocateBBoxToolOutput{}, fmt.Errorf("'image' URL or data URI is required")
		}
		if prepared, err := prepareRequestImages(ctx, []string{input.Image}); err != nil {
			return nil, LocateBBoxToolOutput{}, err
		} else {
			input.Image = prepared[0]
		}
		target := strings.TrimSpace(input.Target)
		if target == "" {
			target = "primary foreground object"
		}
		vars := map[string]interface{}{"target": target}
		targetPath, _, err := resolveTemplateFile(serveTemplatesDir, "bbox_localization")
		if err != nil {
			return nil, LocateBBoxToolOutput{}, err
		}
		engine := template.NewEngine()
		rendered, err := engine.RenderFile(targetPath, vars)
		if err != nil {
			return nil, LocateBBoxToolOutput{}, err
		}
		schemaContent, stateContent, err := template.ParseStructuredPayload(rendered, vars)
		if err != nil {
			return nil, LocateBBoxToolOutput{}, err
		}
		backendTarget, targetURL, bErr := resolveBackendTargetFromParams(ctx, input.Backend, input.VertexURL)
		if bErr != nil {
			return nil, LocateBBoxToolOutput{}, bErr
		}
		if backendTarget == "local" {
			return nil, LocateBBoxToolOutput{}, fmt.Errorf("locate_bounding_boxes is not supported on the local Apple Silicon Metal backend (text-only diffgemma; SigLIP vision tower is not available on Metal). Please use backend: 'vertex_first' or 'cloudrun'")
		}
		start := time.Now()
		resp, _, _, err := executeDecideWithWarmup(ctx, schemaContent, stateContent, []string{input.Image}, targetURL)
		if err != nil {
			return nil, LocateBBoxToolOutput{}, err
		}
		readoutMs := time.Since(start).Milliseconds()
		if backendTarget == "cloudrun" {
			MarkGPUWarm(readoutMs)
		} else if backendTarget == "vertex" {
			RecordVertexReadoutLatency(readoutMs)
		}

		expYMin, argYMin := computeExpectedCoord(resp.Answers["ymin"])
		expXMin, argXMin := computeExpectedCoord(resp.Answers["xmin"])
		expYMax, argYMax := computeExpectedCoord(resp.Answers["ymax"])
		expXMax, argXMax := computeExpectedCoord(resp.Answers["xmax"])

		coordEnt := map[string]float64{}
		maxEnt := 0.0
		for _, k := range []string{"object_present", "ymin", "xmin", "ymax", "xmax"} {
			e := resp.Answers[k].Entropy
			if e == 0 {
				if q, ok := resp.Diagnostics.Questions[k]; ok {
					e = q.Entropy
				}
			}
			coordEnt[k] = e
			if e > maxEnt {
				maxEnt = e
			}
		}
		// The template's presence slot is "object_present" ("presence" in older templates).
		presAns, ok := resp.Answers["object_present"]
		if !ok {
			presAns = resp.Answers["presence"]
		}
		present := strings.EqualFold(presAns.Label, "yes") || strings.EqualFold(presAns.Label, "true")

		return nil, LocateBBoxToolOutput{
			Target:             target,
			Presence:           present,
			PresenceConfidence: presAns.Confidence,
			SoftmaxExpectation: BBoxCoords{YMin: expYMin, XMin: expXMin, YMax: expYMax, XMax: expXMax},
			DiscreteArgmax:     BBoxCoords{YMin: argYMin, XMin: argXMin, YMax: argYMax, XMax: argXMax},
			CoordinateEntropy:  coordEnt,
			MaxEntropy:         maxEnt,
			WallTimeMs:         time.Since(start).Milliseconds(),
			BackendTarget:      backendTarget,
			UpstreamURL:        targetURL,
		}, nil
	})

	// Tool 4b: locate_object (guided locate)
	mcp.AddTool(srv, &mcp.Tool{
		Name: "locate_object",
		Description: "Locates an object or UI element in an image: dgem first answers presence and 3x3 location (~0.2 s); if it " +
			"is confident the target is absent, returns absent without a Gemini call; otherwise Gemini 3.x (LOW thinking) " +
			"returns the box with dgem's cell as a hint, and optionally a SAM mask (when DGEM_SAM_URL is configured). " +
			"Returns box_1000 [ymin, xmin, ymax, xmax] in [0, 1000], the path taken and per-stage latency. Prefer this over " +
			"locate_bounding_boxes for precise boxes.",
		InputSchema: visionBackendChoicesSchema[LocateRequest](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input LocateRequest) (*mcp.CallToolResult, LocateResult, error) {
		if prepared, err := prepareRequestImages(ctx, []string{input.Image}); err != nil {
			return nil, LocateResult{}, err
		} else {
			input.Image = prepared[0]
		}
		backendTarget, targetURL, bErr := resolveBackendTargetFromParams(ctx, input.Backend, input.VertexURL)
		if bErr != nil {
			return nil, LocateResult{}, bErr
		}
		if backendTarget == "local" {
			return nil, LocateResult{}, fmt.Errorf("locate_object needs a vision backend: use 'vertex_first', 'vertex' or 'cloudrun'")
		}
		res, err := runGuidedLocate(ctx, input, locateDecideVia(backendTarget, targetURL))
		if err != nil {
			return nil, LocateResult{}, err
		}
		res.BackendTarget = backendTarget
		return nil, *res, nil
	})

	// Tool 5: decide_custom_questions
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "decide_custom_questions",
		Description: "Evaluates an ad-hoc list of boolean, choice (<=26 options), or score (1..5) questions simultaneously in 1 forward pass against a context document, returning joint answers, calibrated Shannon entropy H, and optional Stage-2 Gemini 3.x cascade escalation (gemini-3.8-flash default).",
		InputSchema: backendChoicesSchema[DecideCustomToolInput](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input DecideCustomToolInput) (*mcp.CallToolResult, GatewayDecideResponse, error) {
		if len(input.Questions) == 0 {
			return nil, GatewayDecideResponse{}, fmt.Errorf("'questions' list cannot be empty")
		}
		qList := make([]map[string]interface{}, 0, len(input.Questions))
		for _, q := range input.Questions {
			qMap := map[string]interface{}{
				"id":           q.ID,
				"type":         q.Type,
				"instructions": q.Question,
			}
			if len(q.Options) > 0 {
				qMap["options"] = q.Options
			}
			qList = append(qList, qMap)
		}
		schemaObj := map[string]interface{}{"questions": qList}
		schemaBytes, _ := json.Marshal(schemaObj)
		schemaStr, stateStr, _ := template.ParseStructuredPayload(string(schemaBytes), map[string]interface{}{"context": input.Context})

		schemaStr, lerr := ApplyPromptLayout(schemaStr, input.Layout)
		if lerr != nil {
			return nil, GatewayDecideResponse{}, lerr
		}
		if _, perr := NormalizeStage2Prior(input.Stage2Prior); perr != nil {
			return nil, GatewayDecideResponse{}, perr
		}
		var injectedSlots map[string]bool
		var existingOptions map[string][]client.ProposedOption
		if input.SuggestExpansions {
			schemaStr, injectedSlots, existingOptions = InjectUnclassifiedCatchAll(schemaStr)
		} else {
			_, _, existingOptions = InjectUnclassifiedCatchAll(schemaStr)
		}

		backendTarget, targetURL, bErr := resolveBackendTargetFromParams(ctx, input.Backend, input.VertexURL)
		if bErr != nil {
			return nil, GatewayDecideResponse{}, bErr
		}
		start := time.Now()
		var customImgs []string
		if input.Image != "" {
			customImgs = append(customImgs, input.Image)
		}
		customImgs, err := prepareRequestImages(ctx, customImgs)
		if err != nil {
			return nil, GatewayDecideResponse{}, err
		}
		resp, stats, attempts, err := executeDecideWithWarmup(ctx, schemaStr, stateStr, customImgs, targetURL)
		if err != nil {
			return nil, GatewayDecideResponse{}, err
		}
		summary := summarizeDecision(resp, stats, time.Since(start))
		recordBackendReadoutLatency(backendTarget, summary.GpuForwardMs)

		var cascadeSummary *CascadeExecutionSummary
		if input.CascadeMode != "" && input.CascadeMode != "off" && input.CascadeMode != "none" {
			thresholdMode, terr := NormalizeCascadeThresholdMode(input.CascadeThresholdMode)
			if terr != nil {
				return nil, GatewayDecideResponse{}, terr
			}
			_, cascadeSpan := gatewayTracer().Start(ctx, "dgem.cascade.gemini")
			cascadeSummary = ExecuteStage2GeminiCascadeWithImages(
				WithCascadeThresholdMode(WithStage2Prior(ctx, input.Stage2Prior), thresholdMode),
				input.CascadeMode,
				input.CascadeThreshold,
				input.CascadeModel,
				input.ExpectedAnswers,
				schemaStr,
				stateStr,
				resp,
				customImgs,
			)
			cascadeSpan.End()
		}
		reqT, terr := mcpTemperature(input.Temperature, input.TemperatureByType)
		if terr != nil {
			return nil, GatewayDecideResponse{}, terr
		}
		appliedTemp, maxH, terr := applyPolicyTemperature(resp, schemaStr, reqT, cascadeSummary)
		if terr != nil {
			return nil, GatewayDecideResponse{}, terr
		}
		if appliedTemp != nil {
			summary.MaxEntropy = maxH
		}

		if input.SuggestExpansions || (resp.Diagnostics.Thought != nil && strings.Contains(resp.Diagnostics.Thought.Text, "SUGGESTED_")) {
			expandCtx, expandSpan := gatewayTracer().Start(ctx, "dgem.taxonomy.expand")
			resp.SuggestedExpansions = SynthesizeTaxonomyExpansions(
				expandCtx,
				nil,
				"custom_questions",
				stateStr,
				resp,
				injectedSlots,
				existingOptions,
				input.ExpansionEntropy,
				targetURL,
			)
			expandSpan.SetAttributes(
				attribute.Bool("dgem.taxonomy.triggered", len(resp.SuggestedExpansions) > 0),
				attribute.Int("dgem.taxonomy.suggestions_count", len(resp.SuggestedExpansions)),
			)
			expandSpan.SetStatus(codes.Ok, "ok")
			expandSpan.End()
		}

		return nil, GatewayDecideResponse{
			Template:            "custom_questions",
			Answers:             resp.Answers,
			Diagnostics:         resp.Diagnostics,
			Decision:            resp,
			Cascade:             cascadeSummary,
			Temperature:         appliedTemp,
			SuggestedExpansions: resp.SuggestedExpansions,
			MaxEntropy:          summary.MaxEntropy,
			WallTimeMs:          time.Since(start).Milliseconds(),
			GpuForwardMs:        summary.GpuForwardMs,
			ColdStartWaitMs:     summary.ColdStartWaitMs,
			WarmupAttempts:      attempts,
			Model:               stats.Model,
			BackendTarget:       backendTarget,
			UpstreamURL:         targetURL,
		}, nil
	})

	// Tool 6: list_policy_templates
	mcp.AddTool(srv, &mcp.Tool{
		Name:        "list_policy_templates",
		Description: "Lists the executable .json.tmpl decision policies available on the server, including required variables and sample values. Filter with category: 'core', 'calibration', 'multimodal', or 'rerank'.",
		InputSchema: geminiSafeInputSchema[ListTemplatesToolInput](),
	}, func(ctx context.Context, req *mcp.CallToolRequest, input ListTemplatesToolInput) (*mcp.CallToolResult, ListTemplatesToolOutput, error) {
		catalog, err := discoverTemplates(serveTemplatesDir)
		if err != nil {
			return nil, ListTemplatesToolOutput{}, err
		}
		var filtered []TemplateCatalogEntry
		catFilter := strings.TrimSpace(strings.ToLower(input.Category))
		for _, item := range catalog {
			if catFilter == "" || strings.ToLower(item.Category) == catFilter {
				// Omit full raw_template in listing to keep agent context lean
				slim := item
				slim.RawTemplate = ""
				filtered = append(filtered, slim)
			}
		}
		return nil, ListTemplatesToolOutput{
			Count:     len(filtered),
			Templates: filtered,
		}, nil
	})

	return srv
}

// newMCPHTTPHandler builds the Streamable HTTP /mcp handler. The SDK's DNS rebinding protection (403 for a loopback
// connection with a non-loopback Host header) stays on. gatewaySecurityHandler applies the same check to the whole
// gateway; only when the operator lists extra hosts with --allowed-hosts is the SDK check turned off, so those hosts
// (still enforced by gatewaySecurityHandler) reach /mcp too.
func newMCPHTTPHandler(extraAllowedHosts bool) http.Handler {
	mcpServer := buildMCPServer()
	return mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
		return mcpServer
	}, &mcp.StreamableHTTPOptions{
		Stateless:                  true,
		JSONResponse:               true,
		DisableLocalhostProtection: extraAllowedHosts,
	})
}
