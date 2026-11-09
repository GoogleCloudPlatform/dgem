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

package dgemgenkit

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	systemonex "github.com/firebase/genkit/go/plugins/systemone/exp"
)

func TestNewSystemOnePlugin_DirectGatewayMode(t *testing.T) {
	var (
		gotPath    string
		gotBackend string
		gotGwKey   string
		gotAuth    string
	)
	mockGateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBackend = r.Header.Get("X-DGem-Backend")
		gotGwKey = r.Header.Get("X-DGem-Api-Key")
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "diffgemma-26b-a4b-it-q4",
			"answers": map[string]any{
				"urgent": map[string]any{
					"type": "noul",
					"noul": 0.99,
				},
				"team": map[string]any{
					"type":          "choice",
					"choice":        "billing",
					"confidence":    0.98,
					"probabilities": map[string]float64{"billing": 0.98, "support": 0.01, "engineering": 0.01},
				},
				"sentiment": map[string]any{
					"type":          "score",
					"score":         2.0,
					"confidence":    0.95,
					"probabilities": map[string]float64{"0": 0.01, "1": 0.04, "2": 0.95},
				},
			},
		})
	}))
	defer mockGateway.Close()

	cfg := EndpointConfig{
		SystemOneURL: mockGateway.URL,
		ModelID:      "diffgemma-26b-a4b-it-q4",
		Provider:     "dgem",
		Backend:      "vertex_first",
		GatewayKey:   "gw-secret-123",
		AuthToken:    "Bearer iam-token-xyz",
	}
	if !cfg.IsDirectSystemOne() {
		t.Fatalf("cfg.IsDirectSystemOne() = false, want true")
	}

	plugin, closeFn := NewSystemOnePlugin(cfg)
	defer closeFn()

	ctx := t.Context()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	tmplDir := ResolveTemplatesDir("../../templates")
	dec, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), filepath.Join(tmplDir, "support_triage.json.tmpl"), map[string]any{
		"ticket":  "Double billed for renewal",
		"samples": 1,
	})
	if err != nil {
		t.Fatalf("DecideWithTemplate(support_triage) unexpected error: %v", err)
	}
	if gotPath != "/v1/systemone" {
		t.Errorf("request path = %q, want %q", gotPath, "/v1/systemone")
	}
	if gotBackend != "vertex_first" {
		t.Errorf("X-DGem-Backend = %q, want %q", gotBackend, "vertex_first")
	}
	if gotGwKey != "gw-secret-123" {
		t.Errorf("X-DGem-Api-Key = %q, want %q", gotGwKey, "gw-secret-123")
	}
	if gotAuth != "Bearer iam-token-xyz" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer iam-token-xyz")
	}
	if dec.Answers["team"].Choice != "billing" {
		t.Errorf("Answers[\"team\"].Choice = %q, want %q", dec.Answers["team"].Choice, "billing")
	}
	if dec.Passes != 1 {
		t.Errorf("dec.Passes = %d, want 1", dec.Passes)
	}
}

func TestEndpointConfigModeDetection(t *testing.T) {
	cases := []struct {
		name       string
		cfg        EndpointConfig
		wantDirect bool
	}{
		{
			name:       "default_local_diffgemma_v1",
			cfg:        EndpointConfig{URL: "http://127.0.0.1:8080/v1"},
			wantDirect: false,
		},
		{
			name:       "explicit_SystemOneURL",
			cfg:        EndpointConfig{URL: "http://127.0.0.1:8080/v1", SystemOneURL: "http://127.0.0.1:8095"},
			wantDirect: true,
		},
		{
			name:       "gateway_root_URL",
			cfg:        EndpointConfig{URL: "https://dgemma-gateway.a.run.app"},
			wantDirect: true,
		},
		{
			name:       "apple_container_host_bridge_gateway_URL",
			cfg:        EndpointConfig{URL: "http://192.168.64.1:8095"},
			wantDirect: true,
		},
		{
			name:       "explicit_v1_systemone_path",
			cfg:        EndpointConfig{URL: "http://192.168.64.1:8095/v1/systemone"},
			wantDirect: true,
		},
		{
			name:       "vertex_rawPredict_endpoint",
			cfg:        EndpointConfig{URL: "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/123:rawPredict"},
			wantDirect: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.IsDirectSystemOne(); got != tc.wantDirect {
				t.Errorf("EndpointConfig(%+v).IsDirectSystemOne() = %v, want %v", tc.cfg, got, tc.wantDirect)
			}
		})
	}
}

func TestDecideWithTemplate_ConditionalDAG(t *testing.T) {
	var (
		callCount     atomic.Int32
		lastPriorAns  map[string]any
		lastSeedToken string
	)
	mockGateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if st, ok := body["seed_token"].(string); ok {
			lastSeedToken = st
		}

		questions, _ := body["questions"].(map[string]any)
		stateMap, _ := body["state"].(map[string]any)
		if pa, ok := stateMap["prior_answers"].(map[string]any); ok {
			lastPriorAns = pa
		}
		payload, _ := stateMap["alert_payload"].(string)

		answers := make(map[string]any)
		if _, asksActive := questions["active_breach"]; asksActive {
			if payload == "benign_backup" {
				answers["active_breach"] = map[string]any{
					"type":          "choice",
					"choice":        "no",
					"confidence":    0.99,
					"probabilities": map[string]float64{"yes": 0.01, "no": 0.99},
				}
				answers["threat_vector"] = map[string]any{
					"type":          "choice",
					"choice":        "benign_operation",
					"confidence":    0.98,
					"probabilities": map[string]float64{"benign_operation": 0.98, "iam_compromise": 0.01, "data_exfiltration": 0.005, "workload_injection": 0.005},
				}
				answers["confidence_score"] = map[string]any{
					"type":          "score",
					"score":         0.02,
					"confidence":    0.98,
					"probabilities": map[string]float64{"0": 0.98, "1": 0.02, "2": 0, "3": 0, "4": 0},
				}
			} else {
				answers["active_breach"] = map[string]any{
					"type":          "choice",
					"choice":        "yes",
					"confidence":    0.97,
					"probabilities": map[string]float64{"yes": 0.97, "no": 0.03},
				}
				answers["threat_vector"] = map[string]any{
					"type":          "choice",
					"choice":        "iam_compromise",
					"confidence":    0.96,
					"probabilities": map[string]float64{"iam_compromise": 0.96, "data_exfiltration": 0.02, "workload_injection": 0.01, "benign_operation": 0.01},
				}
				answers["confidence_score"] = map[string]any{
					"type":          "score",
					"score":         3.95,
					"confidence":    0.95,
					"probabilities": map[string]float64{"0": 0, "1": 0, "2": 0, "3": 0.05, "4": 0.95},
				}
			}
		}
		if _, asksContainment := questions["containment_action"]; asksContainment {
			answers["containment_action"] = map[string]any{
				"type":          "choice",
				"choice":        "revoke_iam_and_snapshots",
				"confidence":    0.95,
				"probabilities": map[string]float64{"revoke_iam_and_snapshots": 0.95, "quarantine_pod_egress": 0.02, "block_edge_asn": 0.02, "freeze_db_replica": 0.01},
			}
			answers["blast_radius"] = map[string]any{
				"type":          "choice",
				"choice":        "account_wide_root",
				"confidence":    0.93,
				"probabilities": map[string]float64{"account_wide_root": 0.93, "production_data_plane": 0.05, "single_service": 0.02},
			}
		}
		if _, asksBool := questions["is_incident"]; asksBool {
			p := 0.95
			if payload == "benign_bool" {
				p = 0.04
			}
			answers["is_incident"] = map[string]any{"type": "noul", "noul": p}
		}
		if _, asksPlaybook := questions["playbook"]; asksPlaybook {
			answers["playbook"] = map[string]any{
				"type":          "choice",
				"choice":        "isolate_host",
				"confidence":    0.94,
				"probabilities": map[string]float64{"isolate_host": 0.94, "page_oncall": 0.06},
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "diffgemma-26b-a4b-it-q4",
			"answers": answers,
		})
	}))
	defer mockGateway.Close()

	cfg := EndpointConfig{
		SystemOneURL: mockGateway.URL,
		ModelID:      "diffgemma-26b-a4b-it-q4",
		Provider:     "dgem",
	}
	plugin, closeFn := NewSystemOnePlugin(cfg)
	defer closeFn()

	ctx := t.Context()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	tmplDir := ResolveTemplatesDir("../../templates")
	secopsDAG := filepath.Join(tmplDir, "secops_conditional_dag.json.tmpl")

	t.Run("benign_skips_stage2_questions", func(t *testing.T) {
		callCount.Store(0)
		dec, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), secopsDAG, map[string]any{
			"alert_payload": "benign_backup",
			"samples":       1,
		})
		if err != nil {
			t.Fatalf("DecideWithTemplate(%q, benign_backup) unexpected error: %v", secopsDAG, err)
		}
		if dec.Passes != 1 || callCount.Load() != 1 {
			t.Errorf("DecideWithTemplate(benign_backup) passes=%d calls=%d, want 1 and 1", dec.Passes, callCount.Load())
		}
		wantSkipped := []string{"containment_action", "blast_radius"}
		if !slices.Equal(dec.Skipped, wantSkipped) {
			t.Errorf("DecideWithTemplate(benign_backup) Skipped = %v, want %v", dec.Skipped, wantSkipped)
		}
		if !dec.IsSkipped("containment_action") {
			t.Errorf("dec.IsSkipped(\"containment_action\") = false, want true")
		}
		if _, present := dec.Answers["containment_action"]; present {
			t.Errorf("Answers[\"containment_action\"] present on skipped stage, want omitted")
		}
	})

	t.Run("active_breach_runs_stage2_with_prior_answers", func(t *testing.T) {
		callCount.Store(0)
		lastPriorAns = nil
		dec, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), secopsDAG, map[string]any{
			"alert_payload": "sts:AssumeRole root snapshot public",
			"samples":       1,
		})
		if err != nil {
			t.Fatalf("DecideWithTemplate(%q, active_breach) unexpected error: %v", secopsDAG, err)
		}
		if dec.Passes != 2 || callCount.Load() != 2 {
			t.Errorf("DecideWithTemplate(active_breach) passes=%d calls=%d, want 2 and 2", dec.Passes, callCount.Load())
		}
		if len(dec.Skipped) != 0 {
			t.Errorf("DecideWithTemplate(active_breach) Skipped = %v, want empty", dec.Skipped)
		}
		if got := dec.Answers["containment_action"].Choice; got != "revoke_iam_and_snapshots" {
			t.Errorf("Answers[\"containment_action\"].Choice = %q, want %q", got, "revoke_iam_and_snapshots")
		}
		if got := dec.Answers["blast_radius"].Choice; got != "account_wide_root" {
			t.Errorf("Answers[\"blast_radius\"].Choice = %q, want %q", got, "account_wide_root")
		}
		if lastPriorAns["active_breach"] != "yes" {
			t.Errorf("stage 2 prior_answers[\"active_breach\"] = %v, want %q", lastPriorAns["active_breach"], "yes")
		}
	})

	t.Run("boolean_noul_ask_if_yes_and_seed_token_forwarding", func(t *testing.T) {
		tmpDir := t.TempDir()
		boolDAG := filepath.Join(tmpDir, "bool_dag.json.tmpl")
		tmplBody := `{
  "schema": {
    "seed_token": "fixed",
    "temperature": 2.0,
    "instructions": "Evaluate alert.",
    "questions": [
      {"id": "is_incident", "type": "boolean", "instructions": "Is this an incident?"},
      {
        "id": "playbook",
        "type": "choice",
        "instructions": "Select playbook.",
        "depends_on": ["is_incident"],
        "ask_if": {"is_incident": ["yes"]},
        "options": [{"name": "isolate_host", "description": "Isolate"}, {"name": "page_oncall", "description": "Page"}]
      }
    ],
    "samples": 1
  },
  "state": {"alert_payload": {{ default "active_bool" .alert_payload | toJson }}}
}`
		if err := os.WriteFile(boolDAG, []byte(tmplBody), 0o644); err != nil {
			t.Fatalf("WriteFile(%q): %v", boolDAG, err)
		}

		lastSeedToken = ""
		decYes, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), boolDAG, map[string]any{"alert_payload": "active_bool"})
		if err != nil {
			t.Fatalf("DecideWithTemplate(bool_dag, active_bool) unexpected error: %v", err)
		}
		if lastSeedToken != "fixed" {
			t.Errorf("forwarded seed_token = %q, want %q", lastSeedToken, "fixed")
		}
		if decYes.Passes != 2 || decYes.Answers["playbook"].Choice != "isolate_host" {
			t.Errorf("DecideWithTemplate(bool_dag, active_bool) passes=%d playbook=%q, want 2 and %q",
				decYes.Passes, decYes.Answers["playbook"].Choice, "isolate_host")
		}
		if decYes.Temperature != 2.0 {
			t.Errorf("decYes.Temperature = %v, want 2.0", decYes.Temperature)
		}
		if decYes.RawAnswers["is_incident"].Probability != 0.95 {
			t.Errorf("RawAnswers[\"is_incident\"].Probability = %v, want 0.95", decYes.RawAnswers["is_incident"].Probability)
		}
		if decYes.Answers["is_incident"].Probability >= 0.95 {
			t.Errorf("scaled Answers[\"is_incident\"].Probability = %v, want softened < 0.95", decYes.Answers["is_incident"].Probability)
		}

		decNo, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), boolDAG, map[string]any{"alert_payload": "benign_bool"})
		if err != nil {
			t.Fatalf("DecideWithTemplate(bool_dag, benign_bool) unexpected error: %v", err)
		}
		if decNo.Passes != 1 || !decNo.IsSkipped("playbook") {
			t.Errorf("DecideWithTemplate(bool_dag, benign_bool) passes=%d skipped=%v, want 1 and [playbook]", decNo.Passes, decNo.Skipped)
		}
	})

	t.Run("rejects_cycle_and_invalid_ask_if", func(t *testing.T) {
		tmpDir := t.TempDir()
		cyclePath := filepath.Join(tmpDir, "cycle.json.tmpl")
		cycleBody := `{
  "schema": {
    "questions": [
      {"id": "q1", "type": "boolean", "instructions": "Q1", "depends_on": ["q2"]},
      {"id": "q2", "type": "boolean", "instructions": "Q2", "depends_on": ["q1"]}
    ]
  },
  "state": {"x": "1"}
}`
		if err := os.WriteFile(cyclePath, []byte(cycleBody), 0o644); err != nil {
			t.Fatalf("WriteFile(%q): %v", cyclePath, err)
		}
		if _, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), cyclePath, nil); err == nil {
			t.Errorf("DecideWithTemplate(%q) succeeded, want cycle error", cyclePath)
		}

		badAskPath := filepath.Join(tmpDir, "bad_ask_if.json.tmpl")
		badAskBody := `{
  "schema": {
    "questions": [
      {"id": "q1", "type": "boolean", "instructions": "Q1"},
      {"id": "q2", "type": "boolean", "instructions": "Q2", "ask_if": {"q1": ["maybe"]}}
    ]
  },
  "state": {"x": "1"}
}`
		if err := os.WriteFile(badAskPath, []byte(badAskBody), 0o644); err != nil {
			t.Fatalf("WriteFile(%q): %v", badAskPath, err)
		}
		if _, err := DecideWithTemplate(ctx, g, cfg.GenkitModelName(), badAskPath, nil); err == nil {
			t.Errorf("DecideWithTemplate(%q) succeeded, want invalid ask_if error", badAskPath)
		}
	})
}

func TestTemplateDecision_TemperatureAndThresholds(t *testing.T) {
	dec := &TemplateDecision{
		Answers: map[string]systemonex.Answer{
			// Overconfident false negative: raw p(yes)=0.0127 looks Clear (H_norm = 9.8% < 16%),
			// but at T=2.625 scales to p_T(yes)=0.1600 with CalibratedHesitation = 63.4% >= 16%.
			"unsafe_request": {
				Type:        "noul",
				Probability: 0.0127,
			},
			"harm_category": {
				Type:       "choice",
				Choice:     "cyberattack_or_malware",
				Confidence: 0.90,
				Probabilities: map[string]float64{
					"cyberattack_or_malware": 0.90,
					"fraud_or_illegal_acts":  0.06,
					"none":                   0.04,
				},
			},
			"severity": {
				Type:       "score",
				Score:      2.80,
				Confidence: 0.85,
				Probabilities: map[string]float64{
					"0": 0.02,
					"1": 0.03,
					"2": 0.10,
					"3": 0.85,
				},
				Legend: map[string]string{
					"0": "compliant",
					"1": "minor_borderline",
					"2": "moderate_violation",
					"3": "severe_violation",
				},
			},
		},
		OptionCount: map[string]int{
			"unsafe_request": 2,
			"harm_category":  3,
			"severity":       4,
		},
		Passes:      1,
		Temperature: 1.0,
	}

	rawH := dec.RawHesitation("unsafe_request")
	if rawH >= 0.16 {
		t.Fatalf("RawHesitation(\"unsafe_request\") = %.4f, want < 0.16 (Clear band before calibration)", rawH)
	}

	calP := dec.CalibratedProb("unsafe_request", "yes", 2.625)
	if math.Abs(calP-0.1600) > 0.002 {
		t.Errorf("CalibratedProb(\"unsafe_request\", \"yes\", 2.625) = %.4f, want ~0.1600", calP)
	}
	calPNo := dec.CalibratedProb("unsafe_request", "no", 2.625)
	if math.Abs((calP+calPNo)-1.0) > 1e-9 {
		t.Errorf("CalibratedProb yes+no = %.9f, want 1.0", calP+calPNo)
	}

	// Operating-point thresholds: strict (tau=0.06) and balanced (tau=0.15) catch it; uncalibrated default (tau=0.50) misses it.
	if !dec.ExceedsThreshold("unsafe_request", "yes", 2.625, 0.06) {
		t.Errorf("ExceedsThreshold(\"unsafe_request\", \"yes\", 2.625, 0.06) = false, want true")
	}
	if !dec.ExceedsThreshold("unsafe_request", "yes", 2.625, 0.15) {
		t.Errorf("ExceedsThreshold(\"unsafe_request\", \"yes\", 2.625, 0.15) = false, want true")
	}
	if dec.ExceedsThreshold("unsafe_request", "yes", 1.0, 0.50) {
		t.Errorf("ExceedsThreshold(\"unsafe_request\", \"yes\", 1.0, 0.50) = true, want false")
	}

	calH := dec.CalibratedHesitation("unsafe_request", 2.625)
	if calH < 0.50 {
		t.Errorf("CalibratedHesitation(\"unsafe_request\", 2.625) = %.4f, want >= 0.50", calH)
	}
	// Non-mutating helpers must leave Answers untouched.
	if dec.Answers["unsafe_request"].Probability != 0.0127 {
		t.Errorf("Answers[\"unsafe_request\"].Probability mutated to %v before ApplyTemperature", dec.Answers["unsafe_request"].Probability)
	}

	// ApplyTemperature mutates Answers idempotently while preserving RawHesitation and argmax Choice.
	dec.ApplyTemperature(2.625)
	dec.ApplyTemperature(2.625) // second call must not double-scale
	if math.Abs(dec.Answers["unsafe_request"].Probability-calP) > 1e-9 {
		t.Errorf("after repeated ApplyTemperature(2.625), Probability = %.6f, want %.6f", dec.Answers["unsafe_request"].Probability, calP)
	}
	if math.Abs(dec.RawHesitation("unsafe_request")-rawH) > 1e-9 {
		t.Errorf("RawHesitation(\"unsafe_request\") after ApplyTemperature = %.6f, want preserved %.6f", dec.RawHesitation("unsafe_request"), rawH)
	}
	if math.Abs(dec.Hesitation("unsafe_request")-calH) > 1e-9 {
		t.Errorf("Hesitation(\"unsafe_request\") after ApplyTemperature = %.6f, want %.6f", dec.Hesitation("unsafe_request"), calH)
	}
	if got := dec.Answers["harm_category"].Choice; got != "cyberattack_or_malware" {
		t.Errorf("ApplyTemperature changed argmax Choice to %q, want %q", got, "cyberattack_or_malware")
	}
	var sumChoice float64
	for _, p := range dec.Answers["harm_category"].Probabilities {
		sumChoice += p
	}
	if math.Abs(sumChoice-1.0) > 1e-9 {
		t.Errorf("scaled choice probabilities sum = %.9f, want 1.0", sumChoice)
	}
	if pSevere := dec.CalibratedProb("severity", "severe_violation", 2.0); pSevere <= 0 || pSevere >= 0.85 {
		t.Errorf("CalibratedProb(\"severity\", \"severe_violation\", 2.0) = %.4f, want in (0, 0.85)", pSevere)
	}
}
