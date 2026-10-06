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
	"math"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestApplyPolicyTemperature(t *testing.T) {
	mk := func() *client.StructuredDecisionResponse {
		return &client.StructuredDecisionResponse{Answers: map[string]client.QuestionAnswer{
			"urgent": {Type: "noul", Label: "yes", Noul: 0.9, Probabilities: map[string]float64{"yes": 0.9, "no": 0.1}},
			"team":   {Type: "choice", Choice: "billing", Probabilities: map[string]float64{"billing": 0.8, "support": 0.2}},
		}}
	}
	schema := `{"questions": [], "temperature": {"noul": 2}}`
	r := mk()
	tt, maxH, err := applyPolicyTemperature(r, schema, nil, nil)
	if err != nil || tt == nil || math.Abs(r.Answers["urgent"].Noul-0.75) > 1e-9 || r.Answers["team"].Probabilities["billing"] != 0.8 || maxH <= 0 {
		t.Fatalf("template temperature: %+v %v %+v", tt, err, r.Answers)
	}
	r = mk()
	if _, _, err := applyPolicyTemperature(r, schema, &client.Temperature{All: 1.5}, nil); err != nil || r.Answers["team"].Probabilities["billing"] == 0.8 {
		t.Fatalf("request override should temper every answer: %v %+v", err, r.Answers["team"])
	}
	r = mk()
	if tt, _, _ := applyPolicyTemperature(r, schema, &client.Temperature{All: 1}, nil); tt != nil || r.Answers["urgent"].Noul != 0.9 {
		t.Fatalf("an explicit temperature 1 must override the template: %+v %+v", tt, r.Answers["urgent"])
	}
	r = mk()
	cas := &CascadeExecutionSummary{Slots: map[string]CascadeSlotTelemetry{"urgent": {Escalated: true}}}
	applyPolicyTemperature(r, schema, nil, cas)
	if r.Answers["urgent"].Noul != 0.9 {
		t.Fatalf("an answer resolved by the cascade must be left as is: %+v", r.Answers["urgent"])
	}
	if _, _, err := applyPolicyTemperature(mk(), `{"temperature": {"colour": 2}}`, nil, nil); err == nil {
		t.Fatal("invalid template temperature must be an error")
	}
	if got, err := mcpTemperature(0, map[string]float64{"boolean": 3}); err != nil || got.For("noul") != 3 {
		t.Fatalf("mcp by-type: %+v %v", got, err)
	}
}
