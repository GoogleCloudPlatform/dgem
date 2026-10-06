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
	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// applyPolicyTemperature rescales a decision's probabilities with the policy's temperature: the request's value if
// given, else the template's "temperature" key. It runs after Stage 1 and after the cascade (so cascade gating keeps
// using raw entropy) and leaves answers resolved by the Stage-2 cascade untouched. Returns the temperature applied
// (nil when none) and the new maximum entropy.
func applyPolicyTemperature(resp *client.StructuredDecisionResponse, schemaJSON string, req *client.Temperature,
	cascade *CascadeExecutionSummary) (*client.Temperature, float64, error) {
	t := client.Temperature{}
	if req != nil && (req.All > 0 || len(req.ByType) > 0) {
		t = *req // an explicit request value wins, including 1 ("no temperature for this request")
	} else {
		st, err := client.ParseSchemaTemperature(schemaJSON)
		if err != nil {
			return nil, 0, err
		}
		t = st
	}
	if resp == nil || !t.IsSet() {
		return nil, maxAnswerEntropy(resp), nil
	}
	escalated := map[string]client.QuestionAnswer{}
	if cascade != nil {
		for id, s := range cascade.Slots {
			if s.Escalated {
				if a, ok := resp.Answers[id]; ok {
					escalated[id] = a
				}
			}
		}
	}
	resp.ApplyTemperature(t)
	for id, a := range escalated {
		resp.Answers[id] = a
	}
	return &t, maxAnswerEntropy(resp), nil
}

func maxAnswerEntropy(resp *client.StructuredDecisionResponse) float64 {
	if resp == nil {
		return 0
	}
	m := 0.0
	for _, a := range resp.Answers {
		if a.Entropy > m {
			m = a.Entropy
		}
	}
	return m
}

// mcpTemperature builds a Temperature from the MCP tools' flat fields.
func mcpTemperature(all float64, byType map[string]float64) (*client.Temperature, error) {
	if all == 0 && len(byType) == 0 {
		return nil, nil
	}
	t := &client.Temperature{All: all}
	if len(byType) > 0 {
		t.ByType = map[string]float64{}
		for k, v := range byType {
			var probe client.Temperature
			if err := probe.UnmarshalJSON([]byte(`{"` + k + `":1}`)); err != nil {
				return nil, err
			}
			for nk := range probe.ByType {
				t.ByType[nk] = v
			}
		}
	}
	return t, t.Validate()
}

// mustMCPTemperature is mcpTemperature for the remote-proxy mapping, where a bad value is reported by the gateway.
func mustMCPTemperature(all float64, byType map[string]float64) *client.Temperature {
	t, err := mcpTemperature(all, byType)
	if err != nil {
		return &client.Temperature{All: all, ByType: byType}
	}
	return t
}
