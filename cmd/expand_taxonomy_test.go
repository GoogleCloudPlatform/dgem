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
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

func TestInjectUnclassifiedCatchAll(t *testing.T) {
	rawSchema := `{
		"instructions": "Triage tickets",
		"questions": [
			{
				"id": "urgent",
				"type": "boolean",
				"instructions": "Urgent?"
			},
			{
				"id": "team",
				"type": "choice",
				"instructions": "Which team?",
				"options": [
					{"name": "billing", "description": "Invoices"},
					{"name": "engineering", "description": "Outages"}
				]
			},
			{
				"id": "already_has_other",
				"type": "choice",
				"instructions": "Category",
				"options": [
					{"name": "known", "description": "Known"},
					{"name": "other", "description": "Catch-all"}
				]
			}
		]
	}`

	modSchema, injected, existing := InjectUnclassifiedCatchAll(rawSchema)
	if !injected["team"] {
		t.Fatalf("expected slot 'team' to have catch-all injected")
	}
	if injected["already_has_other"] {
		t.Fatalf("did not expect slot 'already_has_other' to have catch-all injected")
	}
	if len(existing["team"]) != 2 {
		t.Fatalf("expected 2 existing options recorded for 'team', got %d", len(existing["team"]))
	}
	if !strings.Contains(modSchema, DefaultInjectedCatchAllName) {
		t.Fatalf("expected modified schema to contain %q", DefaultInjectedCatchAllName)
	}
}

func TestParseSuggestedOptionFromThought(t *testing.T) {
	cases := []struct {
		name        string
		thought     string
		wantName    string
		wantDescSub string
	}{
		{
			name:        "json_format",
			thought:     `The input discusses ECCN export controls. SUGGESTED_OPTION: {"name": "Legal & Export Compliance", "description": "Export control classification (ECCN), indemnity addenda, and bilateral contract reviews"}`,
			wantName:    "legal_export_compliance",
			wantDescSub: "Export control classification",
		},
		{
			name:        "pipe_delimited",
			thought:     `SUGGESTED_CLASS: hipaa_data_residency | DESCRIPTION: HIPAA BAA execution and regional CMEK key management`,
			wantName:    "hipaa_data_residency",
			wantDescSub: "HIPAA BAA execution",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opt, ok := ParseSuggestedOptionFromThought(tc.thought)
			if !ok {
				t.Fatalf("expected ParseSuggestedOptionFromThought to succeed for %q", tc.thought)
			}
			if opt.Name != tc.wantName {
				t.Errorf("got Name=%q, want %q", opt.Name, tc.wantName)
			}
			if !strings.Contains(opt.Description, tc.wantDescSub) {
				t.Errorf("got Description=%q, want substring %q", opt.Description, tc.wantDescSub)
			}
		})
	}
}

func TestSynthesizeTaxonomyExpansionsFromPass1Thought(t *testing.T) {
	resp := &client.StructuredDecisionResponse{
		Answers: map[string]client.QuestionAnswer{
			"primary_category": {
				Type:       "choice",
				Choice:     "other",
				Confidence: 0.91,
				Probabilities: map[string]float64{
					"other":                   0.91,
					"security_and_compliance": 0.06,
					"billing_and_invoicing":   0.03,
				},
			},
		},
		Diagnostics: client.Diagnostics{
			Thought: &client.ThoughtDiagnostic{
				Text:   `SUGGESTED_OPTION: {"name": "procurement_legal_indemnity", "description": "Custom AI indemnity addenda, ECCN export classification, and board procurement reviews"}`,
				Tokens: 34,
				Ms:     210,
			},
		},
	}

	suggestions := SynthesizeTaxonomyExpansions(
		context.Background(),
		nil,
		"templates/taxonomy_discovery.json.tmpl",
		`{"input": "Need custom AI indemnity addendum"}`,
		resp,
		map[string]bool{"primary_category": false},
		nil,
		0.35,
	)

	if len(suggestions) != 1 {
		t.Fatalf("expected 1 suggestion, got %d", len(suggestions))
	}
	if suggestions[0].SuggestedOption.Name != "procurement_legal_indemnity" {
		t.Errorf("got suggested name %q, want 'procurement_legal_indemnity'", suggestions[0].SuggestedOption.Name)
	}
}
