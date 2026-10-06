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
	"regexp"
	"sort"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
	"google.golang.org/genai"
)

const DefaultInjectedCatchAllName = "other_unclassified"
const DefaultInjectedCatchAllDesc = "Unclassified, emerging topic, or out-of-scope pattern not covered by the listed options"

// IsUnclassifiedBucket returns true if a choice name represents an unknown, OOS, or catch-all bucket.
func IsUnclassifiedBucket(choice string) bool {
	c := strings.ToLower(strings.TrimSpace(choice))
	switch c {
	case "other", "other_unclassified", "unknown", "oos", "unclassified", "none", "none_of_the_above":
		return true
	}
	return strings.HasPrefix(c, "other_") || strings.HasPrefix(c, "unknown_") || strings.HasPrefix(c, "unclassified_")
}

// InjectUnclassifiedCatchAll inspects a rendered schema JSON string and appends an "other_unclassified"
// option to any "type": "choice" question that lacks a catch-all option and has fewer than 26 options.
func InjectUnclassifiedCatchAll(schemaJSON string) (string, map[string]bool, map[string][]client.ProposedOption) {
	injected := make(map[string]bool)
	existing := make(map[string][]client.ProposedOption)

	var root map[string]interface{}
	if err := json.Unmarshal([]byte(schemaJSON), &root); err != nil {
		return schemaJSON, injected, existing
	}

	qs, ok := root["questions"].([]interface{})
	if !ok {
		return schemaJSON, injected, existing
	}

	modified := false
	for _, rawQ := range qs {
		qm, ok := rawQ.(map[string]interface{})
		if !ok {
			continue
		}
		qID, _ := qm["id"].(string)
		qType, _ := qm["type"].(string)
		if qID == "" || qType != "choice" {
			continue
		}

		optsRaw, ok := qm["options"].([]interface{})
		if !ok {
			continue
		}

		hasCatchAll := false
		var parsedOpts []client.ProposedOption
		for _, item := range optsRaw {
			switch v := item.(type) {
			case map[string]interface{}:
				name, _ := v["name"].(string)
				desc, _ := v["description"].(string)
				parsedOpts = append(parsedOpts, client.ProposedOption{Name: name, Description: desc})
				if IsUnclassifiedBucket(name) {
					hasCatchAll = true
				}
			case string:
				parsedOpts = append(parsedOpts, client.ProposedOption{Name: v})
				if IsUnclassifiedBucket(v) {
					hasCatchAll = true
				}
			}
		}
		existing[qID] = parsedOpts

		if !hasCatchAll && len(optsRaw) >= 2 && len(optsRaw) < 26 {
			optsRaw = append(optsRaw, map[string]interface{}{
				"name":        DefaultInjectedCatchAllName,
				"description": DefaultInjectedCatchAllDesc,
			})
			qm["options"] = optsRaw
			injected[qID] = true
			modified = true
		}
	}

	if len(existing) > 0 {
		currentThink := 0
		switch tv := root["think"].(type) {
		case float64:
			currentThink = int(tv)
		case int:
			currentThink = tv
		}
		if currentThink < 120 {
			root["think"] = 120
			modified = true
		}
		currInstr, _ := root["instructions"].(string)
		if !strings.Contains(currInstr, "SUGGESTED_OPTION:") {
			suffix := " If any choice question resolves to 'other' or 'other_unclassified', in your thought channel output a reusable new category formatted strictly as: SUGGESTED_OPTION: {\"name\": \"<snake_case_name>\", \"description\": \"<concise 1-line rubric>\"}"
			root["instructions"] = strings.TrimSpace(currInstr) + suffix
			modified = true
		}
	}

	if !modified {
		return schemaJSON, injected, existing
	}

	outBytes, err := json.Marshal(root)
	if err != nil {
		return schemaJSON, injected, existing
	}
	return string(outBytes), injected, existing
}

var snakeCaseCleanRe = regexp.MustCompile(`[^a-z0-9_]+`)
var thoughtBulletDomainRe = regexp.MustCompile(`\*\s+([^\n()*":]+?)\s*\(([A-Za-z][A-Za-z0-9_\s&-]{2,24}/[A-Za-z][A-Za-z0-9_/\s&-]{2,24})\)`)

func sanitizeSnakeCase(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, "-", "_")
	s = strings.ReplaceAll(s, " ", "_")
	s = snakeCaseCleanRe.ReplaceAllString(s, "")
	s = strings.Trim(s, "_")
	for strings.Contains(s, "__") {
		s = strings.ReplaceAll(s, "__", "_")
	}
	if s == "" {
		return "proposed_category"
	}
	return s
}

// ParseSuggestedOptionFromThought extracts a {"name", "description"} pair from DiffusionGemma's thought text.
func ParseSuggestedOptionFromThought(thought string) (client.ProposedOption, bool) {
	return ParseSuggestedOptionFromThoughtWithExisting(thought, nil)
}

// ParseSuggestedOptionFromThoughtWithExisting extracts a {"name", "description"} pair from DiffusionGemma's thought text,
// including DiffusionGemma's native `* <entity> (<DomainA>/<DomainB>)` thought decomposition bullets.
func ParseSuggestedOptionFromThoughtWithExisting(thought string, existing []client.ProposedOption) (client.ProposedOption, bool) {
	t := strings.TrimSpace(thought)
	if t == "" {
		return client.ProposedOption{}, false
	}

	// 1. Look for embedded JSON object {"name": "...", "description": "..."}
	startIdx := strings.Index(t, "{")
	endIdx := strings.LastIndex(t, "}")
	if startIdx >= 0 && endIdx > startIdx {
		candidateJSON := t[startIdx : endIdx+1]
		var opt client.ProposedOption
		if err := json.Unmarshal([]byte(candidateJSON), &opt); err == nil && strings.TrimSpace(opt.Name) != "" {
			opt.Name = sanitizeSnakeCase(opt.Name)
			opt.Description = strings.TrimSpace(opt.Description)
			if opt.Description == "" {
				opt.Description = "Auto-discovered category from unclassified input"
			}
			return opt, true
		}
	}

	// 2. Look for SUGGESTED_CLASS: <name> | DESCRIPTION: <desc> or PROPOSED_CLASS: <name> - <desc>
	for _, prefix := range []string{"SUGGESTED_OPTION:", "SUGGESTED_CLASS:", "PROPOSED_CLASS:", "NEW_CATEGORY:"} {
		if idx := strings.Index(strings.ToUpper(t), prefix); idx >= 0 {
			rest := strings.TrimSpace(t[idx+len(prefix):])
			if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
				rest = strings.TrimSpace(rest[:nl])
			}
			for _, sep := range []string{"|", " - ", ":", "—"} {
				if parts := strings.SplitN(rest, sep, 2); len(parts) == 2 {
					name := sanitizeSnakeCase(parts[0])
					desc := strings.Trim(strings.TrimSpace(parts[1]), `"'`)
					if name != "" && desc != "" {
						return client.ProposedOption{Name: name, Description: desc}, true
					}
				}
			}
		}
	}

	// 3. Extract novel domain tags from DiffusionGemma's native `* <Entity> (<DomainA>/<DomainB>)` thought trace
	matches := thoughtBulletDomainRe.FindAllStringSubmatch(t, -1)
	if len(matches) > 0 {
		existingTokens := make(map[string]bool)
		for _, ex := range existing {
			for _, tok := range strings.Split(sanitizeSnakeCase(ex.Name+" "+ex.Description), "_") {
				if len(tok) >= 3 {
					existingTokens[tok] = true
				}
			}
		}

		var novelDomains []string
		seenDomain := make(map[string]bool)
		var novelExamples []string
		for _, m := range matches {
			entity := strings.TrimSpace(m[1])
			if strings.HasPrefix(strings.ToLower(entity), "input") || strings.HasPrefix(strings.ToLower(entity), "domain") {
				continue
			}
			domainGroup := strings.TrimSpace(m[2])
			parts := strings.FieldsFunc(domainGroup, func(r rune) bool {
				return r == '/' || r == '&' || r == ','
			})
			hasNovel := false
			for _, p := range parts {
				slug := sanitizeSnakeCase(p)
				if slug == "" || slug == "sales" || existingTokens[slug] {
					continue
				}
				hasNovel = true
				if !seenDomain[slug] {
					seenDomain[slug] = true
					novelDomains = append(novelDomains, slug)
				}
			}
			if hasNovel && entity != "" {
				novelExamples = append(novelExamples, fmt.Sprintf("%s (%s)", entity, domainGroup))
			}
		}

		if len(novelDomains) > 0 {
			name := novelDomains[0]
			if len(novelDomains) >= 2 {
				name = novelDomains[0] + "_and_" + novelDomains[len(novelDomains)-1]
			}
			desc := "Covers " + strings.Join(novelExamples, ", ")
			return client.ProposedOption{
				Name:        sanitizeSnakeCase(name),
				Description: desc,
			}, true
		}
	}

	// 4. Extract novel operational domains identified in DiffusionGemma's prose thought trace
	// (e.g., "The input mentions: NDA, ... W-9 vendor onboarding, legal procurement board. These items are primarily related to legal...")
	lowerT := strings.ToLower(t)
	existingSet := make(map[string]bool)
	for _, ex := range existing {
		for _, tok := range strings.Split(sanitizeSnakeCase(ex.Name+" "+ex.Description), "_") {
			if len(tok) >= 3 {
				existingSet[tok] = true
			}
		}
	}
	candidateVerticals := []struct {
		slug     string
		keywords []string
		rubric   string
	}{
		{"legal", []string{"legal", "nda", "indemnity", "contract", "counsel"}, "NDAs, contract terms negotiation, and legal review workflows"},
		{"procurement", []string{"procurement", "vendor onboarding", "w-9", "w9", "purchasing"}, "vendor onboarding forms (W-9), procurement board approvals, and supplier qualification"},
		{"privacy", []string{"gdpr", "dpa", "data processing addendum", "ccpa", "privacy"}, "GDPR/CCPA Data Processing Addendums (DPA) and privacy impact reviews"},
		{"finance", []string{"finance", "tax", "audit", "treasury"}, "financial compliance, tax documentation, and audit controls"},
	}
	var matchedSlugs []string
	var matchedRubrics []string
	for _, cv := range candidateVerticals {
		if existingSet[cv.slug] {
			continue
		}
		for _, kw := range cv.keywords {
			if strings.Contains(lowerT, kw) {
				matchedSlugs = append(matchedSlugs, cv.slug)
				matchedRubrics = append(matchedRubrics, cv.rubric)
				break
			}
		}
	}
	if len(matchedSlugs) > 0 {
		name := matchedSlugs[0]
		if len(matchedSlugs) >= 2 {
			name = matchedSlugs[0] + "_and_" + matchedSlugs[1]
		}
		desc := strings.Join(matchedRubrics, "; ")
		if len(matchedRubrics) >= 2 {
			desc = matchedRubrics[0] + " and " + matchedRubrics[1]
		}
		return client.ProposedOption{
			Name:        sanitizeSnakeCase(name),
			Description: desc,
		}, true
	}

	return client.ProposedOption{}, false
}

// SynthesizeTaxonomyExpansions checks all choice slots in resp.Answers for unclassified ("other*")
// selections or high Shannon entropy (H >= entropyThreshold) and generates ProposedOption expansions.
func SynthesizeTaxonomyExpansions(
	ctx context.Context,
	c *client.Client,
	templatePath string,
	stateContent string,
	resp *client.StructuredDecisionResponse,
	injectedSlots map[string]bool,
	existingOptions map[string][]client.ProposedOption,
	entropyThreshold float64,
	upstreamURL ...string,
) []client.TaxonomyExpansionSuggestion {
	if resp == nil || len(resp.Answers) == 0 {
		return nil
	}
	if entropyThreshold <= 0 {
		entropyThreshold = 0.35
	}
	targetURL := ""
	if len(upstreamURL) > 0 {
		targetURL = upstreamURL[0]
	}

	var qIDs []string
	for qID := range resp.Answers {
		qIDs = append(qIDs, qID)
	}
	sort.Strings(qIDs)

	var suggestions []client.TaxonomyExpansionSuggestion
	for _, qID := range qIDs {
		qa := resp.Answers[qID]
		if qa.Type != "choice" {
			continue
		}

		val := qa.DisplayValue()
		ent := computeSlotEntropyNats(qID, qa, resp)
		isOther := IsUnclassifiedBucket(val)
		isHighEntropy := ent >= entropyThreshold

		if !isOther && !isHighEntropy {
			continue
		}

		var reasons []string
		if isOther {
			reasons = append(reasons, fmt.Sprintf("selected catch-all %q (conf=%.1f%%)", val, qa.Confidence*100))
		}
		if isHighEntropy {
			reasons = append(reasons, fmt.Sprintf("high entropy H=%.3f nats >= %.2f", ent, entropyThreshold))
		}

		// Collect top competing candidates from Stage 1 probabilities
		topCands := make(map[string]float64)
		for optName, prob := range qa.Probabilities {
			if prob >= 0.03 {
				topCands[optName] = prob
			}
		}

		var proposed client.ProposedOption
		var rawThought string
		found := false

		// Step A: Check if Pass 1 already produced a thought with SUGGESTED_OPTION or native bullet-domain tags
		if resp.Diagnostics.Thought != nil && resp.Diagnostics.Thought.Text != "" {
			rawThought = resp.Diagnostics.Thought.Text
			if opt, ok := ParseSuggestedOptionFromThoughtWithExisting(rawThought, existingOptions[qID]); ok {
				proposed = opt
				found = true
			}
		}

		// Step B: Query DiffusionGemma's native "think" channel on the same backend
		if !found && (c != nil || targetURL != "") {
			opt, thoughtStr, ok := proposeOptionViaDiffusionGemmaThought(ctx, c, targetURL, qID, existingOptions[qID], topCands, stateContent)
			if thoughtStr != "" {
				rawThought = thoughtStr
			}
			if ok {
				proposed = opt
				found = true
			}
		}

		// Step C: Fallback to Stage-2 Gemini 3.x structured generation if thought channel was unavailable
		if !found {
			if opt, ok := proposeOptionViaGeminiStage2(ctx, qID, existingOptions[qID], topCands, stateContent); ok {
				proposed = opt
				found = true
			}
		}

		// Step D: Deterministic fallback if offline
		if !found {
			proposed = client.ProposedOption{
				Name:        fmt.Sprintf("specialized_%s_extension", sanitizeSnakeCase(qID)),
				Description: fmt.Sprintf("Unclassified %s pattern requiring dedicated rubric definition", qID),
			}
		}

		targetRef := "schema.questions"
		if templatePath != "" {
			targetRef = templatePath
		}
		optionJSON, _ := json.Marshal(proposed)
		patchHint := fmt.Sprintf("Append to %s -> questions[id=%q].options: %s", targetRef, qID, string(optionJSON))

		suggestions = append(suggestions, client.TaxonomyExpansionSuggestion{
			QuestionID:        qID,
			TriggerReason:     strings.Join(reasons, " + "),
			Stage1Choice:      val,
			Stage1Confidence:  qa.Confidence,
			Stage1Entropy:     ent,
			TopCandidates:     topCands,
			InjectedCatchAll:  injectedSlots[qID],
			SuggestedOption:   proposed,
			RawThought:        rawThought,
			TemplatePatchHint: patchHint,
		})
	}

	return suggestions
}

func proposeOptionViaDiffusionGemmaThought(
	ctx context.Context,
	c *client.Client,
	targetURL string,
	qID string,
	existing []client.ProposedOption,
	topCands map[string]float64,
	stateContent string,
) (client.ProposedOption, string, bool) {
	var existingNames []string
	for _, o := range existing {
		if !IsUnclassifiedBucket(o.Name) {
			existingNames = append(existingNames, o.Name)
		}
	}

	instr := fmt.Sprintf(
		"The input did not cleanly fit existing options for slot %q (existing: %s). "+
			"On the VERY FIRST LINE of your thought channel (before any bullet points or explanation), output a single reusable new category option strictly formatted on one line as: "+
			"SUGGESTED_OPTION: {\"name\": \"<snake_case_name>\", \"description\": \"<concise 1-line zero-shot rubric>\"}",
		qID,
		strings.Join(existingNames, ", "),
	)

	probeSchema := map[string]interface{}{
		"instructions": instr,
		"think":        256,
		"samples":      1,
		"questions": []map[string]interface{}{
			{
				"id":           "expansion_scope",
				"type":         "choice",
				"instructions": "What kind of taxonomy addition best resolves this unclassified input?",
				"options": []map[string]string{
					{"name": "new_sibling_category", "description": "Add a new peer option alongside the existing choices"},
					{"name": "refine_existing_description", "description": "Broaden the description rubric of an existing choice"},
					{"name": "out_of_scope_filter", "description": "Filter as out-of-scope noise before classification"},
				},
			},
		},
	}

	schemaBytes, err := json.Marshal(probeSchema)
	if err != nil {
		return client.ProposedOption{}, "", false
	}

	var probeResp *client.StructuredDecisionResponse
	if c != nil {
		probeResp, _, err = c.Decide(ctx, string(schemaBytes), stateContent)
	} else if targetURL != "" {
		probeResp, _, _, err = executeDecideWithWarmup(ctx, string(schemaBytes), stateContent, nil, targetURL)
	}
	if err != nil || probeResp == nil || probeResp.Diagnostics.Thought == nil {
		return client.ProposedOption{}, "", false
	}

	thoughtText := strings.TrimSpace(probeResp.Diagnostics.Thought.Text)
	opt, ok := ParseSuggestedOptionFromThought(thoughtText)
	return opt, thoughtText, ok
}

func proposeOptionViaGeminiStage2(
	ctx context.Context,
	qID string,
	existing []client.ProposedOption,
	topCands map[string]float64,
	stateContent string,
) (client.ProposedOption, bool) {
	genaiClient, _, err := getSharedGenaiClient(ctx)
	if err != nil || genaiClient == nil {
		return client.ProposedOption{}, false
	}

	var existingLines []string
	for _, o := range existing {
		existingLines = append(existingLines, fmt.Sprintf("- %s: %s", o.Name, o.Description))
	}

	prompt := fmt.Sprintf(
		"You are designing a zero-shot decision taxonomy for DiffusionGemma (dgem).\n"+
			"Question slot %q encountered an unclassified or high-entropy input.\n\n"+
			"Existing options:\n%s\n\n"+
			"Input state:\n%s\n\n"+
			"Propose one concise, reusable snake_case option 'name' and a one-line zero-shot 'description' rubric that cleanly captures this category.",
		qID,
		strings.Join(existingLines, "\n"),
		stateContent,
	)

	respSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"name":        {Type: genai.TypeString, Description: "Lowercase snake_case option identifier (e.g., legal_and_compliance)"},
			"description": {Type: genai.TypeString, Description: "Concise 1-line rubric describing items belonging to this category"},
		},
		Required: []string{"name", "description"},
	}

	res, err := genaiClient.Models.GenerateContent(ctx, SanitizeCascadeModel(""), genai.Text(prompt), &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema:   respSchema,
	})
	if err != nil || res == nil {
		return client.ProposedOption{}, false
	}

	var opt client.ProposedOption
	if err := json.Unmarshal([]byte(strings.TrimSpace(res.Text())), &opt); err == nil && opt.Name != "" {
		opt.Name = sanitizeSnakeCase(opt.Name)
		return opt, true
	}
	return client.ProposedOption{}, false
}
