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

package decisionindex

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

const (
	TaxonomyModeMTPNaive         = "mtp_naive"
	TaxonomyModeMTPTrie          = "mtp_trie"
	TaxonomyModeBeam2Pass        = "beam_2pass"
	TaxonomyModeHesitationHybrid = "hesitation_hybrid"
)

// validTaxonomyMode reports whether mode is empty (disabled) or a supported TaxonomyMode constant.
func validTaxonomyMode(mode string) bool {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", TaxonomyModeMTPNaive, TaxonomyModeMTPTrie, TaxonomyModeBeam2Pass, TaxonomyModeHesitationHybrid:
		return true
	default:
		return false
	}
}

// TaxonomyLeaf maps a leaf option key to a valid 2-level (CoarseCode, FineCode) path in a prefix trie.
type TaxonomyLeaf struct {
	Key         string `json:"key"`
	Description string `json:"description,omitempty"`
	CoarseCode  string `json:"coarse_code"`
	FineCode    string `json:"fine_code"`
	IsCatchAll  bool   `json:"is_catch_all,omitempty"`
}

// TaxonomySpec configures opt-in hierarchical taxonomy readout on a wide SystemOneQuestion.
// When SystemOneQuestion.Taxonomy is nil and EngineOptions.TaxonomyMode is empty, ExecuteSystemOne uses its
// unchanged default routing.
type TaxonomySpec struct {
	Mode                string            `json:"mode,omitempty"`
	PrefixDelimiter     string            `json:"prefix_delimiter,omitempty"`
	CoarsePrompt        string            `json:"coarse_prompt,omitempty"`
	FinePrompt          string            `json:"fine_prompt,omitempty"`
	CoarseDescriptions  map[string]string `json:"coarse_descriptions,omitempty"`
	FineDescriptions    map[string]string `json:"fine_descriptions,omitempty"`
	Leaves              []TaxonomyLeaf    `json:"leaves,omitempty"`
	TopB                int               `json:"top_b,omitempty"`
	HesitationThreshold float64           `json:"hesitation_threshold,omitempty"`
	MarginThreshold     float64           `json:"margin_threshold,omitempty"`
}

var opaqueKeyPrefixes = map[string]bool{
	"option": true, "opt": true, "choice": true, "item": true,
	"label": true, "candidate": true, "class": true, "id": true,
}

var genericTaxonomyWords = map[string]bool{
	"option": true, "opt": true, "choice": true, "item": true,
	"label": true, "candidate": true, "class": true, "intent": true,
	"request": true, "user": true, "ask": true, "tell": true,
	"want": true, "need": true, "get": true, "find": true, "make": true,
}

func isDigitsOnly(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func isOpaqueNumberedKey(key, delimiter string) bool {
	idx := strings.Index(key, delimiter)
	if idx <= 0 {
		return false
	}
	prefix := strings.ToLower(key[:idx])
	rest := key[idx+len(delimiter):]
	return opaqueKeyPrefixes[prefix] && isDigitsOnly(rest)
}

func detectTaxonomyCatchAlls(criteria map[string]string) map[string]bool {
	catchSet := make(map[string]bool)
	for _, k := range catchAllKeys(criteria) {
		catchSet[k] = true
	}
	var explicit []string
	for _, k := range sortedOptionKeys(criteria) {
		low := strings.ToLower(strings.TrimSpace(k))
		if low == "oos" || low == "out_of_scope" || low == "other_unclassified" || low == "none_visible" {
			explicit = append(explicit, k)
		}
	}
	if len(catchSet)+len(explicit) <= 3 {
		for _, k := range explicit {
			catchSet[k] = true
		}
	}
	return catchSet
}

func contentStemsForOption(key, desc, delimiter string) []string {
	text := desc
	if !isOpaqueNumberedKey(key, delimiter) {
		text = strings.ReplaceAll(key, delimiter, " ") + " " + desc
	}
	raw := splitGenericSubwords(text)
	var out []string
	seen := make(map[string]bool)
	for _, w := range raw {
		if len(w) < 3 || isDigitsOnly(w) || shortlistStopWords[w] || genericTaxonomyWords[w] {
			continue
		}
		st := simpleStem(w)
		if len(st) < 3 || shortlistStopWords[st] || genericTaxonomyWords[st] || seen[st] {
			continue
		}
		seen[st] = true
		out = append(out, st)
	}
	return out
}

// buildLexicalCoarseGroups partitions non-catchall option keys into <= maxCoarse semantic/lexical coarse buckets
// with <= maxPerLevel leaves per bucket when keys are opaque ("option_0 .. option_150") or prefix counts exceed maxCoarse.
func buildLexicalCoarseGroups(
	nonCatchKeys []string,
	criteria map[string]string,
	delimiter string,
	maxCoarse int,
	maxPerLevel int,
) (map[string][]string, []string, error) {
	if maxCoarse <= 0 || len(nonCatchKeys) > maxCoarse*maxPerLevel {
		return nil, nil, fmt.Errorf("option count (%d) exceeds 2-level taxonomy capacity (%d x %d)", len(nonCatchKeys), maxCoarse, maxPerLevel)
	}

	keyStems := make(map[string][]string, len(nonCatchKeys))
	stemMembers := make(map[string][]string)
	for _, k := range nonCatchKeys {
		stems := contentStemsForOption(k, criteria[k], delimiter)
		keyStems[k] = stems
		for _, st := range stems {
			stemMembers[st] = append(stemMembers[st], k)
		}
	}

	type stemCand struct {
		stem  string
		count int
	}
	var cands []stemCand
	for st, members := range stemMembers {
		if len(members) >= 2 && len(members) <= maxPerLevel {
			cands = append(cands, stemCand{stem: st, count: len(members)})
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].count != cands[j].count {
			return cands[i].count > cands[j].count
		}
		return cands[i].stem < cands[j].stem
	})

	assigned := make(map[string]bool, len(nonCatchKeys))
	grouped := make(map[string][]string)
	var coarseOrder []string
	remainingCount := len(nonCatchKeys)

	ceilDiv := func(a, b int) int {
		if a <= 0 {
			return 0
		}
		return (a + b - 1) / b
	}

	for _, cand := range cands {
		var avail []string
		for _, k := range stemMembers[cand.stem] {
			if !assigned[k] {
				avail = append(avail, k)
			}
		}
		if len(avail) < 2 {
			continue
		}
		if len(avail) > maxPerLevel {
			avail = avail[:maxPerLevel]
		}
		afterRem := remainingCount - len(avail)
		tailNeeded := ceilDiv(afterRem, maxPerLevel)
		if len(coarseOrder)+1+tailNeeded > maxCoarse {
			continue
		}
		code := "topic_" + cand.stem
		if _, clash := grouped[code]; clash {
			continue
		}
		grouped[code] = avail
		coarseOrder = append(coarseOrder, code)
		for _, k := range avail {
			assigned[k] = true
		}
		remainingCount = afterRem
	}

	if remainingCount > 0 {
		var leftovers []string
		for _, k := range nonCatchKeys {
			if !assigned[k] {
				leftovers = append(leftovers, k)
			}
		}
		// Sort leftovers by primary stem then key so related tail intents share a bucket.
		sort.Slice(leftovers, func(i, j int) bool {
			si, sj := "", ""
			if len(keyStems[leftovers[i]]) > 0 {
				si = keyStems[leftovers[i]][0]
			}
			if len(keyStems[leftovers[j]]) > 0 {
				sj = keyStems[leftovers[j]][0]
			}
			if si != sj {
				return si < sj
			}
			return leftovers[i] < leftovers[j]
		})
		numBuckets := ceilDiv(len(leftovers), maxPerLevel)
		if len(coarseOrder)+numBuckets > maxCoarse {
			return nil, nil, fmt.Errorf("lexical bucketing exceeded coarse limit (%d + %d > %d)", len(coarseOrder), numBuckets, maxCoarse)
		}
		chunkSize := ceilDiv(len(leftovers), numBuckets)
		if chunkSize > maxPerLevel {
			chunkSize = maxPerLevel
		}
		for b := 0; b < numBuckets; b++ {
			start := b * chunkSize
			if start >= len(leftovers) {
				break
			}
			end := start + chunkSize
			if end > len(leftovers) {
				end = len(leftovers)
			}
			code := fmt.Sprintf("group_%02d", b+1)
			grouped[code] = leftovers[start:end]
			coarseOrder = append(coarseOrder, code)
		}
	}

	sort.Strings(coarseOrder)
	return grouped, coarseOrder, nil
}

func extractActionRoleTokens(key, coarse, desc, delimiter string) []string {
	var out []string
	seen := make(map[string]bool)
	add := func(t string) {
		if len(t) >= 2 && len(t) <= 18 && !isDigitsOnly(t) && !shortlistStopWords[t] && !genericTaxonomyWords[t] && !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	coarseLow := strings.ToLower(strings.TrimPrefix(coarse, "topic_"))
	coarseStem := simpleStem(coarseLow)

	if !isOpaqueNumberedKey(key, delimiter) {
		if idx := strings.Index(key, delimiter); idx > 0 {
			suf := strings.ToLower(strings.TrimSpace(key[idx+len(delimiter):]))
			if suf != "" && !isDigitsOnly(suf) {
				parts := splitGenericSubwords(suf)
				if len(parts) == 1 && parts[0] != coarseLow {
					add(parts[0])
				} else if len(parts) >= 2 {
					last := parts[len(parts)-1]
					if last != coarseLow {
						add(last)
					}
				}
			}
		}
	}
	for _, w := range splitGenericSubwords(desc) {
		if len(w) < 3 || isDigitsOnly(w) || shortlistStopWords[w] || genericTaxonomyWords[w] {
			continue
		}
		st := simpleStem(w)
		if st != coarseStem && w != coarseLow {
			add(st)
		}
	}
	return out
}

func shortLeafSummary(key, desc, delimiter string) string {
	d := strings.TrimSpace(desc)
	if isOpaqueNumberedKey(key, delimiter) && d != "" {
		if len(d) > 36 {
			return d[:36] + "..."
		}
		return d
	}
	if d != "" && d != strings.ReplaceAll(key, "_", " ") && len(d) <= 32 {
		return key + " (" + d + ")"
	}
	return key
}

// BuildPrefixTaxonomyTrie constructs a 2-level (CoarseCode, FineCode) trie from criteria.
// When option keys share clean domain prefixes (e.g. "alarm_query", "calendar_query") within maxPerLevel,
// it groups by prefix and aligns recurring cross-domain action suffixes into shared semantic fine codes
// ("role_query", "role_set", ...) before assigning residual positional slots ("slot_00", ...).
// When keys are opaque numbered identifiers ("option_0 .. option_150") or prefix counts exceed maxPerLevel,
// it deterministically clusters option descriptions into <= maxPerLevel lexical coarse buckets.
func BuildPrefixTaxonomyTrie(
	criteria map[string]string,
	delimiter string,
	maxPerLevel int,
) ([]TaxonomyLeaf, map[string]string, map[string]string, error) {
	if delimiter == "" {
		delimiter = "_"
	}
	if maxPerLevel <= 0 {
		maxPerLevel = MaxOptionsPerSlot
	}
	keys := sortedOptionKeys(criteria)
	if len(keys) == 0 {
		return nil, nil, nil, fmt.Errorf("empty criteria")
	}
	catchSet := detectTaxonomyCatchAlls(criteria)

	var nonCatchKeys, catchKeys []string
	opaqueCount := 0
	for _, k := range keys {
		if catchSet[k] {
			catchKeys = append(catchKeys, k)
		} else {
			nonCatchKeys = append(nonCatchKeys, k)
			if isOpaqueNumberedKey(k, delimiter) {
				opaqueCount++
			}
		}
	}
	if len(catchKeys) > maxPerLevel {
		return nil, nil, nil, fmt.Errorf("catch-all count (%d) exceeds per-slot limit (%d)", len(catchKeys), maxPerLevel)
	}

	maxNonCatchCoarse := maxPerLevel
	if len(catchKeys) > 0 {
		maxNonCatchCoarse = maxPerLevel - 1
	}

	grouped := make(map[string][]string)
	var coarseOrder []string
	prefixOK := len(nonCatchKeys) > 0 && opaqueCount*2 < len(nonCatchKeys)
	if prefixOK {
		seen := make(map[string]bool)
		for _, k := range nonCatchKeys {
			coarse := "general"
			if idx := strings.Index(k, delimiter); idx > 0 {
				coarse = k[:idx]
			}
			if !seen[coarse] {
				seen[coarse] = true
				coarseOrder = append(coarseOrder, coarse)
			}
			grouped[coarse] = append(grouped[coarse], k)
		}
		sort.Strings(coarseOrder)
		if len(coarseOrder) < 2 || len(coarseOrder) > maxNonCatchCoarse {
			prefixOK = false
		} else {
			for _, c := range coarseOrder {
				if len(grouped[c]) > maxPerLevel {
					prefixOK = false
					break
				}
			}
		}
	}

	if !prefixOK {
		var err error
		grouped, coarseOrder, err = buildLexicalCoarseGroups(nonCatchKeys, criteria, delimiter, maxNonCatchCoarse, maxPerLevel)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	if len(catchKeys) > 0 {
		grouped["catchall"] = catchKeys
		coarseOrder = append(coarseOrder, "catchall")
		sort.Strings(coarseOrder)
	}

	// Align shared cross-domain action suffixes into semantic "role_<tok>" fine codes when capacity permits.
	// A token is eligible inside coarse domain c only when it uniquely identifies a single leaf in c (freq == 1),
	// preventing shared domain boilerplate words from shifting positional indices.
	roleClaims := make(map[string]map[string]string) // roleTok -> coarse -> leafKey
	for _, c := range coarseOrder {
		if c == "catchall" {
			continue
		}
		tokFreq := make(map[string]int)
		leafToks := make(map[string][]string, len(grouped[c]))
		for _, leafKey := range grouped[c] {
			toks := extractActionRoleTokens(leafKey, c, criteria[leafKey], delimiter)
			leafToks[leafKey] = toks
			for _, t := range toks {
				tokFreq[t]++
			}
		}
		for _, leafKey := range grouped[c] {
			for _, tok := range leafToks[leafKey] {
				if tokFreq[tok] == 1 {
					if roleClaims[tok] == nil {
						roleClaims[tok] = make(map[string]string)
					}
					roleClaims[tok][c] = leafKey
					break
				}
			}
		}
	}

	type roleCand struct {
		tok     string
		domains int
	}
	var roleCands []roleCand
	for tok, domMap := range roleClaims {
		if len(domMap) >= 2 {
			roleCands = append(roleCands, roleCand{tok: tok, domains: len(domMap)})
		}
	}
	sort.Slice(roleCands, func(i, j int) bool {
		if roleCands[i].domains != roleCands[j].domains {
			return roleCands[i].domains > roleCands[j].domains
		}
		return roleCands[i].tok < roleCands[j].tok
	})

	leafRole := make(map[string]string)
	unassignedPerDomain := make(map[string]int, len(coarseOrder))
	for _, c := range coarseOrder {
		unassignedPerDomain[c] = len(grouped[c])
	}
	var acceptedRoles []string

	for _, rc := range roleCands {
		domMap := roleClaims[rc.tok]
		eligible := make(map[string]string)
		for c, k := range domMap {
			if _, already := leafRole[k]; !already {
				eligible[c] = k
			}
		}
		if len(eligible) < 2 {
			continue
		}
		maxAfter := 0
		for _, c := range coarseOrder {
			u := unassignedPerDomain[c]
			if _, hit := eligible[c]; hit {
				u--
			}
			if u > maxAfter {
				maxAfter = u
			}
		}
		if len(acceptedRoles)+1+maxAfter <= maxPerLevel {
			roleCode := "role_" + rc.tok
			acceptedRoles = append(acceptedRoles, roleCode)
			for c, k := range eligible {
				leafRole[k] = roleCode
				unassignedPerDomain[c]--
			}
		}
	}
	sort.Strings(acceptedRoles)

	leaves := make([]TaxonomyLeaf, 0, len(keys))
	coarseDesc := make(map[string]string, len(coarseOrder))
	fineExamples := make(map[string][]string)
	maxSlotIdx := 0

	for _, c := range coarseOrder {
		members := grouped[c]
		examples := make([]string, 0, 5)
		slotIdx := 0
		for _, leafKey := range members {
			fineCode, hasRole := leafRole[leafKey]
			if !hasRole {
				fineCode = fmt.Sprintf("slot_%02d", slotIdx)
				slotIdx++
				if slotIdx > maxSlotIdx {
					maxSlotIdx = slotIdx
				}
			}
			desc := criteria[leafKey]
			leaves = append(leaves, TaxonomyLeaf{
				Key:         leafKey,
				Description: desc,
				CoarseCode:  c,
				FineCode:    fineCode,
				IsCatchAll:  catchSet[leafKey],
			})
			if len(examples) < 5 {
				examples = append(examples, shortLeafSummary(leafKey, desc, delimiter))
			}
			if len(fineExamples[fineCode]) < 5 {
				fineExamples[fineCode] = append(fineExamples[fineCode], shortLeafSummary(leafKey, desc, delimiter))
			}
		}
		switch {
		case c == "catchall":
			coarseDesc[c] = fmt.Sprintf("Out-of-scope or none of the listed options (%s)", strings.Join(examples, " | "))
		case strings.HasPrefix(c, "topic_"):
			coarseDesc[c] = fmt.Sprintf("Topic %s: %s", strings.TrimPrefix(c, "topic_"), strings.Join(examples, " | "))
		case strings.HasPrefix(c, "group_"):
			coarseDesc[c] = fmt.Sprintf("Option group %s: %s", strings.TrimPrefix(c, "group_"), strings.Join(examples, " | "))
		default:
			coarseDesc[c] = fmt.Sprintf("Domain %s: %s", c, strings.Join(examples, " | "))
		}
	}

	fineDesc := make(map[string]string, len(acceptedRoles)+maxSlotIdx)
	for _, roleCode := range acceptedRoles {
		roleName := strings.TrimPrefix(roleCode, "role_")
		fineDesc[roleCode] = fmt.Sprintf("Action role %s (%s)", roleName, strings.Join(fineExamples[roleCode], " | "))
	}
	for idx := 0; idx < maxSlotIdx; idx++ {
		fineCode := fmt.Sprintf("slot_%02d", idx)
		fineDesc[fineCode] = fmt.Sprintf("Sub-option index %d (%s)", idx, strings.Join(fineExamples[fineCode], " | "))
	}

	return leaves, coarseDesc, fineDesc, nil
}

// DecodeTaxonomyTrie computes exact joint probabilities P(e | x) proportional to P_coarse(c1(e)) * P_fine(c2(e))
// strictly over valid root-to-leaf paths in leaves, eliminating invalid cross-domain slot combinations.
func DecodeTaxonomyTrie(
	coarseProbs map[string]float64,
	fineProbs map[string]float64,
	leaves []TaxonomyLeaf,
) (map[string]float64, string) {
	if len(leaves) == 0 {
		return map[string]float64{}, ""
	}
	const eps = 1e-12
	logScores := make(map[string]float64, len(leaves))
	maxLog := math.Inf(-1)
	keys := make([]string, 0, len(leaves))

	for _, leaf := range leaves {
		keys = append(keys, leaf.Key)
		p1 := math.Max(eps, coarseProbs[leaf.CoarseCode])
		p2 := math.Max(eps, fineProbs[leaf.FineCode])
		ls := math.Log(p1) + math.Log(p2)
		logScores[leaf.Key] = ls
		if ls > maxLog {
			maxLog = ls
		}
	}
	sort.Strings(keys)

	rawProbs := make(map[string]float64, len(leaves))
	sum := 0.0
	for _, k := range keys {
		w := math.Exp(logScores[k] - maxLog)
		rawProbs[k] = w
		sum += w
	}
	winner := keys[0]
	bestP := -1.0
	for _, k := range keys {
		if sum > 0 {
			rawProbs[k] /= sum
		} else {
			rawProbs[k] = 1.0 / float64(len(keys))
		}
		if rawProbs[k] > bestP {
			bestP = rawProbs[k]
			winner = k
		}
	}
	return rawProbs, winner
}

func topCoarseDomains(coarseProbs map[string]float64, b int) ([]string, float64) {
	type pair struct {
		code string
		prob float64
	}
	items := make([]pair, 0, len(coarseProbs))
	for k, v := range coarseProbs {
		items = append(items, pair{code: k, prob: v})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].prob != items[j].prob {
			return items[i].prob > items[j].prob
		}
		return items[i].code < items[j].code
	})
	if len(items) == 0 {
		return nil, 1.0
	}
	margin := items[0].prob
	if len(items) > 1 {
		margin = items[0].prob - items[1].prob
	}
	const maxTaxonomySlotOptions = 52
	if b <= 0 {
		b = 2
	}
	if b > maxTaxonomySlotOptions {
		b = maxTaxonomySlotOptions
	}
	if b > len(items) {
		b = len(items)
	}
	out := make([]string, 0, 8)
	for i := 0; i < b; i++ {
		out = append(out, items[i].code)
	}
	return out, margin
}

func evaluateHierarchicalQuestion(
	ctx context.Context,
	cli *client.Client,
	stateText string,
	images []string,
	qKey string,
	qSpec SystemOneQuestion,
	opts EngineOptions,
) (SystemOneAnswer, int, error) {
	spec := qSpec.Taxonomy
	explicitSpec := spec != nil
	if spec == nil && strings.TrimSpace(opts.TaxonomyMode) != "" {
		spec = &TaxonomySpec{
			Mode:            opts.TaxonomyMode,
			PrefixDelimiter: opts.TaxonomyDelimiter,
		}
	}
	if spec == nil {
		return evaluateWideQuestionTournament(ctx, cli, stateText, images, qKey, qSpec, opts)
	}

	limit := optionLimit(opts.Labels)
	if opts.MaxOptionsPerSlot > 0 && opts.MaxOptionsPerSlot < limit {
		limit = opts.MaxOptionsPerSlot
	}

	leaves := spec.Leaves
	coarseDesc := spec.CoarseDescriptions
	fineDesc := spec.FineDescriptions
	delim := spec.PrefixDelimiter
	if delim == "" {
		delim = opts.TaxonomyDelimiter
	}
	if len(leaves) == 0 {
		var err error
		leaves, coarseDesc, fineDesc, err = BuildPrefixTaxonomyTrie(qSpec.Criteria, delim, limit)
		if err != nil {
			if explicitSpec {
				return SystemOneAnswer{}, 0, fmt.Errorf("%w: question %q: %v", ErrInvalidRequest, qKey, err)
			}
			// Server-wide --taxonomy-mode falls back to bracket tournament when options exceed 2-level capacity,
			// recording the fallback explicitly on the answer metadata.
			ans, p, tErr := evaluateWideQuestionTournament(ctx, cli, stateText, images, qKey, qSpec, opts)
			if tErr == nil {
				ans.TaxonomyModeExecuted = "tournament_fallback"
				ans.TaxonomyFallbackReason = err.Error()
			}
			return ans, p, tErr
		}
	} else {
		if len(leaves) < 2 {
			return SystemOneAnswer{}, 0, fmt.Errorf("%w: question %q: taxonomy leaves must contain at least 2 entries", ErrInvalidRequest, qKey)
		}
		seenLeafKeys := make(map[string]bool, len(leaves))
		for _, l := range leaves {
			if _, ok := qSpec.Criteria[l.Key]; !ok {
				return SystemOneAnswer{}, 0, fmt.Errorf("%w: question %q: taxonomy leaf %q is not present in criteria", ErrInvalidRequest, qKey, l.Key)
			}
			if strings.TrimSpace(l.CoarseCode) == "" || strings.TrimSpace(l.FineCode) == "" {
				return SystemOneAnswer{}, 0, fmt.Errorf("%w: question %q: taxonomy leaf %q requires non-empty coarse_code and fine_code", ErrInvalidRequest, qKey, l.Key)
			}
			if seenLeafKeys[l.Key] {
				return SystemOneAnswer{}, 0, fmt.Errorf("%w: question %q: duplicate taxonomy leaf %q", ErrInvalidRequest, qKey, l.Key)
			}
			seenLeafKeys[l.Key] = true
		}
		if len(coarseDesc) == 0 {
			coarseDesc = make(map[string]string)
			for _, l := range leaves {
				if _, ok := coarseDesc[l.CoarseCode]; !ok {
					coarseDesc[l.CoarseCode] = "Domain " + l.CoarseCode
				}
			}
		}
		if len(fineDesc) == 0 {
			fineDesc = make(map[string]string)
			for _, l := range leaves {
				if _, ok := fineDesc[l.FineCode]; !ok {
					fineDesc[l.FineCode] = "Sub-role " + l.FineCode
				}
			}
		}
		if len(coarseDesc) > limit || len(fineDesc) > limit {
			if explicitSpec {
				return SystemOneAnswer{}, 0, fmt.Errorf(
					"%w: question %q: taxonomy coarse (%d) and fine (%d) option counts must not exceed %d",
					ErrInvalidRequest, qKey, len(coarseDesc), len(fineDesc), limit,
				)
			}
			return evaluateWideQuestionTournament(ctx, cli, stateText, images, qKey, qSpec, opts)
		}
	}

	mode := strings.ToLower(strings.TrimSpace(spec.Mode))
	if mode == "" {
		mode = strings.ToLower(strings.TrimSpace(opts.TaxonomyMode))
	}
	if mode == "" {
		mode = TaxonomyModeBeam2Pass
	}
	switch mode {
	case TaxonomyModeMTPNaive, TaxonomyModeMTPTrie, TaxonomyModeBeam2Pass, TaxonomyModeHesitationHybrid:
	default:
		return SystemOneAnswer{}, 0, fmt.Errorf(
			"%w: question %q: unknown taxonomy mode %q (want %s, %s, %s, or %s)",
			ErrInvalidRequest, qKey, mode,
			TaxonomyModeMTPNaive, TaxonomyModeMTPTrie, TaxonomyModeBeam2Pass, TaxonomyModeHesitationHybrid,
		)
	}
	topB := spec.TopB
	if topB <= 0 {
		topB = 2
	}
	hesThresh := spec.HesitationThreshold
	if hesThresh <= 0 {
		hesThresh = opts.HesitationGate
		if hesThresh <= 0 {
			hesThresh = 0.16
		}
	}
	marginThresh := spec.MarginThreshold
	if marginThresh <= 0 {
		marginThresh = 0.25
	}

	baseInstr := FormatInstructions(qSpec.Instructions)
	coarsePrompt := strings.TrimSpace(spec.CoarsePrompt)
	if coarsePrompt == "" {
		coarsePrompt = baseInstr + " (Select coarse category)"
	}
	finePrompt := strings.TrimSpace(spec.FinePrompt)
	if finePrompt == "" {
		finePrompt = baseInstr + " (Select fine action or sub-option)"
	}

	coarseQID := qKey + "__coarse"
	fineQID := qKey + "__fine"
	coarseQ := SystemOneQuestion{
		Type:         "choice",
		Instructions: coarsePrompt,
		Criteria:     coarseDesc,
	}

	totalPasses := 0
	var coarseAns, fineAns SystemOneAnswer
	hasFineRead := false

	if mode == TaxonomyModeBeam2Pass {
		batchAns, passes, err := evaluateStandardBatch(
			ctx, cli, stateText, images,
			[]string{coarseQID},
			map[string]SystemOneQuestion{coarseQID: coarseQ},
			map[string]string{coarseQID: kindChoice},
			opts,
		)
		if err != nil {
			return SystemOneAnswer{}, passes, err
		}
		totalPasses += passes
		coarseAns = batchAns[coarseQID]
	} else {
		fineQ := SystemOneQuestion{
			Type:         "choice",
			Instructions: finePrompt,
			Criteria:     fineDesc,
		}
		batchAns, passes, err := evaluateStandardBatch(
			ctx, cli, stateText, images,
			[]string{coarseQID, fineQID},
			map[string]SystemOneQuestion{coarseQID: coarseQ, fineQID: fineQ},
			map[string]string{coarseQID: kindChoice, fineQID: kindChoice},
			opts,
		)
		if err != nil {
			return SystemOneAnswer{}, passes, err
		}
		totalPasses += passes
		coarseAns = batchAns[coarseQID]
		fineAns = batchAns[fineQID]
		hasFineRead = true
	}

	allOptKeys := optionKeys(qSpec, opts.OptionOrder)

	if mode == TaxonomyModeMTPNaive || mode == TaxonomyModeMTPTrie {
		trieProbs, _ := DecodeTaxonomyTrie(coarseAns.Probabilities, fineAns.Probabilities, leaves)
		normProbs, normWinner, h, normH := NormalizeAndScaleProbabilities(trieProbs, allOptKeys, 1.0)
		winner := normWinner
		if mode == TaxonomyModeMTPNaive {
			// Unconstrained ablation: combine independent argmaxes (coarseAns.Choice, fineAns.Choice);
			// when that cell does not exist in the ragged taxonomy, fall back to the first leaf in coarseAns.Choice.
			matched := ""
			firstInCoarse := ""
			for _, l := range leaves {
				if l.CoarseCode == coarseAns.Choice {
					if firstInCoarse == "" {
						firstInCoarse = l.Key
					}
					if l.FineCode == fineAns.Choice {
						matched = l.Key
						break
					}
				}
			}
			switch {
			case matched != "":
				winner = matched
			case firstInCoarse != "":
				winner = firstInCoarse
			default:
				winner = allOptKeys[0]
			}
		}
		conf := normProbs[winner]
		return SystemOneAnswer{
			Type:                 "choice",
			Choice:               winner,
			Probabilities:        normProbs,
			Confidence:           &conf,
			Entropy:              h,
			NormalizedH:          normH,
			RawEntropy:           coarseAns.RawEntropy,
			RawNormalizedH:       coarseAns.RawNormalizedH,
			PassesUsed:           totalPasses,
			TaxonomyModeExecuted: mode,
		}, totalPasses, nil
	}

	var trieProbs map[string]float64
	if hasFineRead {
		trieProbs, _ = DecodeTaxonomyTrie(coarseAns.Probabilities, fineAns.Probabilities, leaves)
	}

	topDomains, topMargin := topCoarseDomains(coarseAns.Probabilities, topB)
	if mode == TaxonomyModeHesitationHybrid {
		if coarseAns.RawNormalizedH < hesThresh && topMargin >= marginThresh {
			normProbs, winner, h, normH := NormalizeAndScaleProbabilities(trieProbs, allOptKeys, 1.0)
			conf := normProbs[winner]
			return SystemOneAnswer{
				Type:                 "choice",
				Choice:               winner,
				Probabilities:        normProbs,
				Confidence:           &conf,
				Entropy:              h,
				NormalizedH:          normH,
				RawEntropy:           coarseAns.RawEntropy,
				RawNormalizedH:       coarseAns.RawNormalizedH,
				PassesUsed:           totalPasses,
				TaxonomyModeExecuted: mode,
			}, totalPasses, nil
		}
	}

	// Pass 2: shortlist leaves across top-b coarse domains (+ any catch-all leaves) with fair per-domain
	// quota allocation so a full 25-leaf rank-1 domain never starves rank-2..b domains.
	inTopDomain := make(map[string]bool, len(topDomains))
	for _, d := range topDomains {
		inTopDomain[d] = true
	}
	byDomain := make(map[string][]TaxonomyLeaf, len(topDomains))
	var catchLeaves []TaxonomyLeaf
	totalTopLeaves := 0
	for _, l := range leaves {
		if l.IsCatchAll {
			catchLeaves = append(catchLeaves, l)
			continue
		}
		if inTopDomain[l.CoarseCode] {
			byDomain[l.CoarseCode] = append(byDomain[l.CoarseCode], l)
			totalTopLeaves++
		}
	}
	var activeTopDomains []string
	for _, d := range topDomains {
		if len(byDomain[d]) > 0 {
			activeTopDomains = append(activeTopDomains, d)
			domList := byDomain[d]
			sort.Slice(domList, func(i, j int) bool {
				var si, sj float64
				if hasFineRead {
					si, sj = trieProbs[domList[i].Key], trieProbs[domList[j].Key]
				} else if len(qSpec.PriorLogits) > 0 {
					si, sj = qSpec.PriorLogits[domList[i].Key], qSpec.PriorLogits[domList[j].Key]
				}
				if si != sj {
					return si > sj
				}
				return domList[i].Key < domList[j].Key
			})
			byDomain[d] = domList
		}
	}

	budget := limit - len(catchLeaves)
	if budget < 1 {
		budget = limit
	}
	if budget > totalTopLeaves {
		budget = totalTopLeaves
	}

	selectedDomainLeaves := make([]TaxonomyLeaf, 0, budget)
	if totalTopLeaves <= budget {
		for _, d := range activeTopDomains {
			selectedDomainLeaves = append(selectedDomainLeaves, byDomain[d]...)
		}
	} else if len(activeTopDomains) > 0 {
		// Allocate probability-proportional quotas with a guaranteed floor per active domain in top-b.
		probMass := 0.0
		for _, d := range activeTopDomains {
			probMass += math.Max(1e-6, coarseAns.Probabilities[d])
		}
		minFloor := budget / (2 * len(activeTopDomains))
		if minFloor < 2 {
			minFloor = 2
		}
		if minFloor*len(activeTopDomains) > budget {
			minFloor = maxInt(1, budget/len(activeTopDomains))
		}
		taken := make(map[string]int, len(activeTopDomains))
		remaining := budget
		for _, d := range activeTopDomains {
			share := int(math.Round(float64(budget) * (math.Max(1e-6, coarseAns.Probabilities[d]) / probMass)))
			if share < minFloor {
				share = minFloor
			}
			if share > len(byDomain[d]) {
				share = len(byDomain[d])
			}
			if share > remaining {
				share = remaining
			}
			taken[d] = share
			remaining -= share
		}
		// Spill over any unused quota in rank order across activeTopDomains.
		for remaining > 0 {
			progressed := false
			for _, d := range activeTopDomains {
				if remaining == 0 {
					break
				}
				if taken[d] < len(byDomain[d]) {
					taken[d]++
					remaining--
					progressed = true
				}
			}
			if !progressed {
				break
			}
		}
		for _, d := range activeTopDomains {
			selectedDomainLeaves = append(selectedDomainLeaves, byDomain[d][:taken[d]]...)
		}
	}

	const maxTaxonomyPass2Cap = 52
	shortCap := limit
	if shortCap < 2 || shortCap > maxTaxonomyPass2Cap {
		shortCap = maxTaxonomyPass2Cap
	}
	shortKeys := make([]string, 0, shortCap)
	coarsePriorLogits := make(map[string]float64, shortCap)
	for _, l := range selectedDomainLeaves {
		shortKeys = append(shortKeys, l.Key)
		p := coarseAns.Probabilities[l.CoarseCode]
		if hasFineRead {
			p = trieProbs[l.Key]
		}
		coarsePriorLogits[l.Key] = math.Log(math.Max(1e-9, p))
	}
	for _, l := range catchLeaves {
		if len(shortKeys) >= limit {
			break
		}
		shortKeys = append(shortKeys, l.Key)
		p := coarseAns.Probabilities[l.CoarseCode]
		if hasFineRead {
			p = trieProbs[l.Key]
		}
		coarsePriorLogits[l.Key] = math.Log(math.Max(1e-9, p))
	}
	if len(shortKeys) < 2 {
		return evaluateWideQuestionTournament(ctx, cli, stateText, images, qKey, qSpec, opts)
	}

	subQ := subQuestion(qSpec, shortKeys)
	if len(subQ.PriorLogits) == 0 {
		subQ.PriorLogits = coarsePriorLogits
	}
	pass2Opts := opts
	if pass2Opts.PriorWeight <= 0 {
		pass2Opts.PriorWeight = 0.25
		pass2Opts.PriorGateMode = "hesitant"
	}

	pass2Batch, passes2, err := evaluateStandardBatch(
		ctx, cli, stateText, images,
		[]string{qKey},
		map[string]SystemOneQuestion{qKey: subQ},
		map[string]string{qKey: kindChoice},
		pass2Opts,
	)
	if err != nil {
		return SystemOneAnswer{}, totalPasses + passes2, err
	}
	totalPasses += passes2
	shortAns := pass2Batch[qKey]

	inShort := make(map[string]bool, len(shortKeys))
	for _, k := range shortKeys {
		inShort[k] = true
	}
	const tailMass = 1e-3
	fullProbs := make(map[string]float64, len(allOptKeys))
	for _, k := range shortKeys {
		fullProbs[k] = (1.0 - tailMass) * math.Max(1e-9, shortAns.Probabilities[k])
	}
	numRest := len(allOptKeys) - len(shortKeys)
	if numRest > 0 {
		perRest := tailMass / float64(numRest)
		for _, k := range allOptKeys {
			if !inShort[k] {
				fullProbs[k] = perRest
			}
		}
	}
	normProbs, winner, h, normH := NormalizeAndScaleProbabilities(fullProbs, allOptKeys, 1.0)
	conf := normProbs[winner]
	return SystemOneAnswer{
		Type:                 "choice",
		Choice:               winner,
		Probabilities:        normProbs,
		Confidence:           &conf,
		Entropy:              h,
		NormalizedH:          normH,
		RawEntropy:           shortAns.RawEntropy,
		RawNormalizedH:       shortAns.RawNormalizedH,
		PassesUsed:           totalPasses,
		TaxonomyModeExecuted: mode,
	}, totalPasses, nil
}
