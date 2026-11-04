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

// DefaultCosinePriorTemperature is the default InfoNCE / bi-encoder temperature tau used when converting
// cosine similarity scores in [-1, 1] into normalized log-probabilities via NormalizeCosinePriorLogits.
const DefaultCosinePriorTemperature = 0.07

// NormalizeCosinePriorLogits converts caller embedding cosine similarities s_k into temperature-scaled
// log-softmax logits:
//
//	l_k = (s_k / tau) - log(sum_j exp(s_j / tau))
//
// Raw cosine scores occupy a narrow interval in [-1, 1] and carry an arbitrary scale across candidate counts K;
// normalizing them with a calibrated bi-encoder temperature (default DefaultCosinePriorTemperature = 0.07 when
// temperature <= 0) puts prior_logits onto a proper log-probability simplex while strictly preserving rank order
// for PriorLogitsShortlist. Non-finite entries (NaN, +-Inf) are ignored; returns nil when no finite entry exists.
func NormalizeCosinePriorLogits(cosines map[string]float64, temperature float64) map[string]float64 {
	if len(cosines) == 0 {
		return nil
	}
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		temperature = DefaultCosinePriorTemperature
	}
	scaled := make(map[string]float64, len(cosines))
	maxScaled := math.Inf(-1)
	for k, s := range cosines {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			continue
		}
		v := s / temperature
		scaled[k] = v
		if v > maxScaled {
			maxScaled = v
		}
	}
	if len(scaled) == 0 {
		return nil
	}
	sumExp := 0.0
	for _, v := range scaled {
		sumExp += math.Exp(v - maxScaled)
	}
	logZ := maxScaled + math.Log(sumExp)
	out := make(map[string]float64, len(scaled))
	for k, v := range scaled {
		out[k] = v - logZ
	}
	return out
}

// ShortlistFunc selects at most topK candidate option keys (and optional stage-1 prior logits)
// for a wide choice question (K > MaxOptionsPerSlot) so it can be resolved in a single diffusion pass.
type ShortlistFunc func(
	ctx context.Context,
	stateText string,
	images []string,
	qKey string,
	q SystemOneQuestion,
	topK int,
) (shortlistKeys []string, priorLogits map[string]float64, err error)

// PriorLogitsShortlist returns a ShortlistFunc that ranks options by q.PriorLogits (descending, tie-broken
// deterministically by option key), retaining the topK highest-prior options plus any catch-all options.
// When q.PriorLogits is empty, it returns nil so ExecuteSystemOne falls back to 2-stage bracket tournament routing.
func PriorLogitsShortlist(defaultTopK int) ShortlistFunc {
	return func(_ context.Context, _ string, _ []string, _ string, q SystemOneQuestion, topK int) ([]string, map[string]float64, error) {
		if len(q.PriorLogits) == 0 {
			return nil, nil, nil
		}
		if topK <= 0 {
			topK = defaultTopK
		}
		if topK <= 0 {
			topK = 6
		}
		allKeys := sortedOptionKeys(q.Criteria)
		if len(allKeys) <= topK {
			return allKeys, q.PriorLogits, nil
		}

		catchAll := catchAllKeys(q.Criteria)
		isCatchAll := make(map[string]bool, len(catchAll))
		for _, k := range catchAll {
			isCatchAll[k] = true
		}

		type scoredKey struct {
			key   string
			score float64
		}
		candidates := make([]scoredKey, 0, len(allKeys))
		for _, k := range allKeys {
			if isCatchAll[k] {
				continue
			}
			candidates = append(candidates, scoredKey{
				key:   k,
				score: q.PriorLogits[k],
			})
		}
		sort.Slice(candidates, func(i, j int) bool {
			if candidates[i].score != candidates[j].score {
				return candidates[i].score > candidates[j].score
			}
			return candidates[i].key < candidates[j].key
		})

		budget := topK
		if len(catchAll) > 0 && budget+len(catchAll) > MaxOptionsPerSlot {
			budget = MaxOptionsPerSlot - len(catchAll)
		}
		if budget > len(candidates) {
			budget = len(candidates)
		}
		out := make([]string, 0, budget+len(catchAll))
		for i := 0; i < budget; i++ {
			out = append(out, candidates[i].key)
		}
		out = append(out, catchAll...)
		return out, q.PriorLogits, nil
	}
}

// BuildContrastiveCriteria enriches each shortlisted option's description with the tokens that distinguish it
// from sibling candidates in the shortlist (e.g., pack counts "6x100ml" vs "1x100ml", variant badges, or compound suffixes).
func BuildContrastiveCriteria(criteria map[string]string, shortlistKeys []string) map[string]string {
	out := make(map[string]string, len(shortlistKeys))
	rawOptionTokens := make(map[string][]string, len(shortlistKeys))
	vocabSet := make(map[string]bool)

	for _, k := range shortlistKeys {
		desc := strings.TrimSpace(criteria[k])
		out[k] = desc
		toks := splitGenericSubwords(k + " " + desc)
		rawOptionTokens[k] = toks
		for _, t := range toks {
			vocabSet[t] = true
		}
	}

	prefixSplits := deriveSharedPrefixSplits(vocabSet)
	tokenFreq := make(map[string]int)
	perOptTokens := make(map[string][]string, len(shortlistKeys))

	for _, k := range shortlistKeys {
		seen := make(map[string]bool)
		for _, rawTok := range rawOptionTokens[k] {
			candidates := []string{rawTok}
			if parts, ok := prefixSplits[rawTok]; ok {
				candidates = append(candidates, parts...)
			}
			for _, tok := range candidates {
				if len(tok) < 2 || seen[tok] || shortlistStopWords[tok] {
					continue
				}
				seen[tok] = true
				perOptTokens[k] = append(perOptTokens[k], tok)
				tokenFreq[tok]++
			}
		}
	}

	for _, k := range shortlistKeys {
		var unique []string
		for _, tok := range perOptTokens[k] {
			if tokenFreq[tok] == 1 {
				unique = append(unique, tok)
				if len(unique) >= 4 {
					break
				}
			}
		}
		if len(unique) > 0 && !strings.Contains(out[k], "[distinguishing:") {
			out[k] = fmt.Sprintf("%s [distinguishing: %s]", out[k], strings.Join(unique, ", "))
		}
	}
	return out
}

var shortlistStopWords = map[string]bool{
	"the": true, "a": true, "an": true, "to": true, "in": true, "on": true,
	"of": true, "for": true, "is": true, "are": true, "my": true, "me": true,
	"i": true, "you": true, "can": true, "please": true, "how": true, "do": true,
	"what": true, "why": true, "it": true, "this": true, "that": true, "with": true,
	"from": true, "at": true, "be": true, "have": true, "has": true, "was": true,
	"were": true, "or": true, "and": true, "by": true, "not": true,
}

func simpleStem(w string) string {
	for _, suf := range []string{"ing", "ed", "es", "s"} {
		if len(w) > len(suf)+2 && strings.HasSuffix(w, suf) {
			return w[:len(w)-len(suf)]
		}
	}
	return w
}

// splitGenericSubwords splits text on punctuation, whitespace, underscores, and camelCase / letter-digit boundaries.
func splitGenericSubwords(text string) []string {
	runes := []rune(text)
	var b strings.Builder
	b.Grow(len(text) + 8)
	for i, r := range runes {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			b.WriteByte(' ')
			continue
		}
		if i > 0 {
			prev := runes[i-1]
			camelBoundary := unicode.IsLower(prev) && unicode.IsUpper(r)
			acronymBoundary := unicode.IsUpper(prev) && unicode.IsUpper(r) && i+1 < len(runes) && unicode.IsLower(runes[i+1])
			digitBoundary := (unicode.IsLetter(prev) && unicode.IsDigit(r)) || (unicode.IsDigit(prev) && unicode.IsLetter(r))
			if camelBoundary || acronymBoundary || digitBoundary {
				b.WriteByte(' ')
			}
		}
		b.WriteRune(unicode.ToLower(r))
	}
	return strings.Fields(b.String())
}

// deriveSharedPrefixSplits decomposes concatenated compound tokens that share a common prefix (>= 4 chars)
// with at least one sibling token in the same option vocabulary and leave a suffix of >= 2 chars.
func deriveSharedPrefixSplits(vocabSet map[string]bool) map[string][]string {
	words := make([]string, 0, len(vocabSet))
	for w := range vocabSet {
		if len(w) >= 6 {
			words = append(words, w)
		}
	}
	sort.Strings(words)
	splits := make(map[string][]string)
	for i := 0; i < len(words); i++ {
		for j := i + 1; j < len(words); j++ {
			w1, w2 := words[i], words[j]
			lcp := 0
			maxL := len(w1)
			if len(w2) < maxL {
				maxL = len(w2)
			}
			for lcp < maxL && w1[lcp] == w2[lcp] {
				lcp++
			}
			if lcp < 4 {
				break
			}
			rem1 := w1[lcp:]
			rem2 := w2[lcp:]
			if len(rem1) >= 2 && len(rem2) >= 2 {
				prefix := w1[:lcp]
				splits[w1] = append(splits[w1], prefix, rem1)
				splits[w2] = append(splits[w2], prefix, rem2)
			}
		}
	}
	return splits
}

func tokenizeWithStemAndPrefixes(text string, prefixSplits map[string][]string) []string {
	raw := splitGenericSubwords(text)
	out := make([]string, 0, len(raw)*2)
	for _, w := range raw {
		if shortlistStopWords[w] {
			continue
		}
		out = append(out, w, simpleStem(w))
		if parts, ok := prefixSplits[w]; ok {
			for _, p := range parts {
				if !shortlistStopWords[p] {
					out = append(out, p, simpleStem(p))
				}
			}
		}
	}
	return out
}

func extractCharNGrams(text string, n int) map[string]bool {
	var b strings.Builder
	b.Grow(len(text) + 2)
	b.WriteByte(' ')
	lastSpace := true
	for _, r := range strings.ToLower(text) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastSpace = false
		} else if !lastSpace {
			b.WriteByte(' ')
			lastSpace = true
		}
	}
	if !lastSpace {
		b.WriteByte(' ')
	}
	s := b.String()
	out := make(map[string]bool)
	if len(s) < n {
		return out
	}
	for i := 0; i <= len(s)-n; i++ {
		out[s[i:i+n]] = true
	}
	return out
}

// LexicalOverlapShortlist returns a ShortlistFunc that delegates to PriorLogitsShortlist when q.PriorLogits is
// populated, and otherwise scores each candidate option using generic IDF-weighted token/subword and character n-gram
// overlap between (stateText + q.Instructions) and the option's key/description, always preserving catch-all options.
func LexicalOverlapShortlist(defaultTopK int) ShortlistFunc {
	priorFn := PriorLogitsShortlist(defaultTopK)
	return func(ctx context.Context, stateText string, images []string, qKey string, q SystemOneQuestion, topK int) ([]string, map[string]float64, error) {
		if len(q.PriorLogits) > 0 {
			return priorFn(ctx, stateText, images, qKey, q, topK)
		}
		if topK <= 0 {
			topK = defaultTopK
		}
		if topK <= 0 {
			topK = 6
		}
		allKeys := sortedOptionKeys(q.Criteria)
		if len(allKeys) <= topK {
			return allKeys, nil, nil
		}

		vocabSet := make(map[string]bool)
		for _, k := range allKeys {
			for _, w := range splitGenericSubwords(k + " " + q.Criteria[k]) {
				vocabSet[w] = true
			}
		}
		prefixSplits := deriveSharedPrefixSplits(vocabSet)

		optTokSets := make(map[string]map[string]bool, len(allKeys))
		tokDocFreq := make(map[string]int)
		for _, k := range allKeys {
			optText := strings.ReplaceAll(k, "_", " ") + " " + q.Criteria[k]
			oTokList := tokenizeWithStemAndPrefixes(optText, prefixSplits)
			oTokSet := make(map[string]bool, len(oTokList))
			for _, t := range oTokList {
				oTokSet[t] = true
			}
			optTokSets[k] = oTokSet
			for t := range oTokSet {
				tokDocFreq[t]++
			}
		}

		queryText := stateText + " " + FormatInstructions(q.Instructions)
		qTokList := tokenizeWithStemAndPrefixes(queryText, prefixSplits)
		qTokSet := make(map[string]bool, len(qTokList))
		for _, t := range qTokList {
			qTokSet[t] = true
		}
		qNGrams := extractCharNGrams(queryText, 3)
		for g := range extractCharNGrams(queryText, 4) {
			qNGrams[g] = true
		}

		numOpts := float64(len(allKeys))
		scores := make(map[string]float64, len(allKeys))
		for _, k := range allKeys {
			oTokSet := optTokSets[k]
			tokWeightedSum := 0.0
			for t := range oTokSet {
				if qTokSet[t] {
					idf := math.Log(1.0 + numOpts/float64(1+tokDocFreq[t]))
					tokWeightedSum += idf
				}
			}
			tokScore := tokWeightedSum / math.Sqrt(math.Max(1.0, float64(len(oTokSet))))

			optText := strings.ReplaceAll(k, "_", " ") + " " + q.Criteria[k]
			oNGrams := extractCharNGrams(optText, 3)
			for g := range extractCharNGrams(optText, 4) {
				oNGrams[g] = true
			}
			ngInter := 0
			for g := range oNGrams {
				if qNGrams[g] {
					ngInter++
				}
			}
			ngScore := float64(ngInter) / math.Sqrt(math.Max(1.0, float64(len(qNGrams)*len(oNGrams))))
			// Heuristic scale balancing root-normalized IDF token overlap (~0-1.5) and character 3/4-gram overlap (~0-0.4).
			scores[k] = 2.5*tokScore + 3.0*ngScore
		}

		qWithPriors := q
		qWithPriors.PriorLogits = scores
		return priorFn(ctx, stateText, images, qKey, qWithPriors, topK)
	}
}

// evaluateWideQuestionShortlist resolves a wide choice question (K > MaxOptionsPerSlot; measured up to K = 151)
// in a single diffusion pass using ShortlistFunc (or PriorLogitsShortlist when q.PriorLogits is populated).
// Non-shortlisted options receive a strictly positive residual tail mass so the returned distribution covers
// all K options and sums to 1.0.
func evaluateWideQuestionShortlist(
	ctx context.Context,
	cli *client.Client,
	stateText string,
	images []string,
	qKey string,
	qSpec SystemOneQuestion,
	opts EngineOptions,
) (SystemOneAnswer, int, error) {
	topK := opts.ShortlistTopK
	if topK <= 0 {
		topK = 6
	}
	limit := optionLimit(opts.Labels)
	if topK > limit {
		topK = limit
	}

	fn := opts.ShortlistFunc
	if fn == nil {
		fn = PriorLogitsShortlist(topK)
	}
	shortKeys, priors, err := fn(ctx, stateText, images, qKey, qSpec, topK)
	if err != nil {
		return SystemOneAnswer{}, 0, err
	}
	if len(shortKeys) < 2 {
		// Fall back to tournament routing if the shortlist produced fewer than 2 options (e.g. no prior_logits).
		return evaluateWideQuestionTournament(ctx, cli, stateText, images, qKey, qSpec, opts)
	}
	if len(shortKeys) > limit {
		shortKeys = shortKeys[:limit]
	}

	subQ := subQuestion(qSpec, shortKeys)
	if len(priors) > 0 && len(subQ.PriorLogits) == 0 {
		subQ.PriorLogits = make(map[string]float64, len(shortKeys))
		for _, k := range shortKeys {
			if v, ok := priors[k]; ok {
				subQ.PriorLogits[k] = v
			}
		}
	}
	if opts.ShortlistContrastive {
		subQ.Criteria = BuildContrastiveCriteria(subQ.Criteria, shortKeys)
	}

	batchMap, passes, err := evaluateStandardBatch(
		ctx,
		cli,
		stateText,
		images,
		[]string{qKey},
		map[string]SystemOneQuestion{qKey: subQ},
		map[string]string{qKey: kindChoice},
		opts,
	)
	if err != nil {
		return SystemOneAnswer{}, passes, err
	}
	shortAns := batchMap[qKey]

	allOptKeys := optionKeys(qSpec, opts.OptionOrder)
	if len(allOptKeys) == len(shortKeys) {
		return shortAns, passes, nil
	}

	// Lift the shortlist distribution to all K options while preserving the shortlist argmax winner
	// and allocating a small residual tail mass (1e-3 total) across non-shortlisted options.
	inShortlist := make(map[string]bool, len(shortKeys))
	for _, k := range shortKeys {
		inShortlist[k] = true
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
			if !inShortlist[k] {
				fullProbs[k] = perRest
			}
		}
	}

	// Normalize exactly over all K options (with tempScale=1.0 since evaluateStandardBatch already scaled).
	// Preserve shortlist NormalizedH when lifting to all K options so dividing by ln(K) does not dilute
	// a near-tie inside the top-M shortlist.
	normProbs, winner, h, normH := NormalizeAndScaleProbabilities(fullProbs, allOptKeys, 1.0)
	if shortAns.NormalizedH > normH {
		normH = shortAns.NormalizedH
	}
	conf := normProbs[winner]
	return SystemOneAnswer{
		Type:           "choice",
		Choice:         winner,
		Probabilities:  normProbs,
		Confidence:     &conf,
		Entropy:        h,
		NormalizedH:    normH,
		RawEntropy:     shortAns.RawEntropy,
		RawNormalizedH: shortAns.RawNormalizedH,
		PassesUsed:     passes,
	}, passes, nil
}
