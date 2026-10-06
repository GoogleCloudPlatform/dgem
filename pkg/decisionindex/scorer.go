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
	"math"
	"sort"
)

// SuiteRow represents one row from apolinario/decision-index selected-rows.jsonl(.gz) or panel_suite.jsonl.
type SuiteRow struct {
	ID         string                       `json:"id"`
	Family     string                       `json:"family,omitempty"`
	CatalogID  int                          `json:"catalog_id"`
	Dataset    string                       `json:"dataset"`
	Area       string                       `json:"area"`
	State      any                          `json:"state"`
	Questions  map[string]SystemOneQuestion `json:"questions"`
	Expected   map[string]string            `json:"expected"`
	ChanceBase float64                      `json:"chance_baseline,omitempty"`
}

// CaseEvalResult stores the evaluation result for a single SuiteRow.
type CaseEvalResult struct {
	ID              string                     `json:"id"`
	CatalogID       int                        `json:"catalog_id"`
	Dataset         string                     `json:"dataset"`
	Area            string                     `json:"area"`
	TotalFields     int                        `json:"total_fields"`
	CorrectFields   int                        `json:"correct_fields"`
	MaxOptions      int                        `json:"max_options"`
	WholeCaseExact  bool                       `json:"whole_case_exact"`
	FieldAccuracy   float64                    `json:"field_accuracy"`
	Supported       bool                       `json:"supported"`
	ErrorMsg        string                     `json:"error_msg,omitempty"`
	MeanConfidence  float64                    `json:"mean_confidence"`
	MeanBrier       float64                    `json:"mean_brier"`
	MeanEntropy     float64                    `json:"mean_entropy"`
	MeanNormalizedH float64                    `json:"mean_normalized_entropy"`
	ForwardPasses   int                        `json:"forward_passes"`
	WallTimeMs      float64                    `json:"wall_time_ms"`
	Answers         map[string]SystemOneAnswer `json:"answers,omitempty"`
	Expected        map[string]string          `json:"expected,omitempty"`
}

// BenchmarkSummary aggregates metrics for one benchmark in the Decision Index panel.
type BenchmarkSummary struct {
	CatalogID        int     `json:"catalog_id"`
	Dataset          string  `json:"dataset"`
	Area             string  `json:"area"`
	InPanel          bool    `json:"in_panel"`
	TotalCases       int     `json:"total_cases"`
	SupportedCases   int     `json:"supported_cases"`
	CoveragePct      float64 `json:"coverage_pct"`
	SupportedScore   float64 `json:"supported_score"`    // 0..100 on supported cases
	CoverageAdjScore float64 `json:"coverage_adj_score"` // 0..100 (unsupported = 0.0, official headline metric)
	ChanceBaseline   float64 `json:"chance_baseline"`    // 0..100
	SkillScore       float64 `json:"skill_score"`        // Chance-normalized skill 0..100
	MeanBrier        float64 `json:"mean_brier"`
	MeanEntropy      float64 `json:"mean_entropy"`
	MeanNormalizedH  float64 `json:"mean_normalized_entropy"`
	MeanPassesPerReq float64 `json:"mean_passes_per_req"`
}

// AreaSummary aggregates one of the 5 equal-weight areas of the Decision Index.
type AreaSummary struct {
	AreaID         string             `json:"area_id"`
	Label          string             `json:"label"`
	BenchmarkCount int                `json:"benchmark_count"`
	CoveragePct    float64            `json:"coverage_pct"`
	PlainScore     float64            `json:"plain_score"`     // Coverage-adjusted 0..100
	SupportedScore float64            `json:"supported_score"` // On supported subset 0..100
	SkillScore     float64            `json:"skill_score"`     // Chance-normalized 0..100
	Benchmarks     []BenchmarkSummary `json:"benchmarks"`
}

// DecisionIndexReport is the complete official Decision Index scorecard (index.json + benchmark-summary.json).
type DecisionIndexReport struct {
	SuiteEdition           string             `json:"suite_edition"`
	EngineMode             string             `json:"engine_mode"`
	TemperatureScale       float64            `json:"temperature_scale"`
	TotalRequests          int                `json:"total_requests"`
	SupportedRequests      int                `json:"supported_requests"`
	CoveragePct            float64            `json:"coverage_pct"`
	TotalFields            int                `json:"total_fields"`
	TotalForwardPasses     int                `json:"total_forward_passes"`
	HeadlineDecisionIndex  float64            `json:"headline_decision_index"`    // Mean of 5 equal-weight areas (coverage-adjusted, 0..100)
	SupportedDecisionIndex float64            `json:"supported_decision_index"`   // Mean of 5 areas on supported subset (0..100)
	SkillDecisionIndex     float64            `json:"skill_decision_index"`       // Chance-normalized skill index (0..100)
	WideOptionAccuracy     float64            `json:"wide_option_accuracy"`       // Accuracy on questions with K > 26 options
	MultiSlotCanvasAcc     float64            `json:"multi_slot_canvas_accuracy"` // Accuracy on requests with M > 10 slots
	ECE10Bin               float64            `json:"ece_10bin"`
	MeanBrier              float64            `json:"mean_brier"`
	P50LatencyMs           float64            `json:"p50_latency_ms"`
	Areas                  []AreaSummary      `json:"areas"`
	DisplayBenchmarks      []BenchmarkSummary `json:"display_benchmarks,omitempty"`
	Results                []CaseEvalResult   `json:"results,omitempty"`
}

// Official 5 Areas from apolinario/decision-index/decision_index/constants.py
var OfficialAreas = []struct {
	ID    string
	Label string
}{
	{ID: "knowledge", Label: "Knowledge & Reasoning"},
	{ID: "language", Label: "Language Understanding"},
	{ID: "retrieval", Label: "Retrieval & Classification"},
	{ID: "tools", Label: "Tools & Automation"},
	{ID: "arts", Label: "Arts & Human Judgment"},
}

// OfficialChanceBaselines stores the frozen random-chance baselines (in 0..1 fraction) from decision_index/data/chance-baselines.json.
var OfficialChanceBaselines = map[string]float64{
	"BFCL":             0.2592,
	"ToolRet":          0.0937,
	"API-Bank":         0.0189,
	"BANKING77":        0.0127,
	"CLINC150+OOS":     0.0066,
	"RouterBench":      0.0909,
	"ContractNLI":      0.3333,
	"ANLI":             0.3333,
	"BPoMP":            0.2500,
	"Humicroedit":      0.5000,
	"POP909-CL":        0.1250,
	"cfcolor":          0.2000,
	"MMLU":             0.2500,
	"GPQA Diamond":     0.2500,
	"GSM8K":            0.2000,
	"ChessBench":       0.2500,
	"BRIGHT":           0.1120,
	"Amazon ESCI":      0.2500,
	"iSarcasmEval":     0.2500,
	"VAST":             0.3333,
	"CRUXEval":         0.2500,
	"CLadder":          0.5000,
	"Habermas Machine": 0.2500,
}

// PanelMembership indicates whether a dataset is in the 19-benchmark scored panel vs display-only.
var PanelMembership = map[string]bool{
	"MMLU":             true,
	"GPQA Diamond":     true,
	"GSM8K":            true,
	"ChessBench":       true,
	"CRUXEval":         true,
	"CLadder":          true,
	"ContractNLI":      true,
	"iSarcasmEval":     true,
	"VAST":             true,
	"BRIGHT":           true,
	"Amazon ESCI":      true,
	"BFCL":             true,
	"ToolRet":          true,
	"RouterBench":      true,
	"BPoMP":            true,
	"Humicroedit":      true,
	"POP909-CL":        true,
	"cfcolor":          true,
	"Habermas Machine": true,
	// Display-only wide-option / NLI benchmarks in the 37-benchmark suite:
	"API-Bank":     false,
	"BANKING77":    false,
	"CLINC150+OOS": false,
	"ANLI":         false,
}

// ScoreSingleCase evaluates a SystemOneResponse against the expected gold answers of a SuiteRow.
func ScoreSingleCase(row SuiteRow, resp *SystemOneResponse, err error) CaseEvalResult {
	totalFields := len(row.Questions)
	maxOpts := 0
	for _, q := range row.Questions {
		if len(q.Criteria) > maxOpts {
			maxOpts = len(q.Criteria)
		}
	}

	if err != nil || resp == nil {
		errMsg := "unsupported capacity rejection"
		if err != nil {
			errMsg = err.Error()
		}
		return CaseEvalResult{
			ID:             row.ID,
			CatalogID:      row.CatalogID,
			Dataset:        row.Dataset,
			Area:           row.Area,
			TotalFields:    totalFields,
			CorrectFields:  0,
			MaxOptions:     maxOpts,
			WholeCaseExact: false,
			FieldAccuracy:  0.0,
			Supported:      false,
			ErrorMsg:       errMsg,
			MeanBrier:      1.0,
			Expected:       row.Expected,
		}
	}

	correct := 0
	sumConf := 0.0
	sumBrier := 0.0
	sumH := 0.0
	sumNormH := 0.0

	for qKey, qSpec := range row.Questions {
		gold := row.Expected[qKey]
		ans, ok := resp.Answers[qKey]
		if !ok {
			sumBrier += 1.0
			continue
		}
		if ans.Choice == gold {
			correct++
		}
		winProb := ans.Probabilities[ans.Choice]
		sumConf += winProb
		sumH += ans.Entropy
		sumNormH += ans.NormalizedH

		// Multi-class Brier score across options
		brier := 0.0
		for optKey := range qSpec.Criteria {
			p := ans.Probabilities[optKey]
			y := 0.0
			if optKey == gold {
				y = 1.0
			}
			diff := p - y
			brier += diff * diff
		}
		sumBrier += brier
	}

	denom := float64(maxInt(1, totalFields))
	passes := 1
	wallMs := 0.0
	if resp.EvaluationTrace != nil {
		passes = resp.EvaluationTrace.ForwardPasses
		wallMs = resp.EvaluationTrace.WallTimeMs
	}

	return CaseEvalResult{
		ID:              row.ID,
		CatalogID:       row.CatalogID,
		Dataset:         row.Dataset,
		Area:            row.Area,
		TotalFields:     totalFields,
		CorrectFields:   correct,
		MaxOptions:      maxOpts,
		WholeCaseExact:  correct == totalFields,
		FieldAccuracy:   float64(correct) / denom,
		Supported:       true,
		MeanConfidence:  sumConf / denom,
		MeanBrier:       sumBrier / denom,
		MeanEntropy:     sumH / denom,
		MeanNormalizedH: sumNormH / denom,
		ForwardPasses:   passes,
		WallTimeMs:      wallMs,
		Answers:         resp.Answers,
		Expected:        row.Expected,
	}
}

// ComputeDecisionIndex aggregates CaseEvalResults into the official 5-Area Decision Index Report.
func ComputeDecisionIndex(results []CaseEvalResult, engineMode string, tempScale float64) DecisionIndexReport {
	byDataset := make(map[string][]CaseEvalResult)
	totalReqs := len(results)
	supportedReqs := 0
	totalFields := 0
	totalPasses := 0
	sumBrier := 0.0

	var wideCorrect, wideTotal int
	var multiSlotCorrect, multiSlotTotal int
	var latencies []float64

	// 10-bin ECE accumulators
	var binCount [10]int
	var binConf [10]float64
	var binAcc [10]float64

	for _, r := range results {
		byDataset[r.Dataset] = append(byDataset[r.Dataset], r)
		totalFields += r.TotalFields
		totalPasses += r.ForwardPasses
		sumBrier += r.MeanBrier

		if r.MaxOptions > 26 {
			wideTotal += r.TotalFields
			wideCorrect += r.CorrectFields
		}
		if r.TotalFields > 10 {
			multiSlotTotal += r.TotalFields
			multiSlotCorrect += r.CorrectFields
		}

		if r.Supported {
			supportedReqs++
			if r.WallTimeMs > 0 {
				latencies = append(latencies, r.WallTimeMs)
			}
			b := int(math.Min(9, math.Max(0, math.Floor(r.MeanConfidence*10))))
			binCount[b]++
			binConf[b] += r.MeanConfidence
			binAcc[b] += r.FieldAccuracy
		}
	}

	ece := 0.0
	if supportedReqs > 0 {
		for b := 0; b < 10; b++ {
			if binCount[b] > 0 {
				avgC := binConf[b] / float64(binCount[b])
				avgA := binAcc[b] / float64(binCount[b])
				ece += (float64(binCount[b]) / float64(supportedReqs)) * math.Abs(avgC-avgA)
			}
		}
	}

	var benchSummaries []BenchmarkSummary
	var displaySummaries []BenchmarkSummary

	for ds, dsResults := range byDataset {
		tot := len(dsResults)
		supp := 0
		sumFieldAccSupp := 0.0
		sumFieldAccAll := 0.0
		sumB := 0.0
		sumH := 0.0
		sumNH := 0.0
		sumP := 0

		for _, r := range dsResults {
			sumFieldAccAll += r.FieldAccuracy // 0.0 if unsupported!
			sumB += r.MeanBrier
			sumP += r.ForwardPasses
			if r.Supported {
				supp++
				sumFieldAccSupp += r.FieldAccuracy
				sumH += r.MeanEntropy
				sumNH += r.MeanNormalizedH
			}
		}

		covPct := 100.0 * float64(supp) / float64(maxInt(1, tot))
		suppScore := 0.0
		if supp > 0 {
			suppScore = 100.0 * (sumFieldAccSupp / float64(supp))
		}
		covAdjScore := 100.0 * (sumFieldAccAll / float64(maxInt(1, tot)))
		chanceFrac, ok := OfficialChanceBaselines[ds]
		if !ok {
			chanceFrac = 0.25
		}
		chanceScore := 100.0 * chanceFrac
		skill := 0.0
		if 100.0-chanceScore > 1e-6 {
			skill = math.Max(0.0, 100.0*(covAdjScore-chanceScore)/(100.0-chanceScore))
		}

		inPanel := PanelMembership[ds]
		bs := BenchmarkSummary{
			CatalogID:        dsResults[0].CatalogID,
			Dataset:          ds,
			Area:             dsResults[0].Area,
			InPanel:          inPanel,
			TotalCases:       tot,
			SupportedCases:   supp,
			CoveragePct:      covPct,
			SupportedScore:   suppScore,
			CoverageAdjScore: covAdjScore,
			ChanceBaseline:   chanceScore,
			SkillScore:       skill,
			MeanBrier:        sumB / float64(maxInt(1, tot)),
			MeanEntropy:      sumH / float64(maxInt(1, supp)),
			MeanNormalizedH:  sumNH / float64(maxInt(1, supp)),
			MeanPassesPerReq: float64(sumP) / float64(maxInt(1, tot)),
		}
		if inPanel {
			benchSummaries = append(benchSummaries, bs)
		} else {
			displaySummaries = append(displaySummaries, bs)
		}
	}

	sort.Slice(benchSummaries, func(i, j int) bool {
		return benchSummaries[i].CatalogID < benchSummaries[j].CatalogID
	})
	sort.Slice(displaySummaries, func(i, j int) bool {
		return displaySummaries[i].CatalogID < displaySummaries[j].CatalogID
	})

	// Group into the 5 equal-weight Official Areas
	var areaSummaries []AreaSummary
	sumAreaPlain := 0.0
	sumAreaSupp := 0.0
	sumAreaSkill := 0.0
	activeAreas := 0

	for _, areaDef := range OfficialAreas {
		var areaBenchs []BenchmarkSummary
		for _, bs := range benchSummaries {
			if bs.Area == areaDef.ID {
				areaBenchs = append(areaBenchs, bs)
			}
		}
		if len(areaBenchs) == 0 {
			continue
		}
		activeAreas++
		var sPlain, sSupp, sSkill, sCov float64
		for _, b := range areaBenchs {
			sPlain += b.CoverageAdjScore
			sSupp += b.SupportedScore
			sSkill += b.SkillScore
			sCov += b.CoveragePct
		}
		n := float64(len(areaBenchs))
		aSum := AreaSummary{
			AreaID:         areaDef.ID,
			Label:          areaDef.Label,
			BenchmarkCount: len(areaBenchs),
			CoveragePct:    sCov / n,
			PlainScore:     sPlain / n,
			SupportedScore: sSupp / n,
			SkillScore:     sSkill / n,
			Benchmarks:     areaBenchs,
		}
		areaSummaries = append(areaSummaries, aSum)
		sumAreaPlain += aSum.PlainScore
		sumAreaSupp += aSum.SupportedScore
		sumAreaSkill += aSum.SkillScore
	}

	areaDenom := float64(maxInt(1, activeAreas))
	wideAcc := 0.0
	if wideTotal > 0 {
		wideAcc = 100.0 * float64(wideCorrect) / float64(wideTotal)
	}
	multiAcc := 0.0
	if multiSlotTotal > 0 {
		multiAcc = 100.0 * float64(multiSlotCorrect) / float64(multiSlotTotal)
	}

	p50 := 0.0
	if len(latencies) > 0 {
		sort.Float64s(latencies)
		p50 = latencies[len(latencies)/2]
	}

	return DecisionIndexReport{
		SuiteEdition:           "release-v1-panel",
		EngineMode:             engineMode,
		TemperatureScale:       tempScale,
		TotalRequests:          totalReqs,
		SupportedRequests:      supportedReqs,
		CoveragePct:            100.0 * float64(supportedReqs) / float64(maxInt(1, totalReqs)),
		TotalFields:            totalFields,
		TotalForwardPasses:     totalPasses,
		HeadlineDecisionIndex:  sumAreaPlain / areaDenom,
		SupportedDecisionIndex: sumAreaSupp / areaDenom,
		SkillDecisionIndex:     sumAreaSkill / areaDenom,
		WideOptionAccuracy:     wideAcc,
		MultiSlotCanvasAcc:     multiAcc,
		ECE10Bin:               ece,
		MeanBrier:              sumBrier / float64(maxInt(1, totalReqs)),
		P50LatencyMs:           p50,
		Areas:                  areaSummaries,
		DisplayBenchmarks:      displaySummaries,
		Results:                results,
	}
}
