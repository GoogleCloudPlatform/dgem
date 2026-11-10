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
	"fmt"
	"math"
	"strings"

	"github.com/GoogleCloudPlatform/dgem/pkg/client"
)

// ConformalFitSummary records 5-fold out-of-sample split conformal prediction metrics on a calibration report.
type ConformalFitSummary struct {
	Method            string    `json:"method"`
	Alpha             float64   `json:"alpha"`
	Folds             int       `json:"folds"`
	FoldQuantiles     []float64 `json:"fold_quantiles"`
	EvaluatedCases    int       `json:"evaluated_cases"`
	EmpiricalCoverage float64   `json:"empirical_coverage"`
	MeanSetSize       float64   `json:"mean_set_size"`
	SingletonRate     float64   `json:"singleton_rate"`
	EscalationRate    float64   `json:"escalation_rate"`
}

func caseConformalProbabilities(c CalibrationCaseResult) (map[string]float64, string, bool) {
	if len(c.TopProbabilities) == 0 || strings.TrimSpace(c.Expected) == "" {
		return nil, "", false
	}
	probs := make(map[string]float64, max(len(c.TopProbabilities), c.VocabCardinality))
	sumTop := 0.0
	for k, v := range c.TopProbabilities {
		if !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 {
			probs[k] = v
			sumTop += v
		}
	}
	if len(probs) == 0 || sumTop <= 0 {
		return nil, "", false
	}

	goldKey, found := client.ResolveGoldProbabilityKey(probs, c.Expected)
	remSlots := c.VocabCardinality - len(probs)
	if !found && remSlots <= 0 {
		return nil, "", false
	}
	if remSlots > 0 {
		remMass := math.Max(1e-9, 1.0-sumTop)
		eachTail := remMass / float64(remSlots)
		added := 0
		if !found {
			goldKey = strings.TrimSpace(c.Expected)
			probs[goldKey] = eachTail
			found = true
			added++
		}
		for added < remSlots {
			tailKey := fmt.Sprintf("__tail_%d", added)
			if _, exists := probs[tailKey]; !exists {
				probs[tailKey] = eachTail
			}
			added++
		}
	}
	if !found {
		return nil, "", false
	}
	return probs, goldKey, true
}

func applyConformalCalibration(cases []CalibrationCaseResult, methodRaw string, alpha float64) ([]CalibrationCaseResult, *ConformalFitSummary, error) {
	trimmed := strings.TrimSpace(methodRaw)
	if trimmed == "" {
		return cases, nil, nil
	}
	method, err := client.ParseConformalMethod(trimmed)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid --conformal %q (use lac, aps, or raps): %w", methodRaw, err)
	}
	if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha <= 0 || alpha >= 1 {
		return nil, nil, fmt.Errorf("invalid --conformal-alpha %v (must lie strictly between 0 and 1)", alpha)
	}

	cfg := client.DefaultConformalConfig()
	cfg.Method = method
	cfg.Alpha = alpha

	type preparedItem struct {
		idx     int
		fold    int
		probs   map[string]float64
		goldKey string
	}
	var valid []preparedItem
	for i, c := range cases {
		probs, goldKey, ok := caseConformalProbabilities(c)
		if !ok {
			continue
		}
		valid = append(valid, preparedItem{
			idx:     i,
			fold:    temperatureFold(c.ID, temperatureFolds),
			probs:   probs,
			goldKey: goldKey,
		})
	}
	if len(valid) == 0 {
		return cases, nil, nil
	}

	foldQ := make([]float64, temperatureFolds)
	for f := 0; f < temperatureFolds; f++ {
		var trainProbs []map[string]float64
		var trainGold []string
		for _, it := range valid {
			if it.fold != f {
				trainProbs = append(trainProbs, it.probs)
				trainGold = append(trainGold, it.goldKey)
			}
		}
		if len(trainProbs) == 0 {
			for _, it := range valid {
				trainProbs = append(trainProbs, it.probs)
				trainGold = append(trainGold, it.goldKey)
			}
		}
		calRes, calErr := client.CalibrateConformalQuantile(trainProbs, trainGold, alpha, cfg)
		if calErr != nil {
			return nil, nil, calErr
		}
		foldQ[f] = calRes.QHat
	}

	out := make([]CalibrationCaseResult, len(cases))
	copy(out, cases)

	coveredCount := 0
	singletonCount := 0
	escalateCount := 0
	totalSize := 0

	for _, it := range valid {
		qHat := foldQ[it.fold]
		setRes, setErr := client.ConformalPredictionSet(it.probs, qHat, cfg)
		if setErr != nil {
			return nil, nil, setErr
		}
		covered := false
		visibleSet := make([]string, 0, len(setRes.PredictionSet))
		for _, opt := range setRes.PredictionSet {
			if opt == it.goldKey {
				covered = true
			}
			if !strings.HasPrefix(opt, "__tail_") {
				visibleSet = append(visibleSet, opt)
			}
		}
		covCopy := covered
		out[it.idx].PredictionSet = visibleSet
		out[it.idx].PredictionSetSize = setRes.SetSize
		out[it.idx].ConformalQHat = qHat
		out[it.idx].ConformalCovered = &covCopy

		if covered {
			coveredCount++
		}
		if setRes.Singleton {
			singletonCount++
		}
		if setRes.ShouldEscalate {
			escalateCount++
		}
		totalSize += setRes.SetSize
	}

	denom := float64(len(valid))
	summary := &ConformalFitSummary{
		Method:            string(method),
		Alpha:             alpha,
		Folds:             temperatureFolds,
		FoldQuantiles:     foldQ,
		EvaluatedCases:    len(valid),
		EmpiricalCoverage: float64(coveredCount) / denom,
		MeanSetSize:       float64(totalSize) / denom,
		SingletonRate:     float64(singletonCount) / denom,
		EscalationRate:    float64(escalateCount) / denom,
	}
	return out, summary, nil
}
