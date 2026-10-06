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

// analysis: baselines that use no image, case-level bootstrap intervals, the split
// of edge error into quantization and localization, paired occlusion deltas, whether
// entropy predicts edge error, run-to-run stability and transform consistency.

import (
	"math"
	"math/rand"
	"sort"
	"strconv"
)

const bboxBootstrapIters = 2000

// BboxStat is a mean with a case-level bootstrap 95% interval.
type BboxStat struct {
	Mean float64 `json:"mean"`
	Lo   float64 `json:"ci95_lo"`
	Hi   float64 `json:"ci95_hi"`
	N    int     `json:"n"`
}

// BboxEdgeStats summarizes one box edge across cases.
type BboxEdgeStats struct {
	SignedBiasExpectation float64 `json:"signed_bias_expectation"`
	SignedBiasArgmax      float64 `json:"signed_bias_argmax"`
	MAEArgmax             float64 `json:"mae_argmax"`
	MAEExpectation        float64 `json:"mae_expectation"`
	MeanHNorm             float64 `json:"mean_h_norm"`
	N                     int     `json:"n"`
}

// BboxGridSplit compares argmax and expectation error on edges whose ground truth sits on a
// label (no quantization error) against edges between labels.
type BboxGridSplit struct {
	OnGridN              int     `json:"on_grid_n"`
	OnGridMAEArgmax      float64 `json:"on_grid_mae_argmax"`
	OnGridMAEExpectation float64 `json:"on_grid_mae_expectation"`
	OffGridN             int     `json:"off_grid_n"`
	OffGridMAEArgmax     float64 `json:"off_grid_mae_argmax"`
	OffGridMAEExpect     float64 `json:"off_grid_mae_expectation"`
	// QuantizationFloorMAE is the mean distance from ground truth to the nearest label: the
	// error a perfect argmax reader would still make. Error above it is localization error.
	QuantizationFloorMAE float64 `json:"quantization_floor_mae"`
	// ArgmaxBinsOff counts edges by how many label steps the argmax sits from the label
	// nearest the ground truth.
	ArgmaxBinsOff map[int]int `json:"argmax_bins_off"`
}

// BboxPairedOcclusion compares an occluded image's entropy with its unoccluded twin.
type BboxPairedOcclusion struct {
	PairID          string  `json:"pair_id"`
	OccludedEdge    string  `json:"occluded_edge"`
	HOccluded       float64 `json:"h_norm_occluded_edge"`
	HTwin           float64 `json:"h_norm_same_edge_in_twin"`
	Delta           float64 `json:"delta_occluded_edge"`
	OtherEdgesDelta float64 `json:"delta_other_edges_mean"`
	// RankInCase is the occluded edge's entropy rank among the 4 edges (1 = highest).
	RankInCase int `json:"occluded_edge_rank_in_case"`
}

// BboxVariantSummary aggregates one variant across cases and repeats.
type BboxVariantSummary struct {
	Variant  string `json:"variant"`
	Runs     int    `json:"runs"`
	Cases    int    `json:"cases"`
	BoxCases int    `json:"box_cases"`
	Skipped  int    `json:"skipped"`
	Errors   int    `json:"errors"`

	MeanArgmaxIoU      BboxStat `json:"mean_argmax_iou"`
	MeanExpectationIoU BboxStat `json:"mean_expectation_iou"`
	AccAt50Argmax      float64  `json:"acc_at_50_argmax_pct"`
	AccAt50Expectation float64  `json:"acc_at_50_expectation_pct"`
	AccAt75Argmax      float64  `json:"acc_at_75_argmax_pct"`
	AccAt75Expectation float64  `json:"acc_at_75_expectation_pct"`
	// CenterHitPct: share of box runs whose expectation-box centre lies inside the true box (a
	// "click" hit, as in ScreenSpot). Unlike IoU it does not punish small targets for edge error.
	CenterHitPct float64 `json:"center_hit_pct"`

	// Expectation-minus-argmax IoU, split by whether every edge is on a label.
	ExpectationGainOnGrid  BboxStat `json:"expectation_gain_on_grid_cases"`
	ExpectationGainOffGrid BboxStat `json:"expectation_gain_off_grid_cases"`

	MissingSlotRuns int     `json:"missing_slot_runs"`
	InvalidBoxRate  float64 `json:"invalid_box_rate"`
	PresenceAsked   int     `json:"presence_asked"`
	// PredictedPresentRate is how often the model answered "present" when asked; on the blank
	// variant it should be near 0.
	PredictedPresentRate *float64 `json:"predicted_present_rate,omitempty"`
	PresenceCorrect      int      `json:"presence_correct"`
	AbsentRuns           int      `json:"absent_runs"`
	FalsePositives       int      `json:"false_positives"`
	ClassAccuracy        *float64 `json:"class_accuracy,omitempty"`

	MeanPredictedBox [4]float64               `json:"mean_predicted_expectation_box"`
	Edges            map[string]BboxEdgeStats `json:"edges"`
	Grid             BboxGridSplit            `json:"grid_split"`

	// EntropyErrorAUROC: does an edge's normalized entropy rank edges whose argmax is more
	// than one label step from the ground truth above the rest? 0.5 = no signal.
	EntropyErrorAUROC    float64 `json:"entropy_error_auroc"` // -1 when undefined (no distributions)
	EntropyErrorPositive int     `json:"entropy_error_positive"`
	EntropyErrorNegative int     `json:"entropy_error_negative"`
	// Spearman correlation between edge entropy and absolute expectation error.
	EntropyErrorSpearman float64 `json:"entropy_error_spearman"` // -1 when undefined

	PairedOcclusion []BboxPairedOcclusion `json:"paired_occlusion,omitempty"`

	// Repeat stability (only with --repeat > 1).
	EdgeArgmaxAgreement *float64 `json:"edge_argmax_agreement,omitempty"`
	RepeatPairwiseIoU   *float64 `json:"repeat_pairwise_expectation_iou,omitempty"`

	// Transform consistency: IoU between the variant's box and the original's box mapped
	// into the variant's frame (same case, same repeat). Needs no ground truth.
	ConsistencyIoU *BboxStat `json:"consistency_iou_vs_transformed_original,omitempty"`
}

// BboxBaseline is a predictor that ignores the image, scored on the original variant.
type BboxBaseline struct {
	Name            string      `json:"name"`
	Description     string      `json:"description"`
	MeanIoU         BboxStat    `json:"mean_iou"`
	AccAt50Pct      float64     `json:"acc_at_50_pct"`
	CenterHitPct    float64     `json:"center_hit_pct"`
	Box             *[4]float64 `json:"box,omitempty"`
	LiftExpectation BboxStat    `json:"model_expectation_lift"`
	LiftArgmax      BboxStat    `json:"model_argmax_lift"`
}

func bootstrapMean(vals []float64, seed int64) BboxStat {
	n := len(vals)
	st := BboxStat{N: n}
	if n == 0 {
		return st
	}
	st.Mean = mean(vals)
	if n == 1 {
		st.Lo, st.Hi = st.Mean, st.Mean
		return st
	}
	rng := rand.New(rand.NewSource(seed))
	means := make([]float64, bboxBootstrapIters)
	for i := range means {
		s := 0.0
		for j := 0; j < n; j++ {
			s += vals[rng.Intn(n)]
		}
		means[i] = s / float64(n)
	}
	sort.Float64s(means)
	st.Lo = means[int(0.025*float64(len(means)))]
	st.Hi = means[int(0.975*float64(len(means)))-1]
	return st
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// aurocScore is the probability that a random positive scores above a random negative.
func aurocScore(pos, neg []float64) float64 {
	if len(pos) == 0 || len(neg) == 0 {
		return math.NaN()
	}
	wins := 0.0
	for _, p := range pos {
		for _, q := range neg {
			if p > q {
				wins++
			} else if p == q {
				wins += 0.5
			}
		}
	}
	return wins / float64(len(pos)*len(neg))
}

func ranks(v []float64) []float64 {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return v[idx[a]] < v[idx[b]] })
	r := make([]float64, len(v))
	for i := 0; i < len(idx); {
		j := i
		for j+1 < len(idx) && v[idx[j+1]] == v[idx[i]] {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			r[idx[k]] = avg
		}
		i = j + 1
	}
	return r
}

func spearman(a, b []float64) float64 {
	if len(a) < 3 || len(a) != len(b) {
		return math.NaN()
	}
	ra, rb := ranks(a), ranks(b)
	ma, mb := mean(ra), mean(rb)
	var num, da, db float64
	for i := range ra {
		num += (ra[i] - ma) * (rb[i] - mb)
		da += (ra[i] - ma) * (ra[i] - ma)
		db += (rb[i] - mb) * (rb[i] - mb)
	}
	if da == 0 || db == 0 {
		return math.NaN()
	}
	return num / math.Sqrt(da*db)
}

// caseEdgeKeys lists the edge telemetry keys of a result with the ground-truth coordinate
// recorded, i.e. every scored edge (both objects for DETR cases).
func scoredEdges(r BboxCaseResult) []BboxEdgeTelemetry {
	keys := make([]string, 0, len(r.Edges))
	for k, e := range r.Edges {
		if e.HasGT && !e.Missing {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	out := make([]BboxEdgeTelemetry, 0, len(keys))
	for _, k := range keys {
		out = append(out, r.Edges[k])
	}
	return out
}

// nearestLabel is the label coordinate closest to v on the slot's regular label grid.
func nearestLabel(v float64, e BboxEdgeTelemetry) float64 {
	w := e.BinWidth
	if w <= 0 {
		w = 5
	}
	lo, hi := e.LabelMin, e.LabelMax
	if hi <= lo {
		lo, hi = 0, 100
	}
	n := math.Round((v - lo) / w)
	n = math.Max(0, math.Min(n, math.Round((hi-lo)/w)))
	return lo + n*w
}

func nearestLabelDistance(gt float64, e BboxEdgeTelemetry) float64 {
	return math.Abs(gt - nearestLabel(gt, e))
}

func isOnGridCase(r BboxCaseResult) bool {
	es := scoredEdges(r)
	if len(es) == 0 {
		return false
	}
	for _, e := range es {
		if nearestLabelDistance(e.GTCoord, e) > 1e-6 {
			return false
		}
	}
	return true
}

func summarizeBboxVariant(variant string, all []BboxCaseResult, original map[string]BboxCaseResult) BboxVariantSummary {
	s := BboxVariantSummary{Variant: variant, Edges: map[string]BboxEdgeStats{}}
	s.Grid.ArgmaxBinsOff = map[int]int{}
	var rs []BboxCaseResult
	repeats := map[int]bool{}
	cases := map[string]bool{}
	for _, r := range all {
		if r.Variant != variant {
			continue
		}
		if r.Skipped != "" {
			s.Skipped++
			continue
		}
		if r.Error != "" {
			s.Errors++
			continue
		}
		rs = append(rs, r)
		repeats[r.Repeat] = true
		cases[r.ID] = true
	}
	s.Runs, s.Cases = len(repeats), len(cases)
	if len(rs) == 0 {
		return s
	}

	// Per-case means over repeats feed the case-level bootstrap.
	type acc struct{ arg, exp []float64 }
	perCase := map[string]*acc{}
	var order []string
	var gainOn, gainOff []float64
	gainCase := map[string][]float64{}
	onGrid := map[string]bool{}
	var a50, e50, a75, e75, boxRuns, invalid, centerHits int
	var predSum [4]float64
	edgeAcc := map[string]*struct{ sb, sa, ma, me, h []float64 }{}
	var posH, negH, allH, allErr []float64
	var qFloor []float64
	var onA, onE, offA, offE []float64
	classHits, classTotal := 0, 0
	presentHits := 0

	for _, r := range rs {
		if len(r.MissingSlots) > 0 {
			s.MissingSlotRuns++
		}
		if r.PresenceAsked {
			s.PresenceAsked++
			if r.ObjectPresentPred {
				presentHits++
			}
			if r.ObjectPresentPred == r.ObjectPresentGT {
				s.PresenceCorrect++
			}
		}
		if !r.ObjectPresentGT {
			s.AbsentRuns++
			if r.ObjectPresentPred {
				s.FalsePositives++
			}
			continue
		}
		boxRuns++
		if r.InvalidBox {
			invalid++
		}
		for i := range predSum {
			predSum[i] += r.ExpectationBox[i]
		}
		if r.ClassTotal > 0 {
			classHits += r.ClassHits
			classTotal += r.ClassTotal
		}
		if perCase[r.ID] == nil {
			perCase[r.ID] = &acc{}
			order = append(order, r.ID)
			onGrid[r.ID] = isOnGridCase(r)
		}
		perCase[r.ID].arg = append(perCase[r.ID].arg, r.ArgmaxIoU)
		perCase[r.ID].exp = append(perCase[r.ID].exp, r.ExpectationIoU)
		gainCase[r.ID] = append(gainCase[r.ID], r.ExpectationIoU-r.ArgmaxIoU)
		if centerHit(r) {
			centerHits++
		}
		if r.ArgmaxIoU >= 0.5 {
			a50++
		}
		if r.ExpectationIoU >= 0.5 {
			e50++
		}
		if r.ArgmaxIoU >= 0.75 {
			a75++
		}
		if r.ExpectationIoU >= 0.75 {
			e75++
		}
		for _, e := range scoredEdges(r) {
			ea := edgeAcc[e.Edge]
			if ea == nil {
				ea = &struct{ sb, sa, ma, me, h []float64 }{}
				edgeAcc[e.Edge] = ea
			}
			errA := e.ArgmaxCoord - e.GTCoord
			errE := e.ExpectedCoord - e.GTCoord
			ea.sb = append(ea.sb, errE)
			ea.sa = append(ea.sa, errA)
			ea.ma = append(ea.ma, math.Abs(errA))
			ea.me = append(ea.me, math.Abs(errE))
			ea.h = append(ea.h, e.NormalizedEntropy)

			qd := nearestLabelDistance(e.GTCoord, e)
			qFloor = append(qFloor, qd)
			if qd < 1e-6 {
				onA = append(onA, math.Abs(errA))
				onE = append(onE, math.Abs(errE))
			} else {
				offA = append(offA, math.Abs(errA))
				offE = append(offE, math.Abs(errE))
			}
			w := e.BinWidth
			if w <= 0 {
				w = 5
			}
			// Steps from the label nearest the ground truth to the argmax label.
			nearest := nearestLabel(e.GTCoord, e)
			s.Grid.ArgmaxBinsOff[int(math.Round(math.Abs(e.ArgmaxCoord-nearest)/w))]++

			allH = append(allH, e.NormalizedEntropy)
			allErr = append(allErr, math.Abs(errE))
			if math.Abs(errA) > w+1e-9 {
				posH = append(posH, e.NormalizedEntropy)
			} else {
				negH = append(negH, e.NormalizedEntropy)
			}
		}
	}
	s.BoxCases = len(order)
	var argCase, expCase []float64
	for _, id := range order {
		argCase = append(argCase, mean(perCase[id].arg))
		expCase = append(expCase, mean(perCase[id].exp))
		g := mean(gainCase[id])
		if onGrid[id] {
			gainOn = append(gainOn, g)
		} else {
			gainOff = append(gainOff, g)
		}
	}
	s.MeanArgmaxIoU = bootstrapMean(argCase, 11)
	s.MeanExpectationIoU = bootstrapMean(expCase, 12)
	s.ExpectationGainOnGrid = bootstrapMean(gainOn, 13)
	s.ExpectationGainOffGrid = bootstrapMean(gainOff, 14)
	if boxRuns > 0 {
		f := 100.0 / float64(boxRuns)
		s.AccAt50Argmax, s.AccAt50Expectation = float64(a50)*f, float64(e50)*f
		s.AccAt75Argmax, s.AccAt75Expectation = float64(a75)*f, float64(e75)*f
		s.InvalidBoxRate = float64(invalid) / float64(boxRuns)
		s.CenterHitPct = 100 * float64(centerHits) / float64(boxRuns)
		for i := range predSum {
			s.MeanPredictedBox[i] = predSum[i] / float64(boxRuns)
		}
	}
	if classTotal > 0 {
		v := float64(classHits) / float64(classTotal)
		s.ClassAccuracy = &v
	}
	if s.PresenceAsked > 0 {
		v := float64(presentHits) / float64(s.PresenceAsked)
		s.PredictedPresentRate = &v
	}
	for name, ea := range edgeAcc {
		s.Edges[name] = BboxEdgeStats{
			SignedBiasExpectation: mean(ea.sb), SignedBiasArgmax: mean(ea.sa),
			MAEArgmax: mean(ea.ma), MAEExpectation: mean(ea.me), MeanHNorm: mean(ea.h), N: len(ea.sb),
		}
	}
	s.Grid.OnGridN, s.Grid.OffGridN = len(onA), len(offA)
	s.Grid.OnGridMAEArgmax, s.Grid.OnGridMAEExpectation = mean(onA), mean(onE)
	s.Grid.OffGridMAEArgmax, s.Grid.OffGridMAEExpect = mean(offA), mean(offE)
	s.Grid.QuantizationFloorMAE = mean(qFloor)
	s.EntropyErrorAUROC = finiteOr(aurocScore(posH, negH), -1)
	s.EntropyErrorPositive, s.EntropyErrorNegative = len(posH), len(negH)
	s.EntropyErrorSpearman = finiteOr(spearman(allH, allErr), -1)
	if allZero(allH) {
		s.EntropyErrorAUROC, s.EntropyErrorSpearman = -1, -1
	}

	s.PairedOcclusion = pairedOcclusion(rs)
	s.EdgeArgmaxAgreement, s.RepeatPairwiseIoU = repeatStability(rs)
	if variant != variantOriginal && original != nil {
		s.ConsistencyIoU = transformConsistency(variant, rs, original)
	}
	return s
}

// meanEdgeEntropy averages one case's per-edge normalized entropy over repeats.
func meanEdgeEntropy(rs []BboxCaseResult, id string) map[string]float64 {
	sum := map[string]float64{}
	n := map[string]int{}
	for _, r := range rs {
		if r.ID != id {
			continue
		}
		for _, e := range scoredEdges(r) {
			sum[e.Edge] += e.NormalizedEntropy
			n[e.Edge]++
		}
	}
	for k := range sum {
		sum[k] /= float64(n[k])
	}
	return sum
}

func pairedOcclusion(rs []BboxCaseResult) []BboxPairedOcclusion {
	type pair struct {
		twin, occ string
		edge      string
	}
	pairs := map[string]*pair{}
	var ids []string
	seen := map[string]bool{}
	for _, r := range rs {
		if r.PairID == "" || seen[r.ID] {
			continue
		}
		seen[r.ID] = true
		p := pairs[r.PairID]
		if p == nil {
			p = &pair{}
			pairs[r.PairID] = p
			ids = append(ids, r.PairID)
		}
		if r.OccludedEdge == "" || r.OccludedEdge == "none" {
			p.twin = r.ID
		} else {
			p.occ, p.edge = r.ID, r.OccludedEdge
		}
	}
	sort.Strings(ids)
	var out []BboxPairedOcclusion
	for _, pid := range ids {
		p := pairs[pid]
		if p.twin == "" || p.occ == "" {
			continue
		}
		ho, ht := meanEdgeEntropy(rs, p.occ), meanEdgeEntropy(rs, p.twin)
		po := BboxPairedOcclusion{PairID: pid, OccludedEdge: p.edge, HOccluded: ho[p.edge], HTwin: ht[p.edge]}
		po.Delta = po.HOccluded - po.HTwin
		var others []float64
		po.RankInCase = 1
		for _, e := range bboxEdges {
			if e == p.edge {
				continue
			}
			others = append(others, ho[e]-ht[e])
			if ho[e] > ho[p.edge] {
				po.RankInCase++
			}
		}
		po.OtherEdgesDelta = mean(others)
		out = append(out, po)
	}
	return out
}

func repeatStability(rs []BboxCaseResult) (*float64, *float64) {
	byCase := map[string][]BboxCaseResult{}
	for _, r := range rs {
		byCase[r.ID] = append(byCase[r.ID], r)
	}
	var agree, ious []float64
	for _, group := range byCase {
		if len(group) < 2 {
			continue
		}
		// Edge argmax agreement with the modal label.
		labels := map[string]map[string]int{}
		for _, r := range group {
			for k, e := range r.Edges {
				if e.Missing {
					continue
				}
				if labels[k] == nil {
					labels[k] = map[string]int{}
				}
				labels[k][e.ArgmaxBin]++
			}
		}
		for _, counts := range labels {
			best, tot := 0, 0
			for _, c := range counts {
				tot += c
				if c > best {
					best = c
				}
			}
			agree = append(agree, float64(best)/float64(tot))
		}
		if group[0].ObjectPresentGT {
			for i := 0; i < len(group); i++ {
				for j := i + 1; j < len(group); j++ {
					ious = append(ious, compute2DIoU(group[i].ExpectationBox, group[j].ExpectationBox))
				}
			}
		}
	}
	if len(agree) == 0 {
		return nil, nil
	}
	a, b := mean(agree), mean(ious)
	return &a, &b
}

func bboxRunKey(id string, repeat int) string { return id + "#" + strconv.Itoa(repeat) }

func transformConsistency(variant string, rs []BboxCaseResult, original map[string]BboxCaseResult) *BboxStat {
	if !isImageVariant(variant) || variant == variantBlank {
		return nil
	}
	perCase := map[string][]float64{}
	var order []string
	for _, r := range rs {
		o, ok := original[bboxRunKey(r.ID, r.Repeat)]
		if !ok || !r.ObjectPresentGT || !o.ObjectPresentPred {
			continue
		}
		if _, seen := perCase[r.ID]; !seen {
			order = append(order, r.ID)
		}
		perCase[r.ID] = append(perCase[r.ID], compute2DIoU(r.ExpectationBox, transformBoxForVariant(o.ExpectationBox, variant)))
	}
	if len(order) == 0 {
		return nil
	}
	var vals []float64
	for _, id := range order {
		vals = append(vals, mean(perCase[id]))
	}
	st := bootstrapMean(vals, 21)
	return &st
}

// computeBboxBaselines scores image-free predictors on the original variant's box cases and
// the model's paired lift over each (case-level bootstrap).
func computeBboxBaselines(all []BboxCaseResult, blank *BboxVariantSummary) []BboxBaseline {
	type caseRow struct {
		gt       [4]float64
		arg, exp float64
	}
	agg := map[string]*struct {
		gt       [4]float64
		arg, exp []float64
	}{}
	var ids []string
	for _, r := range all {
		if r.Variant != variantOriginal || r.Skipped != "" || r.Error != "" || !r.ObjectPresentGT {
			continue
		}
		a := agg[r.ID]
		if a == nil {
			a = &struct {
				gt       [4]float64
				arg, exp []float64
			}{gt: r.GTBox}
			agg[r.ID] = a
			ids = append(ids, r.ID)
		}
		a.arg = append(a.arg, r.ArgmaxIoU)
		a.exp = append(a.exp, r.ExpectationIoU)
	}
	if len(ids) == 0 {
		return nil
	}
	rows := make([]caseRow, len(ids))
	for i, id := range ids {
		rows[i] = caseRow{gt: agg[id].gt, arg: mean(agg[id].arg), exp: mean(agg[id].exp)}
	}
	build := func(name, desc string, box *[4]float64, pred func(i int) [4]float64, seed int64) BboxBaseline {
		var ious, liftE, liftA []float64
		hits, chits := 0, 0
		for i, row := range rows {
			p := pred(i)
			if boxCenterIn(p, row.gt) {
				chits++
			}
			v := compute2DIoU(row.gt, p)
			ious = append(ious, v)
			if v >= 0.5 {
				hits++
			}
			liftE = append(liftE, row.exp-v)
			liftA = append(liftA, row.arg-v)
		}
		return BboxBaseline{
			Name: name, Description: desc, Box: box,
			MeanIoU:         bootstrapMean(ious, seed),
			AccAt50Pct:      100 * float64(hits) / float64(len(rows)),
			CenterHitPct:    100 * float64(chits) / float64(len(rows)),
			LiftExpectation: bootstrapMean(liftE, seed+1),
			LiftArgmax:      bootstrapMean(liftA, seed+2),
		}
	}
	center := [4]float64{25, 25, 75, 75}
	out := []BboxBaseline{
		build("constant_center", "Fixed box [25,25,75,75] for every image", &center,
			func(int) [4]float64 { return center }, 31),
		build("loo_mean_gt", "Mean ground-truth box of the other cases (leave-one-out)", nil,
			func(i int) [4]float64 {
				var b [4]float64
				for j, row := range rows {
					if j == i {
						continue
					}
					for k := range b {
						b[k] += row.gt[k]
					}
				}
				if len(rows) > 1 {
					for k := range b {
						b[k] /= float64(len(rows) - 1)
					}
				}
				return b
			}, 41),
	}
	if blank != nil && blank.BoxCases > 0 {
		blankBoxes := map[string][]float64{}
		for _, r := range all {
			if r.Variant == variantBlank && r.Skipped == "" && r.Error == "" && r.ObjectPresentGT {
				blankBoxes[r.ID] = append(blankBoxes[r.ID], r.ExpectationBox[0], r.ExpectationBox[1], r.ExpectationBox[2], r.ExpectationBox[3])
			}
		}
		mb := blank.MeanPredictedBox
		out = append(out, build("blank_image_prior", "The model's own box for each prompt on a blank image of the same size (its prompt prior)", &mb,
			func(i int) [4]float64 {
				v := blankBoxes[ids[i]]
				if len(v) < 4 {
					return mb
				}
				var b [4]float64
				for j := 0; j < len(v); j += 4 {
					for k := 0; k < 4; k++ {
						b[k] += v[j+k]
					}
				}
				for k := range b {
					b[k] /= float64(len(v) / 4)
				}
				return b
			}, 51))
	}
	return out
}

func finiteOr(v, fallback float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fallback
	}
	return v
}

func allZero(v []float64) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

// BboxTagBreakdown summarizes original-variant results for one tag value of a generated sweep.
type BboxTagBreakdown struct {
	Tag                string   `json:"tag"`
	Value              string   `json:"value"`
	Cases              int      `json:"cases"`
	BoxCases           int      `json:"box_cases"`
	MeanExpectationIoU BboxStat `json:"mean_expectation_iou"`
	AccAt50Pct         float64  `json:"acc_at_50_pct"`
	CenterHitPct       float64  `json:"center_hit_pct"`
	MeanVisibleIoU     *float64 `json:"mean_visible_iou,omitempty"`
	// PresentRate: share of runs answered "present" (on absent targets, the false-positive rate).
	PresentRate float64 `json:"present_rate"`
	// TaggedEdgeHNorm: mean normalized entropy of the edge named by the case's "edge" tag.
	TaggedEdgeHNorm  *float64 `json:"tagged_edge_h_norm,omitempty"`
	MeanMaxEdgeHNorm float64  `json:"mean_max_edge_h_norm"`
}

func computeBboxBreakdowns(all []BboxCaseResult) []BboxTagBreakdown {
	type acc struct {
		ids                []string
		iou                map[string][]float64
		vis, edgeH, maxH   []float64
		hits50, runs, pres int
		presAsked, boxRuns int
		centerHits         int
	}
	groups := map[[2]string]*acc{}
	for _, r := range all {
		if r.Variant != variantOriginal || r.Error != "" || r.Skipped != "" || len(r.Tags) == 0 {
			continue
		}
		for tag, val := range r.Tags {
			if tag == "placement" {
				continue
			}
			k := [2]string{tag, val}
			a := groups[k]
			if a == nil {
				a = &acc{iou: map[string][]float64{}}
				groups[k] = a
			}
			a.runs++
			if r.PresenceAsked {
				a.presAsked++
				if r.ObjectPresentPred {
					a.pres++
				}
			}
			if !r.ObjectPresentGT {
				continue
			}
			a.boxRuns++
			if _, ok := a.iou[r.ID]; !ok {
				a.ids = append(a.ids, r.ID)
			}
			a.iou[r.ID] = append(a.iou[r.ID], r.ExpectationIoU)
			if r.ExpectationIoU >= 0.5 {
				a.hits50++
			}
			if centerHit(r) {
				a.centerHits++
			}
			if r.VisibleIoU != nil {
				a.vis = append(a.vis, *r.VisibleIoU)
			}
			if e := r.Tags["edge"]; e != "" {
				if tel, ok := r.Edges[e]; ok && !tel.Missing {
					a.edgeH = append(a.edgeH, tel.NormalizedEntropy)
				}
			}
			a.maxH = append(a.maxH, r.MaxEdgeEntropy)
		}
	}
	keys := make([][2]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i][0] != keys[j][0] {
			return keys[i][0] < keys[j][0]
		}
		ai, ei := strconv.Atoi(keys[i][1])
		aj, ej := strconv.Atoi(keys[j][1])
		if ei == nil && ej == nil {
			return ai < aj
		}
		return keys[i][1] < keys[j][1]
	})
	var out []BboxTagBreakdown
	for i, k := range keys {
		a := groups[k]
		b := BboxTagBreakdown{Tag: k[0], Value: k[1], Cases: a.runs, BoxCases: len(a.ids), MeanMaxEdgeHNorm: mean(a.maxH)}
		var perCase []float64
		for _, id := range a.ids {
			perCase = append(perCase, mean(a.iou[id]))
		}
		b.MeanExpectationIoU = bootstrapMean(perCase, int64(100+i))
		if a.boxRuns > 0 {
			b.AccAt50Pct = 100 * float64(a.hits50) / float64(a.boxRuns)
			b.CenterHitPct = 100 * float64(a.centerHits) / float64(a.boxRuns)
		}
		if a.presAsked > 0 {
			b.PresentRate = float64(a.pres) / float64(a.presAsked)
		}
		if len(a.vis) > 0 {
			v := mean(a.vis)
			b.MeanVisibleIoU = &v
		}
		if len(a.edgeH) > 0 {
			v := mean(a.edgeH)
			b.TaggedEdgeHNorm = &v
		}
		if repeatsOf := a.runs; repeatsOf > 0 {
			b.Cases = len(a.ids)
			if b.Cases == 0 {
				b.Cases = a.runs
			}
		}
		out = append(out, b)
	}
	return out
}

func boxCenterIn(pred, gt [4]float64) bool {
	cy, cx := (pred[0]+pred[2])/2, (pred[1]+pred[3])/2
	return cy >= gt[0] && cy <= gt[2] && cx >= gt[1] && cx <= gt[3]
}

// centerHit: the model answered present, returned a complete box, and its centre is inside the truth.
func centerHit(r BboxCaseResult) bool {
	return r.ObjectPresentPred && len(r.MissingSlots) == 0 && boxCenterIn(r.ExpectationBox, r.GTBox)
}
