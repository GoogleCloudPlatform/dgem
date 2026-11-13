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
	"sync"
	"testing"
)

func TestNormalizedOrdinalVariance(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		probs     []float64
		want      float64
		tolerance float64
	}{
		{
			name:      "empty_slice",
			probs:     nil,
			want:      0.0,
			tolerance: 1e-12,
		},
		{
			name:      "singleton_slice",
			probs:     []float64{1.0},
			want:      0.0,
			tolerance: 1e-12,
		},
		{
			name:      "point_mass_center_5pt",
			probs:     []float64{0.0, 0.0, 1.0, 0.0, 0.0},
			want:      0.0,
			tolerance: 1e-12,
		},
		{
			name:      "bimodal_extremes_5pt_reaches_popoviciu_bound",
			probs:     []float64{0.5, 0.0, 0.0, 0.0, 0.5},
			want:      1.0,
			tolerance: 1e-12,
		},
		{
			name:      "bimodal_extremes_3pt_scale_invariant",
			probs:     []float64{0.5, 0.0, 0.5},
			want:      1.0,
			tolerance: 1e-12,
		},
		{
			name:      "bimodal_extremes_10pt_scale_invariant",
			probs:     []float64{0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5},
			want:      1.0,
			tolerance: 1e-12,
		},
		{
			name:      "benign_adjacent_split_5pt_is_16x_lower_than_bimodal",
			probs:     []float64{0.0, 0.5, 0.5, 0.0, 0.0},
			want:      0.0625,
			tolerance: 1e-12,
		},
		{
			name:      "uniform_5pt_equals_half_popoviciu",
			probs:     []float64{0.2, 0.2, 0.2, 0.2, 0.2},
			want:      0.5,
			tolerance: 1e-12,
		},
		{
			name:      "unnormalized_weights_normalize_cleanly",
			probs:     []float64{2.0, 0.0, 0.0, 0.0, 2.0},
			want:      1.0,
			tolerance: 1e-12,
		},
		{
			name:      "all_zero_mass_returns_zero",
			probs:     []float64{0.0, 0.0, 0.0},
			want:      0.0,
			tolerance: 1e-12,
		},
		{
			name:      "negative_entry_fails_closed_to_zero",
			probs:     []float64{0.6, -0.1, 0.5},
			want:      0.0,
			tolerance: 1e-12,
		},
		{
			name:      "nan_entry_fails_closed_to_zero",
			probs:     []float64{0.5, math.NaN(), 0.5},
			want:      0.0,
			tolerance: 1e-12,
		},
		{
			name:      "inf_entry_fails_closed_to_zero",
			probs:     []float64{0.5, math.Inf(1), 0.5},
			want:      0.0,
			tolerance: 1e-12,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := NormalizedOrdinalVariance(tc.probs)
			if math.Abs(got-tc.want) > tc.tolerance {
				t.Fatalf("NormalizedOrdinalVariance(%v) = %.12f, want %.12f (tol %g)", tc.probs, got, tc.want, tc.tolerance)
			}
			if got < 0.0 || got > 1.0 {
				t.Fatalf("NormalizedOrdinalVariance(%v) = %v outside [0, 1]", tc.probs, got)
			}
		})
	}
}

func TestParsePackMultiplier(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		title string
		want  int
	}{
		{
			name:  "empty_title",
			title: "",
			want:  1,
		},
		{
			name:  "compound_space_2_x_500ml",
			title: "Mineral Sparkling Water 2 x 500ml",
			want:  2,
		},
		{
			name:  "compound_tight_6x330ml",
			title: "Classic Cola 6x330ml Cans",
			want:  6,
		},
		{
			name:  "compound_unicode_times_12x16oz",
			title: "Cold Brew Coffee 12×16 fl oz",
			want:  12,
		},
		{
			name:  "reverse_compound_500ml_x_4",
			title: "Isotonic Sports Drink 500ml x 4",
			want:  4,
		},
		{
			name:  "hyphenated_6_pack_before_unit_volume",
			title: "Organic Oat Barista Milk 6-pack 1L",
			want:  6,
		},
		{
			name:  "pack_of_24",
			title: "Alkaline AA Batteries, Pack of 24",
			want:  24,
		},
		{
			name:  "case_of_12_500ml",
			title: "Spring Water 500ml (Case of 12)",
			want:  12,
		},
		{
			name:  "espresso_pods_count",
			title: "Dark Roast Espresso 36 Pods",
			want:  36,
		},
		{
			name:  "twin_pack_named",
			title: "Sensitive Toothpaste 75ml Twin-Pack",
			want:  2,
		},
		{
			name:  "triple_pack_named",
			title: "Dishwasher Rinse Aid 250ml Triple Pack",
			want:  3,
		},
		{
			name:  "bare_unit_volume_500ml_stays_single",
			title: "Mineral Sparkling Water 500ml Bottle",
			want:  1,
		},
		{
			name:  "bare_unit_weight_250g_stays_single",
			title: "Whole Bean Espresso 250g Bag",
			want:  1,
		},
		{
			name:  "bare_decimal_liter_1_5l_stays_single",
			title: "Still Spring Water 1.5L",
			want:  1,
		},
		{
			name:  "screen_resolution_not_mistaken_for_pack",
			title: "27-Inch IPS Monitor 1920x1080 144Hz",
			want:  1,
		},
		{
			name:  "4x4_vehicle_not_mistaken_for_pack",
			title: "Remote Control 4x4 Offroad Truck",
			want:  1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ParsePackMultiplier(tc.title)
			if got != tc.want {
				t.Fatalf("ParsePackMultiplier(%q) = %d, want %d", tc.title, got, tc.want)
			}
		})
	}
}

func TestOrdinalAndPackConcurrentSafety(t *testing.T) {
	t.Parallel()

	var wg sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for iter := 0; iter < 100; iter++ {
				gotVar := NormalizedOrdinalVariance([]float64{0.5, 0.0, 0.0, 0.0, 0.5})
				if math.Abs(gotVar-1.0) > 1e-12 {
					t.Errorf("worker %d got NormalizedOrdinalVariance = %v, want 1.0", id, gotVar)
				}
				gotPack := ParsePackMultiplier("Sparkling Water 2 x 500ml")
				if gotPack != 2 {
					t.Errorf("worker %d got ParsePackMultiplier = %d, want 2", id, gotPack)
				}
			}
		}(worker)
	}
	wg.Wait()
}
