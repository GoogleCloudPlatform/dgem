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
	"fmt"
	"math"
	"strings"
)

// CandidateMark represents an interactive UI element or accessibility node on a screen,
// annotated with a Set-of-Marks (SoM) identifier (for example, "M1") and a bounding box
// in normalized [0, 100] percentage coordinates [y_min, x_min, y_max, x_max].
type CandidateMark struct {
	ID    string     `json:"id"`
	Label string     `json:"label"`
	Box   [4]float64 `json:"box"` // [y_min, x_min, y_max, x_max] in [0, 100]
}

var (
	gridRowNames = [3]string{"top", "middle", "bottom"}
	gridColNames = [3]string{"left", "center", "right"}
)

// BoxCenter returns the (cy, cx) center of a [y_min, x_min, y_max, x_max] box in [0, 100] coordinates.
func BoxCenter(box [4]float64) (cy, cx float64) {
	return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
}

// BoxArea returns the non-negative area of a [y_min, x_min, y_max, x_max] box.
func BoxArea(box [4]float64) float64 {
	h := math.Max(0, box[2]-box[0])
	w := math.Max(0, box[3]-box[1])
	return h * w
}

// BoxIoU computes the 2D intersection-over-union of two [y_min, x_min, y_max, x_max] boxes.
func BoxIoU(a, b [4]float64) float64 {
	ih := math.Max(0, math.Min(a[2], b[2])-math.Max(a[0], b[0]))
	iw := math.Max(0, math.Min(a[3], b[3])-math.Max(a[1], b[1]))
	inter := ih * iw
	union := BoxArea(a) + BoxArea(b) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// PointInBox reports whether (cy, cx) lies inside or on the boundary of box [y_min, x_min, y_max, x_max].
func PointInBox(cy, cx float64, box [4]float64) bool {
	return cy >= box[0] && cy <= box[2] && cx >= box[1] && cx <= box[3]
}

// AssignPointToGridCell maps a point (cy, cx) in [0, 100] percentage coordinates to one of
// the nine 3x3 grid cells ("top_left" .. "bottom_right"), matching build_bbox_real_aspects.py.
func AssignPointToGridCell(cy, cx float64) string {
	r := int((cy / 100.0) * 3.0)
	if r < 0 {
		r = 0
	} else if r > 2 {
		r = 2
	}
	c := int((cx / 100.0) * 3.0)
	if c < 0 {
		c = 0
	} else if c > 2 {
		c = 2
	}
	return gridRowNames[r] + "_" + gridColNames[c]
}

// AssignBoxToGridCell maps the center of box [y_min, x_min, y_max, x_max] to its 3x3 grid cell.
func AssignBoxToGridCell(box [4]float64) string {
	cy, cx := BoxCenter(box)
	return AssignPointToGridCell(cy, cx)
}

func gridCellIndices(cell string) (row, col int, ok bool) {
	parts := strings.Split(cell, "_")
	if len(parts) != 2 {
		return 0, 0, false
	}
	row = -1
	for i, name := range gridRowNames {
		if parts[0] == name {
			row = i
			break
		}
	}
	col = -1
	for i, name := range gridColNames {
		if parts[1] == name {
			col = i
			break
		}
	}
	if row < 0 || col < 0 {
		return 0, 0, false
	}
	return row, col, true
}

// GridCellBounds returns the [y_min, x_min, y_max, x_max] box in [0, 100] coordinates for a named 3x3 grid cell.
func GridCellBounds(cell string) ([4]float64, bool) {
	r, c, ok := gridCellIndices(cell)
	if !ok {
		return [4]float64{}, false
	}
	step := 100.0 / 3.0
	return [4]float64{
		float64(r) * step,
		float64(c) * step,
		float64(r+1) * step,
		float64(c+1) * step,
	}, true
}

// FilterMarksByGridCell returns the subset of candidate marks whose center lies in cell
// (or within a Manhattan distance of 1 cell when includeAdjacent is true).
func FilterMarksByGridCell(cell string, marks []CandidateMark, includeAdjacent bool) []CandidateMark {
	targetRow, targetCol, ok := gridCellIndices(cell)
	if !ok {
		return nil
	}
	var out []CandidateMark
	for _, m := range marks {
		mCell := AssignBoxToGridCell(m.Box)
		r, c, valid := gridCellIndices(mCell)
		if !valid {
			continue
		}
		if r == targetRow && c == targetCol {
			out = append(out, m)
			continue
		}
		if includeAdjacent {
			dr := r - targetRow
			if dr < 0 {
				dr = -dr
			}
			dc := c - targetCol
			if dc < 0 {
				dc = -dc
			}
			if dr+dc <= 1 {
				out = append(out, m)
			}
		}
	}
	return out
}

// SelectContainingOrNearestMark maps a query point (cy, cx) in [0, 100] coordinates to a CandidateMark.
// If one or more marks contain (cy, cx), it selects the containing mark with the smallest area (breaking ties
// by Euclidean distance to mark center) and returns distance 0. Otherwise it selects the mark with minimum
// Euclidean center distance and returns that distance.
func SelectContainingOrNearestMark(cy, cx float64, marks []CandidateMark) (CandidateMark, float64, bool) {
	if len(marks) == 0 {
		return CandidateMark{}, 0, false
	}
	var (
		bestContaining     CandidateMark
		bestContainingArea = math.Inf(1)
		bestContainingDist = math.Inf(1)
		foundContaining    bool

		bestNearest     CandidateMark
		bestNearestDist = math.Inf(1)
	)
	for _, m := range marks {
		mcy, mcx := BoxCenter(m.Box)
		dist := math.Hypot(cy-mcy, cx-mcx)
		if PointInBox(cy, cx, m.Box) {
			area := BoxArea(m.Box)
			if !foundContaining || area < bestContainingArea || (area == bestContainingArea && dist < bestContainingDist) {
				bestContaining = m
				bestContainingArea = area
				bestContainingDist = dist
				foundContaining = true
			}
		}
		if dist < bestNearestDist {
			bestNearest = m
			bestNearestDist = dist
		}
	}
	if foundContaining {
		return bestContaining, 0, true
	}
	return bestNearest, bestNearestDist, true
}

// AssignPointToGrid4x4 maps a point (cy, cx) in [0, 100] coordinates to one of the 16 cells of a
// 4x4 screen grid ("r0_c0" .. "r3_c3"), providing an unbadged 16-option single-pass spatial baseline.
func AssignPointToGrid4x4(cy, cx float64) string {
	r := int((cy / 100.0) * 4.0)
	if r < 0 {
		r = 0
	} else if r > 3 {
		r = 3
	}
	c := int((cx / 100.0) * 4.0)
	if c < 0 {
		c = 0
	} else if c > 3 {
		c = 3
	}
	return fmt.Sprintf("r%d_c%d", r, c)
}

// AssignBoxToGrid4x4 maps the center of box [y_min, x_min, y_max, x_max] to its 4x4 grid cell ("r0_c0" .. "r3_c3").
func AssignBoxToGrid4x4(box [4]float64) string {
	cy, cx := BoxCenter(box)
	return AssignPointToGrid4x4(cy, cx)
}

// Grid4x4CellBounds returns the [y_min, x_min, y_max, x_max] bounds in [0, 100] coordinates for a "r<row>_c<col>" 4x4 cell.
func Grid4x4CellBounds(cell string) ([4]float64, bool) {
	var r, c int
	var extra string
	if n, err := fmt.Sscanf(cell, "r%d_c%d%s", &r, &c, &extra); err != nil && n != 2 {
		return [4]float64{}, false
	} else if n != 2 || extra != "" {
		return [4]float64{}, false
	}
	if r < 0 || r > 3 || c < 0 || c > 3 {
		return [4]float64{}, false
	}
	const step = 25.0
	return [4]float64{
		float64(r) * step,
		float64(c) * step,
		float64(r+1) * step,
		float64(c+1) * step,
	}, true
}

// ExpectedMarkCenterFromProbabilities computes the probability-weighted spatial center (cy, cx) across visible
// candidate marks and returns visibleMass (the sum of probabilities assigned to candidate marks, excluding "none_visible").
func ExpectedMarkCenterFromProbabilities(probs map[string]float64, marks []CandidateMark) (cy, cx, visibleMass float64) {
	for _, m := range marks {
		p := probs[m.ID]
		if p <= 0 || math.IsNaN(p) || math.IsInf(p, 0) {
			continue
		}
		mcy, mcx := BoxCenter(m.Box)
		cy += p * mcy
		cx += p * mcx
		visibleMass += p
	}
	if visibleMass > 0 {
		cy /= visibleMass
		cx /= visibleMass
	}
	return cy, cx, visibleMass
}

// BuildSetOfMarksQuestions constructs a 4-slot single-pass SystemOne question map for Set-of-Marks (SoM)
// UI navigation ("next_target_mark", "element_state", "loop_or_error_detected", "goal_progress_score").
// To stay within a single unbracketed [A-Z] pass, marks must contain between 1 and MaxOptionsPerSlot-1 (25)
// candidate marks with unique non-empty IDs, leaving one slot for "none_visible".
func BuildSetOfMarksQuestions(goal string, marks []CandidateMark) (map[string]SystemOneQuestion, error) {
	goal = strings.TrimSpace(goal)
	if goal == "" {
		return nil, fmt.Errorf("goal must not be empty")
	}
	if len(marks) == 0 {
		return nil, fmt.Errorf("marks must not be empty")
	}
	if len(marks) > MaxOptionsPerSlot-1 {
		return nil, fmt.Errorf("marks count %d exceeds single-pass Set-of-Marks limit %d", len(marks), MaxOptionsPerSlot-1)
	}

	markCriteria := make(map[string]string, len(marks)+1)
	markOrder := make([]string, 0, len(marks)+1)
	for _, m := range marks {
		id := strings.TrimSpace(m.ID)
		if id == "" {
			return nil, fmt.Errorf("candidate mark ID must not be empty")
		}
		if id == "none_visible" {
			return nil, fmt.Errorf("candidate mark ID %q is reserved", id)
		}
		if _, exists := markCriteria[id]; exists {
			return nil, fmt.Errorf("duplicate candidate mark ID %q", id)
		}
		cell := AssignBoxToGridCell(m.Box)
		label := strings.TrimSpace(m.Label)
		if label == "" {
			label = "interactive UI element"
		}
		markCriteria[id] = fmt.Sprintf("%s (grid_cell=%s, box=[%.1f, %.1f, %.1f, %.1f])",
			label, cell, m.Box[0], m.Box[1], m.Box[2], m.Box[3])
		markOrder = append(markOrder, id)
	}
	markCriteria["none_visible"] = "Target UI control is off-screen, occluded, or requires scrolling before clicking"
	markOrder = append(markOrder, "none_visible")

	questions := map[string]SystemOneQuestion{
		"next_target_mark": {
			Type:         "choice",
			Instructions: fmt.Sprintf("Which numbered Set-of-Marks (SoM) UI element should be activated next to accomplish the goal: %s?", goal),
			Criteria:     markCriteria,
			Order:        markOrder,
		},
		"element_state": {
			Type:         "choice",
			Instructions: "What is the interactive state of the selected target UI element?",
			Criteria: map[string]string{
				"interactive_ready":   "Visible, enabled, and ready for immediate click or tap",
				"disabled_grayed_out": "Visible but disabled or awaiting prerequisite form input",
				"occluded_by_modal":   "Partially or fully blocked by a modal, toast, or cookie banner",
				"requires_scroll":     "Not visible in the current viewport; scrolling is required",
			},
			Order: []string{"interactive_ready", "disabled_grayed_out", "occluded_by_modal", "requires_scroll"},
		},
		"loop_or_error_detected": {
			Type:         "choice",
			Instructions: "Is the screen stuck in a validation error, modal trap, or repeated state?",
			Criteria: map[string]string{
				"no":  "Screen state is normal with no blocking error or loop",
				"yes": "Inline validation error, blocking dialog, or repeated state detected",
			},
			Order: []string{"no", "yes"},
		},
		"goal_progress_score": {
			Type:         "score",
			Instructions: "How close is the current screen state to completing the user goal?",
			Levels: []string{
				"blocked_or_regressed",
				"initial_screen",
				"intermediate_step_complete",
				"final_confirmation_visible",
			},
		},
	}
	return questions, nil
}
