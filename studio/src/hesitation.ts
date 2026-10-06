/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Plain-language "hesitation" readout shared by the Studio result panel and the Concepts walkthrough.
//
// Hesitation is cardinality-normalized Shannon entropy, H / ln(K), shown as a 0-100% value so that a
// 2-option and a 26-option question can be compared on the same scale. The 16% "Clear" cut-off matches
// the normalized-entropy cascade gate (H~ >= 0.16 escalates).

export type HesitationBand = 'clear' | 'unsure' | 'very-unsure';

export interface Hesitation {
  /** Normalized entropy in [0, 1]. */
  normalized: number;
  /** Rounded percentage for display. */
  pct: number;
  band: HesitationBand;
  /** Short plain-English label. */
  label: string;
  /** Tooltip text with the raw values. */
  tooltip: string;
}

export const HESITATION_CLEAR_MAX = 0.16;
export const HESITATION_UNSURE_MAX = 0.5;

export function hesitation(entropyNats: number, numOptions: number): Hesitation {
  const k = Math.max(2, Math.floor(numOptions || 2));
  const h = Math.max(0, entropyNats || 0);
  const normalized = Math.min(1, h / Math.log(k));
  const band: HesitationBand =
    normalized < HESITATION_CLEAR_MAX ? 'clear' : normalized < HESITATION_UNSURE_MAX ? 'unsure' : 'very-unsure';
  const label = band === 'clear' ? 'Clear' : band === 'unsure' ? 'Somewhat unsure' : 'Very unsure';
  return {
    normalized,
    pct: Math.round(normalized * 100),
    band,
    label,
    tooltip:
      `Hesitation = Shannon entropy ÷ ln(number of options) = ${h.toFixed(3)} nats ÷ ln(${k}). ` +
      `0% means all probability is on one answer; 100% means a perfect tie. ` +
      `Under ${Math.round(HESITATION_CLEAR_MAX * 100)}% is "Clear", ${Math.round(HESITATION_CLEAR_MAX * 100)}–${Math.round(HESITATION_UNSURE_MAX * 100)}% "Somewhat unsure", above that "Very unsure". ` +
      `Low hesitation is not a guarantee: option order can still hide doubt (see Concepts → When to trust an answer).`,
  };
}
