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

import { createHmac } from "node:crypto";
import {
  type ConformalCalibrationResult,
  type ConformalConfig,
  type ConformalMethod,
  type ConformalSetResult,
  DGemValidationError,
  type QuestionAnswer,
  classifyHesitationBand,
} from "./types.js";

export const DEFAULT_COSINE_PRIOR_TEMPERATURE = 0.07;
export const DEFAULT_CONFORMAL_SEED_KEY = "dgem-conformal-v1";
export const CONFORMAL_SCORE_TOLERANCE = 1e-12;
export const MAX_TEMPERATURE = 20.0;

export const DEFAULT_ABSTAIN_LABELS: readonly string[] = [
  "0",
  "abstain",
  "insufficient_evidence",
  "none",
  "none_of_the_above",
  "none_visible",
  "oos",
  "out_of_scope",
  "uncertain",
  "unknown",
];

interface OptionProbability {
  readonly name: string;
  readonly prob: number;
}

function normalizeProbabilities(
  probabilities: Record<string, number>,
): OptionProbability[] {
  const keys = Object.keys(probabilities).sort();
  if (keys.length === 0) {
    throw new DGemValidationError("probability map must be non-empty");
  }
  let totalMass = 0.0;
  const rawItems: OptionProbability[] = [];
  for (const key of keys) {
    const val = probabilities[key];
    if (typeof val !== "number" || !Number.isFinite(val) || val < 0.0) {
      throw new DGemValidationError(
        `probability for ${JSON.stringify(key)} must be finite and >= 0`,
      );
    }
    rawItems.push({ name: key, prob: val });
    totalMass += val;
  }
  if (!Number.isFinite(totalMass) || totalMass <= 0.0) {
    throw new DGemValidationError("probability map total mass must be > 0");
  }
  const normalized = rawItems.map((item) => ({
    name: item.name,
    prob: item.prob / totalMass,
  }));
  normalized.sort((a, b) => {
    if (b.prob !== a.prob) {
      return b.prob - a.prob;
    }
    return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
  });
  return normalized;
}

export function shannonEntropy(probabilities: Record<string, number>): number {
  const ordered = normalizeProbabilities(probabilities);
  let entropyNats = 0.0;
  for (const item of ordered) {
    if (item.prob > 0.0) {
      entropyNats -= item.prob * Math.log(item.prob);
    }
  }
  return Math.max(0.0, entropyNats);
}

export function normalizedEntropy(
  probabilities: Record<string, number>,
): number {
  const ordered = normalizeProbabilities(probabilities);
  if (ordered.length <= 1) {
    return 0.0;
  }
  let entropyNats = 0.0;
  for (const item of ordered) {
    if (item.prob > 0.0) {
      entropyNats -= item.prob * Math.log(item.prob);
    }
  }
  return Math.max(0.0, Math.min(1.0, entropyNats / Math.log(ordered.length)));
}

export function applyTemperature(
  probabilities: Record<string, number>,
  temperature: number,
): Record<string, number> {
  if (
    typeof temperature !== "number" ||
    !Number.isFinite(temperature) ||
    temperature <= 0.0 ||
    temperature > MAX_TEMPERATURE
  ) {
    throw new DGemValidationError(
      `temperature must lie in (0, ${MAX_TEMPERATURE}] (got ${String(temperature)})`,
    );
  }
  const ordered = normalizeProbabilities(probabilities);
  if (Math.abs(temperature - 1.0) <= 1e-12) {
    const copy: Record<string, number> = {};
    for (const item of ordered) {
      copy[item.name] = item.prob;
    }
    return copy;
  }

  let maxLogit = -Infinity;
  const scaled: Array<{ name: string; logit: number }> = [];
  for (const item of ordered) {
    const logit = item.prob <= 0.0 ? -1e9 : Math.log(item.prob) / temperature;
    scaled.push({ name: item.name, logit });
    if (logit > maxLogit) {
      maxLogit = logit;
    }
  }
  let normalizer = 0.0;
  const expScores: Array<{ name: string; score: number }> = [];
  for (const item of scaled) {
    const score = Math.exp(item.logit - maxLogit);
    expScores.push({ name: item.name, score });
    normalizer += score;
  }
  const out: Record<string, number> = {};
  for (const item of expScores) {
    out[item.name] = item.score / normalizer;
  }
  return out;
}

export function applyAnswerTemperature(
  answer: QuestionAnswer,
  temperature: number,
): QuestionAnswer {
  const probKeys = Object.keys(answer.probabilities);
  if (probKeys.length === 0) {
    return answer;
  }
  const scaledProbs = applyTemperature(answer.probabilities, temperature);
  const newEntropy = shannonEntropy(scaledProbs);
  const newHesitation = normalizedEntropy(scaledProbs);

  let winnerKey = probKeys[0] ?? "";
  let winnerProb = -1.0;
  for (const [key, prob] of Object.entries(scaledProbs)) {
    if (prob > winnerProb || (prob === winnerProb && key > winnerKey)) {
      winnerProb = prob;
      winnerKey = key;
    }
  }
  const selectedKey = answer.choice || answer.label;
  const newConfidence = scaledProbs[selectedKey] ?? winnerProb;

  let newNoul = answer.noul;
  if (answer.displayType === "boolean") {
    for (const alias of ["yes", "true", "1"]) {
      if (scaledProbs[alias] !== undefined) {
        newNoul = scaledProbs[alias] ?? newNoul;
        break;
      }
    }
  }

  let newScore = answer.score;
  if (answer.displayType === "score") {
    let expected = 0.0;
    let allNumeric = true;
    for (const [key, prob] of Object.entries(scaledProbs)) {
      const numeric = Number(key);
      if (!Number.isFinite(numeric)) {
        allNumeric = false;
        break;
      }
      expected += numeric * prob;
    }
    if (allNumeric) {
      newScore = expected;
    }
  }

  return {
    ...answer,
    probabilities: scaledProbs,
    confidence: newConfidence,
    entropy: newEntropy,
    hesitation: newHesitation,
    hesitationBand: classifyHesitationBand(newHesitation),
    noul: newNoul,
    score: newScore,
  };
}

export function normalizeCosinePriorLogits(
  cosineScores: Record<string, number>,
  temperature = DEFAULT_COSINE_PRIOR_TEMPERATURE,
): Record<string, number> {
  const entries = Object.entries(cosineScores);
  if (entries.length === 0) {
    return {};
  }
  const tau =
    typeof temperature === "number" &&
    Number.isFinite(temperature) &&
    temperature > 0.0
      ? temperature
      : DEFAULT_COSINE_PRIOR_TEMPERATURE;

  const scaled: Record<string, number> = {};
  let maxScaled = -Infinity;
  for (const [key, rawScore] of entries) {
    if (typeof rawScore !== "number" || !Number.isFinite(rawScore)) {
      continue;
    }
    const val = rawScore / tau;
    scaled[key] = val;
    if (val > maxScaled) {
      maxScaled = val;
    }
  }
  const validKeys = Object.keys(scaled);
  if (validKeys.length === 0) {
    return {};
  }
  let sumExp = 0.0;
  for (const key of validKeys) {
    sumExp += Math.exp((scaled[key] ?? 0.0) - maxScaled);
  }
  const logPartition = maxScaled + Math.log(sumExp);
  const out: Record<string, number> = {};
  for (const key of validKeys) {
    out[key] = (scaled[key] ?? 0.0) - logPartition;
  }
  return out;
}

export function fusePriorProbabilities(
  probabilities: Record<string, number>,
  priorLogits: Record<string, number>,
  priorWeight = 0.2,
): Record<string, number> {
  if (
    typeof priorWeight !== "number" ||
    !Number.isFinite(priorWeight) ||
    priorWeight < 0.0
  ) {
    throw new DGemValidationError(
      `priorWeight must be finite and >= 0 (got ${String(priorWeight)})`,
    );
  }
  const ordered = normalizeProbabilities(probabilities);
  if (priorWeight === 0.0 || Object.keys(priorLogits).length === 0) {
    const unchanged: Record<string, number> = {};
    for (const item of ordered) {
      unchanged[item.name] = item.prob;
    }
    return unchanged;
  }

  let maxLogit = -Infinity;
  const combined: Array<{ name: string; logit: number }> = [];
  for (const item of ordered) {
    const baseLog = item.prob <= 0.0 ? -30.0 : Math.log(item.prob);
    const rawPrior = priorLogits[item.name];
    const priorVal =
      typeof rawPrior === "number" && Number.isFinite(rawPrior) ? rawPrior : 0.0;
    const logit = baseLog + priorWeight * priorVal;
    combined.push({ name: item.name, logit });
    if (logit > maxLogit) {
      maxLogit = logit;
    }
  }
  let normalizer = 0.0;
  const expItems: Array<{ name: string; score: number }> = [];
  for (const item of combined) {
    const score = Math.exp(item.logit - maxLogit);
    expItems.push({ name: item.name, score });
    normalizer += score;
  }
  const out: Record<string, number> = {};
  for (const item of expItems) {
    out[item.name] = item.score / normalizer;
  }
  return out;
}

export function hmacUint64(
  seedKey: string,
  itemKey: string,
  tag: string,
): bigint {
  const effectiveKey = seedKey || DEFAULT_CONFORMAL_SEED_KEY;
  const mac = createHmac("sha256", Buffer.from(effectiveKey, "utf8"));
  mac.update(Buffer.from(itemKey, "utf8"));
  mac.update(Buffer.from([0]));
  mac.update(Buffer.from(tag, "utf8"));
  const digest = mac.digest();
  return digest.readBigUInt64BE(0);
}

export function hmacUniformFloat64(
  seedKey: string,
  itemKey: string,
  tag: string,
): number {
  const top53 = hmacUint64(seedKey, itemKey, tag) >> 11n;
  return Number(top53) * (1.0 / 9007199254740992.0);
}

function validateConformalConfig(config: ConformalConfig): {
  method: ConformalMethod;
  kReg: number;
  lambdaReg: number;
  randomized: boolean;
  seedKey: string;
  itemKey: string;
} {
  const rawMethod = (config.method ?? "raps").trim().toLowerCase();
  let method: ConformalMethod;
  if (rawMethod === "" || rawMethod === "raps") {
    method = "raps";
  } else if (rawMethod === "lac" || rawMethod === "aps") {
    method = rawMethod;
  } else {
    throw new DGemValidationError(
      `unknown conformal method: ${JSON.stringify(config.method)}`,
    );
  }

  const kReg = config.kReg ?? 1;
  const lambdaReg = config.lambdaReg ?? 0.01;
  if (!Number.isInteger(kReg) || kReg < 0) {
    throw new DGemValidationError(`kReg must be an integer >= 0 (got ${kReg})`);
  }
  if (!Number.isFinite(lambdaReg) || lambdaReg < 0.0) {
    throw new DGemValidationError(
      `lambdaReg must be finite and >= 0 (got ${lambdaReg})`,
    );
  }
  return {
    method,
    kReg,
    lambdaReg,
    randomized: config.randomized === true,
    seedKey: config.seedKey ?? DEFAULT_CONFORMAL_SEED_KEY,
    itemKey: config.itemKey ?? "",
  };
}

function partitionTieGroups(
  ordered: readonly OptionProbability[],
): OptionProbability[][] {
  const groups: OptionProbability[][] = [];
  let index = 0;
  while (index < ordered.length) {
    const anchor = ordered[index];
    if (anchor === undefined) {
      break;
    }
    let end = index + 1;
    while (end < ordered.length) {
      const candidate = ordered[end];
      if (
        candidate === undefined ||
        Math.abs(candidate.prob - anchor.prob) > CONFORMAL_SCORE_TOLERANCE
      ) {
        break;
      }
      end += 1;
    }
    groups.push(ordered.slice(index, end));
    index = end;
  }
  return groups;
}

function computeScoresForItem(
  ordered: readonly OptionProbability[],
  method: ConformalMethod,
  kReg: number,
  lambdaReg: number,
  randomized: boolean,
  seedKey: string,
  itemKey: string,
): Record<string, number> {
  if (!randomized) {
    if (method === "lac") {
      const lacScores: Record<string, number> = {};
      for (const item of ordered) {
        lacScores[item.name] = 1.0 - item.prob;
      }
      return lacScores;
    }
    const scores: Record<string, number> = {};
    let cumStrictlyGreater = 0.0;
    let strictCount = 0;
    for (const group of partitionTieGroups(ordered)) {
      const groupStartRank = strictCount + 1;
      let penalty = 0.0;
      if (method === "raps" && groupStartRank > kReg) {
        penalty = lambdaReg * (groupStartRank - kReg);
      }
      const groupScore = cumStrictlyGreater + penalty;
      let groupMass = 0.0;
      for (const item of group) {
        scores[item.name] = groupScore;
        groupMass += item.prob;
      }
      cumStrictlyGreater += groupMass;
      strictCount += group.length;
    }
    return scores;
  }

  const uniformDraw = hmacUniformFloat64(seedKey, itemKey, "u");
  const flattened: OptionProbability[] = [];
  for (const group of partitionTieGroups(ordered)) {
    if (group.length <= 1) {
      flattened.push(...group);
      continue;
    }
    const decorated = group.map((item) => ({
      item,
      digest: hmacUint64(seedKey, itemKey, `tie:${item.name}`),
    }));
    decorated.sort((a, b) => {
      if (a.digest < b.digest) {
        return -1;
      }
      if (a.digest > b.digest) {
        return 1;
      }
      return a.item.name < b.item.name
        ? -1
        : a.item.name > b.item.name
          ? 1
          : 0;
    });
    for (const entry of decorated) {
      flattened.push(entry.item);
    }
  }

  const randScores: Record<string, number> = {};
  let cumBefore = 0.0;
  for (let idx = 0; idx < flattened.length; idx += 1) {
    const item = flattened[idx];
    if (item === undefined) {
      continue;
    }
    const rankOne = idx + 1;
    if (method === "lac") {
      randScores[item.name] = 1.0 - item.prob;
    } else if (method === "aps") {
      randScores[item.name] = cumBefore + uniformDraw * item.prob;
    } else {
      const penalty = rankOne > kReg ? lambdaReg * (rankOne - kReg) : 0.0;
      randScores[item.name] = cumBefore + uniformDraw * item.prob + penalty;
    }
    cumBefore += item.prob;
  }
  return randScores;
}

export function resolveGoldProbabilityKey(
  probabilities: Record<string, number>,
  expectedLabel: string,
): string | null {
  const keys = Object.keys(probabilities);
  if (keys.length === 0) {
    return null;
  }
  if (Object.prototype.hasOwnProperty.call(probabilities, expectedLabel)) {
    return expectedLabel;
  }
  const trimmed = expectedLabel.trim();
  if (Object.prototype.hasOwnProperty.call(probabilities, trimmed)) {
    return trimmed;
  }
  const lowered = trimmed.toLowerCase();
  for (const key of keys) {
    if (key.trim().toLowerCase() === lowered) {
      return key;
    }
  }
  if (lowered === "true" || lowered === "yes" || lowered === "1") {
    for (const alias of ["yes", "true", "1"]) {
      for (const key of keys) {
        if (key.trim().toLowerCase() === alias) {
          return key;
        }
      }
    }
  } else if (lowered === "false" || lowered === "no" || lowered === "0") {
    for (const alias of ["no", "false", "0"]) {
      for (const key of keys) {
        if (key.trim().toLowerCase() === alias) {
          return key;
        }
      }
    }
  }
  const expectedNum = Number(trimmed);
  if (!Number.isFinite(expectedNum) || trimmed === "") {
    return null;
  }
  for (const key of keys) {
    const keyNum = Number(key.trim());
    if (Number.isFinite(keyNum) && Math.abs(expectedNum - keyNum) <= 1e-9) {
      return key;
    }
  }
  return null;
}

export function calibrateConformalQuantile(
  calibrationProbabilities: readonly Record<string, number>[],
  calibrationLabels: readonly string[],
  alpha = 0.10,
  config: ConformalConfig = {},
  itemKeys?: readonly string[],
): ConformalCalibrationResult {
  if (calibrationProbabilities.length === 0) {
    throw new DGemValidationError("calibrationProbabilities must be non-empty");
  }
  if (calibrationProbabilities.length !== calibrationLabels.length) {
    throw new DGemValidationError(
      "calibrationProbabilities and calibrationLabels length mismatch",
    );
  }
  if (itemKeys !== undefined && itemKeys.length !== calibrationProbabilities.length) {
    throw new DGemValidationError("itemKeys length must match calibration rows");
  }
  if (!Number.isFinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
    throw new DGemValidationError(
      `alpha must lie strictly in (0, 1) (got ${alpha})`,
    );
  }

  const validated = validateConformalConfig(config);
  const numItems = calibrationProbabilities.length;
  const rawScores: number[] = [];

  for (let idx = 0; idx < numItems; idx += 1) {
    const probs = calibrationProbabilities[idx];
    const goldLabel = calibrationLabels[idx];
    if (probs === undefined || goldLabel === undefined) {
      throw new DGemValidationError(`missing calibration entry at index ${idx}`);
    }
    const ordered = normalizeProbabilities(probs);
    const resolved = resolveGoldProbabilityKey(probs, goldLabel);
    if (resolved === null) {
      throw new DGemValidationError(
        `gold label ${JSON.stringify(goldLabel)} missing at index ${idx}`,
      );
    }
    const candidateKey = itemKeys?.[idx];
    const key = candidateKey ? candidateKey : `cal:${idx}`;
    const scoreMap = computeScoresForItem(
      ordered,
      validated.method,
      validated.kReg,
      validated.lambdaReg,
      validated.randomized,
      validated.seedKey,
      key,
    );
    rawScores.push(scoreMap[resolved] ?? 1.0);
  }

  const sortedScores = [...rawScores].sort((a, b) => a - b);
  let orderIndex = Math.max(
    1,
    Math.ceil((numItems + 1) * (1.0 - alpha) - CONFORMAL_SCORE_TOLERANCE),
  );
  let qHat: number;
  if (orderIndex > numItems) {
    orderIndex = numItems + 1;
    qHat = Infinity;
  } else {
    qHat = sortedScores[orderIndex - 1] ?? Infinity;
  }

  let covered = 0;
  for (const score of rawScores) {
    if (!Number.isFinite(qHat) || score <= qHat + CONFORMAL_SCORE_TOLERANCE) {
      covered += 1;
    }
  }
  return {
    method: validated.method,
    alpha,
    numCalibration: numItems,
    orderIndex,
    qHat,
    empiricalCoverage: covered / numItems,
    sortedScores,
  };
}

export function conformalPredictionSet(
  probabilities: Record<string, number>,
  qHat: number,
  config: ConformalConfig = {},
): ConformalSetResult {
  const validated = validateConformalConfig(config);
  if (Number.isNaN(qHat) || qHat === -Infinity || qHat < 0.0) {
    throw new DGemValidationError(`qHat must be >= 0 and non-NaN (got ${qHat})`);
  }

  const ordered = normalizeProbabilities(probabilities);
  const scores = computeScoresForItem(
    ordered,
    validated.method,
    validated.kReg,
    validated.lambdaReg,
    validated.randomized,
    validated.seedKey,
    validated.itemKey,
  );

  const abstainSource = config.abstainLabels ?? DEFAULT_ABSTAIN_LABELS;
  const abstainLookup = new Set<string>();
  for (const label of abstainSource) {
    const trimmed = label.trim().toLowerCase();
    if (trimmed !== "") {
      abstainLookup.add(trimmed);
    }
  }

  const predictionSet: string[] = [];
  let containsAbstain = false;
  for (const item of ordered) {
    const itemScore = scores[item.name] ?? Infinity;
    if (!Number.isFinite(qHat) || itemScore <= qHat + CONFORMAL_SCORE_TOLERANCE) {
      predictionSet.push(item.name);
      if (abstainLookup.has(item.name.trim().toLowerCase())) {
        containsAbstain = true;
      }
    }
  }

  const topEntry = ordered[0];
  const topOption = topEntry?.name ?? "";
  const topProbability = topEntry?.prob ?? 0.0;
  const singleton = predictionSet.length === 1;

  return {
    predictionSet,
    scores,
    qHat,
    topOption,
    topProbability,
    setSize: predictionSet.length,
    singleton,
    containsAbstain,
    shouldEscalate: !singleton || containsAbstain,
  };
}
