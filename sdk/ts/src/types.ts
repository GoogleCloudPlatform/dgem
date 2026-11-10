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

export const CLEAR_HESITATION_CEILING = 0.16;
export const SOMEWHAT_UNSURE_CEILING = 0.50;

export type HesitationBand = "clear" | "somewhat_unsure" | "very_unsure";
export type ConformalMethod = "lac" | "aps" | "raps";

export class DGemError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DGemError";
  }
}

export class DGemValidationError extends DGemError {
  constructor(message: string) {
    super(message);
    this.name = "DGemValidationError";
  }
}

export class DGemTransportError extends DGemError {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = "DGemTransportError";
    if (options?.cause !== undefined) {
      this.cause = options.cause;
    }
  }
}

export class DGemHTTPError extends DGemError {
  public readonly statusCode: number;
  public readonly responseBody: string;

  constructor(statusCode: number, message: string, responseBody = "") {
    super(`HTTP ${statusCode}: ${message}`);
    this.name = "DGemHTTPError";
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

export interface QuestionAnswer {
  readonly type: string;
  readonly displayType: string;
  readonly label: string;
  readonly value: string | boolean | number;
  readonly confidence: number;
  readonly probabilities: Record<string, number>;
  readonly logprob: number;
  readonly entropy: number;
  readonly hesitation: number;
  readonly hesitationBand: HesitationBand;
  readonly stderr: number;
  readonly agreement: number;
  readonly choice: string;
  readonly score: number;
  readonly level: string;
  readonly noul: number;
  readonly priorLogits: Record<string, number>;
  readonly priorFused: boolean;
  readonly rawEntropy: number;
  readonly evidence: string;
}

export interface StructuredDecisionResponse {
  readonly answers: Record<string, QuestionAnswer>;
  readonly diagnostics: Record<string, unknown>;
  readonly maxHesitation: number;
}

export interface GatewayDecideRequest {
  readonly template?: string;
  readonly customTemplate?: string;
  readonly variables?: Record<string, unknown>;
  readonly images?: readonly string[];
  readonly backend?: string;
  readonly cascadeMode?: string;
  readonly cascadeThreshold?: number;
  readonly cascadeThresholdMode?: "nats" | "normalized" | "";
  readonly cascadeModel?: string;
  readonly layout?: "document_first" | "schema_first" | "";
  readonly temperature?: number | Record<string, number>;
}

export interface GatewayDecideResponse extends StructuredDecisionResponse {
  readonly template: string;
  readonly maxEntropy: number;
  readonly wallTimeMs: number;
  readonly gpuForwardMs: number;
  readonly coldStartWaitMs: number;
  readonly model: string;
  readonly backendTarget: string;
  readonly backendUsed: string;
  readonly traceId: string;
  readonly cascade?: Record<string, unknown>;
}

export interface SystemOneQuestion {
  readonly type: string;
  readonly instructions: string;
  readonly criteria?: Record<string, string> | readonly string[];
  readonly priorLogits?: Record<string, number>;
  readonly evidencePrompt?: string;
  readonly evidenceCriteria?: Record<string, string>;
}

export interface SystemOneRequest {
  readonly context: string | Record<string, unknown>;
  readonly questions: Record<string, SystemOneQuestion>;
  readonly images?: readonly string[];
  readonly instructions?: string;
  readonly samples?: number;
  readonly layout?: "document_first" | "schema_first" | "";
  readonly priorWeight?: number;
  readonly shortlistTopk?: number;
  readonly backend?: string;
}

export interface ConformalConfig {
  readonly method?: ConformalMethod;
  readonly alpha?: number;
  readonly kReg?: number;
  readonly lambdaReg?: number;
  readonly randomized?: boolean;
  readonly seedKey?: string;
  readonly itemKey?: string;
  readonly abstainLabels?: readonly string[];
}

export interface ConformalCalibrationResult {
  readonly method: ConformalMethod;
  readonly alpha: number;
  readonly numCalibration: number;
  readonly orderIndex: number;
  readonly qHat: number;
  readonly empiricalCoverage: number;
  readonly sortedScores: readonly number[];
}

export interface ConformalSetResult {
  readonly predictionSet: readonly string[];
  readonly scores: Record<string, number>;
  readonly qHat: number;
  readonly topOption: string;
  readonly topProbability: number;
  readonly setSize: number;
  readonly singleton: boolean;
  readonly containsAbstain: boolean;
  readonly shouldEscalate: boolean;
}

export function classifyHesitationBand(hesitation: number): HesitationBand {
  if (hesitation < CLEAR_HESITATION_CEILING) {
    return "clear";
  }
  if (hesitation <= SOMEWHAT_UNSURE_CEILING) {
    return "somewhat_unsure";
  }
  return "very_unsure";
}

function computeNormalizedHesitation(
  probabilities: Record<string, number>,
  entropyNats: number,
): number {
  const keys = Object.keys(probabilities);
  if (keys.length <= 1) {
    return 0.0;
  }
  let resolvedEntropy = entropyNats;
  if (resolvedEntropy <= 0.0) {
    resolvedEntropy = 0.0;
    for (const key of keys) {
      const prob = probabilities[key] ?? 0.0;
      if (Number.isFinite(prob) && prob > 0.0) {
        resolvedEntropy -= prob * Math.log(prob);
      }
    }
  }
  const normalized = resolvedEntropy / Math.log(keys.length);
  return Math.max(0.0, Math.min(1.0, normalized));
}

function normalizeDisplayType(rawType: string): string {
  const lowered = rawType.trim().toLowerCase();
  if (lowered === "noul" || lowered === "bool" || lowered === "boolean") {
    return "boolean";
  }
  return lowered;
}

function resolvePrimaryValue(
  displayType: string,
  label: string,
  choice: string,
  score: number,
  level: string,
): string | boolean | number {
  if (displayType === "boolean") {
    const normalized = label.trim().toLowerCase();
    return normalized === "yes" || normalized === "true" || normalized === "1";
  }
  if (displayType === "choice") {
    return choice || label;
  }
  if (displayType === "score") {
    return score;
  }
  return choice || level || label;
}

function readNumberRecord(raw: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (raw !== null && typeof raw === "object" && !Array.isArray(raw)) {
    for (const [key, val] of Object.entries(raw as Record<string, unknown>)) {
      if (typeof val === "number" && Number.isFinite(val)) {
        out[key] = val;
      }
    }
  }
  return out;
}

function readNumber(
  payload: Record<string, unknown>,
  field: string,
  fallback = 0.0,
): number {
  const raw = payload[field];
  return typeof raw === "number" && Number.isFinite(raw) ? raw : fallback;
}

function readString(payload: Record<string, unknown>, field: string): string {
  const raw = payload[field];
  return typeof raw === "string" ? raw : "";
}

export function parseQuestionAnswer(
  payload: Record<string, unknown>,
): QuestionAnswer {
  const rawType = readString(payload, "type");
  const displayType = normalizeDisplayType(rawType);
  const label = readString(payload, "label");
  const choice = readString(payload, "choice");
  const score = readNumber(payload, "score", 0.0);
  const level = readString(payload, "level");
  const probabilities = readNumberRecord(payload["probabilities"]);
  const entropy = readNumber(payload, "entropy", 0.0);
  const hesitation = computeNormalizedHesitation(probabilities, entropy);

  return {
    type: rawType,
    displayType,
    label,
    value: resolvePrimaryValue(displayType, label, choice, score, level),
    confidence: readNumber(payload, "confidence", 0.0),
    probabilities,
    logprob: readNumber(payload, "logprob", 0.0),
    entropy,
    hesitation,
    hesitationBand: classifyHesitationBand(hesitation),
    stderr: readNumber(payload, "stderr", 0.0),
    agreement: readNumber(payload, "agreement", 1.0),
    choice,
    score,
    level,
    noul: readNumber(payload, "noul", 0.0),
    priorLogits: readNumberRecord(payload["prior_logits"]),
    priorFused: payload["prior_fused"] === true,
    rawEntropy: readNumber(payload, "raw_entropy", 0.0),
    evidence: readString(payload, "evidence"),
  };
}

export function parseStructuredDecisionResponse(
  payload: Record<string, unknown>,
): StructuredDecisionResponse {
  const answers: Record<string, QuestionAnswer> = {};
  const rawAnswers = payload["answers"];
  let maxHesitation = 0.0;

  if (
    rawAnswers !== null &&
    typeof rawAnswers === "object" &&
    !Array.isArray(rawAnswers)
  ) {
    for (const [qid, rawItem] of Object.entries(
      rawAnswers as Record<string, unknown>,
    )) {
      if (
        rawItem !== null &&
        typeof rawItem === "object" &&
        !Array.isArray(rawItem)
      ) {
        const parsed = parseQuestionAnswer(rawItem as Record<string, unknown>);
        answers[qid] = parsed;
        if (parsed.hesitation > maxHesitation) {
          maxHesitation = parsed.hesitation;
        }
      }
    }
  }

  const rawDiagnostics = payload["diagnostics"];
  const diagnostics =
    rawDiagnostics !== null &&
    typeof rawDiagnostics === "object" &&
    !Array.isArray(rawDiagnostics)
      ? { ...(rawDiagnostics as Record<string, unknown>) }
      : {};

  return { answers, diagnostics, maxHesitation };
}

export function parseGatewayDecideResponse(
  payload: Record<string, unknown>,
  backendUsedHeader = "",
): GatewayDecideResponse {
  const structured = parseStructuredDecisionResponse(payload);
  const backendTarget = readString(payload, "backend_target");
  const rawCascade = payload["cascade"];
  const cascade =
    rawCascade !== null &&
    typeof rawCascade === "object" &&
    !Array.isArray(rawCascade)
      ? { ...(rawCascade as Record<string, unknown>) }
      : undefined;

  return {
    ...structured,
    template: readString(payload, "template"),
    maxEntropy: readNumber(payload, "max_entropy", 0.0),
    wallTimeMs: Math.trunc(readNumber(payload, "wall_time_ms", 0)),
    gpuForwardMs: Math.trunc(readNumber(payload, "gpu_forward_ms", 0)),
    coldStartWaitMs: Math.trunc(readNumber(payload, "cold_start_wait_ms", 0)),
    model: readString(payload, "model"),
    backendTarget,
    backendUsed: backendUsedHeader || backendTarget,
    traceId: readString(payload, "trace_id"),
    ...(cascade !== undefined ? { cascade } : {}),
  };
}
