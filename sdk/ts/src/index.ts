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

export {
  CLEAR_HESITATION_CEILING,
  SOMEWHAT_UNSURE_CEILING,
  type ConformalCalibrationResult,
  type ConformalConfig,
  type ConformalMethod,
  type ConformalSetResult,
  DGemError,
  DGemHTTPError,
  DGemTransportError,
  DGemValidationError,
  type GatewayDecideRequest,
  type GatewayDecideResponse,
  type HesitationBand,
  type QuestionAnswer,
  type StructuredDecisionResponse,
  type SystemOneQuestion,
  type SystemOneRequest,
  classifyHesitationBand,
  parseGatewayDecideResponse,
  parseQuestionAnswer,
  parseStructuredDecisionResponse,
} from "./types.js";

export {
  CONFORMAL_SCORE_TOLERANCE,
  DEFAULT_ABSTAIN_LABELS,
  DEFAULT_CONFORMAL_SEED_KEY,
  DEFAULT_COSINE_PRIOR_TEMPERATURE,
  MAX_TEMPERATURE,
  applyAnswerTemperature,
  applyTemperature,
  calibrateConformalQuantile,
  conformalPredictionSet,
  fusePriorProbabilities,
  hmacUint64,
  hmacUniformFloat64,
  normalizeCosinePriorLogits,
  normalizedEntropy,
  resolveGoldProbabilityKey,
  shannonEntropy,
} from "./calibration.js";

export {
  DGemClient,
  type DGemClientOptions,
  type TokenProvider,
  parseUrlOrigin,
} from "./client.js";

export {
  type GenkitDecisionStepOptions,
  type GenkitDecisionStepResult,
  type GenkitToolGuardOptions,
  type GenkitToolGuardVerdict,
  type GuardDecisionOutcome,
  createGenkitDecisionStep,
  createGenkitToolGuard,
} from "./genkit.js";
