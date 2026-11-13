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

import {
  applyAnswerTemperature,
  conformalPredictionSet,
} from "./calibration.js";
import type { DGemClient } from "./client.js";
import {
  CLEAR_HESITATION_CEILING,
  type ConformalConfig,
  DGemError,
  type QuestionAnswer,
} from "./types.js";

export type GuardDecisionOutcome = "allow" | "deny" | "ask_human";

export interface GenkitToolGuardVerdict {
  readonly allowed: boolean;
  readonly decision: GuardDecisionOutcome;
  readonly reason: string;
  readonly hesitation: number;
  readonly predictionSet?: readonly string[];
}

export interface GenkitToolGuardOptions {
  readonly template?: string;
  readonly hazardSlot?: string;
  readonly originSlot?: string;
  readonly denyHazards?: readonly string[];
  readonly denyOrigins?: readonly string[];
  readonly hesitationCeiling?: number;
  readonly temperature?: number;
  readonly conformalQHat?: number;
  readonly conformalConfig?: ConformalConfig;
  readonly failClosed?: boolean;
}

export interface GenkitDecisionStepOptions {
  readonly template: string;
  readonly routeQuestionId: string;
  readonly hesitationCeiling?: number;
  readonly escalateRoute?: string;
  readonly temperature?: number;
}

export interface GenkitDecisionStepResult {
  readonly route: string;
  readonly hesitation: number;
  readonly escalated: boolean;
  readonly answers: Record<string, QuestionAnswer>;
  readonly wallTimeMs: number;
}

function sanitizeUntrustedText(rawText: string): string {
  return rawText.replaceAll("</untrusted_data>", "<\\/untrusted_data>");
}

export function createGenkitToolGuard(
  client: DGemClient,
  options: GenkitToolGuardOptions = {},
): (
  toolName: string,
  toolArgs: Record<string, unknown>,
) => Promise<GenkitToolGuardVerdict> {
  const template = options.template ?? "tool_safety_gate";
  const hazardSlot = options.hazardSlot ?? "action_hazard";
  const originSlot = options.originSlot ?? "intent_origin";
  const blockedHazards = new Set(
    (options.denyHazards ?? ["destructive", "credential_exfiltration"]).map(
      (item) => item.trim().toLowerCase(),
    ),
  );
  const blockedOrigins = new Set(
    (options.denyOrigins ?? ["untrusted_prompt_injection"]).map((item) =>
      item.trim().toLowerCase(),
    ),
  );
  const hesitationCeiling =
    options.hesitationCeiling ?? CLEAR_HESITATION_CEILING;
  const failClosed = options.failClosed ?? true;

  return async (
    toolName: string,
    toolArgs: Record<string, unknown>,
  ): Promise<GenkitToolGuardVerdict> => {
    const serializedArgs = sanitizeUntrustedText(JSON.stringify(toolArgs));
    let responseAnswers: Record<string, QuestionAnswer>;
    try {
      const response = await client.decide({
        template,
        variables: {
          tool_name: toolName,
          tool_args: `<untrusted_data>${serializedArgs}</untrusted_data>`,
        },
      });
      responseAnswers = { ...response.answers };
    } catch (err) {
      if (err instanceof DGemError && !failClosed) {
        return {
          allowed: true,
          decision: "allow",
          reason: "dgem unavailable (failClosed=false)",
          hesitation: 0.0,
        };
      }
      return {
        allowed: false,
        decision: "ask_human",
        reason: `dgem safety gate unavailable: ${String(err)}`,
        hesitation: 1.0,
      };
    }

    if (options.temperature !== undefined) {
      const rescaled: Record<string, QuestionAnswer> = {};
      for (const [qid, ans] of Object.entries(responseAnswers)) {
        rescaled[qid] = applyAnswerTemperature(ans, options.temperature);
      }
      responseAnswers = rescaled;
    }

    const originAns = responseAnswers[originSlot];
    if (originAns !== undefined) {
      const originChoice = String(originAns.value).trim().toLowerCase();
      if (blockedOrigins.has(originChoice)) {
        return {
          allowed: false,
          decision: "deny",
          reason: `blocked intent origin: ${originChoice}`,
          hesitation: originAns.hesitation,
        };
      }
    }

    const hazardAns = responseAnswers[hazardSlot];
    if (hazardAns !== undefined) {
      const hazardChoice = String(hazardAns.value).trim().toLowerCase();
      if (blockedHazards.has(hazardChoice)) {
        return {
          allowed: false,
          decision: "deny",
          reason: `blocked action hazard: ${hazardChoice}`,
          hesitation: hazardAns.hesitation,
        };
      }
      if (
        options.conformalQHat !== undefined &&
        Object.keys(hazardAns.probabilities).length > 0
      ) {
        const conformal = conformalPredictionSet(
          hazardAns.probabilities,
          options.conformalQHat,
          options.conformalConfig,
        );
        if (conformal.shouldEscalate) {
          return {
            allowed: false,
            decision: "ask_human",
            reason: "conformal prediction set requires escalation",
            hesitation: hazardAns.hesitation,
            predictionSet: conformal.predictionSet,
          };
        }
      }
    }

    const answerList = Object.values(responseAnswers);
    const maxHesitation =
      answerList.length === 0
        ? 1.0
        : Math.max(...answerList.map((ans) => ans.hesitation));

    if (maxHesitation >= hesitationCeiling) {
      return {
        allowed: false,
        decision: "ask_human",
        reason: `hesitation ${maxHesitation.toFixed(3)} >= ceiling ${hesitationCeiling.toFixed(3)}`,
        hesitation: maxHesitation,
      };
    }

    return {
      allowed: true,
      decision: "allow",
      reason: "approved",
      hesitation: maxHesitation,
    };
  };
}

export function createGenkitDecisionStep(
  client: DGemClient,
  options: GenkitDecisionStepOptions,
): (variables: Record<string, unknown>) => Promise<GenkitDecisionStepResult> {
  const hesitationCeiling =
    options.hesitationCeiling ?? CLEAR_HESITATION_CEILING;
  const escalateRoute = options.escalateRoute ?? "escalate_stage2";

  return async (
    variables: Record<string, unknown>,
  ): Promise<GenkitDecisionStepResult> => {
    const response = await client.decide({
      template: options.template,
      variables,
    });
    let answers = { ...response.answers };
    if (options.temperature !== undefined) {
      const rescaled: Record<string, QuestionAnswer> = {};
      for (const [qid, ans] of Object.entries(answers)) {
        rescaled[qid] = applyAnswerTemperature(ans, options.temperature);
      }
      answers = rescaled;
    }

    const primary = answers[options.routeQuestionId];
    if (primary === undefined) {
      return {
        route: escalateRoute,
        hesitation: 1.0,
        escalated: true,
        answers,
        wallTimeMs: response.wallTimeMs,
      };
    }
    const escalated = primary.hesitation >= hesitationCeiling;
    return {
      route: escalated ? escalateRoute : String(primary.value),
      hesitation: primary.hesitation,
      escalated,
      answers,
      wallTimeMs: response.wallTimeMs,
    };
  };
}
