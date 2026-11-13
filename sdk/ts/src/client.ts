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
  DGemHTTPError,
  DGemTransportError,
  DGemValidationError,
  type GatewayDecideRequest,
  type GatewayDecideResponse,
  type StructuredDecisionResponse,
  type SystemOneQuestion,
  type SystemOneRequest,
  parseGatewayDecideResponse,
  parseStructuredDecisionResponse,
} from "./types.js";

const CREDENTIAL_HEADER_NAMES = new Set([
  "authorization",
  "x-dgem-key",
  "x-dgem-api-key",
]);

const REDIRECT_STATUS_CODES = new Set([301, 302, 303, 307, 308]);

type TargetRole = "gateway" | "serving";

export type TokenProvider = (
  targetUrl: string,
) => string | undefined | Promise<string | undefined>;

export interface DGemClientOptions {
  readonly baseUrl?: string;
  readonly bearerToken?: string;
  readonly tokenProvider?: TokenProvider;
  readonly gatewayApiKey?: string;
  readonly serverKey?: string;
  readonly defaultBackend?: string;
  readonly timeoutMs?: number;
  readonly maxRedirects?: number;
  readonly fetchImpl?: typeof fetch;
}

export function parseUrlOrigin(rawUrl: string): string {
  if (rawUrl.includes("\r") || rawUrl.includes("\n")) {
    throw new DGemValidationError("URL must not contain CR or LF characters");
  }
  let parsed: URL;
  try {
    parsed = new URL(rawUrl.trim());
  } catch (err) {
    throw new DGemValidationError(`invalid URL ${JSON.stringify(rawUrl)}: ${String(err)}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new DGemValidationError(
      `URL must use http:// or https:// (got ${JSON.stringify(rawUrl)})`,
    );
  }
  if (!parsed.hostname) {
    throw new DGemValidationError(
      `URL must specify a non-empty hostname (got ${JSON.stringify(rawUrl)})`,
    );
  }
  if (parsed.username !== "" || parsed.password !== "") {
    throw new DGemValidationError(
      "URL must not embed credentials in userinfo",
    );
  }
  const defaultPort = parsed.protocol === "https:" ? "443" : "80";
  const effectivePort = parsed.port || defaultPort;
  return `${parsed.protocol}//${parsed.hostname.toLowerCase()}:${effectivePort}`;
}

function stripTrailingSlashes(input: string): string {
  let end = input.length;
  while (end > 0 && input.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return input.slice(0, end);
}

function validateHeaderValue(name: string, value: string): string {
  if (value.includes("\r") || value.includes("\n")) {
    throw new DGemValidationError(
      `header ${JSON.stringify(name)} must not contain CR or LF characters`,
    );
  }
  return value;
}

function stripCredentialHeaders(headers: Record<string, string>): void {
  for (const key of Object.keys(headers)) {
    if (CREDENTIAL_HEADER_NAMES.has(key.toLowerCase())) {
      delete headers[key];
    }
  }
}

function serializeSystemOneQuestion(
  question: SystemOneQuestion,
): Record<string, unknown> {
  const wire: Record<string, unknown> = {
    type: question.type,
    instructions: question.instructions,
  };
  if (question.criteria !== undefined) {
    wire["criteria"] = Array.isArray(question.criteria)
      ? [...question.criteria]
      : { ...question.criteria };
  }
  if (
    question.priorLogits !== undefined &&
    Object.keys(question.priorLogits).length > 0
  ) {
    wire["prior_logits"] = { ...question.priorLogits };
  }
  if (question.evidencePrompt) {
    wire["evidence_prompt"] = question.evidencePrompt;
  }
  if (
    question.evidenceCriteria !== undefined &&
    Object.keys(question.evidenceCriteria).length > 0
  ) {
    wire["evidence_criteria"] = { ...question.evidenceCriteria };
  }
  return wire;
}

export class DGemClient {
  public readonly baseUrl: string;
  private readonly baseOrigin: string;
  private readonly bearerToken: string;
  private readonly tokenProvider?: TokenProvider;
  private readonly gatewayApiKey: string;
  private readonly serverKey: string;
  private readonly defaultBackend: string;
  private readonly timeoutMs: number;
  private readonly maxRedirects: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: DGemClientOptions = {}) {
    const resolvedUrl = (
      options.baseUrl ??
      process.env["DGEM_GATEWAY_URL"] ??
      process.env["DGEM_URL"] ??
      "http://127.0.0.1:8090"
    ).trim();

    this.baseOrigin = parseUrlOrigin(resolvedUrl);
    this.baseUrl = stripTrailingSlashes(resolvedUrl);

    const timeoutMs = options.timeoutMs ?? 60_000;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new DGemValidationError(
        `timeoutMs must be finite and > 0 (got ${String(timeoutMs)})`,
      );
    }
    this.timeoutMs = timeoutMs;

    const maxRedirects = options.maxRedirects ?? 5;
    if (
      !Number.isInteger(maxRedirects) ||
      maxRedirects < 0 ||
      maxRedirects > 10
    ) {
      throw new DGemValidationError(
        `maxRedirects must be an integer in [0, 10] (got ${String(maxRedirects)})`,
      );
    }
    this.maxRedirects = maxRedirects;

    this.bearerToken = (
      options.bearerToken ??
      process.env["DGEM_AUTH_TOKEN"] ??
      ""
    ).trim();
    this.tokenProvider = options.tokenProvider;
    this.gatewayApiKey = (
      options.gatewayApiKey ??
      process.env["DGEM_GATEWAY_KEY"] ??
      ""
    ).trim();

    const rawServerKey =
      options.serverKey ?? process.env["DGEM_SERVER_KEY"] ?? "";
    this.serverKey = (rawServerKey.split(",")[0] ?? "").trim();

    this.defaultBackend = (options.defaultBackend ?? "").trim();
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
  }

  private isRawPredict(): boolean {
    return this.baseUrl.toLowerCase().endsWith(":rawpredict");
  }

  private isServingV1Base(): boolean {
    const lowered = this.baseUrl.toLowerCase();
    return lowered.endsWith("/v1") || lowered.endsWith(":rawpredict");
  }

  private async buildHeaders(
    targetUrl: string,
    targetRole: TargetRole,
    backendOverride = "",
  ): Promise<Record<string, string>> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    const targetOrigin = parseUrlOrigin(targetUrl);

    let token = this.bearerToken;
    if (!token && this.tokenProvider !== undefined) {
      const provided = await this.tokenProvider(targetUrl);
      if (provided) {
        token = provided.trim();
      }
    }
    if (token && targetOrigin === this.baseOrigin) {
      headers["Authorization"] = validateHeaderValue(
        "Authorization",
        `Bearer ${token}`,
      );
    }

    if (
      targetRole === "gateway" &&
      this.gatewayApiKey &&
      targetOrigin === this.baseOrigin
    ) {
      headers["X-DGem-Api-Key"] = validateHeaderValue(
        "X-DGem-Api-Key",
        this.gatewayApiKey,
      );
    } else if (
      targetRole === "serving" &&
      this.serverKey &&
      targetOrigin === this.baseOrigin
    ) {
      headers["X-DGem-Key"] = validateHeaderValue("X-DGem-Key", this.serverKey);
    }

    const selectedBackend = backendOverride.trim() || this.defaultBackend;
    if (selectedBackend && targetRole === "gateway") {
      headers["X-DGem-Backend"] = validateHeaderValue(
        "X-DGem-Backend",
        selectedBackend,
      );
    }
    return headers;
  }

  private async executeJsonRequest(
    method: "GET" | "POST",
    initialUrl: string,
    body: Record<string, unknown> | undefined,
    headers: Record<string, string>,
  ): Promise<{ payload: Record<string, unknown>; headers: Headers }> {
    const firstOrigin = parseUrlOrigin(initialUrl);
    let currentUrl = initialUrl;
    let currentMethod: "GET" | "POST" = method;
    let currentBody = body !== undefined ? JSON.stringify(body) : undefined;
    const currentHeaders: Record<string, string> = { ...headers };
    let crossedOrigin = false;

    for (let hop = 0; hop <= this.maxRedirects; hop += 1) {
      let response: Response;
      try {
        response = await this.fetchImpl(currentUrl, {
          method: currentMethod,
          headers: currentHeaders,
          body: currentBody,
          redirect: "manual",
          signal: AbortSignal.timeout(this.timeoutMs),
        });
      } catch (err) {
        throw new DGemTransportError(
          `failed calling ${currentMethod} ${currentUrl}: ${String(err)}`,
          { cause: err },
        );
      }

      if (REDIRECT_STATUS_CODES.has(response.status)) {
        const locationHeader = response.headers.get("location");
        if (!locationHeader) {
          throw new DGemHTTPError(
            response.status,
            "redirect response missing Location header",
          );
        }
        if (hop === this.maxRedirects) {
          throw new DGemTransportError(
            `exceeded maxRedirects (${this.maxRedirects}) calling ${initialUrl}`,
          );
        }
        let resolvedNextUrl: string;
        try {
          resolvedNextUrl = new URL(locationHeader, currentUrl).toString();
        } catch (err) {
          throw new DGemValidationError(
            `invalid redirect Location ${JSON.stringify(locationHeader)}: ${String(err)}`,
          );
        }
        const nextOrigin = parseUrlOrigin(resolvedNextUrl);
        crossedOrigin = crossedOrigin || nextOrigin !== firstOrigin;
        if (crossedOrigin) {
          stripCredentialHeaders(currentHeaders);
        }
        if (
          (response.status === 301 ||
            response.status === 302 ||
            response.status === 303) &&
          currentMethod === "POST"
        ) {
          currentMethod = "GET";
          currentBody = undefined;
          delete currentHeaders["Content-Type"];
          delete currentHeaders["content-type"];
        }
        currentUrl = resolvedNextUrl;
        continue;
      }

      const rawText = await response.text();
      if (!response.ok) {
        throw new DGemHTTPError(
          response.status,
          response.statusText || "request failed",
          rawText,
        );
      }

      let decoded: unknown;
      try {
        decoded = JSON.parse(rawText);
      } catch (err) {
        throw new DGemTransportError(
          `invalid JSON response from ${currentUrl}: ${String(err)}`,
          { cause: err },
        );
      }
      if (
        decoded === null ||
        typeof decoded !== "object" ||
        Array.isArray(decoded)
      ) {
        throw new DGemTransportError("expected top-level JSON object");
      }
      return {
        payload: decoded as Record<string, unknown>,
        headers: response.headers,
      };
    }

    throw new DGemTransportError(`redirect loop calling ${initialUrl}`);
  }

  public async health(): Promise<Record<string, unknown>> {
    if (this.isRawPredict()) {
      const headers = await this.buildHeaders(this.baseUrl, "serving");
      const { payload } = await this.executeJsonRequest(
        "POST",
        this.baseUrl,
        { health: true },
        headers,
      );
      return payload;
    }

    const baseWithoutV1 = this.baseUrl.toLowerCase().endsWith("/v1")
      ? this.baseUrl.slice(0, -3)
      : this.baseUrl;
    const role: TargetRole = this.isServingV1Base() ? "serving" : "gateway";
    const targetUrl = `${baseWithoutV1}/health`;
    const headers = await this.buildHeaders(targetUrl, role);
    const { payload } = await this.executeJsonRequest(
      "GET",
      targetUrl,
      undefined,
      headers,
    );
    return payload;
  }

  public async decide(
    request: GatewayDecideRequest,
  ): Promise<GatewayDecideResponse> {
    const template = (request.template ?? "").trim();
    const customTemplate = request.customTemplate ?? "";
    if (!template && !customTemplate.trim()) {
      throw new DGemValidationError(
        "either template or customTemplate must be non-empty",
      );
    }
    if (this.isServingV1Base()) {
      throw new DGemValidationError(
        "decide() requires a dgem gateway URL; use decideSystemOne() on /v1",
      );
    }

    const wireBody: Record<string, unknown> = {};
    if (template) {
      wireBody["template"] = template;
    }
    if (customTemplate.trim()) {
      wireBody["custom_template"] = customTemplate;
    }
    if (request.variables !== undefined) {
      wireBody["variables"] = { ...request.variables };
    }
    if (request.images !== undefined && request.images.length > 0) {
      wireBody["images"] = [...request.images];
    }
    if (request.backend?.trim()) {
      wireBody["backend"] = request.backend.trim();
    }
    if (request.cascadeMode?.trim()) {
      wireBody["cascade_mode"] = request.cascadeMode.trim();
    }
    if (request.cascadeThreshold !== undefined) {
      wireBody["cascade_threshold"] = request.cascadeThreshold;
    }
    if (request.cascadeThresholdMode) {
      wireBody["cascade_threshold_mode"] = request.cascadeThresholdMode;
    }
    if (request.cascadeModel?.trim()) {
      wireBody["cascade_model"] = request.cascadeModel.trim();
    }
    if (request.layout) {
      wireBody["layout"] = request.layout;
    }
    if (request.temperature !== undefined) {
      wireBody["temperature"] =
        typeof request.temperature === "number"
          ? request.temperature
          : { ...request.temperature };
    }

    const targetUrl = `${this.baseUrl}/api/decide`;
    const headers = await this.buildHeaders(
      targetUrl,
      "gateway",
      request.backend ?? "",
    );
    const { payload, headers: responseHeaders } = await this.executeJsonRequest(
      "POST",
      targetUrl,
      wireBody,
      headers,
    );
    const backendUsedHeader =
      responseHeaders.get("x-dgem-backend-used") ?? "";
    return parseGatewayDecideResponse(payload, backendUsedHeader);
  }

  public async decideSystemOne(
    request: SystemOneRequest,
  ): Promise<StructuredDecisionResponse> {
    const questionIds = Object.keys(request.questions);
    if (questionIds.length === 0) {
      throw new DGemValidationError("questions must be non-empty");
    }
    const samples = request.samples ?? 1;
    if (!Number.isInteger(samples) || samples < 1) {
      throw new DGemValidationError(
        `samples must be an integer >= 1 (got ${String(samples)})`,
      );
    }

    const serializedQuestions: Record<string, unknown> = {};
    for (const qid of questionIds) {
      const spec = request.questions[qid];
      if (spec !== undefined) {
        serializedQuestions[qid] = serializeSystemOneQuestion(spec);
      }
    }

    const wireBody: Record<string, unknown> = {
      context:
        typeof request.context === "string"
          ? request.context
          : { ...request.context },
      questions: serializedQuestions,
      samples,
    };
    if (request.instructions?.trim()) {
      wireBody["instructions"] = request.instructions.trim();
    }
    if (request.images !== undefined && request.images.length > 0) {
      wireBody["images"] = [...request.images];
    }
    if (request.layout) {
      wireBody["layout"] = request.layout;
    }
    if (request.priorWeight !== undefined) {
      wireBody["prior_weight"] = request.priorWeight;
    }
    if (request.shortlistTopk !== undefined) {
      wireBody["shortlist_topk"] = request.shortlistTopk;
    }

    let targetUrl: string;
    let role: TargetRole;
    if (this.isRawPredict()) {
      targetUrl = this.baseUrl;
      role = "serving";
    } else if (this.baseUrl.toLowerCase().endsWith("/v1")) {
      targetUrl = `${this.baseUrl}/systemone`;
      role = "serving";
    } else {
      targetUrl = `${this.baseUrl}/v1/systemone`;
      role = "gateway";
    }

    const headers = await this.buildHeaders(
      targetUrl,
      role,
      request.backend ?? "",
    );
    const { payload } = await this.executeJsonRequest(
      "POST",
      targetUrl,
      wireBody,
      headers,
    );
    return parseStructuredDecisionResponse(payload);
  }
}
