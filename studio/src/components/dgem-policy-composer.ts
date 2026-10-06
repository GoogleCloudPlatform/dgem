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

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { TemplateEntry } from '../types.js';

export type ComposerViewMode = 'inputs' | 'instance' | 'template' | 'split';

@customElement('dgem-policy-composer')
export class DgemPolicyComposer extends LitElement {
  @property({ type: Array }) templates: TemplateEntry[] = [];
  @property({ type: String }) selectedTemplateName = 'support_triage';
  @property({ type: Object }) variableValues: Record<string, string> = {};
  @property({ type: Boolean }) loading = false;
  @property({ type: String }) gpuState = 'scaled_to_zero';
  @property({ type: Number }) warmupElapsedSec = 0;
  @property({ type: String }) errorMessage = '';
  @property({ type: String, reflect: true }) resolvedTheme: 'light' | 'dark' = 'light';

  @state() private viewMode: ComposerViewMode = 'inputs';
  @state() private splitRightTab: 'instance' | 'template' = 'instance';
  @state() private copiedKey = '';

  static styles = css`
    :host {
      display: block;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

      --card-bg: #ffffff;
      --card-header-bg: #f8fafc;
      --card-border: #e2e8f0;
      --surface-soft: #f8fafc;
      --surface-tertiary: #f1f5f9;
      --text-heading: #0f172a;
      --text-body: #334155;
      --text-muted: #64748b;
      --brand: #1447e6;
      --brand-hover: #1d4ed8;
      --brand-soft: #eff6ff;
      --brand-border: #bfdbfe;
    }

    :host([resolvedTheme='dark']) {
      --card-bg: #0f172a;
      --card-header-bg: #1e293b;
      --card-border: #1e293b;
      --surface-soft: #1e293b;
      --surface-tertiary: #334155;
      --text-heading: #f8fafc;
      --text-body: #cbd5e1;
      --text-muted: #94a3b8;
      --brand: #3b82f6;
      --brand-hover: #2563eb;
      --brand-soft: rgba(59, 130, 246, 0.15);
      --brand-border: rgba(59, 130, 246, 0.35);
    }

    *,
    *::before,
    *::after {
      box-sizing: border-box;
    }

    .material-symbols-outlined {
      font-family: 'Material Symbols Outlined';
      font-size: 16px;
      line-height: 1;
      vertical-align: middle;
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      overflow: hidden;
      min-width: 0;
      max-width: 100%;
    }

    .card-header {
      padding: 0.78rem 1.05rem;
      border-bottom: 1px solid var(--card-border);
      background: var(--card-header-bg);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.65rem;
      flex-wrap: wrap;
    }

    .card-title {
      font-size: 0.86rem;
      font-weight: 700;
      color: var(--text-heading);
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .card-body {
      padding: 1.05rem 1.15rem;
    }

    .segmented {
      display: inline-flex;
      padding: 0.18rem;
      border-radius: 7px;
      background: var(--surface-tertiary);
      border: 1px solid var(--card-border);
      gap: 0.15rem;
    }

    .seg {
      border: none;
      background: transparent;
      color: var(--text-muted);
      padding: 0.28rem 0.6rem;
      border-radius: 5px;
      font-family: inherit;
      font-size: 0.72rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 120ms ease;
    }

    .seg[aria-selected='true'] {
      background: var(--card-bg);
      color: var(--brand);
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }

    .field {
      margin-bottom: 0.85rem;
    }

    .field-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-heading);
      margin-bottom: 0.35rem;
      gap: 0.5rem;
    }

    .field-var-badge {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      padding: 0.1rem 0.42rem;
      border-radius: 4px;
      background: var(--brand-soft);
      color: var(--brand);
      border: 1px solid var(--brand-border);
    }

    select,
    input[type='text'],
    textarea {
      width: 100%;
      padding: 0.55rem 0.72rem;
      border-radius: 6px;
      border: 1px solid var(--card-border);
      background: var(--card-bg);
      color: var(--text-heading);
      font-family: 'Inter', sans-serif;
      font-size: 0.82rem;
      line-height: 1.45;
    }

    textarea {
      field-sizing: content;
      min-height: 78px;
      resize: vertical;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
    }

    select:focus,
    input:focus,
    textarea:focus {
      outline: none;
      border-color: var(--brand);
      box-shadow: 0 0 0 3px var(--brand-soft);
    }

    .split-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1rem;
      align-items: start;
    }

    @media (max-width: 960px) {
      .split-grid {
        grid-template-columns: 1fr;
      }
    }

    .code-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 0.45rem;
      flex-wrap: wrap;
    }

    pre.code-block {
      margin: 0;
      padding: 0.8rem 0.95rem;
      border-radius: 8px;
      background: #0f172a;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      overflow-x: hidden;
      max-width: 100%;
      max-height: 440px;
      overflow-y: auto;
      border: 1px solid #1e293b;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      padding: 0.45rem 0.85rem;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--card-border);
      background: var(--card-bg);
      color: var(--text-heading);
      transition: all 120ms ease;
    }

    .btn--sm {
      padding: 0.28rem 0.58rem;
      font-size: 0.71rem;
    }

    .btn--brand {
      background: var(--brand);
      border-color: var(--brand);
      color: #ffffff;
    }

    .btn--brand:hover:not(:disabled) {
      background: var(--brand-hover);
    }

    .btn:disabled {
      opacity: 0.65;
      cursor: not-allowed;
    }

    .error-box {
      margin-top: 0.85rem;
      padding: 0.7rem 0.85rem;
      border-radius: 6px;
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #ef4444;
      font-size: 0.78rem;
    }
  `;

  private get activeTemplate(): TemplateEntry | undefined {
    return (
      this.templates.find((t) => t.name === this.selectedTemplateName) ||
      this.templates[0]
    );
  }

  /**
   * Renders the Go .json.tmpl source in the browser with the user's current variableValues
   * so the "Instance JSON" view updates live on every keystroke.
   */
  private renderCompiledInstance(): string {
    const raw = this.activeTemplate?.raw_source || '';
    if (!raw) {
      return JSON.stringify(
        {
          template: this.selectedTemplateName,
          variables: this.variableValues,
        },
        null,
        2
      );
    }

    // Replace {{ default "fallback" .varName | toJson }}
    let out = raw.replace(
      /\{\{\s*default\s+"([^"]*)"\s+\.([a-zA-Z0-9_]+)\s*\|\s*toJson\s*\}\}/g,
      (_match, fallback: string, varName: string) => {
        const val = this.variableValues[varName];
        const effective = val !== undefined && val.trim() !== '' ? val : fallback;
        return JSON.stringify(effective);
      }
    );

    // Replace {{ .varName | toJson }}
    out = out.replace(
      /\{\{\s*\.([a-zA-Z0-9_]+)\s*\|\s*toJson\s*\}\}/g,
      (_match, varName: string) => {
        const val = this.variableValues[varName] ?? '';
        return JSON.stringify(val);
      }
    );

    // Replace {{ .varName }}
    out = out.replace(/\{\{\s*\.([a-zA-Z0-9_]+)\s*\}\}/g, (_match, varName: string) => {
      const val = this.variableValues[varName] ?? '';
      return val.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
    });

    try {
      const parsed = JSON.parse(out);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return out;
    }
  }

  private renderCliSnippet(): string {
    const tmpl = this.activeTemplate;
    const path = tmpl?.path ? `templates/${tmpl.path}` : `templates/${this.selectedTemplateName}.json.tmpl`;
    const varFlags = Object.entries(this.variableValues)
      .map(([k, v]) => `  -v ${JSON.stringify(`${k}=${v}`)}`)
      .join(' \\\n');
    return `./bin/dgem decide -u "${window.location.origin}/v1" --gcp-auth \\\n  -t ${path}${varFlags ? ' \\\n' + varFlags : ''}`;
  }

  private copyText(key: string, text: string) {
    navigator.clipboard.writeText(text);
    this.copiedKey = key;
    setTimeout(() => {
      if (this.copiedKey === key) this.copiedKey = '';
    }, 1800);
  }

  private onTemplateChange(e: Event) {
    const nextName = (e.target as HTMLSelectElement).value;
    this.dispatchEvent(
      new CustomEvent<string>('template-change', {
        detail: nextName,
        bubbles: true,
        composed: true,
      })
    );
  }

  private onVarInput(varName: string, val: string) {
    this.dispatchEvent(
      new CustomEvent<{ name: string; value: string }>('variable-change', {
        detail: { name: varName, value: val },
        bubbles: true,
        composed: true,
      })
    );
  }

  private onEvaluateClick() {
    this.dispatchEvent(
      new CustomEvent('evaluate-decision', {
        bubbles: true,
        composed: true,
      })
    );
  }

  private renderInputsPane() {
    const currentTmpl = this.activeTemplate;
    const vars = currentTmpl?.variables || Object.keys(this.variableValues);

    return html`
      <div class="field">
        <label class="field-label">
          <span>Decision Policy (.json.tmpl)</span>
          <span class="field-var-badge">${currentTmpl?.category || 'core'}</span>
        </label>
        <select .value=${this.selectedTemplateName} @change=${this.onTemplateChange}>
          ${this.templates.map(
            (t) => html`
              <option value=${t.name} ?selected=${t.name === this.selectedTemplateName}>
                ${t.name} — ${t.description.slice(0, 62)}
              </option>
            `
          )}
        </select>
      </div>

      ${vars.map(
        (v) => html`
          <div class="field">
            <label class="field-label">
              <span>Input Variable</span>
              <span class="field-var-badge">.{{${v}}}</span>
            </label>
            <textarea
              .value=${this.variableValues[v] || ''}
              placeholder=${`Enter ${v} context for policy evaluation...`}
              @input=${(e: Event) =>
                this.onVarInput(v, (e.target as HTMLTextAreaElement).value)}
            ></textarea>
          </div>
        `
      )}

      <!-- Slot for Multimodal Image Upload & BBox Overlay Canvas -->
      <slot name="multimodal"></slot>
    `;
  }

  private renderInstancePane() {
    const compiled = this.renderCompiledInstance();
    const cli = this.renderCliSnippet();
    return html`
      <div>
        <div class="code-header">
          <span style="font-size:0.74rem;font-weight:600;color:var(--text-heading)">
            Compiled Policy Instance (<code>schema</code> + <code>state</code> with current variables)
          </span>
          <div style="display:flex;gap:0.35rem">
            <button
              class="btn btn--sm"
              @click=${() => this.copyText('instance-json', compiled)}
            >
              <span class="material-symbols-outlined">content_copy</span>
              ${this.copiedKey === 'instance-json' ? 'Copied!' : 'Copy Instance JSON'}
            </button>
            <button class="btn btn--sm" @click=${() => this.copyText('instance-cli', cli)}>
              <span class="material-symbols-outlined">terminal</span>
              ${this.copiedKey === 'instance-cli' ? 'Copied!' : 'Copy CLI'}
            </button>
          </div>
        </div>
        <pre class="code-block">${compiled}</pre>
      </div>
    `;
  }

  private renderTemplateSourcePane() {
    const currentTmpl = this.activeTemplate;
    const raw = currentTmpl?.raw_source || '// Select a policy template to view its .json.tmpl source';
    return html`
      <div>
        <div class="code-header">
          <span style="font-size:0.74rem;font-weight:600;color:var(--text-heading)">
            Parameterized Policy Source (<code>templates/${currentTmpl?.path || `${this.selectedTemplateName}.json.tmpl`}</code>)
          </span>
          <button class="btn btn--sm" @click=${() => this.copyText('raw-tmpl', raw)}>
            <span class="material-symbols-outlined">content_copy</span>
            ${this.copiedKey === 'raw-tmpl' ? 'Copied!' : 'Copy .json.tmpl'}
          </button>
        </div>
        <pre class="code-block">${raw}</pre>
      </div>
    `;
  }

  render() {
    const isColdOrWarming = this.gpuState !== 'warm_and_ready';
    const buttonLabel = this.loading
      ? isColdOrWarming
        ? `Waking GPU (${this.warmupElapsedSec}s / ~90s) & Evaluating Policy...`
        : 'Evaluating Joint Diffusion Slots (Single Forward Pass)...'
      : isColdOrWarming
        ? 'Evaluate Decision Policy (Auto-Wakes GPU + Single Pass)'
        : 'Evaluate Decision Policy (Single Forward Pass)';

    return html`
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">
            <span class="material-symbols-outlined" style="color:var(--brand)">description</span>
            Policy Template &amp; Input Context
          </h2>

          <div class="segmented" role="tablist" aria-label="Policy composer view mode">
            <button
              class="seg"
              role="tab"
              aria-selected=${this.viewMode === 'inputs' ? 'true' : 'false'}
              @click=${() => (this.viewMode = 'inputs')}
              title="Edit policy variables and inputs"
            >
              <span class="material-symbols-outlined">tune</span>
              Inputs
            </button>
            <button
              class="seg"
              role="tab"
              aria-selected=${this.viewMode === 'instance' ? 'true' : 'false'}
              @click=${() => (this.viewMode = 'instance')}
              title="Inspect compiled JSON instance with current variables"
            >
              <span class="material-symbols-outlined">data_object</span>
              Instance JSON
            </button>
            <button
              class="seg"
              role="tab"
              aria-selected=${this.viewMode === 'template' ? 'true' : 'false'}
              @click=${() => (this.viewMode = 'template')}
              title="Inspect raw .json.tmpl policy template"
            >
              <span class="material-symbols-outlined">code</span>
              .json.tmpl
            </button>
            <button
              class="seg"
              role="tab"
              aria-selected=${this.viewMode === 'split' ? 'true' : 'false'}
              @click=${() => (this.viewMode = 'split')}
              title="Side-by-side Inputs + Live Instance JSON"
            >
              <span class="material-symbols-outlined">vertical_split</span>
              Split
            </button>
          </div>
        </div>

        <div class="card-body">
          ${this.viewMode === 'inputs'
            ? this.renderInputsPane()
            : this.viewMode === 'instance'
              ? this.renderInstancePane()
              : this.viewMode === 'template'
                ? this.renderTemplateSourcePane()
                : html`
                    <div class="split-grid">
                      <div>${this.renderInputsPane()}</div>
                      <div>
                        <div style="display:flex;justify-content:flex-end;margin-bottom:0.45rem">
                          <div class="segmented">
                            <button
                              class="seg"
                              aria-selected=${this.splitRightTab === 'instance' ? 'true' : 'false'}
                              @click=${() => (this.splitRightTab = 'instance')}
                            >
                              Instance JSON
                            </button>
                            <button
                              class="seg"
                              aria-selected=${this.splitRightTab === 'template' ? 'true' : 'false'}
                              @click=${() => (this.splitRightTab = 'template')}
                            >
                              .json.tmpl
                            </button>
                          </div>
                        </div>
                        ${this.splitRightTab === 'instance'
                          ? this.renderInstancePane()
                          : this.renderTemplateSourcePane()}
                      </div>
                    </div>
                  `}

          <div style="display:flex;gap:0.65rem;align-items:center;margin-top:1rem">
            <button
              class="btn btn--brand"
              style="flex:1;padding:0.65rem 1rem"
              ?disabled=${this.loading}
              @click=${this.onEvaluateClick}
            >
              <span class="material-symbols-outlined">
                ${this.loading && isColdOrWarming ? 'hourglass_top' : 'bolt'}
              </span>
              ${buttonLabel}
            </button>
          </div>

          ${this.errorMessage
            ? html`
                <div class="error-box">
                  <strong>Execution Error:</strong> ${this.errorMessage}
                </div>
              `
            : null}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'dgem-policy-composer': DgemPolicyComposer;
  }
}
