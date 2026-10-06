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
import { customElement, property } from 'lit/decorators.js';
import type { PresetSample } from '../types.js';

@customElement('dgem-preset-selector')
export class DgemPresetSelector extends LitElement {
  @property({ type: Array }) presets: PresetSample[] = [];
  @property({ type: String }) activePresetId = '';
  @property({ type: String, reflect: true }) resolvedTheme: 'light' | 'dark' = 'light';

  static styles = css`
    :host {
      display: block;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

      --card-bg: #ffffff;
      --card-header-bg: #f8fafc;
      --card-border: #e2e8f0;
      --chip-bg: #f8fafc;
      --chip-hover-bg: #eff6ff;
      --chip-active-bg: #eff6ff;
      --chip-border: #e2e8f0;
      --chip-active-border: #1447e6;
      --text-heading: #0f172a;
      --text-body: #334155;
      --text-muted: #64748b;
      --brand: #1447e6;
    }

    :host([resolvedTheme='dark']) {
      --card-bg: #0f172a;
      --card-header-bg: #1e293b;
      --card-border: #1e293b;
      --chip-bg: #1e293b;
      --chip-hover-bg: rgba(59, 130, 246, 0.14);
      --chip-active-bg: rgba(59, 130, 246, 0.18);
      --chip-border: #334155;
      --chip-active-border: #3b82f6;
      --text-heading: #f8fafc;
      --text-body: #cbd5e1;
      --text-muted: #94a3b8;
      --brand: #3b82f6;
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

    .preset-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 1rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .preset-header {
      padding: 0.65rem 1rem;
      border-bottom: 1px solid var(--card-border);
      background: var(--card-header-bg);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }

    .preset-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text-heading);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .preset-subtitle {
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    .preset-grid {
      padding: 0.85rem 1rem;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.55rem;
    }

    @media (max-width: 720px) {
      .preset-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    .preset-chip {
      text-align: left;
      padding: 0.55rem 0.68rem;
      border-radius: 7px;
      border: 1px solid var(--chip-border);
      background: var(--chip-bg);
      cursor: pointer;
      transition:
        border-color 120ms ease,
        background 120ms ease,
        box-shadow 120ms ease;
      display: flex;
      flex-direction: column;
      gap: 0.18rem;
    }

    .preset-chip:hover {
      border-color: var(--chip-active-border);
      background: var(--chip-hover-bg);
    }

    .preset-chip--active {
      border-color: var(--chip-active-border);
      background: var(--chip-active-bg);
      box-shadow: 0 0 0 1px var(--chip-active-border);
    }

    .preset-chip-badge {
      font-size: 0.62rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--brand);
    }

    .preset-chip-title {
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text-heading);
      line-height: 1.25;
    }

    .preset-chip-tmpl {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.65rem;
      color: var(--text-muted);
    }
  `;

  private select(p: PresetSample) {
    this.dispatchEvent(
      new CustomEvent<PresetSample>('preset-select', {
        detail: p,
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    return html`
      <div class="preset-card">
        <div class="preset-header">
          <div class="preset-title">
            <span class="material-symbols-outlined" style="color:var(--brand)">bolt</span>
            Quick Challenge Presets
          </div>
          <span class="preset-subtitle">1-Click Policy + Payload</span>
        </div>
        <div class="preset-grid">
          ${this.presets.map(
            (p) => html`
              <button
                class="preset-chip ${this.activePresetId === p.id ? 'preset-chip--active' : ''}"
                title=${p.description}
                @click=${() => this.select(p)}
              >
                <span class="preset-chip-badge">${p.badge}</span>
                <span class="preset-chip-title">${p.title}</span>
                <span class="preset-chip-tmpl">${p.template}.json.tmpl</span>
              </button>
            `
          )}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'dgem-preset-selector': DgemPresetSelector;
  }
}
