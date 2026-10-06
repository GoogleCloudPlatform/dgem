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

export type StudioTab = 'studio' | 'batch' | 'concepts' | 'catalog' | 'mcp';
export type ThemePreference = 'auto' | 'light' | 'dark';

@customElement('dgem-nav-rail')
export class DgemNavRail extends LitElement {
  @property({ type: String }) activeTab: StudioTab = 'studio';
  @property({ type: Number }) templateCount = 24;
  @property({ type: String, reflect: true }) themePref: ThemePreference = 'auto';
  @property({ type: String, reflect: true }) resolvedTheme: 'light' | 'dark' = 'light';

  static styles = css`
    :host {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      width: 84px;
      min-height: 100vh;
      position: sticky;
      top: 0;
      height: 100vh;
      background: var(--rail-bg, #ffffff);
      border-right: 1px solid var(--rail-border, #e2e8f0);
      padding: 0.85rem 0.45rem;
      box-sizing: border-box;
      user-select: none;
      z-index: 40;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

      --rail-bg: #ffffff;
      --rail-border: #e2e8f0;
      --rail-text: #64748b;
      --rail-text-active: #0f172a;
      --rail-hover-bg: #f1f5f9;
      --rail-active-bg: #eff6ff;
      --rail-active-border: #bfdbfe;
      --rail-brand: #1447e6;
    }

    :host([resolvedTheme='dark']) {
      --rail-bg: #0f172a;
      --rail-border: #1e293b;
      --rail-text: #94a3b8;
      --rail-text-active: #f8fafc;
      --rail-hover-bg: #1e293b;
      --rail-active-bg: rgba(59, 130, 246, 0.16);
      --rail-active-border: rgba(59, 130, 246, 0.38);
      --rail-brand: #3b82f6;
    }

    .material-symbols-outlined {
      font-family: 'Material Symbols Outlined';
      font-weight: normal;
      font-style: normal;
      font-size: 21px;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .rail-top,
    .rail-bottom {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.45rem;
    }

    .brand-logo {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: var(--rail-brand);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Google Sans', sans-serif;
      font-weight: 700;
      font-size: 1.05rem;
      margin-bottom: 0.65rem;
      box-shadow: 0 1px 2px rgba(20, 71, 230, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.22);
      cursor: pointer;
    }

    .nav-btn {
      width: 100%;
      border: 1px solid transparent;
      background: transparent;
      color: var(--rail-text);
      padding: 0.55rem 0.25rem;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.28rem;
      cursor: pointer;
      transition: all 0.14s ease;
      font-family: inherit;
    }

    .nav-btn:hover {
      background: var(--rail-hover-bg);
      color: var(--rail-text-active);
    }

    .nav-btn[aria-current='page'] {
      background: var(--rail-active-bg);
      border-color: var(--rail-active-border);
      color: var(--rail-brand);
    }

    .nav-label {
      font-size: 0.66rem;
      font-weight: 600;
      line-height: 1.15;
      text-align: center;
      letter-spacing: -0.01em;
    }

    .divider {
      width: 36px;
      height: 1px;
      background: var(--rail-border);
      margin: 0.25rem 0;
    }

    @media (max-width: 768px) {
      :host {
        width: 100%;
        min-height: auto;
        height: auto;
        flex-direction: row;
        padding: 0.5rem 0.75rem;
        border-right: none;
        border-bottom: 1px solid var(--rail-border);
      }
      .rail-top,
      .rail-bottom {
        flex-direction: row;
      }
      .brand-logo {
        margin-bottom: 0;
        width: 34px;
        height: 34px;
      }
      .nav-btn {
        flex-direction: row;
        padding: 0.4rem 0.65rem;
        width: auto;
      }
      .divider {
        display: none;
      }
    }
  `;

  private selectTab(tab: StudioTab) {
    this.dispatchEvent(
      new CustomEvent<StudioTab>('tab-change', {
        detail: tab,
        bubbles: true,
        composed: true,
      })
    );
  }

  private cycleTheme() {
    const order: ThemePreference[] = ['auto', 'light', 'dark'];
    const next = order[(order.indexOf(this.themePref) + 1) % order.length];
    this.dispatchEvent(
      new CustomEvent<ThemePreference>('theme-change', {
        detail: next,
        bubbles: true,
        composed: true,
      })
    );
  }

  private openAbout() {
    this.dispatchEvent(
      new CustomEvent('open-about', {
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    const themeIcon =
      this.themePref === 'auto'
        ? 'brightness_auto'
        : this.themePref === 'dark'
          ? 'dark_mode'
          : 'light_mode';
    const themeLabel =
      this.themePref === 'auto' ? 'Auto' : this.themePref === 'dark' ? 'Dark' : 'Light';

    return html`
      <div class="rail-top" role="navigation" aria-label="Primary Workspace Navigation">
        <div
          class="brand-logo"
          title="DiffusionGemma Decision Studio"
          @click=${() => this.selectTab('studio')}
        >
          dG
        </div>

        <button
          class="nav-btn"
          aria-current=${this.activeTab === 'studio' ? 'page' : 'false'}
          @click=${() => this.selectTab('studio')}
          title="Decision Studio & Multimodal BBox Lab"
        >
          <span class="material-symbols-outlined">tune</span>
          <span class="nav-label">Decision Studio</span>
        </button>

        <button
          class="nav-btn"
          aria-current=${this.activeTab === 'batch' ? 'page' : 'false'}
          @click=${() => this.selectTab('batch')}
          title="Live Batch Evaluation & Streaming Benchmark Table"
        >
          <span class="material-symbols-outlined">dynamic_feed</span>
          <span class="nav-label">Batch Eval</span>
        </button>

        <button
          class="nav-btn"
          aria-current=${this.activeTab === 'concepts' ? 'page' : 'false'}
          @click=${() => this.selectTab('concepts')}
          title="Interactive Concept Walkthrough (1-Pass Diffusion, Entropy Gate & Safety Stencil)"
        >
          <span class="material-symbols-outlined">auto_awesome</span>
          <span class="nav-label">Concepts</span>
        </button>

        <button
          class="nav-btn"
          aria-current=${this.activeTab === 'catalog' ? 'page' : 'false'}
          @click=${() => this.selectTab('catalog')}
          title="Policy Catalog (${this.templateCount}) & Entropy Cascade Simulator"
        >
          <span class="material-symbols-outlined">inventory_2</span>
          <span class="nav-label">Policy Catalog</span>
        </button>

        <button
          class="nav-btn"
          aria-current=${this.activeTab === 'mcp' ? 'page' : 'false'}
          @click=${() => this.selectTab('mcp')}
          title="HTTP API & Model Context Protocol (MCP) Playground"
        >
          <span class="material-symbols-outlined">hub</span>
          <span class="nav-label">API / MCP</span>
        </button>
      </div>

      <div class="rail-bottom">
        <div class="divider"></div>

        <button
          class="nav-btn"
          @click=${this.cycleTheme}
          title="Switch color theme (Auto / Light / Dark)"
        >
          <span class="material-symbols-outlined">${themeIcon}</span>
          <span class="nav-label">${themeLabel}</span>
        </button>

        <button
          class="nav-btn"
          @click=${this.openAbout}
          title="About This App & Engine Specifications"
        >
          <span class="material-symbols-outlined">info</span>
          <span class="nav-label">About</span>
        </button>
      </div>
    `;
  }
}
