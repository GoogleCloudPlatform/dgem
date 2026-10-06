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
import type { GPUHealthStatus } from '../types.js';

@customElement('dgem-about-modal')
export class DgemAboutModal extends LitElement {
  @property({ type: Boolean, reflect: true }) open = false;
  @property({ type: String, reflect: true }) resolvedTheme: 'light' | 'dark' = 'light';
  @property({ type: Object }) gpuStatus: GPUHealthStatus | null = null;
  @property({ type: Number }) templateCount = 26;

  static styles = css`
    :host {
      display: none;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

      --modal-bg: #ffffff;
      --modal-surface: #f8fafc;
      --modal-border: #e2e8f0;
      --modal-heading: #0f172a;
      --modal-body: #334155;
      --modal-muted: #64748b;
      --modal-brand: #1447e6;
      --modal-brand-soft: #eff6ff;
      --modal-brand-border: #bfdbfe;
    }

    :host([open]) {
      display: block;
    }

    :host([resolvedTheme='dark']) {
      --modal-bg: #0f172a;
      --modal-surface: #1e293b;
      --modal-border: #334155;
      --modal-heading: #f8fafc;
      --modal-body: #cbd5e1;
      --modal-muted: #94a3b8;
      --modal-brand: #3b82f6;
      --modal-brand-soft: rgba(59, 130, 246, 0.15);
      --modal-brand-border: rgba(59, 130, 246, 0.35);
    }

    .material-symbols-outlined {
      font-family: 'Material Symbols Outlined';
      font-weight: normal;
      font-style: normal;
      font-size: 19px;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .backdrop {
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(15, 23, 42, 0.55);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.25rem;
    }

      .dialog {
      width: 100%;
      max-width: 680px;
      max-height: 88vh;
      overflow-y: auto;
      background: var(--modal-bg);
      color: var(--modal-body);
      border: 1px solid var(--modal-border);
      border-radius: 12px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.25);
    }

    .dialog-header {
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--modal-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--modal-surface);
    }

    .dialog-title {
      margin: 0;
      font-family: 'Google Sans', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      color: var(--modal-heading);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .close-btn {
      border: 1px solid var(--modal-border);
      background: var(--modal-bg);
      color: var(--modal-muted);
      width: 30px;
      height: 30px;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .close-btn:hover {
      color: var(--modal-heading);
    }

    .dialog-body {
      padding: 1.25rem;
      font-size: 0.83rem;
      line-height: 1.55;
    }

    .spec-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.65rem;
      margin: 1rem 0;
    }

    .spec-item {
      padding: 0.7rem 0.85rem;
      border-radius: 8px;
      border: 1px solid var(--modal-border);
      background: var(--modal-surface);
    }

    .spec-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--modal-muted);
      margin-bottom: 0.2rem;
    }

    .spec-val {
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--modal-heading);
      word-break: break-all;
    }

    .pillar-list {
      margin: 0.75rem 0 0;
      padding-left: 1.15rem;
      color: var(--modal-body);
    }

    .pillar-list li {
      margin-bottom: 0.35rem;
    }

    .glossary-box {
      margin-top: 1rem;
      padding: 0.9rem 1rem;
      border-radius: 10px;
      border: 1px solid var(--modal-brand-border);
      background: var(--modal-brand-soft);
    }

    .glossary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.55rem;
      margin-top: 0.6rem;
    }

    .glossary-card {
      background: var(--modal-bg);
      border: 1px solid var(--modal-border);
      border-radius: 7px;
      padding: 0.55rem 0.7rem;
      font-size: 0.76rem;
    }

    .glossary-card strong {
      color: var(--modal-heading);
      display: block;
      font-size: 0.78rem;
      margin-bottom: 0.12rem;
    }

    .glossary-btn {
      margin-top: 0.75rem;
      width: 100%;
      padding: 0.55rem 0.85rem;
      border-radius: 7px;
      border: 1px solid var(--modal-brand);
      background: var(--modal-brand);
      color: #ffffff;
      font-weight: 600;
      font-size: 0.8rem;
      cursor: pointer;
    }

    .glossary-btn:hover {
      opacity: 0.92;
    }
  `;

  private closeModal() {
    this.dispatchEvent(
      new CustomEvent('close-about', {
        bubbles: true,
        composed: true,
      })
    );
  }

  private openGlossaryTab() {
    this.dispatchEvent(
      new CustomEvent('open-glossary', {
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    if (!this.open) return null;

    return html`
      <div class="backdrop" @click=${this.closeModal}>
        <div
          class="dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby="about-title"
          @click=${(e: Event) => e.stopPropagation()}
        >
          <div class="dialog-header">
            <h2 class="dialog-title" id="about-title">
              <span class="material-symbols-outlined" style="color:var(--modal-brand)">info</span>
              About DiffusionGemma Decision Studio (dgem)
            </h2>
            <button class="close-btn" @click=${this.closeModal} title="Close">
              <span class="material-symbols-outlined">close</span>
            </button>
          </div>

          <div class="dialog-body">
            <p style="margin-top:0">
              <strong>DiffusionGemma (<code>dgemma</code>)</strong> is a Zero-Shot Decision Model that
              evaluates structured multi-slot policies (<code>.json.tmpl</code>) jointly in
              <strong>O(1) forward passes (~490ms)</strong> on a bidirectional discrete-diffusion canvas,
              returning a probability for every allowed answer and a <strong>hesitation score</strong>
              (normalized Shannon entropy) for every question, without writing any text. These scores are a
              strong uncertainty signal, not automatically calibrated probabilities for your traffic.
            </p>

            <div class="glossary-box">
              <div style="font-weight:700;color:var(--modal-heading);font-size:0.83rem;display:flex;justify-content:space-between;align-items:center;">
                <span>📖 Glossary</span>
                <span style="font-size:0.7rem;color:var(--modal-brand);">Jargon-Free Guide</span>
              </div>
              <div class="glossary-grid">
                <div class="glossary-card">
                  <strong>⚖️ Hesitation-gating</strong>
                  Like a triage nurse: answer clear cases straight away and hand hesitant ones to a larger model or a person. Set it with the cascade mode and threshold on the Decide and Batch Eval panels.
                </div>
                <div class="glossary-card">
                  <strong>🌡️ Hesitation (entropy)</strong>
                  How torn the model is, from 0% (one clear answer) to 100% (a perfect tie). Under 16% is treated as clear; above that, consider handing off. Option order can occasionally hide hesitation, so dgem measures order effects on every release.
                </div>
                <div class="glossary-card">
                  <strong>🅰️ First-choice bias ("Box A")</strong>
                  First-Choice Favoritism: like humans on a multiple-choice test, raw models naturally favor Option A when guessing. dgem measures this on every release rather than correcting it per request.
                </div>
                <div class="glossary-card">
                  <strong>📝 The form is part of the question</strong>
                  The model reads the whole answer form. Reusing the same letters for different options across questions, or naming a question "reverse", changes the answers. Use neutral question names.
                </div>
                <div class="glossary-card">
                  <strong>📄 Document first</strong>
                  Show the input before the questions, like reading a letter before filling in a form. The default prompt layout since serving v0.2.0.
                </div>
                <div class="glossary-card">
                  <strong>🌦️ Calibration (Brier &amp; ECE)</strong>
                  Weather-Forecaster Honesty: measures whether "90% confident" really means right about 9 times out of 10. Check it on your own labeled data; the right correction differs by domain.
                </div>
              </div>
              <button class="glossary-btn" @click=${this.openGlossaryTab}>
                📖 Open the full glossary and walkthrough in Concepts ➔
              </button>
            </div>

            <div class="spec-grid">
              <div class="spec-item">
                <div class="spec-label">Inference Engine &amp; Vision</div>
                <div class="spec-val">vLLM (TRITON_ATTN) + Gemma 4 SigLIP</div>
              </div>
              <div class="spec-item">
                <div class="spec-label">Primary &amp; Failover Backends</div>
                <div class="spec-val">Vertex AI Dedicated G4 · RTX PRO 6000 (0s wakeup) + Cloud Run</div>
              </div>
              <div class="spec-item">
                <div class="spec-label">Model Checkpoint</div>
                <div class="spec-val">nvidia/diffusiongemma-26B-A4B-it-NVFP4</div>
              </div>
              <div class="spec-item">
                <div class="spec-label">Embedded Policy Templates</div>
                <div class="spec-val">${this.templateCount} (.json.tmpl policies)</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }
}
