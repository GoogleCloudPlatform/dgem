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

export interface BatchExpectedSlot {
  question: string;
  label: string;
  type: string;
  expected: string;
}

export interface BatchPresetItem {
  id: string;
  domain: string;
  tier: string;
  preview: string;
  template?: string;
  custom_template?: string;
  variables: Record<string, any>;
  expected_slots: BatchExpectedSlot[];
  /** Optional image (URL or data URI) for multimodal items; also sent to the Stage-2 cascade. */
  image_url?: string;
}

export interface BatchPresetSuite {
  id: string;
  title: string;
  category: string;
  badge: string;
  description: string;
  default_concurrency: number;
  total_items: number;
  total_questions: number;
  items: BatchPresetItem[];
}

interface BatchTableRow {
  rowKey: string;
  itemIndex: number; // 1-based item index
  slotIndex: number; // 1-based slot index within item
  totalSlotsForItem: number;
  item: BatchPresetItem;
  slot: BatchExpectedSlot;
  state: 'idle' | 'queued' | 'running' | 'pass' | 'miss' | 'error';
  predicted?: string;
  confidence?: number;
  entropy?: number;
  serverMs?: number;
  roundTripMs?: number;
  errorMsg?: string;
  escalated?: boolean;
  stage1Predicted?: string;
  cascadeModel?: string;
  cascadeExplanation?: string;
  cascadeLatencyMs?: number;
}

interface CustomBuilderSlot {
  id: string;
  type: 'boolean' | 'choice' | 'score';
  instructions: string;
  optionsSpec: string; // "name:description | name2:description2"
  levelsSpec: string;  // "1, 2, 3, 4, 5"
}

const DEFAULT_CUSTOM_SLOTS: CustomBuilderSlot[] = [
  {
    id: 'compliant',
    type: 'boolean',
    instructions: 'Does the contract clause satisfy standard enterprise legal policy without uncapped risk or lock-in?',
    optionsSpec: '',
    levelsSpec: '1, 2, 3, 4, 5',
  },
  {
    id: 'risk_category',
    type: 'choice',
    instructions: 'Select the primary legal risk category present in the clause',
    optionsSpec:
      'indemnity:Uncapped indemnification or third-party IP liability | auto_renewal:Evergreen automatic renewal without 30-day opt-out | data_residency:Cross-border data transfer without GDPR/HIPAA DPA | none:Standard balanced commercial terms with no elevated risk',
    levelsSpec: '1, 2, 3, 4, 5',
  },
];

const DEFAULT_CUSTOM_DATASET_JSONL = [
  JSON.stringify({
    id: 'clause-01',
    clause_text: 'Either party may terminate this Agreement upon sixty (60) days prior written notice for convenience without penalty.',
    expected_compliant: 'yes',
    expected_risk_category: 'none',
  }),
  JSON.stringify({
    id: 'clause-02',
    clause_text: 'Customer shall indemnify, defend, and hold harmless Vendor against any and all claims or damages of any kind without limitation or cap.',
    expected_compliant: 'no',
    expected_risk_category: 'indemnity',
  }),
  JSON.stringify({
    id: 'clause-03',
    clause_text: 'This Subscription Term shall automatically renew for successive three (3) year periods unless Customer gives notice at least 180 days prior.',
    expected_compliant: 'no',
    expected_risk_category: 'auto_renewal',
  }),
  JSON.stringify({
    id: 'clause-04',
    clause_text: 'Vendor may transfer and process Customer EU personal data in any global jurisdiction at its sole discretion without Standard Contractual Clauses.',
    expected_compliant: 'no',
    expected_risk_category: 'data_residency',
  }),
  JSON.stringify({
    id: 'clause-05',
    clause_text: 'Total aggregate liability of either party under this Agreement shall not exceed the fees paid by Customer in the preceding 12 months.',
    expected_compliant: 'yes',
    expected_risk_category: 'none',
  }),
  JSON.stringify({
    id: 'clause-06',
    clause_text: 'All Customer data shall remain encrypted at rest (AES-256) in US-Central1 and EU-West1 regions in accordance with the signed GDPR DPA.',
    expected_compliant: 'yes',
    expected_risk_category: 'none',
  }),
  JSON.stringify({
    id: 'clause-07',
    clause_text: 'Unless cancelled 90 days before anniversary, agreement renews automatically at a mandatory 25% annual price increase.',
    expected_compliant: 'no',
    expected_risk_category: 'auto_renewal',
  }),
  JSON.stringify({
    id: 'clause-08',
    clause_text: 'Customer assumes unlimited financial responsibility for any third-party patent infringement allegation arising from Vendor software.',
    expected_compliant: 'no',
    expected_risk_category: 'indemnity',
  }),
].join('\n');

@customElement('dgem-batch-runner')
export class DgemBatchRunner extends LitElement {
  @property({ type: String, reflect: true }) resolvedTheme: 'light' | 'dark' = 'light';
  @property({ type: String }) backendTarget: 'vertex_first' | 'cloudrun' | 'vertex' | 'local' = 'vertex_first';
  /** Prompt layout override from the Studio ('' = policy or server default). */
  @property({ type: String }) promptLayout = '';
  /** Post-hoc temperature from the Studio (0 = policy or none). */
  @property({ type: Number }) temperature = 0;
  @property({ type: String }) vertexUrl = '';

  @state() private suites: BatchPresetSuite[] = [];
  @state() private selectedSuiteId = 'enterprise_multislot_25';
  @state() private concurrency = 4;
  @state() private cascadeMode: 'off' | 'on_miss' | 'entropy' = 'off';
  @state() private cascadeModel = 'gemini-3.8-flash';
  /** How much of Stage 1's answer Gemini sees: '' = server default, 'full', 'soft' or 'none' (issue #75). */
  @state() private stage2Prior: '' | 'full' | 'soft' | 'none' = '';
  @state() private cascadeModels: string[] = [
    'gemini-3.8-flash',
    'gemini-3.5-flash-lite',
  ];
  @state() private cascadeThreshold = 0.35;
  @state() private rowFilter: 'all' | 'miss' | 'high_entropy' = 'all';
  @state() private rows: BatchTableRow[] = [];
  @state() private running = false;
  @state() private completedItems = 0;
  @state() private totalItems = 0;
  @state() private wallElapsedSec = 0;
  @state() private loadingPresets = true;

  // Custom Template Builder & Dataset Uploader State
  @state() private builderMode: 'visual' | 'raw' = 'visual';
  @state() private customPolicyName = 'batch/contract_audit';
  @state() private customInstructions =
    'You are an enterprise policy auditor evaluating a vendor contract clause.';
  @state() private customStateVars = 'clause_text';
  @state() private customSlots: CustomBuilderSlot[] = [...DEFAULT_CUSTOM_SLOTS];
  @state() private customRawTemplate = '';
  @state() private customDatasetText = DEFAULT_CUSTOM_DATASET_JSONL;
  @state() private customParseMessage = '';
  @state() private copiedTemplateBadge = false;

  private abortRun = false;
  private wallTimer?: number;
  private wallStartMs = 0;

  static styles = css`
    :host {
      display: block;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text-primary, #0f172a);

      --bg-surface: #ffffff;
      --bg-subtle: #f8fafc;
      --bg-muted: #f1f5f9;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --text-primary: #0f172a;
      --text-secondary: #475569;
      --text-muted: #64748b;
      --brand: #1447e6;
      --brand-soft: #eff6ff;
      --brand-border: #bfdbfe;
      --pass-fg: #15803d;
      --pass-bg: #dcfce7;
      --pass-border: #86efac;
      --miss-fg: #b91c1c;
      --miss-bg: #fee2e2;
      --miss-border: #fca5a5;
      --warn-fg: #b45309;
      --warn-bg: #fef3c7;
    }

    :host([resolvedTheme='dark']) {
      --bg-surface: #0f172a;
      --bg-subtle: #1e293b;
      --bg-muted: #0b1120;
      --border: #1e293b;
      --border-strong: #334155;
      --text-primary: #f8fafc;
      --text-secondary: #cbd5e1;
      --text-muted: #94a3b8;
      --brand: #3b82f6;
      --brand-soft: rgba(59, 130, 246, 0.15);
      --brand-border: rgba(59, 130, 246, 0.4);
      --pass-fg: #4ade80;
      --pass-bg: rgba(22, 163, 74, 0.2);
      --pass-border: rgba(74, 222, 128, 0.35);
      --miss-fg: #f87171;
      --miss-bg: rgba(220, 38, 38, 0.2);
      --miss-border: rgba(248, 113, 113, 0.35);
      --warn-fg: #fbbf24;
      --warn-bg: rgba(245, 158, 11, 0.2);
    }

    .batch-shell {
      display: flex;
      flex-direction: column;
      gap: 1.15rem;
    }

    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.25rem 1.4rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .header-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }

    .title-group h2 {
      margin: 0;
      font-size: 1.15rem;
      font-weight: 700;
      font-family: 'Google Sans', 'Inter', sans-serif;
      color: var(--text-primary);
    }

    .title-group p {
      margin: 0.3rem 0 0;
      font-size: 0.84rem;
      color: var(--text-secondary);
      line-height: 1.45;
    }

    .suite-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.1rem;
    }

    .suite-card {
      text-align: left;
      background: var(--bg-subtle);
      border: 1px solid var(--border);
      border-radius: 11px;
      padding: 0.85rem 1rem;
      cursor: pointer;
      transition: all 0.14s ease;
      font-family: inherit;
      color: inherit;
    }

    .suite-card:hover {
      border-color: var(--brand);
    }

    .suite-card.active {
      background: var(--brand-soft);
      border-color: var(--brand);
      box-shadow: 0 0 0 1px var(--brand);
    }

    .suite-card-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.35rem;
    }

    .suite-cat {
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--brand);
    }

    .suite-badge {
      font-size: 0.68rem;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      padding: 0.14rem 0.45rem;
      border-radius: 999px;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-secondary);
    }

    .suite-title {
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 0.25rem;
    }

    .suite-desc {
      font-size: 0.76rem;
      color: var(--text-muted);
      line-height: 1.38;
    }

    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.85rem;
      padding-top: 0.85rem;
      border-top: 1px solid var(--border);
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .control-label {
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .seg-group {
      display: inline-flex;
      background: var(--bg-subtle);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 2px;
    }

    .seg-btn {
      border: none;
      background: transparent;
      color: var(--text-secondary);
      font-size: 0.76rem;
      font-weight: 600;
      padding: 0.34rem 0.65rem;
      border-radius: 6px;
      cursor: pointer;
      font-family: inherit;
    }

    .seg-btn.active {
      background: var(--bg-surface);
      color: var(--brand);
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
    }

    .run-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.6rem 1.2rem;
      border-radius: 9px;
      border: none;
      background: var(--brand);
      color: #ffffff;
      font-size: 0.86rem;
      font-weight: 700;
      cursor: pointer;
      font-family: 'Google Sans', 'Inter', sans-serif;
      box-shadow: 0 1px 3px rgba(20, 71, 230, 0.3);
      transition: all 0.14s ease;
    }

    .run-btn:hover {
      filter: brightness(1.07);
    }

    .run-btn.stop {
      background: #dc2626;
    }

    /* Live Scoreboard Strip */
    .scoreboard {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 0.75rem;
    }

    .metric-tile {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.9rem 1.05rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .metric-label {
      font-size: 0.71rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    .metric-val {
      font-size: 1.35rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-primary);
      display: flex;
      align-items: baseline;
      gap: 0.4rem;
    }

    .metric-sub {
      font-size: 0.74rem;
      font-weight: 500;
      color: var(--text-muted);
      font-family: 'Inter', sans-serif;
    }

    .progress-track {
      width: 100%;
      height: 5px;
      background: var(--bg-subtle);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 0.25rem;
    }

    .progress-fill {
      height: 100%;
      background: var(--brand);
      transition: width 0.2s ease;
    }

    /* Results Table */
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--bg-surface);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.81rem;
    }

    thead th {
      text-align: left;
      padding: 0.7rem 0.85rem;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      background: var(--bg-subtle);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }

    tbody tr {
      border-bottom: 1px solid var(--border);
      transition: background 0.12s ease;
    }

    tbody tr:last-child {
      border-bottom: none;
    }

    tbody tr:hover {
      background: var(--bg-subtle);
    }

    tbody td {
      padding: 0.65rem 0.85rem;
      vertical-align: middle;
    }

    .col-idx {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      white-space: nowrap;
    }

    .state-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.28rem;
      padding: 0.18rem 0.52rem;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
    }

    .state-pill.idle,
    .state-pill.queued {
      background: var(--bg-subtle);
      color: var(--text-muted);
      border: 1px solid var(--border);
    }

    .state-pill.running {
      background: var(--brand-soft);
      color: var(--brand);
      border: 1px solid var(--brand-border);
      animation: pulse 1.2s infinite ease-in-out;
    }

    .state-pill.pass {
      background: var(--pass-bg);
      color: var(--pass-fg);
      border: 1px solid var(--pass-border);
    }

    .state-pill.miss,
    .state-pill.error {
      background: var(--miss-bg);
      color: var(--miss-fg);
      border: 1px solid var(--miss-border);
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.55; }
    }

    .q-cell {
      display: flex;
      flex-direction: column;
      gap: 0.18rem;
      max-width: 460px;
    }

    .q-head {
      display: flex;
      align-items: center;
      gap: 0.45rem;
      flex-wrap: wrap;
    }

    .q-slot {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 0.79rem;
      color: var(--text-primary);
    }

    .q-domain {
      font-size: 0.66rem;
      font-weight: 600;
      padding: 0.08rem 0.38rem;
      border-radius: 4px;
      background: var(--bg-subtle);
      border: 1px solid var(--border);
      color: var(--text-muted);
    }

    .q-preview {
      font-size: 0.75rem;
      color: var(--text-secondary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .type-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      font-weight: 600;
      padding: 0.14rem 0.45rem;
      border-radius: 5px;
      background: var(--bg-subtle);
      border: 1px solid var(--border);
      color: var(--text-secondary);
    }

    .val-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.18rem 0.5rem;
      border-radius: 6px;
      display: inline-block;
    }

    .val-pill.match {
      background: var(--pass-bg);
      color: var(--pass-fg);
    }

    .val-pill.mismatch {
      background: var(--miss-bg);
      color: var(--miss-fg);
    }

    .val-pill.expected {
      background: var(--bg-subtle);
      color: var(--text-primary);
      border: 1px solid var(--border);
    }

    .conf-cell {
      display: flex;
      flex-direction: column;
      gap: 0.22rem;
      min-width: 115px;
    }

    .conf-top {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
    }

    .conf-pct {
      font-weight: 700;
      color: var(--text-primary);
    }

    .conf-ent {
      font-size: 0.67rem;
      color: var(--text-muted);
    }

    .conf-bar {
      width: 100%;
      height: 4px;
      background: var(--bg-subtle);
      border-radius: 999px;
      overflow: hidden;
    }

    .conf-fill {
      height: 100%;
      border-radius: 999px;
    }

    .ms-cell {
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
    }

    .ms-primary {
      font-weight: 700;
      font-size: 0.78rem;
      color: var(--text-primary);
    }

    .ms-sub {
      font-size: 0.67rem;
      color: var(--text-muted);
    }

    .inspect-btn {
      border: 1px solid var(--border);
      background: var(--bg-subtle);
      color: var(--text-secondary);
      border-radius: 6px;
      padding: 0.22rem 0.5rem;
      font-size: 0.7rem;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
      white-space: nowrap;
    }

    .inspect-btn:hover {
      border-color: var(--brand);
      color: var(--brand);
    }

    /* Custom Template & Dataset Builder Drawer */
    .custom-drawer {
      margin-top: 1.1rem;
      padding: 1.15rem;
      border-radius: 12px;
      border: 1px solid var(--brand-border);
      background: var(--bg-subtle);
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 1.25rem;
    }

    @media (max-width: 1020px) {
      .custom-drawer {
        grid-template-columns: 1fr;
      }
    }

    .builder-pane {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .pane-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .pane-title {
      font-size: 0.84rem;
      font-weight: 700;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .field-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.65rem;
    }

    .field-group {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .field-label {
      font-size: 0.69rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
    }

    .text-input,
    .code-area,
    .select-input {
      width: 100%;
      padding: 0.45rem 0.65rem;
      border-radius: 7px;
      border: 1px solid var(--border-strong);
      background: var(--bg-surface);
      color: var(--text-primary);
      font-size: 0.78rem;
      font-family: inherit;
    }

    .code-area {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.74rem;
      line-height: 1.45;
      resize: vertical;
      min-height: 185px;
    }

    .slot-builder-card {
      border: 1px solid var(--border);
      background: var(--bg-subtle);
      border-radius: 8px;
      padding: 0.65rem;
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }

    .slot-builder-top {
      display: grid;
      grid-template-columns: 130px 110px 1fr auto;
      gap: 0.45rem;
      align-items: center;
    }

    .mini-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      border: 1px solid var(--border-strong);
      background: var(--bg-surface);
      color: var(--text-secondary);
      border-radius: 7px;
      padding: 0.32rem 0.65rem;
      font-size: 0.73rem;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
      white-space: nowrap;
    }

    .mini-btn:hover {
      border-color: var(--brand);
      color: var(--brand);
    }

    .mini-btn.danger:hover {
      border-color: var(--miss-fg);
      color: var(--miss-fg);
    }

    .action-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 0.25rem;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this.customRawTemplate = this.compileVisualTemplate();
    this.loadBatchPresets();
    this.loadBackendCascadeConfig();
  }

  private async loadBackendCascadeConfig() {
    try {
      const resp = await fetch('/api/backend-config');
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data?.cascade_models) && data.cascade_models.length > 0) {
        this.cascadeModels = data.cascade_models;
      }
      if (typeof data?.default_cascade_model === 'string' && data.default_cascade_model.trim()) {
        this.cascadeModel = data.default_cascade_model.trim();
      }
    } catch {
      // keep default ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.wallTimer) {
      window.clearInterval(this.wallTimer);
    }
  }

  private compileVisualTemplate(): string {
    const questions = this.customSlots.map((s) => {
      const q: Record<string, any> = {
        id: (s.id || 'slot').trim(),
        type: s.type || 'boolean',
        instructions: s.instructions || 'Evaluate this slot.',
      };
      if (s.type === 'choice') {
        const parts = (s.optionsSpec || 'option_a:First option | option_b:Second option')
          .split('|')
          .map((p) => p.trim())
          .filter(Boolean);
        q.options = parts.map((part) => {
          const colonIdx = part.indexOf(':');
          if (colonIdx > 0) {
            return {
              name: part.slice(0, colonIdx).trim(),
              description: part.slice(colonIdx + 1).trim(),
            };
          }
          return { name: part.trim(), description: part.trim() };
        });
      } else if (s.type === 'score') {
        q.levels = (s.levelsSpec || '1, 2, 3, 4, 5')
          .split(',')
          .map((l) => l.trim())
          .filter(Boolean);
      }
      return q;
    });

    const varNames = (this.customStateVars || 'input')
      .split(',')
      .map((v) => v.trim())
      .filter(Boolean);

    const stateLines = varNames.map(
      (v, i) => `    "${v}": {{ default "" .${v} | toJson }}${i < varNames.length - 1 ? ',' : ''}`
    );

    const schemaJson = JSON.stringify(
      {
        instructions: this.customInstructions,
        questions,
        samples: 'auto',
      },
      null,
      4
    );

    return `{\n  "schema": ${schemaJson.replace(/\n/g, '\n  ')},\n  "state": {\n${stateLines.join('\n')}\n  }\n}`;
  }

  private getEffectiveCustomTemplate(): string {
    return this.builderMode === 'raw' && this.customRawTemplate.trim()
      ? this.customRawTemplate
      : this.compileVisualTemplate();
  }

  private parseCustomDatasetItems(): BatchPresetItem[] {
    const raw = (this.customDatasetText || '').trim();
    if (!raw) return [];
    const tmplStr = this.getEffectiveCustomTemplate();
    const lines = raw.split(/\r?\n/).filter((l) => l.trim().length > 0);
    const items: BatchPresetItem[] = [];

    // Detect if CSV vs JSONL
    const firstTrimmed = lines[0].trim();
    let parsedRecords: Record<string, any>[] = [];

    if (firstTrimmed.startsWith('{') || firstTrimmed.startsWith('[')) {
      if (firstTrimmed.startsWith('[')) {
        try {
          const arr = JSON.parse(raw);
          if (Array.isArray(arr)) parsedRecords = arr;
        } catch {
          // fallback to line-by-line
        }
      }
      if (parsedRecords.length === 0) {
        lines.forEach((line) => {
          try {
            parsedRecords.push(JSON.parse(line));
          } catch {
            // skip malformed line
          }
        });
      }
    } else {
      // Simple CSV parser (handles quoted strings)
      const parseCSVLine = (line: string): string[] => {
        const out: string[] = [];
        let cur = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
          const ch = line[i];
          if (ch === '"') {
            inQuotes = !inQuotes;
          } else if (ch === ',' && !inQuotes) {
            out.push(cur.trim());
            cur = '';
          } else {
            cur += ch;
          }
        }
        out.push(cur.trim());
        return out;
      };
      const headers = parseCSVLine(lines[0]);
      for (let i = 1; i < lines.length; i++) {
        const cols = parseCSVLine(lines[i]);
        const obj: Record<string, any> = {};
        headers.forEach((h, idx) => {
          if (h) obj[h] = cols[idx] ?? '';
        });
        parsedRecords.push(obj);
      }
    }

    parsedRecords.forEach((rec, idx) => {
      const id = String(rec.id || `custom-${String(idx + 1).padStart(2, '0')}`);
      const expectedObj = typeof rec.expected === 'object' && rec.expected !== null ? rec.expected : {};
      const variables: Record<string, any> =
        typeof rec.variables === 'object' && rec.variables !== null ? { ...rec.variables } : {};

      if (Object.keys(variables).length === 0) {
        for (const [k, v] of Object.entries(rec)) {
          if (k !== 'id' && k !== 'expected' && !k.startsWith('expected_')) {
            variables[k] = v;
          }
        }
      }

      const firstVal = Object.values(variables)[0];
      const preview = String(rec.preview || firstVal || JSON.stringify(variables)).slice(0, 140);

      const expected_slots: BatchExpectedSlot[] = this.customSlots.map((s) => {
        const slotId = (s.id || 'slot').trim();
        const expVal =
          expectedObj[slotId] !== undefined
            ? String(expectedObj[slotId])
            : rec[`expected_${slotId}`] !== undefined
              ? String(rec[`expected_${slotId}`])
              : '—';
        return {
          question: slotId,
          label: s.instructions || slotId,
          type: s.type === 'boolean' ? 'bool' : s.type,
          expected: expVal,
        };
      });

      items.push({
        id,
        domain: this.customPolicyName || 'custom_experiment',
        tier: 'custom',
        preview,
        custom_template: tmplStr,
        variables,
        expected_slots,
      });
    });

    this.customParseMessage = `Ready: ${items.length} dataset rows × ${this.customSlots.length} question slots (${items.length * this.customSlots.length} total evaluations)`;
    return items;
  }

  private buildCustomSuiteObject(): BatchPresetSuite {
    const items = this.parseCustomDatasetItems();
    return {
      id: 'custom_builder',
      title: '🛠️ Custom Template & Dataset (.jsonl / .csv)',
      category: 'Self-Service Experiment Builder',
      badge: `${items.length} items · ${items.length * this.customSlots.length} slots`,
      description:
        'Design your own multi-slot Policy-as-Template (.json.tmpl), upload or paste any .jsonl / .csv dataset, and run live evaluations.',
      default_concurrency: 4,
      total_items: items.length,
      total_questions: items.length * this.customSlots.length,
      items,
    };
  }

  private refreshCustomSuiteIfActive() {
    if (this.builderMode === 'visual') {
      this.customRawTemplate = this.compileVisualTemplate();
    }
    const customSuite = this.buildCustomSuiteObject();
    const existingIdx = this.suites.findIndex((s) => s.id === 'custom_builder');
    if (existingIdx >= 0) {
      const updated = [...this.suites];
      updated[existingIdx] = customSuite;
      this.suites = updated;
    } else {
      this.suites = [...this.suites, customSuite];
    }
    if (this.selectedSuiteId === 'custom_builder') {
      this.selectSuite('custom_builder');
    }
  }

  private async handleDatasetFileUpload(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const text = await file.text();
    this.customDatasetText = text.trim();

    // Auto-detect state variable columns from first record
    const items = this.parseCustomDatasetItems();
    if (items.length > 0) {
      const cols = Object.keys(items[0].variables);
      if (cols.length > 0) {
        this.customStateVars = cols.join(', ');
      }
    }
    this.refreshCustomSuiteIfActive();
  }

  private copyCustomTemplate() {
    const tmpl = this.getEffectiveCustomTemplate();
    navigator.clipboard.writeText(tmpl);
    this.copiedTemplateBadge = true;
    setTimeout(() => {
      this.copiedTemplateBadge = false;
    }, 1800);
  }

  private downloadCustomTemplate() {
    const tmpl = this.getEffectiveCustomTemplate();
    const blob = new Blob([tmpl], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const cleanName = (this.customPolicyName || 'custom_policy').replace(/[^a-zA-Z0-9_-]/g, '_');
    a.href = url;
    a.download = `${cleanName}.json.tmpl`;
    a.click();
    URL.revokeObjectURL(url);
  }

  private exportResults(format: 'csv' | 'jsonl') {
    if (this.rows.length === 0) return;
    let content = '';
    let mime = 'text/plain';
    let ext = format;

    if (format === 'jsonl') {
      content =
        this.rows
          .map((r) =>
            JSON.stringify({
              item_index: r.itemIndex,
              item_id: r.item.id,
              domain: r.item.domain,
              question: r.slot.question,
              type: r.slot.type,
              state: r.state,
              predicted: r.predicted ?? '',
              expected: r.slot.expected,
              confidence: r.confidence ?? null,
              entropy_nats: r.entropy !== undefined ? Number(r.entropy.toFixed(4)) : null,
              server_ms: r.serverMs ?? null,
              round_trip_ms: r.roundTripMs ?? null,
              preview: r.item.preview,
            })
          )
          .join('\n') + '\n';
      mime = 'application/x-ndjson';
    } else {
      const esc = (v: any) => `"${String(v ?? '').replace(/"/g, '""')}"`;
      const header = [
        'item_index',
        'item_id',
        'domain',
        'question',
        'type',
        'state',
        'predicted',
        'expected',
        'confidence',
        'entropy_nats',
        'server_ms',
        'round_trip_ms',
        'preview',
      ].join(',');
      const lines = this.rows.map((r) =>
        [
          r.itemIndex,
          esc(r.item.id),
          esc(r.item.domain),
          esc(r.slot.question),
          esc(r.slot.type),
          esc(r.state),
          esc(r.predicted ?? ''),
          esc(r.slot.expected),
          r.confidence !== undefined ? r.confidence.toFixed(4) : '',
          r.entropy !== undefined ? r.entropy.toFixed(4) : '',
          r.serverMs ?? '',
          r.roundTripMs ?? '',
          esc(r.item.preview),
        ].join(',')
      );
      content = [header, ...lines].join('\n') + '\n';
      mime = 'text/csv';
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dgem_batch_${this.selectedSuiteId}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  private async loadBatchPresets() {
    this.loadingPresets = true;
    try {
      const res = await fetch('/api/batch/presets');
      if (res.ok) {
        const data = await res.json();
        const serverSuites: BatchPresetSuite[] = data.suites || [];
        this.suites = [...serverSuites, this.buildCustomSuiteObject()];
        if (this.suites.length > 0) {
          this.selectSuite(this.suites[0].id);
        }
      } else {
        this.suites = [this.buildCustomSuiteObject()];
        this.selectSuite('custom_builder');
      }
    } catch (e) {
      console.error('Failed to load batch presets:', e);
      this.suites = [this.buildCustomSuiteObject()];
      this.selectSuite('custom_builder');
    } finally {
      this.loadingPresets = false;
    }
  }

  private selectSuite(suiteId: string) {
    if (this.running) return;
    this.selectedSuiteId = suiteId;
    const suite = this.suites.find((s) => s.id === suiteId);
    if (!suite) return;
    this.concurrency = suite.default_concurrency || 4;
    this.totalItems = suite.items.length;
    this.completedItems = 0;
    this.wallElapsedSec = 0;

    const newRows: BatchTableRow[] = [];
    suite.items.forEach((item, idx) => {
      const slots = item.expected_slots || [];
      slots.forEach((slot, sIdx) => {
        newRows.push({
          rowKey: `${item.id}:${slot.question}`,
          itemIndex: idx + 1,
          slotIndex: sIdx + 1,
          totalSlotsForItem: slots.length,
          item,
          slot,
          state: 'idle',
        });
      });
    });
    this.rows = newRows;
  }

  private normalizeAnswer(val: any): string {
    if (val === undefined || val === null) return '';
    const s = String(val).trim().toLowerCase();
    if (s === 'true') return 'yes';
    if (s === 'false') return 'no';
    return String(val).trim();
  }

  private isMatch(predicted: string | undefined, expected: string): boolean {
    if (!predicted) return false;
    if (!expected || expected === '—') return true; // Unlabeled dataset inference mode
    const p = this.normalizeAnswer(predicted).toLowerCase();
    const e = this.normalizeAnswer(expected).toLowerCase();
    return p === e;
  }

  private renderCustomBuilderDrawer() {
    if (this.selectedSuiteId !== 'custom_builder') return '';

    return html`
      <div class="custom-drawer">
        <!-- Left Pane: Step 1 — Policy Template Builder (.json.tmpl) -->
        <div class="builder-pane">
          <div class="pane-header">
            <span class="pane-title">1. Policy Template Builder (<code>.json.tmpl</code>)</span>
            <div class="seg-group">
              <button
                class="seg-btn ${this.builderMode === 'visual' ? 'active' : ''}"
                @click=${() => {
                  this.builderMode = 'visual';
                  this.refreshCustomSuiteIfActive();
                }}
              >
                Visual Slot Builder
              </button>
              <button
                class="seg-btn ${this.builderMode === 'raw' ? 'active' : ''}"
                @click=${() => {
                  this.customRawTemplate = this.compileVisualTemplate();
                  this.builderMode = 'raw';
                }}
              >
                Raw .json.tmpl
              </button>
            </div>
          </div>

          <div class="field-row">
            <div class="field-group">
              <span class="field-label">Experiment Name (Cloud Logging Tag)</span>
              <input
                class="text-input"
                .value=${this.customPolicyName}
                placeholder="e.g. batch/contract_audit"
                @input=${(e: Event) => {
                  this.customPolicyName = (e.target as HTMLInputElement).value;
                  this.refreshCustomSuiteIfActive();
                }}
              />
            </div>
            <div class="field-group">
              <span class="field-label">Dataset Variable Columns (comma-separated)</span>
              <input
                class="text-input"
                .value=${this.customStateVars}
                placeholder="e.g. clause_text, jurisdiction"
                @input=${(e: Event) => {
                  this.customStateVars = (e.target as HTMLInputElement).value;
                  this.refreshCustomSuiteIfActive();
                }}
              />
            </div>
          </div>

          ${this.builderMode === 'visual'
            ? html`
                <div class="field-group">
                  <span class="field-label">Global Policy Instructions (Static vLLM Prefix Cache)</span>
                  <input
                    class="text-input"
                    .value=${this.customInstructions}
                    @input=${(e: Event) => {
                      this.customInstructions = (e.target as HTMLInputElement).value;
                      this.refreshCustomSuiteIfActive();
                    }}
                  />
                </div>

                <div class="field-group">
                  <div class="pane-header">
                    <span class="field-label">
                      Simultaneous $O(1)$ Question Slots (${this.customSlots.length})
                    </span>
                    <button
                      class="mini-btn"
                      @click=${() => {
                        this.customSlots = [
                          ...this.customSlots,
                          {
                            id: `slot_${this.customSlots.length + 1}`,
                            type: 'boolean',
                            instructions: 'Is this condition satisfied?',
                            optionsSpec: 'option_a:First choice | option_b:Second choice',
                            levelsSpec: '1, 2, 3, 4, 5',
                          },
                        ];
                        this.refreshCustomSuiteIfActive();
                      }}
                    >
                      + Add Question Slot
                    </button>
                  </div>

                  ${this.customSlots.map(
                    (s, idx) => html`
                      <div class="slot-builder-card">
                        <div class="slot-builder-top">
                          <input
                            class="text-input"
                            title="Slot ID (matches expected_<id> in dataset)"
                            placeholder="slot_id"
                            .value=${s.id}
                            @input=${(e: Event) => {
                              const next = [...this.customSlots];
                              next[idx] = { ...s, id: (e.target as HTMLInputElement).value };
                              this.customSlots = next;
                              this.refreshCustomSuiteIfActive();
                            }}
                          />
                          <select
                            class="select-input"
                            .value=${s.type}
                            @change=${(e: Event) => {
                              const next = [...this.customSlots];
                              next[idx] = {
                                ...s,
                                type: (e.target as HTMLSelectElement).value as any,
                              };
                              this.customSlots = next;
                              this.refreshCustomSuiteIfActive();
                            }}
                          >
                            <option value="boolean">boolean</option>
                            <option value="choice">choice</option>
                            <option value="score">score</option>
                          </select>
                          <input
                            class="text-input"
                            placeholder="Slot question instructions..."
                            .value=${s.instructions}
                            @input=${(e: Event) => {
                              const next = [...this.customSlots];
                              next[idx] = {
                                ...s,
                                instructions: (e.target as HTMLInputElement).value,
                              };
                              this.customSlots = next;
                              this.refreshCustomSuiteIfActive();
                            }}
                          />
                          ${this.customSlots.length > 1
                            ? html`
                                <button
                                  class="mini-btn danger"
                                  title="Remove slot"
                                  @click=${() => {
                                    this.customSlots = this.customSlots.filter((_, i) => i !== idx);
                                    this.refreshCustomSuiteIfActive();
                                  }}
                                >
                                  ✕
                                </button>
                              `
                            : ''}
                        </div>

                        ${s.type === 'choice'
                          ? html`
                              <input
                                class="text-input"
                                placeholder="Options: name:description | name2:description2 (max 26 [A-Z])"
                                .value=${s.optionsSpec}
                                @input=${(e: Event) => {
                                  const next = [...this.customSlots];
                                  next[idx] = {
                                    ...s,
                                    optionsSpec: (e.target as HTMLInputElement).value,
                                  };
                                  this.customSlots = next;
                                  this.refreshCustomSuiteIfActive();
                                }}
                              />
                            `
                          : s.type === 'score'
                            ? html`
                                <input
                                  class="text-input"
                                  placeholder="Score levels (comma-separated): 1, 2, 3, 4, 5"
                                  .value=${s.levelsSpec}
                                  @input=${(e: Event) => {
                                    const next = [...this.customSlots];
                                    next[idx] = {
                                      ...s,
                                      levelsSpec: (e.target as HTMLInputElement).value,
                                    };
                                    this.customSlots = next;
                                    this.refreshCustomSuiteIfActive();
                                  }}
                                />
                              `
                            : ''}
                      </div>
                    `
                  )}
                </div>
              `
            : html`
                <div class="field-group">
                  <span class="field-label">Raw Go Policy Template (<code>schema</code> + <code>state</code>)</span>
                  <textarea
                    class="code-area"
                    .value=${this.customRawTemplate}
                    @input=${(e: Event) => {
                      this.customRawTemplate = (e.target as HTMLTextAreaElement).value;
                      this.refreshCustomSuiteIfActive();
                    }}
                  ></textarea>
                </div>
              `}

          <div class="action-bar">
            <span style="font-size: 0.72rem; color: var(--text-muted);">
              Static <code>"schema"</code> + dynamic <code>"state"</code> enables ~75% vLLM prefix-cache reuse.
            </span>
            <div style="display: flex; gap: 0.45rem;">
              <button class="mini-btn" @click=${this.copyCustomTemplate}>
                ${this.copiedTemplateBadge ? '✓ Copied!' : '📋 Copy .json.tmpl'}
              </button>
              <button class="mini-btn" @click=${this.downloadCustomTemplate}>
                ⬇ Download .json.tmpl
              </button>
            </div>
          </div>
        </div>

        <!-- Right Pane: Step 2 — Dataset Uploader (.jsonl / .csv) -->
        <div class="builder-pane">
          <div class="pane-header">
            <span class="pane-title">2. Dataset Rows (<code>.jsonl</code> or <code>.csv</code>)</span>
            <div style="display: flex; gap: 0.45rem; flex-wrap: wrap;">
              <label class="mini-btn" style="cursor: pointer;">
                📂 Upload .jsonl / .csv
                <input
                  type="file"
                  accept=".jsonl,.json,.csv,.tsv,.txt"
                  style="display: none;"
                  @change=${this.handleDatasetFileUpload}
                />
              </label>
              <button
                class="mini-btn"
                @click=${() => {
                  this.customSlots = [...DEFAULT_CUSTOM_SLOTS];
                  this.customStateVars = 'clause_text';
                  this.customDatasetText = DEFAULT_CUSTOM_DATASET_JSONL;
                  this.refreshCustomSuiteIfActive();
                }}
              >
                ⚡ Reset 8-Row Sample
              </button>
            </div>
          </div>

          <div class="field-group">
            <span class="field-label">
              Paste or Edit JSONL / CSV Rows (include <code>expected_&lt;slot_id&gt;</code> for live accuracy grading, or omit for unlabeled inference)
            </span>
            <textarea
              class="code-area"
              style="min-height: 235px;"
              .value=${this.customDatasetText}
              @input=${(e: Event) => {
                this.customDatasetText = (e.target as HTMLTextAreaElement).value;
                this.refreshCustomSuiteIfActive();
              }}
            ></textarea>
          </div>

          <div class="action-bar">
            <span style="font-size: 0.73rem; font-weight: 600; color: var(--brand);">
              ${this.customParseMessage}
            </span>
          </div>
        </div>
      </div>
    `;
  }

  private median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
      ? sorted[mid]
      : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
  }

  private async runBatch() {
    const suite = this.suites.find((s) => s.id === this.selectedSuiteId);
    if (!suite || this.running) return;

    this.abortRun = false;
    this.running = true;
    this.completedItems = 0;
    this.totalItems = suite.items.length;
    this.wallStartMs = performance.now();
    this.wallElapsedSec = 0;

    // Reset all rows to 'queued'
    this.rows = this.rows.map((r) => ({
      ...r,
      state: 'queued',
      predicted: undefined,
      confidence: undefined,
      entropy: undefined,
      serverMs: undefined,
      roundTripMs: undefined,
      errorMsg: undefined,
    }));

    this.dispatchEvent(new CustomEvent('batch-started', { bubbles: true, composed: true }));

    if (this.wallTimer) window.clearInterval(this.wallTimer);
    this.wallTimer = window.setInterval(() => {
      this.wallElapsedSec = Number(((performance.now() - this.wallStartMs) / 1000).toFixed(1));
    }, 100);

    const queue = suite.items.map((item, idx) => ({ item, itemIndex: idx + 1 }));
    let cursor = 0;

    const worker = async () => {
      while (!this.abortRun && cursor < queue.length) {
        const currentIdx = cursor++;
        const { item } = queue[currentIdx];

        // Mark this item's rows as 'running'
        this.rows = this.rows.map((r) =>
          r.item.id === item.id ? { ...r, state: 'running' } : r
        );

        const t0 = performance.now();
        try {
          const expectedMap: Record<string, string> = {};
          for (const s of item.expected_slots || []) {
            if (s.question && s.expected && s.expected !== '—') {
              expectedMap[s.question] = s.expected;
            }
          }

          const body: Record<string, any> = {
            variables: item.variables || {},
            backend: this.backendTarget,
            cascade_mode: this.cascadeMode,
            cascade_model: this.cascadeModel,
            ...(this.stage2Prior ? { stage2_prior: this.stage2Prior } : {}),
            cascade_threshold: this.cascadeThreshold,
            expected_answers: expectedMap,
          };
          if (this.promptLayout) {
            body.layout = this.promptLayout;
          }
          if (this.temperature > 0) {
            body.temperature = this.temperature;
          }
          if (this.vertexUrl) {
            body.vertex_url = this.vertexUrl;
          }
          if (item.image_url) {
            body.image_url = item.image_url;
          }
          if (item.custom_template) {
            body.custom_template = item.custom_template;
          } else {
            body.template = item.template || 'support_triage';
          }

          const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            'X-DGem-Surface': 'web_studio_batch',
            'X-DGem-Template': item.template || `batch/${item.domain || suite.id}`,
            'X-DGem-Backend': this.backendTarget,
            'X-DGem-Cascade-Mode': this.cascadeMode,
            'X-DGem-Cascade-Model': this.cascadeModel,
          };
          if (this.vertexUrl) {
            headers['X-DGem-Vertex-Url'] = this.vertexUrl;
          }

          const res = await fetch('/api/decide', {
            method: 'POST',
            headers,
            body: JSON.stringify(body),
          });
          const rttMs = Math.round(performance.now() - t0);

          if (!res.ok) {
            const errText = await res.text();
            this.rows = this.rows.map((r) =>
              r.item.id === item.id
                ? { ...r, state: 'error', roundTripMs: rttMs, errorMsg: `HTTP ${res.status}: ${errText}` }
                : r
            );
          } else {
            const payload = await res.json();
            const answers = payload.answers || {};
            const cascadeInfo = payload.cascade || null;
            const timing = payload.diagnostics?.timing || {};
            const serverMs =
              timing.total_ms ||
              timing.denoise_ms ||
              payload.gpu_forward_ms ||
              payload.diagnostics?.server_denoise_ms ||
              rttMs;
            const qDiags = payload.diagnostics?.questions || {};

            this.rows = this.rows.map((r) => {
              if (r.item.id !== item.id) return r;
              const slotAns = answers[r.slot.question];
              const slotCascade = cascadeInfo?.slots?.[r.slot.question];
              const rawVal =
                slotAns?.value !== undefined
                  ? slotAns.value
                  : slotAns?.choice !== undefined && slotAns?.choice !== ''
                    ? slotAns.choice
                    : slotAns?.level !== undefined && slotAns?.level !== ''
                      ? slotAns.level
                      : slotAns?.label;
              const predStr = this.normalizeAnswer(rawVal);
              const matched = this.isMatch(predStr, r.slot.expected);
              const conf = typeof slotAns?.confidence === 'number' ? slotAns.confidence : 0;
              const qDiag = qDiags[r.slot.question];
              let ent =
                typeof slotCascade?.stage1_entropy === 'number' && slotCascade.stage1_entropy > 0
                  ? slotCascade.stage1_entropy
                  : typeof slotAns?.entropy === 'number' && slotAns.entropy > 0
                    ? slotAns.entropy
                    : typeof qDiag?.entropy === 'number' && qDiag.entropy > 0
                      ? qDiag.entropy
                      : 0;
              if (ent === 0 && slotAns?.probabilities && typeof slotAns.probabilities === 'object') {
                for (const p of Object.values(slotAns.probabilities)) {
                  const prob = Number(p);
                  if (prob > 1e-12) {
                    ent -= prob * Math.log(prob);
                  }
                }
              }

              return {
                ...r,
                state: matched ? 'pass' : 'miss',
                predicted: predStr || '—',
                confidence: conf,
                entropy: ent,
                serverMs: Math.round(serverMs),
                roundTripMs: rttMs,
                escalated: Boolean(slotCascade?.escalated),
                stage1Predicted: slotCascade?.stage1_value || predStr || '—',
                cascadeModel: cascadeInfo?.model || this.cascadeModel,
                cascadeExplanation: slotCascade?.explanation || '',
                cascadeLatencyMs: cascadeInfo?.latency_ms || 0,
              };
            });
          }
        } catch (err: any) {
          const rttMs = Math.round(performance.now() - t0);
          this.rows = this.rows.map((r) =>
            r.item.id === item.id
              ? { ...r, state: 'error', roundTripMs: rttMs, errorMsg: err?.message || 'Network error' }
              : r
          );
        } finally {
          this.completedItems += 1;
        }
      }
    };

    const workers = Array.from({ length: Math.min(this.concurrency, queue.length) }, () => worker());
    await Promise.all(workers);

    if (this.wallTimer) {
      window.clearInterval(this.wallTimer);
      this.wallTimer = undefined;
    }
    this.wallElapsedSec = Number(((performance.now() - this.wallStartMs) / 1000).toFixed(1));
    this.running = false;
    this.dispatchEvent(new CustomEvent('batch-completed', { bubbles: true, composed: true }));
  }

  private stopBatch() {
    this.abortRun = true;
    this.running = false;
    if (this.wallTimer) {
      window.clearInterval(this.wallTimer);
      this.wallTimer = undefined;
    }
  }

  private openRowInStudio(row: BatchTableRow) {
    this.dispatchEvent(
      new CustomEvent('inspect-batch-item', {
        detail: {
          template: row.item.template,
          customTemplate: row.item.custom_template,
          variables: row.item.variables,
          preview: row.item.preview,
        },
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    const completedRows = this.rows.filter((r) => r.state === 'pass' || r.state === 'miss');
    const passRows = this.rows.filter((r) => r.state === 'pass');
    const escalatedRows = completedRows.filter((r) => r.escalated);
    const stage1PassCount = completedRows.filter((r) =>
      this.isMatch(r.stage1Predicted || r.predicted, r.slot.expected)
    ).length;
    const accPct =
      completedRows.length > 0 ? ((passRows.length / completedRows.length) * 100).toFixed(1) : '—';
    const stage1AccPct =
      completedRows.length > 0
        ? ((stage1PassCount / completedRows.length) * 100).toFixed(1)
        : '—';

    // Collect per-item latencies (1 per itemIndex)
    const seenItems = new Set<number>();
    const serverLatencies: number[] = [];
    const rttLatencies: number[] = [];
    for (const r of completedRows) {
      if (!seenItems.has(r.itemIndex)) {
        seenItems.add(r.itemIndex);
        if (r.serverMs !== undefined) serverLatencies.push(r.serverMs);
        if (r.roundTripMs !== undefined) rttLatencies.push(r.roundTripMs);
      }
    }
    const serverP50 = this.median(serverLatencies);
    const rttP50 = this.median(rttLatencies);
    const pctDone = this.totalItems > 0 ? Math.round((this.completedItems / this.totalItems) * 100) : 0;

    const visibleRows = this.rows.filter((r) => {
      if (this.rowFilter === 'miss') return r.state === 'miss' || r.state === 'error';
      if (this.rowFilter === 'high_entropy') return (r.entropy || 0) >= 0.25;
      return true;
    });

    return html`
      <div class="batch-shell">
        <!-- Top Suite Selector & Execution Controls Card -->
        <div class="card">
          <div class="header-row">
            <div class="title-group">
              <h2>Batch Decision Evaluation & Live Streaming Telemetry</h2>
              <p>
                Select a ground-truth challenge suite below and run concurrent single-pass DiffusionGemma evaluations with optional Stage-2 Vertex AI Gemini 3.x escalation (<code>${this.cascadeModel}</code>).
              </p>
            </div>
            <div>
              ${this.running
                ? html`
                    <button class="run-btn stop" @click=${this.stopBatch}>
                      <span>■ Stop Batch (${this.completedItems}/${this.totalItems})</span>
                    </button>
                  `
                : html`
                    <button class="run-btn" @click=${this.runBatch}>
                      <span>▶ Run Batch (${this.totalItems} items)</span>
                    </button>
                  `}
            </div>
          </div>

          <div class="suite-grid">
            ${this.suites.map(
              (s) => html`
                <button
                  class="suite-card ${this.selectedSuiteId === s.id ? 'active' : ''}"
                  @click=${() => this.selectSuite(s.id)}
                >
                  <div class="suite-card-top">
                    <span class="suite-cat">${s.category}</span>
                    <span class="suite-badge">${s.badge}</span>
                  </div>
                  <div class="suite-title">${s.title}</div>
                  <div class="suite-desc">${s.description}</div>
                </button>
              `
            )}
          </div>

          ${this.renderCustomBuilderDrawer()}

          <div class="toolbar">
            <div class="control-group">
              <span class="control-label">Concurrency</span>
              <div class="seg-group">
                ${[1, 4, 8].map(
                  (c) => html`
                    <button
                      class="seg-btn ${this.concurrency === c ? 'active' : ''}"
                      @click=${() => {
                        if (!this.running) this.concurrency = c;
                      }}
                    >
                      ${c}x Workers
                    </button>
                  `
                )}
              </div>
            </div>

            <div class="control-group">
              <span class="control-label">Stage-2 Gemini Cascade</span>
              <div class="seg-group">
                <button
                  class="seg-btn ${this.cascadeMode === 'off' ? 'active' : ''}"
                  title="Stage-1 DiffusionGemma only (no Gemini escalation)"
                  @click=${() => {
                    if (!this.running) this.cascadeMode = 'off';
                  }}
                >
                  Off (dgemma Only)
                </button>
                <button
                  class="seg-btn ${this.cascadeMode === 'on_miss' ? 'active' : ''}"
                  title="Forward missed cases to Vertex AI ${this.cascadeModel} with dgemma's Tier-1 prior distribution"
                  @click=${() => {
                    if (!this.running) this.cascadeMode = 'on_miss';
                  }}
                >
                  ⚡ Forward on Miss → ${this.cascadeModel}
                </button>
                <button
                  class="seg-btn ${this.cascadeMode === 'entropy' ? 'active' : ''}"
                  title="Forward high-entropy slots (H >= 0.35 nats) to Vertex AI ${this.cascadeModel} (production cascade)"
                  @click=${() => {
                    if (!this.running) this.cascadeMode = 'entropy';
                  }}
                >
                  ⚡ Entropy H ≥ 0.35 → ${this.cascadeModel}
                </button>
              </div>
            </div>

            ${this.cascadeMode !== 'off'
              ? html`
                  <div class="control-group">
                    <span class="control-label">Stage-2 Gemini 3.x Model</span>
                    <div class="seg-group">
                      ${this.cascadeModels.map(
                        (m) => html`
                          <button
                            class="seg-btn ${this.cascadeModel === m ? 'active' : ''}"
                            @click=${() => {
                              if (!this.running) this.cascadeModel = m;
                            }}
                          >
                            ${m}
                          </button>
                        `
                      )}
                    </div>
                  </div>
                  <div class="control-group">
                    <span class="control-label" title="How much of dgem's answer Gemini sees. Full: answer, confidence and distribution. Soft: only that dgem leaned toward an answer and may be wrong. None: the question alone.">Stage-1 hint to Gemini</span>
                    <div class="seg-group">
                      ${(['', 'full', 'soft', 'none'] as const).map(
                        (m) => html`
                          <button
                            class="seg-btn ${this.stage2Prior === m ? 'active' : ''}"
                            @click=${() => {
                              if (!this.running) this.stage2Prior = m;
                            }}
                          >
                            ${m || 'default'}
                          </button>
                        `
                      )}
                    </div>
                  </div>
                `
              : ''}

            <div class="control-group">
              <span class="control-label">Filter Rows</span>
              <div class="seg-group">
                <button
                  class="seg-btn ${this.rowFilter === 'all' ? 'active' : ''}"
                  @click=${() => (this.rowFilter = 'all')}
                >
                  All (${this.rows.length})
                </button>
                <button
                  class="seg-btn ${this.rowFilter === 'miss' ? 'active' : ''}"
                  @click=${() => (this.rowFilter = 'miss')}
                >
                  ✗ Misses (${this.rows.filter((r) => r.state === 'miss' || r.state === 'error').length})
                </button>
                <button
                  class="seg-btn ${this.rowFilter === 'high_entropy' ? 'active' : ''}"
                  @click=${() => (this.rowFilter = 'high_entropy')}
                >
                  High H ≥ 0.25 (${this.rows.filter((r) => (r.entropy || 0) >= 0.25).length})
                </button>
              </div>
            </div>

            <div class="control-group">
              <span class="control-label">Export Receipt</span>
              <div class="seg-group">
                <button
                  class="seg-btn"
                  title="Download results table as CSV"
                  @click=${() => this.exportResults('csv')}
                >
                  ⬇ Export .csv
                </button>
                <button
                  class="seg-btn"
                  title="Download structured results receipt as JSONL"
                  @click=${() => this.exportResults('jsonl')}
                >
                  ⬇ Export .jsonl
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Live Scoreboard Strip -->
        <div class="scoreboard">
          <div class="metric-tile">
            <span class="metric-label">Items Evaluated</span>
            <div class="metric-val">
              <span>${this.completedItems}/${this.totalItems}</span>
              <span class="metric-sub">items</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width: ${pctDone}%"></div>
            </div>
          </div>

          <div class="metric-tile">
            <span class="metric-label">${this.cascadeMode !== 'off' ? 'Cascade Accuracy' : 'Accuracy'}</span>
            <div class="metric-val">
              <span>${accPct}${accPct !== '—' ? '%' : ''}</span>
              <span class="metric-sub">
                ${this.cascadeMode !== 'off' && escalatedRows.length > 0
                  ? `Stage 1: ${stage1AccPct}% · ${escalatedRows.length} escalated to ${this.cascadeModel}`
                  : `(${passRows.length}/${completedRows.length || this.rows.length} questions)`}
              </span>
            </div>
          </div>

          <div class="metric-tile">
            <span class="metric-label">Server p50</span>
            <div class="metric-val">
              <span>${serverP50 > 0 ? `${serverP50} ms` : '—'}</span>
              <span class="metric-sub">GPU forward</span>
            </div>
          </div>

          <div class="metric-tile">
            <span class="metric-label">Round Trip p50</span>
            <div class="metric-val">
              <span>${rttP50 > 0 ? `${rttP50} ms` : '—'}</span>
              <span class="metric-sub">end-to-end HTTP</span>
            </div>
          </div>

          <div class="metric-tile">
            <span class="metric-label">Wall Time</span>
            <div class="metric-val">
              <span>${this.wallElapsedSec > 0 ? `${this.wallElapsedSec} s` : '0.0 s'}</span>
              <span class="metric-sub">
                ${this.wallElapsedSec > 0 && this.completedItems > 0
                  ? `${(this.completedItems / this.wallElapsedSec).toFixed(1)} items/s`
                  : `${this.concurrency}x parallel`}
              </span>
            </div>
          </div>
        </div>

        <!-- Live Results Table -->
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>State</th>
                <th>Question</th>
                <th>Type</th>
                <th>Predicted</th>
                <th>Expected</th>
                <th>p / conf</th>
                <th>Server ms</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${visibleRows.map((r) => {
                const rowIdxLabel =
                  r.totalSlotsForItem > 1 ? `${r.itemIndex}.${r.slotIndex}` : `${r.itemIndex}`;
                const stateLabel =
                  r.state === 'pass'
                    ? '✓ PASS'
                    : r.state === 'miss'
                      ? '✗ MISS'
                      : r.state === 'running'
                        ? '⚡ RUNNING'
                        : r.state === 'queued'
                          ? '⏳ QUEUED'
                          : r.state === 'error'
                            ? '⚠ ERROR'
                            : 'READY';
                const confPct =
                  r.confidence !== undefined ? `${(r.confidence * 100).toFixed(1)}%` : '—';
                const confWidth = r.confidence !== undefined ? Math.round(r.confidence * 100) : 0;
                const confColor =
                  r.state === 'miss'
                    ? '#ef4444'
                    : (r.confidence || 0) >= 0.85
                      ? '#16a34a'
                      : '#f59e0b';

                return html`
                  <tr>
                    <td class="col-idx">${rowIdxLabel}</td>
                    <td>
                      <span class="state-pill ${r.state}">${stateLabel}</span>
                    </td>
                    <td>
                      <div class="q-cell" title="${r.item.preview}">
                        <div class="q-head">
                          <span class="q-slot">${r.slot.question}</span>
                          <span class="q-domain">${r.item.domain} · ${r.item.tier}</span>
                        </div>
                        <div class="q-preview">${r.item.preview}</div>
                      </div>
                    </td>
                    <td>
                      <span class="type-pill">${r.slot.type}</span>
                    </td>
                    <td>
                      ${r.predicted !== undefined
                        ? html`
                            <div style="display:flex; flex-direction:column; align-items:flex-start; gap:0.2rem;">
                              <span class="val-pill ${r.state === 'pass' ? 'match' : 'mismatch'}">
                                ${r.predicted}
                              </span>
                              ${r.escalated
                                ? html`
                                    <span
                                      title="${r.cascadeExplanation || `Escalated to Stage-2 ${r.cascadeModel}`}"
                                      style="font-size:0.65rem; font-weight:700; padding:0.12rem 0.42rem; border-radius:999px; background:var(--brand-soft); color:var(--brand); border:1px solid var(--brand-border);"
                                    >
                                      ⚡ ${r.cascadeModel} (Stage 1: ${r.stage1Predicted})
                                    </span>
                                  `
                                : ''}
                            </div>
                          `
                        : html`<span style="color: var(--text-muted)">—</span>`}
                    </td>
                    <td>
                      <span class="val-pill expected">${r.slot.expected}</span>
                    </td>
                    <td>
                      <div class="conf-cell">
                        <div class="conf-top">
                          <span class="conf-pct">${confPct}</span>
                          ${r.entropy !== undefined
                            ? html`<span class="conf-ent">H=${r.entropy.toFixed(2)}</span>`
                            : ''}
                        </div>
                        <div class="conf-bar">
                          <div
                            class="conf-fill"
                            style="width: ${confWidth}%; background: ${confColor};"
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td class="ms-cell">
                      ${r.serverMs !== undefined
                        ? html`
                            <div class="ms-primary">${r.serverMs} ms</div>
                            <div class="ms-sub">
                              RTT ${r.roundTripMs} ms${r.escalated && r.cascadeLatencyMs ? ` (+${r.cascadeLatencyMs}ms S2)` : ''}
                            </div>
                          `
                        : html`<span style="color: var(--text-muted)">—</span>`}
                    </td>
                    <td>
                      <button
                        class="inspect-btn"
                        title="Open this item in the Decision Studio"
                        @click=${() => this.openRowInStudio(r)}
                      >
                        Studio ↗
                      </button>
                    </td>
                  </tr>
                `;
              })}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }
}
