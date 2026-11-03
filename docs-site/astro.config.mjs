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

import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

function remarkMermaid() {
  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
  function walk(node) {
    if (!node) return;
    if (node.type === 'code' && node.lang === 'mermaid') {
      node.type = 'html';
      node.value = `<div class="mermaid-wrapper" title="Click to expand diagram in full-screen lightbox"><button type="button" class="mermaid-zoom-btn" aria-label="Expand diagram">🔍 Click to Zoom</button><pre class="mermaid" data-mermaid-src="${ encodeURIComponent(node.value) }">${ escapeHtml(node.value) }</pre></div>`;
      delete node.lang;
      delete node.meta;
      return;
    }
    if (Array.isArray(node.children)) {
      for (const child of node.children) walk(child);
    }
  }
  return (tree) => walk(tree);
}

export default defineConfig({
  site: 'https://googlecloudplatform.github.io',
  base: '/dgem',
  vite: {
    build: {
      cssTarget: 'esnext',
    },
  },
  markdown: {
    remarkPlugins: [remarkMermaid, remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  // Old flat URLs -> reorganized pages (2026-09 docs reorganization)
  redirects: {
    '/path-to-production/': '/dgem/deploy/',
    '/quickstart/': '/dgem/deploy/laptop/',
    '/setup/': '/dgem/reference/metal-engine/',  // Metal install details; the laptop quickstart links here
    '/remote-endpoints/': '/dgem/deploy/remote-gpu/',
    '/deploy-your-own-gpu/': '/dgem/deploy/cloud-run/',
    '/public-image/': '/dgem/deploy/public-images/',
    '/observability-traces/': '/dgem/operate/observability/',
    '/vertex-ai-vs-cloudrun/': '/dgem/reference/vertex-vs-cloud-run/',
    '/cli-reference/': '/dgem/reference/cli/',
    '/studio-mcp-api/': '/dgem/reference/studio-mcp-api/',
    '/user-guide/': '/dgem/policies/first-policy/',
    '/experiment-authoring-guide/': '/dgem/policies/authoring/',
    '/custom-dataset-guide/': '/dgem/policies/datasets/',
    '/templates/': '/dgem/policies/templates/',
    '/taxonomy-discovery/': '/dgem/policies/taxonomy-discovery/',
    '/applications/': '/dgem/policies/applications/',
    '/architecture/': '/dgem/confidence/architecture/',
  },
  integrations: [
    starlight({
      title: 'dgem - Decision Gemma',
      description: 'DiffusionGemma as a Zero-Shot Decision Model & Declarative Policy-as-Template Engine across Apple Silicon, GCE, and Cloud Run',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/GoogleCloudPlatform/dgem' },
      ],
      head: [
        {
          tag: 'script',
          attrs: { type: 'module' },
          content: `
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

            let zoomScale = 1;
            let panX = 0;
            let panY = 0;
            let isDragging = false;
            let dragStartX = 0;
            let dragStartY = 0;

            function ensureLightboxModal() {
              let modal = document.getElementById('mermaid-lightbox-modal');
              if (modal) return modal;
              modal = document.createElement('div');
              modal.id = 'mermaid-lightbox-modal';
              modal.className = 'mermaid-lightbox';
              modal.innerHTML = \`
                <div class="mermaid-lightbox-toolbar">
                  <span class="mermaid-lightbox-title">🔍 Interactive Diagram View (Scroll to Zoom · Drag to Pan)</span>
                  <div class="mermaid-lightbox-controls">
                    <button type="button" id="mm-zoom-out" title="Zoom Out (-)">−</button>
                    <button type="button" id="mm-zoom-reset" title="Reset Zoom (100%)">100%</button>
                    <button type="button" id="mm-zoom-in" title="Zoom In (+)">+</button>
                    <button type="button" id="mm-close" class="mm-close-btn" title="Close (Esc)">✕ Close</button>
                  </div>
                </div>
                <div class="mermaid-lightbox-stage" id="mm-stage">
                  <div class="mermaid-lightbox-canvas" id="mm-canvas"></div>
                </div>
              \`;
              document.body.appendChild(modal);

              const canvas = modal.querySelector('#mm-canvas');
              const stage = modal.querySelector('#mm-stage');
              const resetBtn = modal.querySelector('#mm-zoom-reset');

              function applyTransform() {
                canvas.style.transform = \`translate(\${panX}px, \${panY}px) scale(\${zoomScale})\`;
                resetBtn.textContent = \`\${Math.round(zoomScale * 100)}%\`;
              }

              modal.querySelector('#mm-zoom-in').addEventListener('click', (e) => {
                e.stopPropagation();
                zoomScale = Math.min(4, +(zoomScale + 0.25).toFixed(2));
                applyTransform();
              });

              modal.querySelector('#mm-zoom-out').addEventListener('click', (e) => {
                e.stopPropagation();
                zoomScale = Math.max(0.4, +(zoomScale - 0.25).toFixed(2));
                applyTransform();
              });

              resetBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                zoomScale = 1;
                panX = 0;
                panY = 0;
                applyTransform();
              });

              modal.querySelector('#mm-close').addEventListener('click', () => {
                modal.classList.remove('open');
              });

              modal.addEventListener('click', (e) => {
                if (e.target === modal || e.target === stage) {
                  modal.classList.remove('open');
                }
              });

              stage.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY < 0 ? 0.15 : -0.15;
                zoomScale = Math.min(4, Math.max(0.4, +(zoomScale + delta).toFixed(2)));
                applyTransform();
              }, { passive: false });

              stage.addEventListener('mousedown', (e) => {
                isDragging = true;
                dragStartX = e.clientX - panX;
                dragStartY = e.clientY - panY;
                stage.style.cursor = 'grabbing';
              });

              window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                panX = e.clientX - dragStartX;
                panY = e.clientY - dragStartY;
                applyTransform();
              });

              window.addEventListener('mouseup', () => {
                isDragging = false;
                if (stage) stage.style.cursor = 'grab';
              });

              window.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && modal.classList.contains('open')) {
                  modal.classList.remove('open');
                }
              });

              return modal;
            }

            function openMermaidLightbox(wrapper) {
              const svg = wrapper.querySelector('svg');
              if (!svg) return;
              const modal = ensureLightboxModal();
              const canvas = modal.querySelector('#mm-canvas');
              canvas.innerHTML = '';
              const clone = svg.cloneNode(true);
              clone.removeAttribute('style');
              clone.style.width = '100%';
              clone.style.height = 'auto';
              clone.style.maxHeight = '82vh';
              canvas.appendChild(clone);
              zoomScale = 1;
              panX = 0;
              panY = 0;
              canvas.style.transform = 'translate(0px, 0px) scale(1)';
              modal.querySelector('#mm-zoom-reset').textContent = '100%';
              modal.classList.add('open');
            }

            async function renderAllMermaid() {
              const isDark = document.documentElement.dataset.theme !== 'light';
              mermaid.initialize({
                startOnLoad: false,
                theme: isDark ? 'dark' : 'default',
                securityLevel: 'loose',
                fontFamily: 'Google Sans Flex, system-ui, sans-serif',
                flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis' },
                themeVariables: { fontSize: '15px' },
              });
              const nodes = document.querySelectorAll('pre.mermaid');
              for (const el of nodes) {
                const raw = el.getAttribute('data-mermaid-src');
                if (raw) {
                  el.removeAttribute('data-processed');
                  el.textContent = decodeURIComponent(raw);
                }
              }
              if (nodes.length > 0) {
                await mermaid.run({ nodes });
              }
              document.querySelectorAll('.mermaid-wrapper').forEach((wrapper) => {
                if (!wrapper.dataset.lightboxBound) {
                  wrapper.dataset.lightboxBound = 'true';
                  wrapper.addEventListener('click', () => openMermaidLightbox(wrapper));
                }
              });
            }

            document.addEventListener('DOMContentLoaded', renderAllMermaid);
            document.addEventListener('astro:page-load', renderAllMermaid);
            const obs = new MutationObserver(() => renderAllMermaid());
            obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
          `,
        },
      ],
      customCss: [
        'katex/dist/katex.min.css',
        './src/styles/custom.css',
      ],
      sidebar: [
        { label: 'Overview', link: '/' },
        {
          label: 'Build, Deploy & Operate',
          items: [
            { label: 'From Laptop to Production', slug: 'deploy' },
            { label: '1. Run on Your Laptop', slug: 'deploy/laptop' },
            { label: '2. Use a Remote GPU', slug: 'deploy/remote-gpu' },
            { label: '3. Deploy on Cloud Run', slug: 'deploy/cloud-run' },
            { label: '4. Production on Vertex AI', slug: 'deploy/vertex' },
            { label: '5. Gateway and Routing', slug: 'deploy/gateway' },
            { label: 'To Batch or Not to Batch', slug: 'deploy/batch' },
            { label: 'Evaluate on Your Own GPU', slug: 'deploy/evaluate' },
            { label: 'Latency and Capacity', slug: 'operate/latency-capacity' },
            { label: 'Operations Runbook', slug: 'operate/runbook' },
            { label: 'Regression Matrix', slug: 'operate/regression-matrix' },
            { label: 'Monitoring and Alerts', slug: 'operate/monitoring' },
            { label: 'Observability and Traces', slug: 'operate/observability' },
            { label: 'Security and Access', slug: 'operate/security' },
          ],
        },
        {
          label: 'Confidence & Calibration',
          items: [
            { label: 'Confidence Beyond Shannon', slug: 'confidence/overview' },
            { label: 'Confidence and Calibration', slug: 'confidence' },
            { label: 'Calibrate Your Policy', slug: 'confidence/calibrate-your-policy' },
            { label: 'The Journey to Decision Models', slug: 'decision-models-primer' },
            { label: 'Glossary & Mental Models', slug: 'glossary' },
          ],
        },
        {
          label: 'Policies & Decisions',
          items: [
            { label: 'Your First Decision Policy', slug: 'policies/first-policy' },
            { label: 'Authoring and Stage 2 Cascades', slug: 'policies/authoring' },
            { label: 'Run a Dataset', slug: 'policies/datasets' },
            { label: 'Template Catalog', slug: 'policies/templates' },
            { label: 'Real-World Applications', slug: 'policies/applications' },
            { label: 'Taxonomy Discovery', slug: 'policies/taxonomy-discovery' },
            { label: 'Prompt Layout', slug: 'policies/prompt-layout' },
            { label: 'What dgem Can Do With Images', slug: 'policies/images' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'CLI, HTTP Gateway & MCP', slug: 'reference/cli' },
            { label: 'Decision Studio, MCP & HTTP API', slug: 'reference/studio-mcp-api' },
            { label: 'Build the Serving Image', slug: 'deploy/public-images' },
            { label: 'Vertex AI vs. Cloud Run', slug: 'reference/vertex-vs-cloud-run' },
            { label: 'Apple Silicon Engine (diffgemma)', slug: 'reference/metal-engine' },
            { label: 'How the Model Decides in One Pass', slug: 'confidence/architecture' },
          ],
        },
      ],
    }),
  ],
});
