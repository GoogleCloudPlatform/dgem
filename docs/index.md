---
title: "Documentation Overview & Index"
description: "DiffusionGemma (dgem) documentation, organized by audience: build and deploy, confidence and calibration, and policies and decisions."
---

# DiffusionGemma (`dgem`) Documentation

**`dgem`** turns DiffusionGemma into a zero-shot **decision model**: you describe the decisions you need in a
policy file, and the model answers all of its questions in one forward pass, with a probability for every allowed
answer and a hesitation score that tells you when to hand the decision off.

## Choose your path

### 🛠️ Build, deploy and operate
For developers taking `dgem` from a laptop to production, and for the SREs who run it.

1. [From laptop to production](deploy/index.md): the journey, and how to measure latency and cold start at each stage
2. [Run on your laptop](deploy/laptop.md) · [Use a remote GPU](deploy/remote-gpu.md)
3. [Deploy on Cloud Run](deploy/cloud-run.md) (scale to zero) · [Production on Vertex AI](deploy/vertex.md) (always warm)
4. [Gateway and routing](deploy/gateway.md) · [Latency and capacity](operate/latency-capacity.md) ·
   [Operations runbook](operate/runbook.md) · [Observability](operate/observability.md)
5. [Evaluate dgem on your own GPU](deploy/evaluate.md): GPU requirements, install check, custom evaluation

### 📐 Confidence and calibration
For teams who need to know when a decision can be trusted.

1. [Confidence and calibration](confidence/index.md): what to measure, what we recommend, and the caveats
2. [Calibrate your policy](confidence/calibrate-your-policy.md) on your own labelled data
3. [Confidence beyond Shannon](confidence/overview.md) · [The journey to decision models](decision-models-primer.md) ·
   [Glossary](glossary.md)
4. [Regression matrix](operate/regression-matrix.md): check a serving change against your own baseline

### 📝 Policies and decisions
For people who write policies and use the answers.

1. [Your first decision policy](policies/first-policy.md): a hands-on tutorial
2. [Authoring and Stage 2 cascades](policies/authoring.md) · [Run a dataset](policies/datasets.md)
3. [Template catalog](policies/templates.md) · [Real-world applications](policies/applications.md) ·
   [Taxonomy discovery](policies/taxonomy-discovery.md)

## Reference

- [CLI, HTTP gateway and MCP reference](reference/cli.md)
- [Decision Studio, MCP and HTTP API](reference/studio-mcp-api.md)
- [Build the serving image](deploy/public-images.md)
- [Vertex AI vs. Cloud Run](reference/vertex-vs-cloud-run.md)
- [Apple Silicon engine (`diffgemma`)](reference/metal-engine.md)
- [How the model decides in one pass](confidence/architecture.md)
