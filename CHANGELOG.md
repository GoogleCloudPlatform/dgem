# Changelog

Versioning: `vMAJOR.MINOR.PATCH`. **Major:** breaking API or response changes. **Minor:** vLLM runtime or model
changes, new features. **Patch:** fixes. Releases are git tags (`make release VERSION=vX.Y.Z`); build the serving
image for a release in your own project with `make cloudrun-build`.

## Unreleased

## v0.4.0 — first public release

Earlier versions were developed internally; this is the first release in this repository.

- **Decision engine:** `dgem decide` compiles `.json.tmpl` decision policies into one-pass multi-question readouts
  (`boolean`, `choice`, `score`) with per-option probabilities, Shannon entropy and Hesitation %; conditional
  questions (`depends_on` / `ask_if`), per-policy `temperature`, and taxonomy-expansion suggestions.
- **Hesitation-gated cascade:** low-hesitation answers are returned directly; the rest go to a Stage 2 Gemini model
  (`gemini-3.8-flash` default, `gemini-3.5-flash-lite`) with a soft hint of dgem's answer by default
  (`stage2_prior: soft | full | none`).
- **Surfaces:** CLI, HTTP gateway (`dgem serve`: `/api/decide`, `/v1/systemone`, `/api/locate`), MCP server
  (`dgem mcp`, Streamable HTTP `/mcp`), and the embedded Decision Studio. Backend routing `vertex_first | vertex |
  cloudrun | local` with automatic failover.
- **Images:** approximate one-pass boxes, and guided locate (`dgem locate`, MCP `locate_object`): dgem answers
  presence and grid cell, then Gemini returns the box; optional SAM masks.
- **Serving image:** vLLM structured generation with the dgem server patch (`deploy/cloudrun/`): `document_first`
  prompt layout, yes/no isolation (`DEFAULT_ISOLATE=auto`), in-flight limiting, 8k context on RTX PRO 6000,
  idle-connection timeout. Deploy scripts for Vertex AI dedicated endpoints, Cloud Run GPU and GCE.
- **Evaluation:** `dgem bench-*` harnesses and the regression matrix (`scripts/bench_matrix.py`,
  `benchmarks/matrix/matrix_example.json`) to compare a candidate deployment with a baseline in one session.
