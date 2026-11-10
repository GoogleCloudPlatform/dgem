# Genkit Consistency & Multi-Agent Lock Arbiters (`genkit-consistency-arbiters`)

Demonstrates pairing single-pass **`dgem`** `.json.tmpl` decision policies (via Genkit Go `v1.14.0`'s `systemonex` plugin) with deterministic post-decision graph, pack-quantity, and file-lock arbiters:

1. **Multimodal Entity Resolution + Pack-Quantity & 3-Cycle Transitivity Guardrail** (`templates/multimodal/entity_resolution_dedup.json.tmpl`):
   - Evaluates `entity_relation`, `pack_quantity_match`, `visual_evidence_cue`, and `merge_action` in 1 forward pass.
   - Verifies multipack multipliers (`ParsePackMultiplier` / `ValidatePackQuantityConsistency`, such as `6x330ml` vs. `Single Can`) to block false merges, and ranks transitive 3-cycle graph contradictions across pairwise edges with `ScoreProbabilisticTriangles` and `CheckTransitiveTriangles`.
2. **Multi-Agent Swarm Lock Arbitration** (`templates/swarm_lock_arbiter.json.tmpl`):
   - Scores 4 concurrent worker proposals in 1 pass and feeds per-slot confidence and normalized hesitation into `ArbitrateSwarmLocks` so disjoint paths acquire locks in parallel while overlapping file edits serialize or escalate when hesitant (`H_norm >= 0.16`).
3. **Sliding-Window Operational Checklist Verification** (`templates/live_checklist_verify.json.tmpl`):
   - Evaluates 6 operational protocol items, commitment detection, and interruption urgency simultaneously on a 30-second transcript window.

## Run Offline Unit Tests (No GPU Required)

```bash
go -C examples test -race -v ./genkit-consistency-arbiters/...
```

## Run Against a Live `dgem` Server or Gateway

```bash
# Against local diffgemma (127.0.0.1:8080/v1) or a dgem serve gateway:
go -C examples/genkit-consistency-arbiters run . -gateway-url http://127.0.0.1:8095
```
