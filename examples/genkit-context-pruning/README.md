# Genkit 8-Slot Agent Context Pruning (`genkit-context-pruning`)

Demonstrates single-pass trajectory context pruning using [`templates/agent_context_pruning.json.tmpl`](../../templates/agent_context_pruning.json.tmpl) and an exact Multiple-Choice Knapsack (`SelectContextPruningActions`) over `dgem`'s per-block retention probabilities.

## Architecture

1. **Single-Pass 8-Slot Scoring (`dgemgenkit.DecideWithTemplate`)**: Evaluates up to 8 candidate context blocks (`b0`..`b7`) simultaneously across `{"full", "stub", "drop"}` in a single `dgem` pass.
2. **Exact Token-Budget Knapsack (`SelectContextPruningActions`)**: Maximizes expected utility across `{full, stub, drop}` subject to `TokenBudget`, enforcing `Pinned: true` blocks to stay in `{"full", "stub"}` and optionally shifting uncertain blocks toward compact signature stubs via `UncertaintyStubBonus`.

## Run Offline Tests

```bash
go -C examples test -race -v ./genkit-context-pruning/...
```

## Run Against a Live Endpoint or Gateway

```bash
# Against a running dgem serve gateway or local diffgemma server:
go -C examples/genkit-context-pruning run . -token-budget 1400
```
