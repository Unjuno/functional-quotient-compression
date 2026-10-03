# Phase II — Mirror-native conditional models

**Active, unfinished research.** Phase II asks whether one learned shared representation can support multiple functional states through low-description state/view coordinates and parallel routing, without storing a separate full expert for every state.

The current work has moved beyond the initial single-router feasibility study. The main evidence summary is [RESEARCH_STATE_THROUGH_MN007.md](RESEARCH_STATE_THROUGH_MN007.md). The current conceptual target is recorded separately in [SEMANTIC_MIRROR_HYPOTHESIS.md](SEMANTIC_MIRROR_HYPOTHESIS.md).

## Evidence map

| Experiment | Role | Status |
|---|---|---|
| MN001 | Initial 2-layer causal Transformer / 4-rule toy | Learned, router ablation and cache feasibility; ordinary Dense also solved it. |
| [MN002](../../experiments/mirror_native/mn002_20261003/README.md) | 16-rule task with matched Dense controls | Mirror4 positive on one table, weaker fresh-table result; no general advantage. |
| MN003 | Fixed disjoint block factorization + packed parallel router | Packed routing preserves tested function/gradients and reduces Python-loop routing overhead; Dense still strong. |
| MN004 | Overlapping sparse supports + mean/sum aggregation | Where/how routing works; averaging is not consistently superior; fixed-support sum/mean are reparameterization-equivalent under stated assumptions. |
| MN005 | Candidate-region where-routing | Lower parameter/file cost; runtime benefit depends on shape; fixed disjoint routing remains a strong control. |
| MN006 | Router early-lock-in diagnosis | Strong collapse explanation not supported; router-only tuning helps, full joint tuning helps much more. |
| MN007 | Shared semantic bases + state/view codes | Compact shared representation works; strong semantic self-organization and Mirror-specific advantage remain unproven. |

See the [experiment index](../../experiments/mirror_native/INDEX.md) for local evidence-package hashes.

## Current architectural interpretation

The evidence favors **structured shared subspaces + parallel routing + joint optimization** over unrestricted per-token support search.

Three distinctions are mandatory:

1. **Functional mechanism:** can the model execute distinct input-dependent states?
2. **Resource efficiency:** does it do so at lower stored bytes and acceptable compute?
3. **Representation semantics:** do related concepts actually share the same parameter identity with reusable low-description transformations?

The first has small-scale evidence. The second is mixed. The third is the current open problem.

## Current representation hypothesis

The central candidate factorization is:

```math
concept \approx (shared\ canonical\ parameter,\ Mirror/view\ coordinate)
```

The dual parameterization must also be tested:

```math
concept \approx (concept\ coordinate,\ shared\ transformation\ basis/directions)
```

Task loss alone did not recover strong semantic grouping in MN007. The next experiment therefore needs a direct **transformation-consistency** objective: the same state/view code should induce a consistent functional change across multiple bases.

## Evidence boundaries

Do not conflate:

- router usage with superiority over a retrained static model;
- routing combinations with independent-expert capacity;
- lower parameter count with lower serialized bytes;
- CPU loop-overhead speedups with production throughput;
- positive ARI with successful semantic-factor recovery;
- compact shared representation with Mirror-specific advantage over additive/low-rank factorization.

## Next gate

The next falsifiable comparison should use held-out family x state combinations and matched serialized bytes across:

1. independent embedding;
2. additive factorization;
3. CP/low-rank factorization;
4. shared-base Mirror;
5. shared-transformation-basis Mirror;
6. optional small private residual.

Measure task NLL/accuracy, state-swap consistency, cross-family transfer, semantic clustering, actual bytes, and active compute. A Mirror-specific claim requires improvement beyond the additive/low-rank controls.

Phase I remains [preserved and paused](../phase1/STATUS.md). Existing Apache-2.0 terms are unchanged.
