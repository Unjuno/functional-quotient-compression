# Phase II — Mirror-native conditional models

**Active, unfinished research.** Phase II now focuses on a single wide Dense model controlled by one context-aware router and a small bank of global Mirror states.

The active state is [RESEARCH_STATE_THROUGH_MN010.md](RESEARCH_STATE_THROUGH_MN010.md). The architecture is specified in [GLOBAL_DENSE_MIRROR_ARCHITECTURE.md](GLOBAL_DENSE_MIRROR_ARCHITECTURE.md). The next experiment is pre-registered in [MN011_PLAN.md](../../experiments/mirror_native/MN011_PLAN.md).

## Evidence map

| Experiment | Role | Status |
|---|---|---|
| MN001 | Initial causal shared-state toy | Feasibility only; Dense also solved it. |
| [MN002](../../experiments/mirror_native/mn002_20261003/README.md) | Matched Dense/shared-state comparison | Mixed; no general advantage. |
| MN003 | Packed parallel block routing | Numerical equivalence and loop-overhead reduction demonstrated. |
| MN004 | Overlapping sparse supports | Averaging not consistently superior; fixed-support sum/mean are reparameterization-equivalent under stated assumptions. |
| MN005 | Candidate-region routing | Lower metadata/parameter cost; quality/runtime benefits are shape- and task-dependent. |
| MN006 | Router early-lock-in diagnosis | Strong collapse explanation not supported; joint optimization mattered more. |
| MN007 | Semantic shared bases | Compact representation works; strong semantic self-organization and Mirror-specific advantage not demonstrated. |
| MN008 | Functional recombination supervision | Reusable factor structure improves strongly with relation supervision; additive control remains stronger on median NLL. |
| MN009 | Shared Dense vs duplicated experts | Mirror beats a near-byte-matched narrow soft-MoE; not consistently better than Dense/simple gate and not quality-equivalent to a much larger MoE. |
| MN010 | Single Dense + single router + dynamic global Mirror | Low-dimensional state-count growth ineffective; broad 256-dim global modulation gives a discovery signal but replication is mixed. |
| MN011 | Mirror breadth × Mirror count | **Pre-registered next experiment.** |

## Main architectural interpretation

The evidence currently favors:

> **wide shared Dense computation + context-aware global control**

over aggressively fragmenting the model into narrow independent experts.

Three resources must be optimized separately:

1. **Dense width** — shared capacity;
2. **Mirror breadth** — how many functional directions one state can change;
3. **Mirror count** — how many alternative states can be routed.

MN010 is the first experiment to separate these explicitly. Increasing count with a small 8-dimensional control did not help; increasing state breadth to direct 256-dimensional modulation produced a stronger signal in one data world.

## Router rule

The router must see enough causal context to distinguish the functions it is meant to select.

MN010 found that:

- raw final-token routing was context-blind on the synthetic task;
- prefix mean removed operation order;
- an order-sensitive causal prefix summary corrected the information defect.

Router design is therefore part of the model's information interface, not merely a latency component.

## Dynamic growth

A Mirror state can be split function-preservingly under soft routing. This allows state count to grow during training without an insertion-time output jump.

Growth is not automatic evidence of capacity. Every proposed addition must be compared from the same parent checkpoint against:

- no structural change;
- similar-byte Dense widening.

The current automatic-growth rule correctly rejected state additions when measured development gain was too small.

## Semantic-sharing side lane

MN008 showed that explicit recombination supervision can produce strong semantic grouping and held-out recombination. However, additive factorization also benefited and remained strong. Semantic Mirror work is preserved as an auxiliary representation-learning lane in [SEMANTIC_MIRROR_HYPOTHESIS.md](SEMANTIC_MIRROR_HYPOTHESIS.md), not the current main architecture.

## Current evidence boundaries

Do not conflate:

- state count with functional breadth;
- path count with independent-expert capacity;
- parameter count with actual serialized bytes;
- discovery-world improvements with replication;
- router usage with superiority over retrained Dense;
- soft-MoE CPU timing with optimized sparse-MoE timing;
- synthetic arithmetic/compositional tasks with natural-language quality.

## Next gate

MN011 will test a factorial grid of Mirror breadth 32/64/128/256 and state count E=1/2/4/8, plus dynamic state growth with no-change and similar-byte Dense-widening controls.

A Mirror-specific claim requires a reproducible rate-quality advantage beyond both ordinary extra training and Dense widening.

Phase I remains [preserved and paused](../phase1/STATUS.md). Apache-2.0 terms are unchanged.
