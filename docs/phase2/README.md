# Phase II — Mirror-native conditional models

**Active, unfinished research.**

## Current integrated state — 2026-10-06

The latest integrated architecture proposal is:

- [Mirror Transformer — current architecture proposal](MIRROR_TRANSFORMER_CURRENT_ARCHITECTURE.md)
- [Research state through 2026-10-06](RESEARCH_STATE_2026-10-06.md)

The current proposal is **router-free by default**: one shared Transformer is trained through deterministic Mirror coordinates derived from known sensor/token/layer/global state. This is a next architecture hypothesis, not a replacement of the historical MN results.

The integrated evidence summary keeps Fact / Interpretation / Hypothesis separate. In particular:

- narrow gain/rotation Mirror families have not shown a stable capacity advantage;
- RF1 showed an unseen-view robustness signal but not a Dense-beating capacity result;
- reachability-only Mirror selection did not reliably improve small-FFN continuation;
- token-period and sensor-parallel Mirror are proposed next directions and remain untested.

## Historical canonical records

- [Research state through MN010](RESEARCH_STATE_THROUGH_MN010.md)
- [Global Dense Mirror architecture](GLOBAL_DENSE_MIRROR_ARCHITECTURE.md) — historical router-based architecture hypothesis
- [Reachability-adjusted Mirror analysis](REACHABILITY_ADJUSTED_MIRROR_ANALYSIS_JA.md)
- [MN011 preregistration](../../experiments/mirror_native/MN011_PLAN.md)
- [Semantic Mirror hypothesis](SEMANTIC_MIRROR_HYPOTHESIS.md)

The MN010/MN011 documents are preserved as provenance. The new integrated architecture does not retroactively change their claims or experimental conditions.

## Evidence map

| Line | Role | Current status |
|---|---|---|
| Phase I / FQC | Post-training functional sharing/compression | Strong sharing controls did not establish a general win; motivates native-training approach. |
| MN007 | Semantic shared bases | Compact representation, weak spontaneous grouping. |
| MN008 | Relation-supervised factor reuse | Reusable factors trainable; not Mirror-specific. |
| MN009 | Shared wide path vs duplicated experts | Mirror beat near-byte narrow soft-MoE, but Dense/simple gate remained strong. |
| MN010 | Global Mirror count vs breadth | Count-only weak; broad state gave discovery signal with mixed replication. |
| MT001 | Multi-view paired learning | NLL signal confounded by repeated exposure; packed execution useful. |
| MT002A/B | Routing diagnosis | Learned bridge unstable; oracle routing exposed representation bottleneck. |
| MT003A | Gain-Mirror breadth sweep | Gain family too narrow; independent experts much stronger under oracle routing. |
| RF1 | Router-free deterministic rotation Mirror | No stable Dense quality advantage; unseen-view robustness signal; packed execution works. |
| RA-Mirror | Reachability-adjusted local analysis | Algebra verified; not a model-level capacity proof. |
| Small-FFN RA pilot | Reachability-selected direction continuation | No stable learning advantage; 2/6 paired wins. |
| Token Mirror | Periodic computation coordinate | Proposed; not yet validated. |
| Sensor Mirror | Sensor/view observation coordinate | Proposed; not yet validated. |

## Current interpretation

The active hypothesis is no longer “more states are better.”

It is:

> **A small number of low-description Mirror transformations may improve effective capacity only if they expose task-useful functional directions that are expensive/interfering for ordinary shared-weight updates and still survive actual multi-view training.**

The proposed next architecture is therefore:

> **one shared Transformer + deterministic token/layer/sensor Mirror coordinates + structured non-rotation transforms + explicit capacity-frontier evaluation.**

## Evidence boundaries

Do not conflate:

- Mirror state count with independent-expert capacity;
- robustness to view changes with capacity;
- local reachability cost with task value;
- parameter count with serialized bytes;
- deterministic transform diversity with new Shannon information;
- small synthetic CPU results with natural-language or real-world sensor performance.

Phase I remains preserved. Apache-2.0 terms are unchanged.
