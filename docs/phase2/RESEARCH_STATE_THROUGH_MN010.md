> **HISTORICAL SNAPSHOT.**\n> This was canonical through MN010, but is no longer the current Phase II state. See [CURRENT_STATE_2026-10-07.md](CURRENT_STATE_2026-10-07.md).\n\n# Phase II research state through MN010

Date: 2026-10-04

This is the canonical Phase II state after MN008, MN009, and MN010. It supersedes `RESEARCH_STATE_THROUGH_MN007.md` as the active summary while preserving that file as historical provenance.

All MN008-MN010 results are small CPU synthetic experiments. They do **not** establish a natural-language-model advantage, a production MoE result, or a general capacity multiplier.

## Current main direction

The active architecture is now:

> **one dense model + one context-aware router + a bank of global Mirror states**

The router selects/soft-mixes how the same dense model is read. Mirror states are intended to add conditional functional variation without duplicating full expert weights.

The semantic-sharing line remains useful as a side experiment, but it is no longer the main architecture target.

## Why the direction changed

MN007 showed that shared semantic bases can be compact, but task optimization did not reliably recover the intended semantic quotient and strong additive/low-rank controls remained competitive.

MN008 then showed that explicit functional recombination supervision can strongly improve semantic grouping and held-out recombination, but the same supervision also improved additive controls. This demonstrated that **factor reuse is trainable**, not that phase-Mirror is uniquely superior.

MN009 moved the question back to MoE-like conditional computation: can a wide shared FFN replace multiple independent experts? A shared Mirror model beat a near-byte-matched small independent soft-MoE, but not Dense/simple-gate controls consistently, and it did not fully match a much larger independent MoE.

MN010 identified a more specific bottleneck: **Mirror count and Mirror breadth are different resources**. Increasing the number of small low-dimensional Mirror codes did little; increasing the functional breadth of one Mirror state produced measurable gains in one world, though replication was mixed.

## MN008 — functional recombination supervision

Question: if the model is explicitly trained to combine the content from one seen concept with the state from another, can shared bases / shared transformation directions become reusable?

Result:

- shared-base phase Mirror: held-out recombination accuracy improved from 63.12% to 89.56% in the main world and from 63.60% to 85.55% in a fresh world;
- family ARI improved from about 0.08 to about 0.72-0.76;
- the additive shared-base control remained stronger on median NLL in both worlds;
- the dual parameterization (shared transformation direction + concept coordinate) also learned state-aligned structure, but did not outperform the additive/shared-base controls.

Interpretation:

- functional relation supervision can organize reusable factors;
- semantic alignment and functional reuse can be trained;
- this is **not** evidence that Mirror-Phase is uniquely better than ordinary factorization;
- semantic grouping is now an auxiliary representation-learning lane, not the main deployment architecture.

## MN009 — expert duplication versus shared dense computation

Question: can a wide shared FFN plus Mirror control replace duplicated experts more efficiently than simply shrinking each independent expert?

Compared:

- Dense;
- simple feature gate;
- shared Mirror FFN;
- near-byte-matched small 4-expert soft-MoE;
- same-width large 4-expert soft-MoE.

After 3,000 total updates on the composite-operation task:

- Mirror file: 86,127 B;
- small independent soft-MoE: 85,665 B;
- large independent soft-MoE: 181,367 B;
- Mirror beat the small independent MoE on IID and OOD NLL in all 3 paired seeds;
- Mirror reduced file size by 52.51% versus the large MoE but did not satisfy the preregistered quality-equivalence gate;
- Dense and simple-gate controls remained competitive or better on median quality;
- Mirror routing was used by the trained models, but the tested soft-MoE computed all experts and was not an optimized sparse-MoE runtime baseline.

Interpretation:

- preserving a wide shared computation can be better than splitting the same storage budget into many narrow experts;
- the result does not yet show a Mirror-specific advantage over Dense or a simple gate.

## Analytic bridge — function-preserving global-state growth

Before MN010, the project reconnected the current architecture to the earlier holographic/shared-parameter analysis.

Key analytic results used by MN010:

- deterministic Mirror views do not create new Shannon information by themselves;
- Mirror count, state/control dimension, and shared Dense width are distinct resources;
- under soft routing a state can be split into two symmetric children while preserving the mixed control exactly at insertion time;
- the two children can still receive different router gradients after the split;
- a local router-control tangent has at most \(\min(q,E-1)\) independent directions for control dimension \(q\) and state count \(E\).

Verified local artifacts:

- analysis ZIP SHA-256: `7b3bb4b14d5a56c4576431bf5d049f5fbdb1a7bbc5f34881e05f2a3364671932`;
- analysis report SHA-256: `e952186b11703373c21420fae48acee8baf30e118a3a84dc65f7c64bbcddbff8`.

## MN010 — single Dense, single router, dynamically added global Mirror states

Question: can one dense model be kept intact while global Mirror states are added during training only when they provide additional value?

### Analytic split used by the experiment

A state (j) with router logit (a_j) and state code (m_j) is split as:

```math
a_j \rightarrow (a_j-\log 2,\ a_j-\log 2)
```

```math
m_j \rightarrow (m_j+\varepsilon v,\ m_j-\varepsilon v)
```

For a soft mixture, this preserves the effective control exactly at the instant of the split. The split direction (v) was chosen from a development-set gradient covariance so that first-order child-router differentiation is large.

### Router-information correction

MN010 found and corrected two implementation issues:

1. routing from the raw final token was context-blind on the synthetic task because every sequence ended with the same ASK token;
2. a simple prefix mean removed operation order.

The final router used an order-sensitive causal exponential-decay prefix summary with no added trainable parameters.

### Low-dimensional Mirror code

With an 8-dimensional Mirror code expanded through a shared map:

- increasing E from 1 to 8 produced almost no split-specific development gain;
- at approximately the same extra-parameter budget, increasing control dimension or dense width was a separate resource;
- automatic Mirror addition stopped at E=1 under the predeclared gain rule.

### Rich global Mirror

A richer state directly modulated 256 values across the two-layer dense model:

- attention residual gain: 32/layer;
- FFN hidden gain: 64/layer;
- FFN output gain: 32/layer;
- total: 256 scalars/state.

Adding E=2 cost 289 parameters; widening the dense FFN by two neurons cost 264 parameters.

On the original world, a split-amplitude sweep gave positive paired development NLL gains in all 3 seeds, increasing with epsilon from 0.25 to 2.0. At epsilon=2.0 the gains were:

- +0.00819;
- +0.00646;
- +0.00478 nat.

With epsilon=2.0 fixed after development selection, Mirror E=2 beat the continue-Dense and width+2 controls on OOD NLL in all 3 seeds in that world.

On a new dataset seed with new training seeds, the result was mixed:

- IID NLL: Mirror beat continue-Dense in 2/3 and width+2 in 2/3;
- OOD NLL: Mirror beat each control in 1/3.

Interpretation:

- the first clear positive signal came from **increasing functional breadth per Mirror state**, not from increasing the number of low-dimensional states;
- the positive result is not yet stable across data worlds;
- Mirror count, Mirror breadth, and dense width must be treated as separate budget axes.

## Current strongest hypothesis

The current hypothesis is:

> A dense model may support useful conditional specialization when a small number of sufficiently broad global Mirror states are routed contextually, while full expert duplication is unnecessary for some tasks.

The evidence does **not** yet show:

- that more Mirror states monotonically improve quality;
- that automatic state addition is stable;
- that Mirror beats Dense widening at equal bytes across data worlds;
- that Mirror beats a simple gate generally;
- that this carries to natural language or large models.

## Current design constraints

1. **One dense path remains primary.** Do not fragment the core into narrow independent experts merely to imitate MoE.
2. **One context-aware router is the default.** It may output a distribution over global Mirror states.
3. **Router input must contain the information needed for the decision.** Raw final-token routing is not acceptable when context determines the task.
4. **Mirror count and breadth are separate knobs.**
5. **State addition should be function-preserving at insertion.**
6. **Every structural change must be compared with a no-change shadow and a dense-widening control at similar bytes.**
7. **Audit data must not select state count, breadth, epsilon, or stopping point.**

## Claims that should NOT be made

Do not claim from MN008-MN010 that:

- semantic grouping has been solved;
- phase Mirror is better than additive factorization;
- Mirror is a drop-in replacement for a strong sparse MoE;
- global Mirror is generally better than Dense;
- the original-world MN010 OOD 3/3 result replicated;
- dynamic state addition is already a robust training algorithm;
- CPU synthetic results predict production LLM behavior.

## Evidence-package provenance

| Experiment | Evidence ZIP SHA-256 | Report SHA-256 |
|---|---|---|
| MN008 | `b8f7b4e86002c654e137ad00faf1f9ba481eeddd012ffeb5275f4887ad8bc727` | `5d737135d08604c42daccb2b94575543c114dbdd959b966b87820a53df08fbae` |
| MN009 | `76d26a0f89323ece2d1d980d06ac10219ae5cd5f84ebe73a56251b2f9e9a7055` | `ab36cc47b5d06b315c71c9cdc31a13e8e097c3e150ca7032881d866e467ae1cc` |
| MN010 | `309f39d3450f42a158497be6a4732367dd203871e764ca22139db8368e327fb3` | `f49c8be5dfd9cde6599429c7f424016bf1b344167cc86c82dd56170bdd150302` |

Large model/checkpoint evidence remains outside ordinary Git history.
