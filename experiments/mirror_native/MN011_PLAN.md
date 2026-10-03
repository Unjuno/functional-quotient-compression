# MN011 experiment plan — Mirror breadth x Mirror count

Status: **pre-registered next experiment**

## Goal

Determine whether the useful resource is primarily:

- Mirror count (E);
- Mirror breadth (B);
- dense width;
- or ordinary extra optimization.

The experiment must separate these factors under explicit storage accounting.

## Base architecture

Use the MN010 single-Dense/single-router model with the corrected order-sensitive causal router.

The base dense model and router architecture are fixed before training.

## Mirror breadth

Use four state breadths:

- 32;
- 64;
- 128;
- 256.

The full modulation space is 256-dimensional.

For breadth below 256, expand the state through a fixed decoder-known orthogonal basis over the 256 modulation coordinates. The prototype will use fixed Hadamard-derived directions so the shared expansion matrix does not become an uncounted learned parameter.

This basis choice is a control, not a claim of optimality.

## Mirror count

For each breadth, evaluate:

- E=1;
- E=2;
- E=4;
- E=8.

State additions use the MN010 function-preserving split.

## Factorial comparison

Primary grid:

| breadth | E=1 | E=2 | E=4 | E=8 |
|---:|---:|---:|---:|---:|
| 32 | test | test | test | test |
| 64 | test | test | test | test |
| 128 | test | test | test | test |
| 256 | test | test | test | test |

This grid is descriptive. It is not itself the dynamic-growth algorithm.

## Dynamic-growth lane

Separately, start from E=1 and propose one state at a time.

For every proposal, train from the same parent checkpoint:

1. **Mirror-add candidate**;
2. **no-change shadow**;
3. **dense-widening control** at similar added serialized bytes.

Use four disjoint development minibanks and 3 independent training seeds.

Provisional adoption gate:

- median development NLL gain over no-change >= 0.003 nat;
- Mirror candidate better than the dense-widening control on median development NLL;
- at least 2/3 seeds positive;
- at least 3/4 development minibanks positive.

Stop after two consecutive rejected state additions.

The numerical threshold is an experiment-management criterion, not a field-standard significance threshold.

## Split amplitude

Do not tune epsilon on the audit.

Use a small development-only discrete set fixed before the run:

- 0.5;
- 1.0;
- 2.0.

If a single epsilon must be used in confirmatory replication, use 2.0 because MN010 selected it in the discovery world, and label the replication accordingly.

## Controls

Required controls:

1. continue training with no structural change;
2. dense widening with similar added serialized bytes;
3. simple feature gate with similar stored size where practical;
4. optional independent soft-MoE reference, clearly marked as a reference rather than a matched runtime baseline.

## Metrics

Primary:

- audit NLL;
- audit accuracy;
- actual serialized bytes.

Secondary:

- active MAC estimate;
- measured CPU latency after training;
- router entropy / state utilization;
- pairwise functional output distance between states;
- development-vs-audit selection regret.

## Decision logic

The main hypothesis is supported only if there is a region where increasing Mirror breadth/count improves the rate-quality frontier beyond both:

- same-budget dense widening;
- no-change continued optimization.

A positive point on one synthetic world is not enough. Confirm on a fresh dataset seed with fixed hyperparameters.

## Failure interpretation

If breadth helps but count does not, retain a small number of rich states.

If count helps only at high breadth, Mirror count is conditional on state functional capacity.

If dense widening dominates at matched bytes, prefer Dense.

If all structural changes match the no-change shadow, treat the result as optimization noise rather than architecture gain.

## Evidence discipline

- no audit-based hyperparameter selection;
- preserve negative stages;
- record exact file bytes and hashes;
- distinguish discovery from confirmatory replication;
- do not convert path-count arguments into capacity claims.
