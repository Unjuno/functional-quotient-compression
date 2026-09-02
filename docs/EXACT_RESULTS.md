# Exact / Deterministic Codec Results

These results establish mechanisms and exact optimizer behavior in finite toy systems. They are **not** real-Transformer compression results.

## E1 — Serializer-aware Bellman / Pareto

- 4-leaf system with 152 legal codec states.
- Additive logical-bit Bellman matched brute force only before serializer overhead.
- Serializer rule `8 * ceil((B_raw + 5) / 8)` changed the optimum.
- A raw 76-bit state serialized to 88 bits and therefore falsely passed an 80-bit raw-budget test.
- True serializer-aware 80-bit optimum: distortion `23.5505` at 72 serialized bits.
- Exact Pareto pruning: `152 -> 11` states while preserving the serializer-aware optimum.

**Durable result:** hard-budget decisions must be made against the actual serializer or an exactly equivalent serializer-state model.

## E2 — Cross-block coupling

A 4-leaf PSD task metric with strong cross terms was used.

- naive diagonal distortion underestimated true distortion in all 38 tested codec states;
- at `D <= 20`, the naive model produced a false 32-bit pass whose true distortion was `24.0726`;
- a microcluster/block majorizer reduced conservatism and recovered the true minimum bit counts at tested thresholds;
- topology affected true distortion: a coupling-aligned partition achieved 40-bit true distortion `10.615208`, while crossed alternatives were `22.344664` and `24.072600`.

**Durable result:** topology and cross-block coupling belong in task/compression geometry; independent block accounting can be unsafe.

## E3 — Tree co-design and decoded-state quotient

For a 16-leaf toy:

- recursive optimizer state count: `2,090,918`;
- distinct decoded binary masks: `65,536`;
- quotienting by decoded mask yields about `31.9×` state reduction.

A second result showed a conflict between certificate geometry and compression geometry: a tree with tighter coupling majorizers need not be the tree that encodes the best hard-budget private/shared mask.

**Durable result:** `tight certificate tree != best compression tree` in general.

## E4 — Tree-rotation surrogate audit

A single predicted-mask surrogate could overfit. Ensemble surrogates improved average ranking but retained only modest rank correlation with the exact hard-budget objective in the tested neighborhood.

A validated proposal strategy was therefore used:

1. surrogate ranks candidate rotations;
2. only exact hard-budget improvement is committed.

Top-3 exact validation produced no worsening in the reported 30-seed test and, in a 10-seed subset, matched the full local NNI oracle while reducing exact validations from 40.2 to 5.2 on average.

**Boundary:** this is empirical; Top-3 is not a theorem.

## E5 — Description-only rotation transfer

Across all `65,536` masks in a 16-leaf toy, the tested local description-cost identity held:

`C_Tr(M) = C_T(M) + delta_r(M ∩ R)`.

The conditional-frontier transfer reproduced direct hard-budget enumeration when decoded distortion was tree-independent.

**Durable result:** description-only operations can reuse exact conditional frontiers when future decoded semantics are unchanged and the locality assumptions hold.

## E6 — Operation-local Pareto cache

In the 16-leaf / 65,536-mask reference system, exact conditional Pareto entries were:

| local support m | entries | reduction vs full universe |
|---:|---:|---:|
| 2 | 110 | 595.8× |
| 4 | 332 | 197.4× |
| 6 | 933 | 70.2× |
| 8 | 2,476 | 26.5× |

**Durable result:** operation locality was more valuable than aggressive approximate bit bucketing; exact bits near a hard budget should not be coarsened casually.

## E7 — Joint tree × precision bundle

- 8 leaves;
- 5 local precision states;
- full assignment universe `5^8 = 390,625`;
- 24-bit hard-budget constructed example.

Results:

- crossed-tree optimum: all state-2, `D = 15.23899`;
- aligned-tree optimum: `33330000`, `D = 14.26436`;
- same serialized budget;
- joint improvement: `0.97464`, or `6.40%`;
- from the crossed incumbent, precision-only gain = 0 and tree-only task gain = 0, while joint tree+precision gain > 0.

Exact conditional cache sizes:

| local support m | entries |
|---:|---:|
| 1 | 68 |
| 2 | 294 |
| 3 | 1,546 |
| 4 | 4,662 |

The full universe divided by the `m=4` cache is about `83.8×`.

**Durable result:** structural variables can be complementary; a canonical optimizer must support joint local codec bundles rather than relying only on coordinate descent.

## What these results do not establish

They do not establish:

- a real-model compression ratio;
- Transformer task-quality retention;
- training stability;
- practical wall-clock gains;
- 64× feasibility on a real model.

The next requirement is to reconstruct these mechanisms as executable tests and then test the resulting optimizer on a real Transformer structural audit.
