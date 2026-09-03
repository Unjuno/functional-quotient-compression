# TinyStories-8M architecture-aware bundle graph results

This note records a narrow follow-up to the actual TinyStories-8M interaction experiments. It tests whether QCO becomes simpler if same-layer Q/K actions are represented as a single architecture-coupled bundle node.

## Setup

- uploaded TinyStories 8M GPT-Neo checkpoint;
- manual local PyTorch forward and GPT-2 BPE tokenizer;
- four fixed next-token probes;
- Q and K weights use 32-scalar block VQ, K=128;
- `B_l` denotes the joint same-layer action `{Q_l, K_l}`.

## 1. Bundle nodes absorb the strong same-layer Q/K complementarity

The separate-action experiment showed that same-layer Q/K has joint/additive KL ratio about `0.62–0.72`, unlike cross-layer Q/K which is usually super-additive. Treating `{Q_l,K_l}` as one node therefore removes a known architecture-internal interaction from the outer optimizer state.

## 2. Bundle nodes are still not independent across layers

All 28 two-bundle combinations were evaluated.

`D(B_i,B_j) / [D(B_i)+D(B_j)]`:

- minimum: `0.9709`;
- median: **`1.1160`**;
- maximum: `1.2712`;
- 23/28 are more than 5% super-additive.

Thus bundling simplifies the graph but does not make layer-level task costs additive.

## 3. Pairwise bundle edges predict triple-bundle damage well

All 56 choices of three bundle nodes were evaluated; each triple contains six underlying Q/K weight actions.

- median actual / sum-of-bundle-costs: **`1.1972`**;
- range: `1.0486–1.4026`.

Using measured pairwise bundle interaction edges,

`D_pair(S)=sum_i D(B_i)+sum_{i<j}[D(B_i,B_j)-D(B_i)-D(B_j)]`,

produced:

- median relative error: **`0.91%`**;
- maximum error: `2.18%`;
- predicted-vs-actual ranking Spearman: **`0.99959`**.

For the actual safest top-5 triple bundles:

- independent bundle-cost ranking recalled `4/5`;
- pairwise bundle ranking recalled **`5/5`**.

## Optimizer implication

The observed structure supports a hierarchical QCO representation:

1. form architecture-coupled atomic bundles for strongly interacting operations such as same-layer Q/K;
2. attach pairwise task-interaction edges between bundle nodes;
3. use independent node costs only for cheap initial screening;
4. use the pairwise graph for shortlist selection;
5. retain exact model replay as the commit authority.

This is actual-model evidence for one checkpoint/action family, not a theorem that all useful Transformer interactions are pairwise or that Q/K must always be bundled.
