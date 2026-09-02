# Research History and Consolidation Map

This file preserves the scientific lineage without making historical checkpoint numbering the repository structure.

## Legacy origin: “Vector Mirror”

The project began by exploring whether multiple vectors/parameters related by mirror or orthogonal transforms could share a representation. The durable result from this phase is a correction: reversible coordinate changes are not themselves information compression. Mirror operations remain useful only as possible alignment/gauge tools inside a broader functional-equivalence framework.

## D1–D19 — geometry, gauge, bases, packet structure

Topics included shared bases, Givens search, block gauges, partition/signature geometry, certified rank, tree critical prices, orthogonal packetization, structural shells, anchor/master representations, approximate joint diagonalization, and commutator closure.

Canonical contribution to FQC:

- functional/gauge equivalence;
- shared low-description geometry;
- diagnostics for when a common representation is or is not justified.

## D20–D38 — stable structure and exact/priced combinatorial search

Topics included commutator-Laplacian diagnostics, separator stability, recursive structure, rate-distortion-priced split/merge operations, topology trees, laminar/crossing structure, branch-and-bound, LP guidance, conflict cuts, and lifted covers.

Canonical contribution:

- structural candidate search;
- exact/certified combinatorial optimization machinery;
- separation of proposal heuristics from certified acceptance.

## D39–D49 — certified search work and replay

The project explicitly separated search/proof/replay cost from codec payload bits. Work envelopes, proof-frontier compression, replay traces, caches, and adaptive controllers were studied.

Canonical contribution:

- research-compute optimization;
- accounting discipline: reducing search work is not reducing the serialized model.

## D50–D69 — hard 64× contract and codec-oriented pricing

Key milestones:

- D50: hard 64× accounting target for a 16-bit scalar baseline;
- D56: real-pilot evidence contract;
- D57: architecture-aware Transformer extraction contract;
- D58–D69: diagnostics-to-codec mapping, rate-distortion frontiers, functional sensitivity, bit allocation, shared atoms, learned dictionaries, root pricing, and replacement bounds.

Canonical contribution:

- hard paid-bit accounting;
- real-model evidence requirements;
- decoder-DAG prerequisite discipline.

## D70–D101 — scheduler/validation and reproducibility lessons

Adaptive bound escalation, value-of-information scheduling, confidence bounds, audits, candidate-impact localization, and stateful counterfactuals were studied. A major reproducibility result was negative: the original V2 simulator was not fully serialized, and the reconstructed V2-R could not be treated as the original evidence source.

Canonical contribution:

- strict synthetic-vs-real evidence boundary;
- reproducibility as a hard gate;
- stateful replay/counterfactual caution.

## D102–D112 — fully serialized synthetic V3

A new synthetic environment (`VM_SYNTH_V3_20260902`) was introduced with explicit constants/replay identity. The line studied recalibration, sufficient statistics, structural pseudo-targets, branchless fallback, and selective queries.

Canonical contribution:

- a cleaner synthetic research-scheduler test bed;
- evidence that query/controller complexity can be economically dominated by representation choices.

This entire line remains `SYNTHETIC`, not codec evidence.

## D113–D120 — representation audit and decision geometry

The late line progressively weakened information interfaces, then found a much larger effect from representation choice itself.

- D115: global E ablation strongly improved the tested synthetic policy; the adaptive branch became harmful afterward.
- D116: common E contribution was identified as a decision-null gauge in the tested pairwise decision system.
- D117: a tiny spectral-energy mode was decision-critical under hard deletion; soft shrink worked better.
- D118: an automatic normalized spectral-tail shrink rule improved the base but did not beat fixed q=0.75.
- D119: jackknife amplitude uncertainty did not recover the best tested shrink.
- D120: hard-support robust-shift geometry improved the base but still did not beat the simpler D118 rule or fixed q=0.75.

Canonical contribution:

- decision-equivalence / quotient perspective;
- warning against energy-only rank truncation;
- representation simplification before controller complexity.

## Parallel COMPLETE branch — exact codec mechanism experiments

A separate handoff branch contains the E1–E7 deterministic codec experiments and a consolidated optimizer specification. This branch is not fully superseded by D120 and must be integrated separately.

Key retained mechanisms:

- E1 serializer-aware Bellman/Pareto optimization;
- E2 cross-block coupling and safe majorization;
- E3 tree co-design and decoded-mask quotient;
- E4 exact validation of surrogate-ranked rotations;
- E5 exact transfer for description-only rotations under stated assumptions;
- E6 operation-local conditional Pareto caches;
- E7 joint tree/precision local bundles and complementarity.

## Canonical project name transition

The project is now called **Functional Quotient Compression** because the scientifically durable object is not a mirror transform. The unifying question is whether the model can be compressed by quotienting functionally/task-equivalent states and coding only the remaining task-relevant structure.

Historical “Vector Mirror” names are retained only in provenance metadata and source citations.
