# MA-014 — four-layer layer-specific Mirror expert bank

Status: **PROMISING**; registered aligned quality/storage gate passed 3/3.
Branch: `research/ma-014-layer-specific-expert-bank-20261007`.
Base commit: `d8cd702778026c114d1a1f48dc1eee4f66582eed`.

## H — Hypothesis

One shared four-expert bank plus a small learned view per layer can recover aligned layer-specific routed functions across four layers at lower actual payload than sixteen independent experts.

## T — Test

Four 8→16→8 GELU layers, four experts, balanced oracle top-1 routing. This extends MA-241's two-layer soft-evaluation screen to four layers with exactly one active expert per example. Controls include independent banks, hard tying, gates, and rank-1/rank-2 residuals. Development world 140000 selects LR; fresh worlds remain sealed until both pre-access gates pass.

## Development result

- Selected LR 0.003 on development world 140000.
- Mirror MSE was 5.18e-5 vs 7.10e-5 full independent (0.73x); actual payload was 6,372B vs 18,289B (0.348x). The registered pre-fresh gate passed.
- Rank-2 residual reached 6.55e-5 at 9,443B. Mirror used 32.5% fewer bytes and had 20.9% lower MSE in development. Hard tie was smaller (5,930B) but had 7.1x Mirror MSE.
- On independent functions, Mirror MSE was 5.37e-3 vs 8.38e-5 independent, marking a private-function boundary.

## D — Decision

**PASS** on the registered fresh mechanism gate. Mirror/full MSE was 0.384–0.508 and actual payload was 6,372B vs 18,289B independent (0.348x, 65.2% fewer bytes). Mirror used 7.5% more bytes than hard tying and reduced MSE by 86.7–92.6% in all worlds. At near bytes, Mirror MSE was also 86–93% below the scalar gate; rank-2 residual used more bytes and had higher MSE.

This was not a runtime win: median eager CPU throughput was 1.65M examples/s vs 2.63M hard-tied and 3.10M independent; median training wall was 1.65s vs 0.84s tied. The MAC proxy was 256 per example plus about 12 Givens coordinate FLOPs, compared with 256 for tying and 256 for the one active expert in the independent model. On independent functions, Mirror MSE was 0.00579–0.00627 vs 0.000078–0.000089 for full independent. The teacher is deliberately view-aligned; this fixed-update screen is not capacity or language evidence.

All 36 deterministic fresh rows replayed exactly (timing excluded); eight frozen input hashes verified against commit `0f852c9f9408a210fd4d6b4ddf60281ed5c76550`.

## C — Strongest counter-hypothesis

As in MA-241, the aligned teacher is generated from a shared expert bank and layer Givens views. The design may be much less representative of naturally trained layers; scaling from two to four layers does not establish language benefit.

## U — Unconfirmed

Whether naturally trained layers contain this structure, learned routing, near-convergence capacity, fixed-byte frontiers, and optimized kernels remain untested.

## Fact / interpretation / hypothesis

- **Fact:** all three fresh worlds passed the aligned quality/storage and simple-control gates with 65.2% fewer bytes than independent and 7.5% more than hard tie.
- **Interpretation:** scaling to four layers and sparse top-1 execution retained the aligned synthetic view advantage, but eager runtime regressed and unrelated layers needed private weights.
- **Hypothesis:** naturally trained MoE layers may share a similar low-description layer orbit; this has not been measured.
