# MA-014 — four-layer layer-specific Mirror expert bank

Status: SCREENING; development passed, fresh settings frozen.  
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

Fresh results pending; fixed-update development is not capacity evidence.

## C — Strongest counter-hypothesis

As in MA-241, the aligned teacher is generated from a shared expert bank and layer Givens views. The design may be much less representative of naturally trained layers; scaling from two to four layers does not establish language benefit.

## U — Unconfirmed

Natural MoE layers, learned routing, near-convergence capacity, fixed-byte frontiers, and optimized kernels remain untested.

## Fact / interpretation / hypothesis

- **Fact:** the protocol fixes four layers and one active expert per example; development passed quality/storage access conditions.
- **Interpretation:** this isolates scaling the layer-view idea from two layers while reducing expert compute vs MA-241's all-expert evaluation.
- **Hypothesis:** layer-specific views may retain their storage/quality value when expert execution is sparse.
