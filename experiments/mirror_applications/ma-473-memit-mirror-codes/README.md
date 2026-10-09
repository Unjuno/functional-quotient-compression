# MA-473 — MEMIT layerwise edit bank with Mirror memory coordinates

Status: **PROMISING for aligned edit-bank storage; merged multi-fact utility is not established**
Evidence lane: multilayer edit storage / efficacy / specificity / interference
Branch: `research/ma-473-memit-mirror-codes-20261009`
Protocol frozen: `2a8d0407`; fresh worlds 47310–47312 × seeds 0–2.

## H — Hypothesis

For a multi-fact update bank whose per-layer ROME/MEMIT rank-one factors lie on fixed-norm 2D orbits, shared layerwise bases and two angles per fact/layer can reduce serialized edit state relative to direct factor storage and Cartesian basis codes while preserving single-fact behavior and adding no merged interference beyond the direct factor control.

## T — Test

An analytic 16D linear model used four mediating layers and 64 fact updates per world. Each layer had a shared key plane and a shared value plane. Each fact/layer update was `ΔW[f,l] = v[f,l] k[f,l]ᵀ / ||k[f,l]||²`. We compared no edit, direct per-layer rank-one factors, generic Cartesian coefficients over the shared planes, and Mirror key/value angles. Fresh N values were 1, 8, 32 and 64, across three worlds and three seeds. Actual Torch serialized payloads charged bases, codes/factors, fact IDs, scales and metadata. We measured single-fact efficacy, orthogonal-key locality, merged-bank error, bytes, MAC proxies and small CPU timing probes.

## D — PROMISING for storage only

**Facts:** At N=64, Mirror used 5,973B (93.3B/fact), generic coefficients 8,021B (125.3B/fact), and direct MEMIT-style rank-one factors 35,105B (548.5B/fact). Mirror was 17.4% of direct-factor bytes and 74.5% of generic-coefficient bytes. Across all nine fresh pairs, maximum single-edit efficacy NRMSE was below 2e-7 and mean orthogonal-key locality drift was numerically zero. Mirror's N64 merged-bank error matched direct factors to within 4.7e-8 relative difference.

At the same time, the **absolute** merged-bank error grew to mean NRMSE 4.80 at N=64 (range 3.68–5.87), from 1.50 at N=8. All three representations shared this degradation. This screen therefore establishes compact storage for an aligned update bank and no extra representational interference, but it does not show that naive additive mass edits preserve useful factual behavior.

**Interpretation:** Angle coordinates compress the known fixed-norm orbit. This is a strong storage Pareto point against direct factors, and a smaller payload than the generic Cartesian coefficient control. The data do not establish naturally aligned facts or usable simultaneous edits; the common 2D key orbit causes large cross-edit interference as N grows.

## C — Strongest counter-hypothesis

The result is an oracle-aligned synthetic factor bank with hand-specified low-dimensional planes. The key vectors share only a 2D subspace, so cumulative rank-one updates collide. A learned MEMIT editor, realistic key covariance, or private components could change both compression and interference. The improvement may be ordinary polar compression of a known manifold rather than a Mirror-specific functional advantage.

## U — Unknown

Natural factual memories, learned shared-basis discovery, editing paraphrase generalization, adaptive residual allocation, and Transformer runtime are untested. This does not reproduce MEMIT optimization. MA-472's order and composition tests remain separate.

## Decision

**FACT:** Fresh single-edit reconstructions are exact to float precision; the N64 payload is far smaller than both direct per-layer factors and generic shared-plane coefficients. Merged interference is the same as direct factors but grows to a large absolute error.
**INTERPRETATION:** This is a storage-only result for aligned updates, not evidence of useful mass editing.
**HYPOTHESIS:** Natural MEMIT updates may admit compact layerwise coordinates if their directions cluster without collapsing fact specificity.
**BOUNDARY:** Analytic linear multi-layer update bank; no learned MEMIT or language-model factual evaluation.
