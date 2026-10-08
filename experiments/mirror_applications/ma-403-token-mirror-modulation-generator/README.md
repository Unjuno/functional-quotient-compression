# MA-403 — token-wise Mirror modulation generator

Status: SCREENING; protocol and implementation frozen before development runs.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Prior art: PA63 (FiLM), PA64 (SPADE/spatially adaptive normalization).

## H — Hypothesis
A token-position-generated Mirror coordinate can express smooth token-specific rotations over one shared nonlinear feature block with lower held-out error and payload than native token-wise FiLM. The full-affine generator is a higher-cost conditional control.

## Mirror insertion
> **Mirror insertion:** this experiment adds a token-position input `p` and an eight-angle code `m(p)` to one shared 16-dimensional feature block, so one physical block can provide a continuum of token-specific feature rotations.

The task uses a frozen shared feature block and target angles affine in token position. Compare global FiLM, token-wise affine FiLM, token-wise rank-one residual, Mirror angle generator, and full-affine generator. The protocol defines coefficients, held-out positions, bytes, compute, and gates.

## Freeze boundary

Development seeds 40301/40302. Fresh seeds 40311–40313 remain sealed unless the full development gate passes. Actual deterministic NPZ payload bytes, including shared weights, generator parameters, and metadata, are authoritative. The benchmark includes code generation and transformation.
