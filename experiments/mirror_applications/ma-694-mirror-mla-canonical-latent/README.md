# MA-694 — Mirror-MLA canonical latent cache

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `61041d10`

## Hypothesis

H: One MLA latent cache plus a compact shared basis of view-conditioned K/V up-projections can provide multiple logical attention roles at lower complete serialized bytes, with private residual rank revealing the heterogeneity frontier.

## Mirror insertion

> **Mirror insertion:** adds per-view coefficients and optional private residuals to the MLA latent-to-K/V up-projection read, keeping the physical latent sequence shared.

- Baseline: hard-tied MLA uses one latent cache and one projection pair for every role.
- Independent control: one K/V up-projection pair per logical role.
- Cheapest native equivalent: shared basis + direct coefficients/PCA, included as an exact attribution control.

## Prior-art delta

PA161 establishes MLA's shared low-dimensional latent cache and absorbed up-projections. This experiment isolates eight logical up-projection roles, measures complete state plus cache bytes and compute, and sweeps private residual rank. It does not treat view combinations as independent capacity.

## Controls and gates

Frozen dimensions, seeds, error/byte/compute gates, and residual grid are in [PROTOCOL.json](PROTOCOL.json). Mirror attribution fails if ordinary shared-basis/PCA coordinates match.

## Storage and compute

Actual serialized NPZ payload includes the latent cache, up-projection basis, coefficients, residual factors, and metadata. Independent full matrices are serialized separately. Report fit/decode and attention MAC proxy and CPU wall time descriptively.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic FP32 attention only.
