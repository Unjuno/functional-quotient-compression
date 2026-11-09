# MA-694 — Mirror-MLA canonical latent cache

Status: FAIL  
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

## Results and decision

**FACT:** Across three audit seeds, aligned rho=0/q=0 views reconstructed with zero measured output NRMSE at 4,426 B complete serialized state versus 6,406 B independent (69.1%) and 1.04x the independent MAC proxy. At rho=.1, q<=2 NRMSE was .130–.142; q=4 was .092–.130. Exact reconstruction required q=8 at 8,810 B (137.5% of independent). At rho=.25, q<=4 NRMSE was .231–.342; q=8 again exceeded independent bytes. Every Mirror payload hash and output exactly matched the native shared-basis/PCA control. Hard-tied MLA missed view-specific outputs.

**INTERPRETATION:** A shared latent and low-rank view basis can store a deliberately aligned bank compactly. Mild private heterogeneity already requires more than q=2 to approach the frozen error gate; exact private recovery erases the byte saving. The candidate is ordinary shared-basis coefficient coding, not Mirror-specific value.

**HYPOTHESIS:** Canonical MLA latents can support many logical projections when the map bank is aligned, but meaningful private variation sets a sharp state frontier. The preregistered rho=.1 quality/bytes gate failed.

**BOUNDARY:** Synthetic FP32 attention operators, T=96, D=16, latent rank 4, eight views; no trained MLA model, language NLL, GPU kernel, or production throughput claim. Audit seeds were locked and untouched during tuning.
