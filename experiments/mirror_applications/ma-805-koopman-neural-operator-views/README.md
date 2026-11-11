# MA-805 — Koopman neural-operator parameter Views

Status: **FAIL for Mirror-specific value** (known Fourier operator screen)

## Prior-art boundary

PA215 motivates combining Koopman latent evolution with operator learning; PA186 motivates Fourier-domain operators for PDE families. This experiment is a known linear periodic advection-diffusion semigroup in a fixed Fourier basis. It does not implement or train a KNO/FNO neural network.

## Fact

Seed 80501; 32 spatial points; nine diffusivities, with interior indices 2 and 6 held out; 512 test fields/regime; zero optimizer updates. Independent per-regime Fourier kernels use 2,191 B and have zero field nRMSE. An unconditioned shared KNO multiplier uses 1,201 B and heldout nRMSE .00712, already below the .02 tolerance. A scalar diffusivity View uses 1,444 B, zero heldout error, and max multiplier magnitude 1; it saves 34.1% bytes versus independent kernels but costs 20.2% more bytes than the unconditioned shared control and computes 153 exponentials per regime. The native conditional Fourier/KNO parameterization has the exact same 1,444 B payload/hash and output. Payload roundtrips are exact.

## Interpretation

The physical parameter code generalizes the known PDE semigroup to heldout diffusivities, and compresses the independent kernel bank. However, the shared unconditioned kernel already meets the quality tolerance with fewer bytes and no exponentiation per regime. The native conditional Fourier formula exactly aliases the Mirror View. Neither comparison establishes Mirror-specific improvement or a trained neural-operator result.

## H / T / D / C / U

- **H:** One shared Fourier-latent Koopman operator plus a scalar physical regime code recovers heldout PDE solutions at lower actual storage than independent FNO kernels while maintaining stable multipliers.
- **T:** Seed 80501; periodic 1D advection-diffusion semigroup, N=32; nine diffusivities, two held out; 512 fields/regime. Compare independent FNO kernels, shared unconditioned KNO, Mirror diffusivity code, and native conditional Fourier control. Measure actual compressed bytes/hash, field nRMSE, spectral magnitude, active modes, code-generation ops, test fields and CPU transform time.
- **D:** **FAIL** Mirror-specific value. It saves 34.1% vs independent and hits exact heldout fields, but the shared unconditioned KNO is smaller and already meets .02 error; native physical-parameter conditioning exactly aliases the View.
- **C:** The known PDE law directly maps diffusivity to Fourier multipliers. Standard physical-parameter conditioning is the simplest native explanation.
- **U:** Trained KNO/FNO, nonlinear PDEs, learned encoders, task-parameter inference, real dataset transfer, larger grids and GPU throughput.

Fresh stayed sealed after the shared baseline met the tolerance and native alias. This is a known-physics mechanism screen only.
