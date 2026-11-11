# MA-805 status

- Status: FAIL for Mirror-specific value (known Fourier semigroup only).
- Branch: `research/ma-805-koopman-neural-operator-views-20261111`
- Base: `0849d4f7`
- Development: seed 80501; 2 diffusivity values held out.
- Fresh/audit: sealed after unconditioned shared KNO met tolerance and native alias.
- Verification: one focused test passes; payloads roundtrip exactly.

## H / T / D / C / U

- **H:** Shared Fourier operator plus regime code improves storage/quality over independent FNO kernels and native controls.
- **T:** Known N=32 advection-diffusion Fourier semigroup; nine diffusivities; independent FNO, shared KNO, View and native physical-law controls.
- **D:** **FAIL.** View saves 34.1% bytes vs independent with zero heldout error, but unconditioned KNO already reaches .00712 at 1,201 B vs View's 1,444 B; native conditional formula aliases exactly.
- **C:** The physical diffusivity parameter deterministically specifies the Fourier multiplier; ordinary conditional Fourier evaluation is sufficient.
- **U:** Trained neural operators, nonlinear PDEs, natural transfer, larger grids and accelerator runtime.

## Next action

Verify the serialization/test and record the scoped result; continue to next registry candidate.
