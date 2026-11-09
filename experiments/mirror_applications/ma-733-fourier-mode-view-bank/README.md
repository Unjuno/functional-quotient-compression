# MA-733 — Fourier-mode Mirror transform bank

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

The fixed four-atom band bank serialized to 2,382–2,383 B, versus 4,123–4,137 B for independent spectra and 3,128–3,133 B for the native rank-4 PCA bank. At rho=0 its mean total/high-band solution errors were numerical zero; at rho=.05 they were .0054/.0149. At rho=.2 they were .0218/.0593 and missed both .02 limits. The native shared-band control and Mirror rows have identical payload hashes, bytes and outputs for every rho, residual budget and resolution. Full measurements are in `RESULTS_CORE.csv`.

## Interpretation

The predeclared synthetic band family has a useful compact representation: about 42% fewer bytes than the independent bank. For rho=.05 it meets the absolute quality limits. Sparse private entries improve the off-band fit, but at high rho the tested k<=16 budget does not restore the high-band gate. The same band-coordinate representation is ordinary shared-band coefficients; it adds no Mirror-specific function, storage or compute advantage. It also does not beat native PCA on a strict quality/bytes frontier at rho=.05: PCA is more accurate while band coding is smaller.

## Hypothesis result

**FAIL** for incremental Mirror value. The generic structured spectral compression feasibility case is positive only for the deliberately band-aligned synthetic family. No trained FNO, physical PDE or extrapolation claim is established. Fresh/audit was not opened because the exact native shared-band alias falsified the Mirror-specific hypothesis.

## Frozen protocol

D=64 retained positive Fourier modes, 16 logical domain operators (12 basis-fit, 4 held out), seed73301, grids128/256. Generate a shared smooth base spectrum plus four band-specific response atoms; sweep private residual rho={0,.05,.2}, top-k={0,2,4,8,16}. Compare independent kernels, band-code representation, native shared PCA, native shared-band coefficient control, and matched magnitude/random private residual controls. Apply to 128 random smooth input fields/domain and report total and high-band k>32 relative solution error, complete serialized bank bytes, active multiply proxy and CPU multiplication time.

Gates: total and high-band errors <=.02, >=25% complete byte savings versus independent kernels, and strictly better quality/byte frontier than native shared-basis controls. Exact native alias is FAIL for Mirror-specific value. No fresh/audit if the development result falsifies specificity.

## Scope

Synthetic linear spectral operators only. No optimizer, training updates or token count apply to this mechanism screen. The 128/256 grids use the same retained 64-mode response and do not establish resolution extrapolation. No trained FNO, real PDE or accelerator runtime claim.

## H / T / D / C / U

- **H:** Shared Fourier base and low-description band coordinates preserve multiple domain responses below independent-bank bytes and improve over native shared-basis controls.
- **T:** One deterministic development world (seed 73301), 64 modes, 16 domains (12 basis-fit, 4 held out), 128 fields/domain, two grid sizes; rho 0/.05/.2; independent, PCA, shared-band, Mirror-band and sparse-private controls. No fresh worlds.
- **D:** FAIL for Mirror-specific value; scoped synthetic band compression meets the quality/byte gate at rho 0 and .05 against independent kernels but exactly aliases native shared-band coefficients.
- **C:** Gains come from the intentionally aligned four-band family and ordinary shared-band coding. The native shared-band method has the same function and serialized payload.
- **U:** Trained operator learning, natural domain shifts, true resolution extrapolation, optimized inference kernels and physical PDE quality remain untested.
