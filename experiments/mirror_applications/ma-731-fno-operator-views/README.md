# MA-731 — FNO parameter-family Mirror operator views

Status: SCREENING

## Hypothesis

H: A shared spectral operator basis plus a low-description parameter View code represents unseen diffusion operators with lower complete bytes than storing one FNO kernel per parameter, while beating native per-mode polynomial conditioning.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, a compact coefficient code over shared Fourier-mode operator atoms, to one physical spectral kernel so the same learned basis can represent multiple PDE-parameter operators.

Prior art: Fourier Neural Operator PA186. Native controls: independent spectral multiplier bank, direct per-mode polynomial in the physical parameter, and analytic diffusion multiplier formula. The analytic formula is an oracle/physics upper control, not a learned-model baseline.

## Frozen protocol

1D periodic diffusion with spectral multiplier `exp(-alpha*k^2*t)`, 16 retained positive modes and t=.04. Fit representation using 20 alpha values in [.02,.16]; evaluate 7 unseen alpha values spanning interpolation and one mild extrapolation. At each alpha, use 128 random real periodic input fields on resolutions 64 and 128. Fit shared operator basis from training multipliers only; sweep ranks 2/4/8/16; predict coefficient View codes with a cubic polynomial in alpha. Compare independent kernel bank, Mirror basis+code, native per-mode cubic alpha conditioning, scalar linear interpolation, and analytic physics formula. Report relative solution error, actual serialized operator bank bytes and multiply proxy.

### Gates

PASS requires relative solution error <=.02 at both resolutions, >=25% complete bytes saved versus independent kernels, and a strictly better quality/byte frontier than native polynomial conditioning. Exact native function/state alias or failure is scoped FAIL. No audit if development shows native control dominance.

## Scope

Synthetic linear diffusion only; not trained FNO or learned PDE generalization evidence. Analytic physics control is known-law upper bound.

## Results — development seed 73101

**FACT:** Independent bank of seven held-out spectral kernels is 874 B with zero relative solution error. Native per-mode cubic conditioning is 908 B and mean relative errors .000461/.000469 at resolutions 64/128. Mirror q=2 is 1,267 B with errors .000598/.000604; q=4 is 1,436 B with errors .000461/.000469; q=8 is 1,767 B and q=16 2,387 B. Mirror outputs and payload hashes exactly match the native PCA operator-code control at each rank. Parameter interpolation is 1,655 B with error .00688/.006998. Analytic known-law control is 437 B and exact (this assumes the PDE formula is known and is an oracle upper bound). All learned methods meet the .02 error tolerance, but no Mirror rank saves bytes against independent or native polynomial controls. Per-method CPU screen wall times are noisy single-run measurements and are not used as a performance claim.

**INTERPRETATION:** Low-rank basis coordinates can describe the diffusion family accurately, but in this known parameter family they store more than explicit kernels and native per-mode polynomials. The analytic PDE formula is even smaller. The Mirror-specific mechanism is an exact ordinary PCA/shared-coordinate alias.

**HYPOTHESIS:** A learned FNO on an unknown parameterized PDE may have operator variations not captured by a known analytic law; this test does not establish that case.

**D:** FAIL for the frozen storage and Mirror-attribution gates; audit unopened.

**C:** The law is known and synthetic, and the analytic control is intentionally very strong. The generated family has low-rank structure by physics.

**U:** Trained FNO weights, nonlinear PDE families, natural parameter generalization, grid extrapolation beyond 128, multi-seed quality and fused accelerator runtime.

### H / T / D / C / U

- **H:** Shared Mirror operator atoms encode multiple PDE parameter kernels with lower state than independent FNO kernels and native conditioning.
- **T:** 16 spectral modes; 20 train alpha values after amendment 01 (8 were insufficient to identify rank16); 7 held-out alphas; 128 fields each at resolutions64/128; dev seed73101; ranks2/4/8/16.
- **D:** FAIL. All errors are small, but Mirror payloads 1,267–2,387B exceed independent 874B and native cubic 908B; exact native PCA code alias.
- **C:** Known diffusion law / analytic formula control and low-rank physics structure.
- **U:** Learned FNO and unknown nonlinear PDE families.
