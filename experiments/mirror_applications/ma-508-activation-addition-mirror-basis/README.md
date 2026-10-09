# MA-508 — Activation-addition Mirror basis

Status: FROZEN_SCREENING  
Evidence lane: MECHANISM / STORAGE / FUNCTIONALITY  
Base commit: `f85b2963` (integrated through MA-504)  
Prior art: PA97 (Representation Engineering), PA98 (Activation Addition)

## Hypothesis

H: A shared rank-4 activation-steering basis plus a per-behavior low-description Givens View can store and apply a bank of fixed-norm steering behaviors with lower actual bytes than explicit steering vectors and generic per-behavior rank-4 coefficients, while retaining on-target efficacy and off-target specificity.

## Mirror insertion

> **Mirror insertion:** this experiment adds two per-behavior Givens angles `m_i` to one shared activation-addition vector so that multiple logical behavior directions are represented by Views of one physical steering code rather than storing one full vector per behavior.

- Physical object: one shared synthetic 64×4 orthonormal representation basis and one rank-4 seed vector.
- Mirror coordinate: two per-behavior Givens angles; optional FP16 serialization is charged.
- Logical multiplicity: 64 behavior vectors applied as additive activation interventions.
- Simple controls: direct full vectors; generic shared basis plus FP32 or FP16 per-behavior coefficients.
- Task reads out behavior efficacy and off-target effects as well as vector reconstruction.

## Frozen protocol

- D=64, rank=4, 64 behavior vectors; 20% of behavior IDs are held out from any basis/seed fitting.
- Fresh worlds: 50820/50821/50822 × seeds 0/1/2. Development worlds: 50800/50801 × seeds 0/1/2.
- Two regimes: fixed-norm two-plane Givens orbit (`rho=0`) and 10% coefficient-space private residual (`rho=.1`).
- `B` is a known shared synthetic basis charged in each low-rank payload. The shared basis and seed are fixed by a public deterministic synthetic construction and charged in every payload. Development validates the encoder; fresh behavior angles are not used to tune anything and their compact codes are paid inference state.
- A fixed random set of 32 unit readout probes is frozen per world. Efficacy is target-readout activation change relative to the explicit-vector teacher. Off-target drift is RMS change under 31 non-target probes. Also report vector NRMSE and exact payload bytes.

## Methods

1. Explicit full 64D steering-vector bank (exact control).
2. Shared basis + per-behavior FP32 rank-4 coefficient table.
3. Shared basis + per-behavior FP16 coefficient table.
4. Mirror shared basis + one FP32 seed code + per-behavior two FP16 Givens angles.
5. Mirror plus per-behavior FP16 rank-4 private residual fallback; this maps the private-state boundary and charges every residual.

## Gates

- PASS for aligned compression if every fresh `rho=0` world has vector NRMSE ≤0.01, efficacy ≥0.99, off-target drift ≤0.01, and payload ≤80% of both the explicit bank and the cheapest generic coefficient control.
- PROMISING if it beats explicit storage/quality but a generic coefficient control matches the rate-quality point, or if CPU runtime is worse.
- FAIL if aligned quality or byte gates fail or generic coefficients dominate.
- `rho=.1` is a boundary test; record the residual fraction and cost needed to recover efficacy.

## Boundaries

This is a controlled synthetic Activation Addition screen, not a pretrained-language-model behavior-steering result. The aligned orbit is deliberately favorable to Mirror. Codes for the 64 test behavior directions are paid inference state. Logical behavior count is not capacity.


### Amendment A1 (before development/fresh)

The protocol now fixes one shared basis/seed across development and fresh banks instead of fitting a separate basis per world. This prevents per-world shared-model changes from being confused with behavior-code storage. Both tensors are included in every applicable serialized payload. No fresh data has been opened.

## Results

### H / T / D / C / U

**H:** A shared seed plus per-behavior Givens Views compresses a bank of activation-addition directions while preserving efficacy and specificity better than explicit vectors and generic low-rank coefficients.

**T:** One deterministic 64×4 basis and shared seed, 64 steering behaviors, aligned `rho=0` and 10% private coefficient residual `rho=.1`, three fresh worlds × three seeds. Compared explicit vectors, shared-basis FP32/FP16 coefficients, Mirror FP16 angles, and Mirror plus FP16 private residuals. A1 fixed the same physical basis/seed across development/fresh; all are charged. Fresh angle/residual codes are stored per behavior.

**D: FAIL for the registered Mirror-specific gate; shared-basis storage is promising as a generic compression result.** At `rho=0`, Mirror NRMSE was .000385, efficacy 1.0 and off-target drift 4.9e-5 at 3,361B versus generic FP16 coefficients at 3,365B (.000213 NRMSE, 2.7e-5 drift). The 4B saving is far short of the required 20% advantage over the cheapest generic control. Both compressed representations use ~18.6% of explicit-vector bytes (18,025B). Mirror CPU decode took 0.049 ms/bank versus 0.0044 ms for generic FP16 coefficients. At `rho=.1`, Mirror-only NRMSE rose to .141 and off-target drift .0179; adding all per-behavior private residuals restored NRMSE to .000361 but raised payload to 4,061B.

**C:** Generic FP16 coefficients already encode the same rank-4 behavior family nearly optimally; the Mirror phase format adds decode compute while saving only 4 bytes. Results are on a deliberately Givens-aligned synthetic bank.

**U:** Pretrained-language behavior efficacy, natural steering vectors, learned basis selection, selective residual allocation and device-optimized decode kernels are untested.

### Facts

- 90 fresh rows: 2 residual regimes × 3 worlds × 3 seeds × 5 methods.
- At `rho=0`, Mirror uses 3,361B vs FP16 generic coefficient control 3,365B; explicit full vectors use 18,025B.
- At `rho=.1`, Mirror-only error is .1411 mean. Private FP16 residuals reduce it to .000361 at 4,061B.
- All serialized replay errors are zero; payload lengths and SHA-256 digests are checked.
- Container is CPU-only; no CUDA device was available.

### Interpretation

Shared low-rank activation bases compress this synthetic steering bank substantially relative to storing full vectors. The Mirror phase parameterization does not provide a useful marginal storage advantage over generic FP16 basis coefficients and is much slower in eager CPU decoding. Off-orbit behaviors need private state.

### Hypothesis boundary

The experiment fails the registered Mirror-specific 20% byte advantage over the cheapest generic code control. It supports only generic shared-basis compression on an aligned synthetic orbit; it does not establish behavior steering on a pretrained LM.
