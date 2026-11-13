# MA-996 — MACE-MP shared force field with material Mirror code

Status: SCREENING
Branch: `research/ma-996-mace-material-mirror-20261113`

## Hypothesis

H: A compact material-specific coordinate on a shared conservative invariant potential can recover useful held-out material functions with fewer inference bytes than independent coefficients, while beating native low-rank and sparse controls.

## Mirror insertion

- Native method: shared scalar energy function expanded in pair-distance radial basis features.
- `m` insertion: two-value material coordinate selects a coefficient vector in a shared rank-2 material subspace.
- `m` is persistent per material.
- Simplest controls: shared-only, scalar gate, native low-rank coefficient factorization, sparse coefficient update, and independent coefficients.

## Physical-to-logical claim

- Shared object: radial basis and shared coefficient subspace.
- Logical multiplicity: distinct conservative synthetic material pair potentials.
- Storage claim uses actual serialized payload bytes.
- Main risk: this is an analytic toy and the Mirror factorization may exactly alias ordinary low-rank adaptation.

## Prior art

PA296 establishes MACE's equivariant energy/force modeling; PA297 establishes a pretrained shared MACE-MP material potential; PA300 establishes structured sparse adaptation as a strong native control. The present screen tests only a conservative, rotation-invariant radial mechanism and does not reproduce any of those systems.

## Tuning boundary

Development seed: 99601 only. Fresh worlds: 99611, 99613, 99617, opened after protocol freeze.

## Decision

FACT: pending run.
INTERPRETATION: pending run.
HYPOTHESIS: pending run.
BOUNDARY: no MACE-MP or real-material claim.

## Protocol amendment 01

Before opening fresh worlds, development review found an implementation error in delta-code fitting: the shared-base prediction was not subtracted from calibration targets. The Mirror and sparse fit now use residual targets, and the scalar-gate payload stores all per-target gates. No fresh result was accessed before this correction.
