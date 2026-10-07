# MS008 — View decomposition and shared-world-core specification

Date: 2026-10-07 JST. Evidence: small synthetic MLP sensor worlds only. No Transformer, real-sensor, or general compression claim.

## Executive result

The strongest result is an architectural separation rule: **after calibration, do not feed sensor identity directly into the shared world core unless sensor-specific world laws are actually desired.**

In 8 fresh affine worlds, the no-sensor-ID shared core beat the ID-conditioned shared core on seen-law error in 8/8 and changed-law transfer in 8/8. Median transfer-error ratio (ID-conditioned / no-ID) was **2.430x**. A post-hoc intervention that forced receiver sensors through the source/reference address reduced that ratio to **1.028x**, supporting sensor-conditioned pathway divergence as the mechanism; this does not prove duplicated world laws.

Residual Views were not useful when affine calibration already removed almost all sensor difference: a dev-selected shear View worsened seen-law NMSE in 8/8 fresh affine worlds (median ratio **1.246**). Transfer effect was tiny (median ratio **0.997**).

When sensors contained nonlinear observation residuals that affine calibration could not remove, a useful View regime emerged. At beta=0.6, a combined shear+stretch residual View improved seen NMSE in **4/5** fresh worlds and changed-law transfer in **4/5**. Median ratios versus shared-only were **0.941** (seen NMSE) and **0.961** (transfer). New-sensor reference onboarding was neutral overall (median ratio **1.002**).

A fixed beta sweep with the same View specification showed a transition:
- beta=.2, post-calibration residual RMSE ~0.016: seen ratio **1.069** (worse, 0/3 wins);
- beta=.4, residual RMSE ~0.028: seen ratio **1.034** (worse, 1/3 wins);
- beta=.6, residual RMSE ~0.047: seen ratio **0.916** (better, 3/3 wins).

Residual RMSE above was computed post hoc on synthetic test states for diagnosis and was not used for model selection. A deployable threshold must be estimated from calibration/development data.

## Design consequence

1. sensor-specific encoder/calibration handles observation-system differences first;
2. the shared world core receives canonical state only, **not sensor ID**;
3. View is optional and residual, activated only when measurable post-calibration residual remains;
4. View code is learned with Backprop; ES and loops are absent;
5. sensor-specific View capacity stays small and structured;
6. combined shear+stretch is the best balanced candidate so far; stretch gave larger median seen-law gain but slightly worsened new-sensor onboarding;
7. label-free new-sensor View-code fitting from paired readings was weak/inconsistent and is not yet a solved onboarding mechanism.

## Storage

- no-ID shared payload: **11,690 B**
- residual View payload: **11,949–11,952 B** (~+2.2%)
- ID-conditioned shared control: **12,458 B** (~+6.6% vs no-ID shared)

All are deterministic uncompressed FP32 payload lengths including learned tensors, calibration, config metadata, and seeds required to reconstruct fixed structure.

## Decision

- **PASS:** calibration -> no-ID shared core -> optional residual View, for this synthetic family.
- **FAIL:** always-on View for well-calibrated affine sensors.
- **PROMISING / UNCERTAIN:** residual View under irreducible nonlinear observation mismatch.
- **FAIL / UNCERTAIN:** current label-free new-sensor View-code calibration.

Fresh-world ranges are not confidence intervals.