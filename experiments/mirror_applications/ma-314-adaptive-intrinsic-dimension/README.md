# MA-314 — Adaptive intrinsic dimension Mirror allocation

Status: SCREENING. Prior art: PA33 intrinsic-dimensional fine-tuning.

## H

Per-task active dimensions may reduce storage over a fixed full intrinsic vector. A shared Mirror orbit plus sparse private fallback may add another sharing level; generic PCA and direct adaptive coordinates are controls.

## T

Synthetic regression with D=128, intrinsic d=32, 24 tasks and active dimension levels 4/8/16/32. Compare direct adaptive coordinates, Mirror shared-vector/angle with private residual fallback, no-fallback Mirror, and generic PCA. Two strata: shared Givens orbit and independent task vectors. Development worlds 31400–31401; fresh 31410–31412; three seeds each. U and all coordinates, dimension codes, private residual indices/values, PCA state and metadata are charged.

## D

Pending development/fresh results.

## C

The aligned tasks are teacher-generated from the same Givens orbit, while the direct adaptive-coordinate control may already capture all savings. Independent tasks may force private state.

## U

Synthetic linear tasks only; no pretrained model or natural task adaptation.

Development: aligned tasks select mean active dimension 15/32. Mirror with variable rank uses 8,557 B at query NRMSE ~3.18e-7, about 6% fewer bytes than direct adaptive coordinates (9,110 B) and 8.5% fewer than generic adaptive PCA (9,352 B), with similar quality. Independent tasks select full dimension; the Mirror view splits 23/24 tasks and uses 12,978 B vs direct 9,926 B. Without private residuals its query NRMSE is ~1.19. Fresh checks the fixed rule.
