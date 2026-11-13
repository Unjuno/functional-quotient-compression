# MA-997 — Structured sparse equivariant support × Mirror code

Status: SCREENING. This candidate isolates shared sparse support from MA-996's dense rank-2 coefficient code. It uses the same conservative radial toy harness only as a mechanism screen; no MACE-MP/PA300 result is claimed.

## Hypothesis

H: A common sparse support with per-material values compresses an aligned bank of conservative material residuals beyond independent sparse masks and native sparse fine-tuning.

## Controls

Shared-only, a matched native shared-mask coefficient table, per-material sparse masks, and independent full coefficients. All comparison payloads include shared radial basis/base and serialization metadata.

## Protocol

Development seed 99701 selects six support coordinates. Fresh worlds 99711/99713/99717 remain locked until protocol commit. Each fresh world contains four target potentials, eight calibration and 24 held-out configurations per target.

## Decision

FACT: pending.\nINTERPRETATION: pending.\nHYPOTHESIS: pending.\nBOUNDARY: synthetic radial-potential mechanism only.

Implementation amendment 01: before fresh access, fixed serialization to store the independent sparse-fit values corresponding to its selected indices. No fresh world was opened before this correction. Radial width is fixed at 0.65 to match MA-996.
