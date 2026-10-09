# MA-327 — Factorized layer × expert Tucker address

Status: SCREENING. Prior art: PA35 Tucker matrix banks.

## H

Factor layer and expert coefficients instead of storing one Tucker vector per pair. A Mirror expert address may reduce the expert factor further while predicting held-out layer-expert combinations.

## T

Synthetic 16×16 Cartesian layer/expert bank (16 layers × 16 experts), K=8 orthonormal 8×8 matrix bases. 75% of non-anchor combinations observed; hold out the remainder. Compare independent full matrices, flat Tucker coefficients, generic free layer×expert factors, and Mirror Givens expert codes. Test aligned and independent expert factors. Development worlds 32700–32701; fresh 32710–32712; three seeds each. Held-out coefficient values are inferred from observed factors, not stored.

## D

Pending development/fresh results.

## C

Generic free factorization may explain the held-out generalization; independent expert factors may require private coordinates.

## U

Synthetic linear matrix functions only; no learned routing or Transformer quality.

Development: on 60 held-out combinations, aligned Mirror and generic free factorization both reconstruct at NRMSE ~3.0e-4. Mirror uses 1,552 B vs generic 1,745 B (11% fewer, below the frozen 20% gate); flat Tucker coefficients fail held-out combinations (NRMSE 1). For independent expert factors, generic free factorization remains near exact while Mirror NRMSE is ~1.05. Fresh tests the frozen rule.
