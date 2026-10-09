# MA-576 — SmoothQuant + residual rotation

Status: FAIL
Prior art: PA114 SmoothQuant; PA112 QuaRot

## H

After cheap diagonal conditioning, a small residual rotation may improve held-out output-row int4 reconstruction over either method alone at bounded actual payload and compute.

## T

Pinned Pythia-70M attention output and MLP-up matrices from layers 0–5, groupwise int4, SmoothQuant-style channel scaling, and 16 residual block-Hadamard/sign/permutation views. Each matrix uses 20% output rows for calibration and the complementary rows for audit. Compare identity, scale-only, rotation-only, combined global rotation, per-matrix residual selection, random residual rotation and exact native sequential control. Fresh seeds remain sealed until the development gate passes without native alias.
