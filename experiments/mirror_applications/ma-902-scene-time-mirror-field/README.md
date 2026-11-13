# MA-902 — scene identity × time Mirror field

Status: **SCREENING — synthetic mechanism only**
Evidence lane: synthetic neural graphics / compositional generalization / storage / compute
Base commit: `daf15aac3f68cf78453788f1f39f989e04d874b3` (MA-900 research branch)

## H — falsifiable hypothesis

A small multiplicative scene × time coordinate over shared spatial atoms predicts held-out scene-time combinations more compactly and accurately than an additive scene+time code. Mirror-specific value requires improvement over native rank-4 scene×time tensor factorization, which uses the same mathematical factorization.

## T — frozen protocol

See `PROTOCOL.json`. Eight scenes × six timestamps, 96 spatial samples, rank four. Each scene has four observed timestamps; the other 16 cells are held out while every scene and timestamp appears in training. Development seeds 90201/90207. The initial fresh set was opened under the unregularized run and remains exploratory. Amendment 01 selects weight decay 1e-4 using development only and uses fresh seeds 90221/90223/90229, with no fresh tuning. Compare native factorization, additive shared-basis code, multiplicative Mirror code, and independent cell payload. Actual compressed NPZ bytes are authoritative.

## Boundary

This is a synthetic functional screen, not evidence for C-NGP/K-Planes or real neural rendering. The independent cell upper control stores held-out truth and is an oracle payload, not a generalizing predictor. The key attribution question is whether multiplicative Mirror beats native tensor factorization.
