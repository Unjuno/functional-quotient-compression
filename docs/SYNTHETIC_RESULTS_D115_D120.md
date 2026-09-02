# Synthetic Representation / Decision-Geometry Results

This document summarizes the late D115–D120 synthetic line. These are **not model-compression bits** and **not real-Transformer evidence**.

## D115 — Global representation ablation

The tested synthetic V3 predictor previously used an `E` coordinate in its routing representation. Dropping that coordinate globally produced a much larger improvement than the preceding controller refinements.

On Confirmation1 (`n=12,000`):

- `DROP_E_OFF` vs previous update policy: mean advantage `46,055.58 work/stream`, 95% CI `[45,704.87, 46,406.29]`;
- `DROP_E_OFF` vs previous off policy: mean advantage `47,380.01`, 95% CI `[47,022.17, 47,737.85]`.

The adaptive update became harmful after the representation change.

**Interpretation:** representation mismatch dominated controller complexity in the tested environment.

## D116 — Exact decision-null common mode in the tested model

D116 showed that a common `E` contribution across actions can improve absolute cost prediction while leaving pairwise action-score differences unchanged.

Confirmation1 (`n=10,000`):

- COMMON vs DROP pairwise decision advantage: exactly `0`, all 10,000 ties;
- calibration MSE: COMMON `792.45` vs DROP `1608.76`;
- DROP vs ordinary exact-E routing: mean advantage `47,364.13 work/stream`, 95% CI `[46,970.07, 47,758.19]`.

Replication (`n=8,000`) again produced exact decision ties between COMMON and DROP.

**Interpretation:** predictive fidelity and decision-relevant geometry are different objects. A shared nuisance component may be retained for absolute prediction yet quotiented out of the decision representation.

## D117 — Tiny spectral energy, material decision value

After separating common-mode information, the centered 3-feature decision-contrast matrix was almost rank-2 in coefficient energy. The smallest singular mode contributed only about **0.019%** of centered coefficient Frobenius energy.

However:

- exact hard rank-2 truncation was about `790–815 work/stream` worse than the full-rank base on confirmations;
- a preselected soft shrink `q=0.85` improved the base by `69.20 work/stream` on Confirmation1;
- after that result, `q=0.75` was frozen as a new hypothesis and improved the base by `74.41 work/stream` on an independent 8k confirmation, 95% CI `[64.67, 84.15]`;
- a final independent 6k replication improved the base by `67.98 work/stream`, 95% CI `[56.68, 79.28]`, with all ten 600-stream blocks positive.

**Interpretation:** low spectral energy is not a safe deletion criterion. Weak modes require decision-aware validation; soft shrink can outperform hard truncation.

## D118 — Automatic spectral shrink

An automatic per-model rule based only on the already-paid calibration fit was selected:

`q = clip(1 - 16*s3/(s1+s2+s3), 0.5, 1)`.

Confirmation1 (`n=8,000`):

- AUTO vs base: `+67.89 work/stream`, 95% CI `[59.41, 76.36]`;
- AUTO vs fixed `q=0.75`: `-4.91`, 95% CI `[-8.80, -1.02]`.

Replication (`n=6,000`):

- AUTO vs base: `+73.84`, 95% CI `[63.64, 84.05]`;
- AUTO vs fixed `q=0.75`: `-2.91`, 95% CI `[-7.43, 1.60]` (unresolved).

**Interpretation:** automatic shrink captures most of the benefit but does not establish superiority over the best tested fixed rule.

## D119 — Jackknife uncertainty

A live-feasible jackknife-amplitude rule improved the unshrunk base but did not recover the best shrink.

- Confirmation1 (`n=5,000`): D119 vs base `+64.59`, 95% CI `[50.51, 78.67]`; vs fixed `q=0.75`: `-15.38`, CI `[-21.70, -9.06]`.
- Replication (`n=4,000`): D119 vs base `+72.15`, CI `[56.07, 88.23]`; vs fixed: `-13.08`, CI `[-20.42, -5.74]`.

**Interpretation:** same-regime estimator uncertainty is not sufficient to explain the cross-phase regularization benefit.

## D120 — Robust support-shift proxy

A support-robust rule used worst-case smallest-mode pairwise distortion over the known clipped feature support, normalized by calibration top-2 margin.

Confirmation1 (`n=4,500`):

- D120 vs base: `+73.16`, 95% CI `[60.50, 85.81]`;
- D120 vs D118: `-3.85`, CI `[-6.97, -0.73]`;
- D120 vs fixed `q=0.75`: `-4.59`, CI `[-8.90, -0.29]`.

Replication (`n=3,500`):

- D120 vs base: `+73.58`, CI `[59.32, 87.83]`;
- D120 vs D118: `-4.79`, CI `[-8.37, -1.21]`;
- D120 vs fixed: `-9.89`, CI `[-14.71, -5.07]`.

**Interpretation:** support-aware shift geometry is more mechanism-targeted than generic variance, but it still does not recover the best tested fixed shrink. The next synthetic question would be anisotropic/phase-dependent shift geometry; this is lower priority than the real Transformer audit.

## Durable conclusions from D115–D120

1. Simplifying the representation can dominate adding controller complexity.
2. Common nuisance components can be decision-null even when they improve absolute prediction.
3. Energy/rank alone is not a reliable task-importance criterion.
4. The useful object is the **decision/task quotient**, not the raw coefficient matrix.
5. Automatic shrink is possible, but the tested rules do not establish a universal optimal shrink.
6. None of these work/stream gains update codec bit bounds or real-model 64× feasibility.
