# MA-003 — Mirror top-k expert

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `3826b43b4f474e8739392322d399e50c1ac11e6b`

## Hypothesis

A top-1 router over four logical expert views of one shared matrix can recover a four-expert teacher whose experts are shared-base Givens views, at lower actual model bytes than standard top-1 MoE. Independent expert functions should require private residuals or full weights.

## Prior art delta

PA01 motivates hard expert tying, PA02 motivates path/router-sharing and low-rank router controls, and PA03 motivates low-rank routing. This screen fixes the router task and compares standard dense routing against a rank-2 factorized router, while varying expert parameterization: full, hard-tied, scalar-gated, rank-1 residual, and Mirror view.

## Task

Four role regions are determined by the signs of the first two input coordinates. Each logical expert maps a 16D input to 12D output. In the aligned teacher, the four experts are one shared matrix under Givens views. In the independent teacher, each expert is unrelated. Students route top-1 and are trained with a fixed auxiliary role cross-entropy plus regression loss. No combination count is used as a capacity metric.

## Development amendment

Initial dev_v1 (router auxiliary CE weight 0.2) achieved Mirror routed MSE 0.0184 vs full-MoE 0.0175, but router accuracy was 98.3%, below the registered 99% gate. Since quadrant role labels are deterministically linearly separable, the pre-fresh protocol was amended to raise router CE weight to 1.0. Dev_v1 rows remain preserved and tagged; only matched-method dev_v2 selects settings and may open fresh worlds. No data, methods, LR candidates, or update budget changed.

## Results

Pending final v2 development and fresh evaluation.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: pending.
