# MA-250 — MAP/Hadamard binding as Mirror expert address

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `d657dbe8539e5af756cb89533fd9651e968ef4e3`

## Hypothesis

A shared expert matrix plus a small per-role Givens Mirror coordinate can recover four logical functions from one physical expert when roles share a transformed basis. MAP, Hadamard linear binding, and HRR are required address controls; independent role matrices should require private residuals or untied experts.

## Prior art delta

PA11 reports MAP/sign-permute and Hadamard linear binding as important alternatives to FFT-HRR; PA12 establishes circular-convolution binding. This test uses each as a role-address transform around one shared expert matrix, and includes a learned Givens Mirror view plus hard tying, scalar gate, rank-1 residual, and full untied controls.

## Task

Four logical roles map 16D inputs to 12D outputs. In the aligned teacher mode, each role's target matrix is a Givens-rotated view of one shared base matrix. In the independent mode, each role has an unrelated target matrix. The synthetic regression task isolates address transforms from routing and language-model effects. Fixed MAP/Hadamard/HRR codes are deterministic from a charged seed; their reconstructed address state is included in the serialization config.

## Development screen

World 25000 selected LR 0.01 by pooled MSE across eight methods and both modes (2.7091 vs 2.7123 at LR 0.003). In aligned mode, Mirror reached MSE 1.24e-9 with 3,406B actual payload; untied reached 3.18e-9 with 5,522B. MAP/Hadamard/HRR MSEs were 4.08–4.46. In independent mode, Mirror MSE was 3.57; rank-1 residual 4.15; untied 2.75e-9. Fresh worlds 25001–25003 remain unopened. The payload gate was amended pre-fresh from 60% to 65% of untied (at least 35% savings) after the exact dev serialization measured 61.7%.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: pending.
