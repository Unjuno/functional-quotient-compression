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

## Results

Pending development and fresh evaluations.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: pending.
