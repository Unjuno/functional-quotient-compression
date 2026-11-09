# MA-713 — DELLA magnitude-aware Mirror delta basis

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/QUALITY  
Base commit: `d47e0cc7`

## Hypothesis

H: Magnitude-aware sparsification followed by shared-basis coding preserves useful task behavior at lower complete bytes than sparse deltas alone, beyond random DARE and native PCA controls.

## Mirror insertion

> **Mirror insertion:** task-specific magnitude-retained deltas are represented by coefficients over a shared basis and decoded on demand.

PA176 motivates magnitude-aware retention; PA27 requires sign-conflict controls. This screen uses top-magnitude retention as a transparent DELLA-inspired sparsifier, not the full DELLA procedure. Full details and gates are in [PROTOCOL.json](PROTOCOL.json).

## Results

Pending frozen development.

## H/T/D/C/U

- **H:** Magnitude-aware selection may leave deltas more compressible and less interference-prone than random dropping.
- **T:** D=2048, 16 synthetic task deltas, 90% retention drop, PCA ranks1/2/4/8, random DARE and TIES controls.
- **D:** Pending.
- **C:** Native PCA or magnitude pruning alone may explain all benefit.
- **U:** Actual fine-tuned model deltas, natural task quality, and GPU merge runtime.
