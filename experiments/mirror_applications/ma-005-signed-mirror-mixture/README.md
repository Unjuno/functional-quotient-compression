# MA-005 — signed Mirror expert mixture

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Branch: `research/ma-005-signed-mirror-mixture-20261007`  
Base commit: `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`

## H — Hypothesis

A signed mixture of four Givens-coordinate views of one shared expert matrix can reproduce an aligned four-expert mixture teacher with fewer actual serialized bytes than independent signed full experts. Independently parameterized expert mixtures should expose a private-parameter boundary.

## Physical-to-logical claim

- Physical object: one 16D-to-12D matrix.
- View coordinate: eight disjoint Givens angles per logical expert.
- Logical multiplicity: four outputs composed per example with signed Walsh coefficients derived from two input signs.
- Potential storage saving: one shared matrix plus angle codes instead of four matrices.
- Potential failure: arbitrary independent matrices cannot generally be encoded by rotations of one shared matrix; runtime has to evaluate all four views.

## Prior art delta

PA01 establishes expert tying as the closest hard-sharing baseline. PA02/PA03 establish that path-constrained and low-rank routing are meaningful MoE controls. MA-003 tested top-1 selection, while MA-005 isolates signed joint composition: all methods receive the same deterministic signed coefficients, and there is no learned router.

## T — Protocol

A synthetic 16D-to-12D linear task, four signed components, aligned Givens teacher and independent-matrix teacher. Development world 50000 selects a common LR from {0.003, 0.01}; fresh worlds 50001–50003 are reserved. Each method trains for 1,200 AdamW updates, batch 64, on matched minibatches. Controls are full independent experts, hard tying, scalar amplitudes, rank-1 per-expert residuals, and Mirror views. See `PROTOCOL.json` for gates and storage/compute accounting.

## Decision

FACT: pending development and fresh verification.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic signed mixtures only; not a learned-router, natural-language, or capacity claim.
