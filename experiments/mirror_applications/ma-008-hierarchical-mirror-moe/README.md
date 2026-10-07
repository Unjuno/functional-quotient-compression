# MA-008 — hierarchical Mirror-MoE

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-008-hierarchical-mirror-moe-20261007`  
Base commit: `993c4ce` (verified MA-007 parent)

## H — hypothesis

For four related nonlinear expert roles arranged as two groups of two, a hierarchical group→expert router plus one shared FFN and role-specific Givens views can recover untied hierarchical MoE quality with fewer serialized bytes. It must beat hard tying and byte-near FiLM/rank-2 residual controls; flat token-choice and factorized routing test whether the hierarchy contributes beyond router structure.

## Prior-art delta

MA-001/002/004/006/007 measured flat token-choice, sparse top-k, dense soft and expert-choice variants. This experiment adds a two-level route tree over balanced roles. PA02 studies shared/path-constrained routing across blocks; PA03 factorized routers are included as a strong parameterization control.

## T — protocol

See `PROTOCOL.json`. Every 64-example batch has 16 examples per role. Roles are grouped by the first input coordinate; a child router distinguishes the two roles within the selected group. Compare full flat, full hierarchical, rank-2 flat, tied hierarchical, tied+FiLM, tied+rank-2 residual, and tied+Mirror. The aligned teacher uses one FFN under Givens views; independent mode uses four unrelated FFNs.

## Storage and compute

Use actual serialized payload bytes including both hierarchy levels of router, experts/views/control state and metadata. Report router MACs separately from one active FFN, route/group accuracy, examples/updates, wall-clock and CPU throughput.

## Results

Pending implementation and development. Fresh remains sealed until the registered gate passes.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic two-level routing; no language-model or capacity claim.
