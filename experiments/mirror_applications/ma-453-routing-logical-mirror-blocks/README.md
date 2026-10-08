# MA-453 — Dynamic Routing Network with logical Mirror blocks

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/RUNTIME  
Branch: `research/ma-453-routing-logical-mirror-blocks-20261008`  
Base commit: `c6bc140`  
Prior art: PA81 (Routing Networks)

## Hypothesis

H: A router can select one of two shared physical blocks and an additional two-angle role address can make that routed block serve a second logical function at a useful quality/bytes/compute point. Native Givens and rank-2 controls determine whether the extra address is distinctive.

**Mirror insertion:** this experiment adds a context-addressed Givens coordinate after the Routing Network selects a shared linear block, so two physical blocks can express four logical block-role functions.

## Task and controls

The synthetic input provides a one-hot module cue and one-hot role cue, and the teacher applies a role rotation to the selected module function. Compare a single shared block, two routed physical blocks without a View, routed Mirror blocks, rank-2 additive residuals, full role residual vectors, native Givens conditioning, and four independent logical blocks. The router is fixed deterministic context routing; it is paid in the inference payload and routing accuracy/compute are reported.

Two development seeds (45301, 45302) use 180 outer updates and 64 held-out examples per function. Fresh seeds 45311–45313 remain sealed unless every frozen gate passes. The protocol fixes all compute, quality and actual `.npz` byte conditions.

## Results

Facts, interpretation and hypothesis will be added after frozen development and payload replay.
