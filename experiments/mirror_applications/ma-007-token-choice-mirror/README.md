# MA-007 — token-choice Mirror routing crossover

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-007-token-choice-mirror-20261007`  
Base commit: `a556e63` (verified MA-006 parent)

## H — hypothesis

With one top-1 token-choice assignment per input, a shared nonlinear FFN plus per-role Givens views can recover the full model's quality while retaining every token and using fewer serialized bytes. On matched role data, token-choice Mirror should reduce the 18–19% no-route loss observed for expert-choice Mirror in MA-006. It must still beat tied/FiLM/residual controls for a Mirror-specific result.

## Prior evidence delta

- **MA-001** already tested nonlinear top-1 Givens experts on sign-quadrant inputs; it passed aligned quality/storage but was slower than tying. This experiment uses exactly balanced overlapping role clouds and directly contrasts routing policies.
- **MA-006** found expert-choice coverage around 81%, with token-choice control at 100% coverage and 2.5–3.4x lower MSE. This experiment tests whether a Mirror view bank under token-choice removes that failure mode while retaining compression.
- PA01 hard tying and PA03 low-rank routing are included as context/control.

## T — protocol

See `PROTOCOL.json`. Eight methods include dense token-choice and expert-choice controls, rank-2 token-choice router, tied/FiLM/residual token-choice, token-choice Mirror, and expert-choice Mirror. Each 64-example batch has 16 examples per role sampled from overlapping Gaussian clouds. The aligned teacher uses one FFN under Givens views; the independent teacher uses four unrelated FFNs.

## Storage and compute

Actual serialized CPU state dict plus compact config is authoritative. Report one active expert FFN per token for token-choice; expert-choice dispatch slots, overlap/coverage, coordinate FLOPs, updates/examples, wall time and CPU throughput are separate.

## Results

Pending implementation and development. Fresh data remains sealed until the development gate passes and settings are frozen.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic balanced role clouds; no natural-language or capacity claim.
