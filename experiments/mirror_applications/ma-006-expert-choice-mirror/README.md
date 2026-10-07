# MA-006 — capacity-balanced expert-choice Mirror

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-006-expert-choice-mirror-20261007`  
Base commit: `eca36e2` (verified MA-004 parent)

## H — hypothesis

Under expert-choice routing with a fixed token capacity per expert, one nonlinear FFN plus per-expert Mirror views can retain the routed quality of a full expert bank while reducing serialized bytes. The effect must beat ordinary expert tying and byte-near FiLM/rank-2 residual controls; standard token-choice routing is a route-policy comparator.

## Prior-art delta

MA-001/002/004 studied token-choice top-1/top-2/dense-soft routing. This experiment fixes how many tokens each expert may select, allowing overlap and unassigned tokens. The score router chooses tokens per expert; each expert has capacity exactly batch_size/4. PA01 hard tying and PA03 low-rank routers are included; PA02 path constraints are outside this single-layer task.

## Physical-to-logical claim

One 16→32→16 GELU FFN is shared across four expert-choice slots. Each slot chooses a fixed-capacity set, then uses its expert-specific view. A token selected by multiple experts receives a normalized weighted output; an unselected token receives zero on the MoE branch. This explicit no-route behavior is charged in quality, not repaired by an oracle fallback.

## T — protocol

See `PROTOCOL.json`. Each training batch has 16 examples per role; role labels are supervised router targets and inputs have overlapping Gaussian class clouds. The aligned teacher uses one FFN under Givens views; independent mode uses four unrelated FFNs. Compare full expert-choice, dense token-choice, rank-2 router variants, hard tying, FiLM, rank-2 residual, and Mirror. Development chooses LR; fresh worlds are gated and frozen.

## Storage and compute

Count actual serialized inference payload bytes including router, full/shared experts, views/control tensors and metadata. Record capacity, expert loads, token coverage, role recall, active FFN assignments, coordinate FLOPs, wall time and CPU throughput.

## Results

Pending implementation and development.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic balanced routing; no natural-language or scale claim.
