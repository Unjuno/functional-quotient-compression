# MA-002 — nonlinear top-2 Mirror experts

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-002-mirror-top2-expert-20261007`  
Base commit: `4236bb6` (verified MA-001 parent)

## H — hypothesis

For a learned top-2 MoE whose nonlinear expert functions are related by per-role Givens coordinates, one shared FFN with charged role-specific views can recover the teacher's probability-weighted two-expert output at lower serialized bytes than untied top-2 MoE. The view model must beat hard tying and byte-near FiLM/rank-2 residual controls. Independent role FFNs test when private weights become necessary.

## Prior-art delta

- **MA-003** tested hard top-1 routing over linear Givens experts; **MA-001** tested nonlinear hard top-1 experts. This experiment changes the functional composition: two router-selected roles contribute with their normalized router weights for each example.
- **MA-005** evaluated all four linear views under a deterministic signed Walsh mixture. Here routing is learned, sparse top-2, input-conditioned, and nonlinear; it is not an all-expert signed mixture.
- **PA01** requires ordinary tying; **PA02/PA03** require a factorized router control. This single-layer experiment does not test path constraints.

## Physical-to-logical claim

- Physical object: one 16→32→16 GELU FFN.
- Address: four per-role four-angle Givens codes applied before and inverted after the shared FFN.
- Logical multiplicity: four top-2 router-addressable roles; two weighted role outputs are active per input.
- Controls: dense untied top-2 FFNs, rank-2 factorized-router untied FFNs, hard tying, tied+FiLM, tied+rank-2 output residual, and tied+Mirror.

## T — frozen mechanism screen

The protocol is in `PROTOCOL.json`. Development world 20000 selects one LR for all methods and teacher modes; fresh worlds 20001–20003 run only if the preregistered development gate passes. The teacher router is a frozen linear score model whose full softmax is distilled by the learned student router. Top-2 roles and normalized selected probabilities define the teacher/student mixture. Aligned teacher experts are Givens conjugates of one frozen nonlinear FFN; independent mode uses four separately initialized FFNs.

## Storage and compute

Storage is actual serialized CPU state dict plus deterministic JSON metadata, charging router, expert/view/control state and reconstruction metadata. Report two-active-expert MAC proxy, router/coordinate operations, updates/examples, wall time, and CPU inference throughput.

## Results

Pending development screen. Fresh data remains sealed until the development gate is evaluated and fresh settings are frozen.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic one-layer nonlinear MoE; no natural-language or independent-capacity claim.
