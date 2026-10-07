# MA-004 — nonlinear soft Mirror expert mixture

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-004-soft-mirror-expert-mixture-20261007`  
Base commit: `7ff5430` (verified MA-002 parent)

## H — hypothesis

For a dense softmax MoE whose nonlinear expert functions are related by per-role Givens coordinates, one shared FFN with four charged views can recover the all-expert probability-weighted output at lower serialized bytes than untied MoE. It must outperform hard tying and byte-near FiLM/rank-2 residual controls to support a Mirror-specific claim. Independent role FFNs test the private-parameter boundary.

## Prior-art delta

- MA-001/MA-002 tested nonlinear top-1 and sparse top-2 routing. This experiment sends every token through all four experts and mixes all four outputs with learned softmax weights.
- MA-005 tested an all-expert linear signed mixture with deterministic Walsh coefficients. Here experts are nonlinear, coefficients are learned input-conditioned router probabilities, and all role contributions are positive and normalized.
- PA01 requires hard expert tying; PA02/PA03 motivate router sharing/factorization controls. This single-layer screen does not test path constraints.

## Physical-to-logical claim

One 16→32→16 GELU FFN plus four role-specific four-angle Givens views creates four router-addressable expert functions. Each example composes all four with softmax probabilities. Controls are full untied experts, rank-2 factorized-router full experts, hard tying, tied+FiLM, tied+rank-2 residual, and tied+Mirror.

## T — protocol

See `PROTOCOL.json`. A frozen teacher linear router provides a full softmax distribution; the learned student router is distilled on all four probabilities. The aligned teacher uses Givens-conjugated views of one nonlinear FFN; independent mode uses four unrelated FFNs. Development world 40000 selects one LR. Fresh worlds 40001–40003 are opened only after the preregistered quality/byte gate passes.

## Storage and compute

Measure actual serialized state-dict plus deterministic config bytes, charging router, all experts or shared FFN, views/controls, and metadata. Report four-active-expert MAC proxy, router/coordinate operations, updates/examples, wall time, and CPU throughput.

## Results

Pending development. Fresh data remains sealed until the development gate and configuration freeze.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic one-layer dense soft MoE; no natural-language or capacity claim.
