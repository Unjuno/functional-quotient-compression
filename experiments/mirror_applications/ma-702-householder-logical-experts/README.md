# MA-702 — Householder logical experts from one shared FFN

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `45a2bf56`

## Hypothesis

H: A small Householder code per routed expert can create useful distinct FFN outputs from one physical shared FFN, with lower serialized state and modest compute overhead.

## Mirror insertion

> **Mirror insertion:** applies per-expert output-space reflection codes after one shared FFN, while expert routing selects the logical View.

- Native baseline: independent full FFN experts.
- Hard tying control: one shared FFN with identical outputs for every expert.
- Simple control: same-budget per-channel FiLM gains.
- Exact structural control: native Householder/BOFT-style transform.

## Prior-art delta

PA171 describes Householder parameterization; PA21 describes BOFT. MA-701 showed task-view storage/compute feasibility but exact native Householder alias. MA-702 tests whether the same transform creates routed logical expert outputs and whether diagonal FiLM suffices.

## Gates

See frozen [PROTOCOL.json](PROTOCOL.json). Expert combinations are evaluated by actual outputs, bytes, and compute; their count is not capacity.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic routed FFN only.
