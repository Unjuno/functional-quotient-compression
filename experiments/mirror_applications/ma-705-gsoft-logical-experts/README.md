# MA-705 — GSOFT logical experts from one shared FFN

Status: SCREENING  
Evidence lane: MECHANISM  
Base commit: `dba727b6`

## Hypothesis

H: GSOFT codes can create distinct routed expert outputs from one shared FFN at lower bytes and modest compute than independent experts.

## Mirror insertion

> **Mirror insertion:** adds expert-indexed group-shuffle factors after a shared FFN, with routing choosing a logical view.

- Independent baseline: full FFN per expert.
- Hard tying: same FFN output for every route.
- Simple controls: IA3 channel gains and rank-one residuals.
- Exact structural control: native GSOFT coordinates.

## Prior-art delta

PA01 establishes expert tying and requires that Views add behavior beyond shared experts. PA172 establishes GSOFT group-shuffle orthogonal parameterization. MA-705 tests whether that transform can replace routed expert duplication and compares simple/native controls.

## Gates

See [PROTOCOL.json](PROTOCOL.json) for frozen outputs, byte, compute and seed gates. Expert counts alone are not capacity evidence.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic routed FFN only.
