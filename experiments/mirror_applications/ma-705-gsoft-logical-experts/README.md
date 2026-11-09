# MA-705 — GSOFT logical experts from one shared FFN

Status: FAIL  
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

## Results and decision

**FACT:** Across three audit seeds, GSOFT views reproduced independent expert outputs below 1e-7. At one/two/four stages, payload was 12.69%/16.85%/25.17% of independent and MAC proxy was 1.0625x/1.125x/1.25x. The two-stage GSOFT NumPy wall time was 5.8 ms per bank versus .41 ms dense; stage 1/4 were 2.2/10.4 ms versus .35/.46 ms. Same-budget IA3 output NRMSE was .93–.97; rank-one residual also missed. Native GSOFT matched function and payload exactly.

**INTERPRETATION:** GSOFT replaces expert duplication with a compact structured bank on this generated family, while transform runtime is much slower in this unfused NumPy implementation. IA3/rank-one do not reproduce the orthogonal teacher. However, the entire result is native GSOFT, so it adds no Mirror-specific evidence.

**HYPOTHESIS:** A shared FFN can host many GSOFT logical experts at low storage cost, but usefulness depends on native structured-transform execution.

**BOUNDARY:** Synthetic routed FFN generated from GSOFT maps; no trained expert, language utility, or GPU claim.

## H/T/D/C/U

- **H:** GSOFT output Views can replace physical expert copies at lower bytes while retaining routed outputs.
- **T:** D=64, 16 experts, 128 examples; 1/2/4 group-shuffle stages; three audit seeds; full experts, hard tie, IA3, rank-one, native GSOFT controls.
- **D:** FAIL for Mirror-specific attribution; GSOFT compactly shares the expert but exactly aliases native GSOFT and runs slower in NumPy.
- **C:** This is GSOFT structured expert tying, with no additional Mirror parameterization.
- **U:** Training utility, realistic router learning, and fused/GPU runtime remain untested.
