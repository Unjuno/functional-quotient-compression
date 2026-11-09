# MA-702 — Householder logical experts from one shared FFN

Status: FAIL  
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

## Results and decision

**FACT:** Across three audit seeds, Householder-view experts matched independent full experts within 9.31e-8. At r=1, bytes were 71,038 vs 787,590 (9.02%) and compute 1.016x independent; at r=4, 83,326 B (10.58%) and 1.062x compute. Mean pairwise functional output distance increased from .32–.34 at r=1 to .88–.91 at r=8. Hard tying had output NRMSE .35–.76. Same-budget FiLM had NRMSE .263–.324 at r=1 while using 71,018 B, 20 B less than Householder. Native Householder exactly matched function and payload hash.

**INTERPRETATION:** Householder views create diverse routed FFN outputs from shared physical weights with large storage savings and modest operation overhead. The result is exactly the known native Householder/BOFT-style parameterization, not Mirror-specific. FiLM is smaller by 20 B at r=1 but fails output quality.

**HYPOTHESIS:** Reflection codes can substitute for independent FFN expert weights on this deliberately aligned family.

**BOUNDARY:** Synthetic routed FFN outputs only; no training, downstream language task, or GPU inference claim. The teacher functions were generated from Householder codes by construction.


## H/T/D/C/U

- **H:** A short Householder code recovers distinct logical experts from one shared FFN at lower bytes than independent experts.
- **T:** D=64, hidden=64, 16 routed views, 128 examples; r=1/2/4/8; three locked audit seeds; full experts, hard tying, same-budget FiLM, native Householder controls.
- **D:** FAIL for Mirror-specific attribution: native Householder exactly aliases it; generic shared-expert storage/compute feasibility is strong.
- **C:** The full result is ordinary Householder/BOFT parameterization around a shared FFN, an established orthogonal PEFT family.
- **U:** Trained experts, task utility, generalization to unconstrained expert functions, and GPU runtime remain untested.
