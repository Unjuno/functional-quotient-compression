# MA-003 — Mirror top-k expert

Status: **PROMISING** (aligned synthetic mechanism; strict quality gate passed in 2/3 fresh worlds).
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-003-mirror-topk-expert-20261007`.
Base commit: `3826b43b4f474e8739392322d399e50c1ac11e6b`.

## H — Hypothesis

A top-1 router over four logical views of one shared matrix can recover a four-expert teacher generated from that shared matrix plus Givens coordinates at lower serialized payload bytes than untied MoE. Independent experts should require private parameters.

## T — Test

A synthetic 16D-to-12D regression MoE with four roles selected by signs of the first two input coordinates. Trained six methods for 1,200 AdamW updates at selected LR 0.01 over fresh worlds 30001–30003: standard full MoE, rank-2 factorized-router full MoE, hard tying, scalar gate, rank-1 per-role residual, and Mirror views. Tested both a shared-base Givens teacher and independent role matrices. Development amendments and all v1/v2/v3 rows are preserved in `RESULTS_CORE.csv`; v3 protocol and sources were frozen before fresh access.

## D — Decision: PROMISING

### Fact

- On aligned worlds, Mirror routed MSE was 0.02683 / 0.01195 / 0.03107; full MoE was 0.02345 / 0.01347 / 0.03197. The <=1.10x gate passed 2/3 worlds (ratios 1.145x / 0.887x / 0.972x).
- Router accuracy was within 1 percentage point of full MoE in all 3 worlds.
- Serialized inference payload was 3,920B vs 6,102B, 35.8% fewer bytes.
- Mirror beat hard tying, scalar gate, and rank-1 residual quality in all aligned worlds. It did not pass the all-world quality gate, so overall result is PROMISING.
- On independent experts, Mirror MSE was 3.42–3.63 vs full MoE 0.19–0.29. Shared methods did not recover arbitrary role matrices.
- Mirror's active-compute proxy was 73.7M vs 191.7M for full MoE. Median training wall time was 1.23s vs 1.21s; median CPU inference throughput was 0.85M vs 1.17M examples/s.
- Fresh replay covered 36 rows: max MSE delta 4.8e-10, router-accuracy delta 5.0e-9, R² delta 4.9e-9, and payload-byte delta 0. Tests: 4 passed.

### Interpretation

The aligned teacher supports four useful logical expert roles from one shared physical expert with lower payload. The 3/3 formal quality requirement was missed once, and the simple shared controls were not strong matches on this specifically Givens-aligned teacher. The independent teacher demonstrates a private-parameter boundary for arbitrary role functions. Lower compute proxy did not translate to faster measured CPU execution in this small eager implementation.

### Strongest counter-hypothesis

The teacher is deliberately generated from the Mirror coordinate family, and the screen uses fixed-budget linear regression rather than near-converged nonlinear language modeling. The result may not transfer beyond this aligned mechanism.

### Unconfirmed

Near-convergence or fixed-byte capacity, natural-language quality, nonlinear experts, top-k greater than one, optimized GPU kernels, and equal-payload learned LoRA/FiLM controls remain untested. Do not interpret logical combinations as independent capacity.

## Prior art and amendments

PA01 motivates hard expert tying, PA02 path/router-sharing controls, and PA03 low-rank routing. Before fresh access, development amendments raised auxiliary router CE from 0.2 to 1.0, removed router weight decay, and replaced an absolute route-accuracy gate with a relative gate because every method shared a ~98% learned-router ceiling. The frozen protocol records the amendments and original rows.
