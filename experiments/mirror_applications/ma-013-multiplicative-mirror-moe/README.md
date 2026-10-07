# MA-013 — multiplicative Mirror-MoE

Status: **FAIL** for Mirror-specific benefit; aligned quality/storage passed 3/3.
Branch: `research/ma-013-multiplicative-mirror-moe-20261007`.
Base commit: `62a801584be2cf504e5142b9e7689f6ee44b5ea5`.

## H — Hypothesis

Two low-description hidden scale views multiplied channelwise can create four useful routed experts from one shared nonlinear FFN at lower actual payload than independent experts.

## T — Test

Four balanced oracle-routed 16→32→16 GELU experts. The aligned teacher uses a shared FFN and factorized scale `(1 + alpha_i*u) * (1 + beta_j*v)`. Controls are full independent experts, hard tie, role-FiLM, and rank-2 output residual. Development world 130000 selects learning rate; fresh remains sealed until the preregistered access gate passes.

## Development result

- Selected learning rate was 0.01 on development world 130000.
- Aligned multiplicative Mirror MSE was 1.29e-5 vs 4.67e-4 independent (0.028×); actual payload was 7,001B vs 18,214B (0.384×). The preregistered pre-fresh access gate passed.
- Role-FiLM had MSE 7.93e-5 at 6,619B; multiplicative Mirror used 382B more payload and had 6.1× lower MSE. Rank-2 residual MSE was 4.23e-4 at 7,832B. Hard tie used 5,919B but had 67× Mirror's MSE.
- On independent functions, multiplicative Mirror MSE was 7.66e-3 vs 4.23e-4 for full independent weights; unrelated roles remain a private-function boundary.
- Fresh worlds 130001–130003 used frozen LR 0.01. Deterministic fields replayed exactly in all 30 fresh rows.

## D — Decision

Fresh aligned quality/storage passed in 3/3 worlds, with multiplicative Mirror MSE 0.221–0.274x independent and actual payload 7,001B vs 18,214B (0.384x). The full success gate failed: Mirror uses 18.3% more bytes than hard tying, above the 10% allowance. FiLM is 4.5% lower MSE with fewer bytes in one world and 34–48x lower MSE with fewer bytes in the other two, so the observed quality/storage frontier is dominated by the simpler control. This is a FAIL for the candidate and does not establish Mirror-specific value. Independent teacher functions also required private weights.

## C — Strongest counter-hypothesis

Role-FiLM is a simpler direct hidden gate. It matched or beat the Mirror candidate while storing fewer bytes in two worlds, showing that factorized multiplication may not be needed for the observed useful quality/storage effect.

## U — Unconfirmed

Learned routing, larger experts, near-convergence capacity, optimized inference, and language quality remain untested.

## Fact / interpretation / hypothesis

- **Fact:** fresh quality/storage versus independent passed 3/3; FiLM had lower MSE and bytes in all three worlds. Median Mirror CPU throughput was 1.62M examples/s vs 2.08M for FiLM; median training wall was 1.26s vs 0.94s. Independent functions had much lower full-model MSE than Mirror.
- **Interpretation:** the factorized scale model is a compact aligned representation, but its gains are not Mirror-specific against role-FiLM.
- **Hypothesis:** this multiplication could still be useful if it improves a broader task where role-FiLM does not match quality.
