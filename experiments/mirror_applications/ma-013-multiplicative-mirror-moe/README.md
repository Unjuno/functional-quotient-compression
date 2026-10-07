# MA-013 — multiplicative Mirror-MoE

Status: SCREENING; development gate passed and fresh settings frozen.  
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
- Fresh worlds 130001–130003 are frozen at LR 0.01. Do not tune from them.

## D — Decision

Fresh results pending; development is not capacity evidence.

## C — Strongest counter-hypothesis

The teacher is generated from this exact factorized multiplicative form. Ordinary FiLM or a small residual may match its behavior with simpler parameters.

## U — Unconfirmed

Fresh replication, learned routing, larger experts, near-convergence capacity, optimized inference, and language quality are untested.

## Fact / interpretation / hypothesis

- **Fact:** the development aligned quality/storage gate passed; multiplicative Mirror also beat role-FiLM MSE at 5.8% more payload.
- **Interpretation:** factorized hidden modulation captured the deliberately aligned teacher compactly, but independent experts were much better on unrelated functions.
- **Hypothesis:** product coordinates could capture useful multiplicative interactions among related learned roles.
