# Conditional modulation family pause — 2026-10-08

## Scope

Pause MA-401, MA-403, MA-405 and related untested P0 siblings MA-407/408/411/413 pending an inference-kernel redesign. Continue work in another family. This is a family pause, not a general conclusion about Mirror views.

## Facts

| MA | Mirror mechanism | Development outcome relevant to the repeat failure |
|---|---|---|
| 401 | Givens transform after shared features | Throughput was 0.113–0.149x FiLM across two seeds. Also missed the strict independent-byte threshold. |
| 403 | Token-position-generated Givens transform | Throughput was 0.049–0.255x token FiLM across two seeds. Total bytes also missed the frozen ratios. |
| 405 | Givens transform before shared FFN projection | Throughput was 0.633x StyleGAN2 in one seed and 0.996x in the other. It missed the two-seed runtime gate; actual payload limits passed. |

Each mechanism reached very low development NRMSE on its aligned rotation target. In these CPU runs, Mirror forward paths evaluated per-context trigonometric rotations in eager PyTorch; FiLM/StyleGAN controls used simpler vector scale operations. Development checks and serialized payload replay passed for all three experiments. All fresh seeds remained sealed.

Reports: [MA-401](ma-401-film-mirror-feature-conditioning/README.md), [MA-403](ma-403-token-mirror-modulation-generator/README.md), [MA-405](ma-405-ffn-weight-modulation-mirror/README.md).

## Interpretation and stop-rule application

The repeated limiting factor is CPU inference work for applying the structured rotation coordinate, rather than inability to represent the aligned functions. MA-401 and MA-403 have clear throughput deficits in both seeds; MA-405 shows the same direction in one seed but near parity in the other. Together the consecutive family screens identify a runtime implementation bottleneck that should be redesigned before spending more experiments in this family.

This diagnosis is scoped to the eager CPU implementation and these small widths. It does not establish that optimized/fused Givens application is slow, nor that the Mirror function family is unhelpful. Total-byte ratios also depend on how much shared state is amortized.

## Reopen criteria

A new preregistered family experiment should compare an optimized/fused rotation path and any cached sin/cos reconstruction state against native FiLM/StyleGAN2. Count cached tables as paid or deterministically reconstructed state, measure load plus inference separately, and use stable repeated timing. Do not open the sealed seeds from MA-401/403/405 as new tuning data.

## Queue action

The MA-407/408/411/413 conditional-modulation siblings remain UNTESTED and paused. Resume the global P0 queue in the next untested family after a live branch refresh.
