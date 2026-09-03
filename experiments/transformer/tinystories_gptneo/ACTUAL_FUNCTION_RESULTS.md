# TinyStories GPT-Neo actual-weight / actual-function experiments

Scope: user-provided `1M.zip`, `3M.zip`, `8M.zip`, `28M.zip`; local container only. Manual PyTorch GPT-Neo forward and dependency-free GPT-2 byte-level BPE were used. Probe sequences are shorter than the 256-token local-attention window, so local/global masking reduces to ordinary causal attention on these probes.

## Model inventory / scaling

The four checkpoints form an 8-layer GPT-Neo family with hidden sizes 64 / 128 / 256 / 512. Trainable-float parameter accounting excludes the serialized 2048x2048 boolean causal-mask buffers.

Embedding share falls strongly with scale:
- 1M: 85.9%
- 3M: 77.7%
- 8M: 65.3%
- 28M: 49.5%

Actual embedding top-35 explained variance also falls with scale:
- 1M: 0.799
- 3M: 0.533
- 8M: 0.347
- 28M: 0.216

A synthetic spectrum-only codec-family gate did not transfer: at ~0.25 bps, actual embedding PQ beat the tested same-budget low-rank codec on all four sizes. Actual rate/factor overhead matters in addition to spectrum concentration.

## Real-weight layer sharing at K128

On 8M actual weights, one shared K128 / 32-scalar codebook across all eight layers was close to per-layer local codebooks for corresponding roles while reducing codebook metadata 8x. Representative reconstruction-NMSE penalty of shared vs local:
- WQ: +0.9%
- WK: roughly +3–4.5%
- WV: roughly +3–4%
- WO: +1.5%
- MLP FC/PROJ: shared was approximately equal or slightly better in the tested sample-pooled setup.

A fairer 28M WQ rerun found shared about 2.24% worse than local, so sample-pooling must not be confused with intrinsic sharing gain.

## Actual function: same-rate component hierarchy

For 28M layer 7, isolated K128 / 32-scalar block-VQ (`7/32 = 0.21875` index bps) on six held-out TinyStories-style passages produced mean KL approximately:
- WQ: 0.00372
- WK: 0.00356
- WV: 0.5625
- WO: 0.3401
- MLP FC: 1.0420
- MLP PROJ: 0.7661

Thus Q/K are in a qualitatively different functional regime from V/O and MLP. Uniform sub-bit treatment by module class is not justified.

Increasing K from 32 through 512 steadily improved weight NMSE for V/O/MLP, but task KL was weakly monotone or non-monotone and remained large. Representation mismatch, not merely insufficient codebook cardinality, is the main problem in this rate range.

## Actual function: Q/K critical-layer structure

For 28M, all eight layers' Q+K matrices were first block-quantized to K=2 (`1/32 = 0.03125` index bps). Mean KL was about 0.252, essentially the same as setting all Q/K matrices to zero. Raising every layer uniformly through K=32 barely changed this floor; K128 reduced it only to about 0.228.

Single-layer diagnostics show extreme layer dependence:
- restoring layer 0 Q/K exactly: KL 0.252 -> 0.065
- single exact restores for other layers each gave only roughly 0.011–0.022 improvement.

Representation family matters more than nominal rate at critical layer 0. With all other layers kept at K2:
- layer-0 block-VQ K128 (~0.21875 bps): KL ~0.234
- layer-0 low-rank rank16 (4-bit-factor count proxy ~0.25 bps, factors not quantized in this diagnostic): KL ~0.179
- layer-0 row-wise 2-bit scalar quantization: KL ~0.071
- exact layer-0 restore: KL ~0.065

Increasing layer-0 row quantization above 2 bits did not materially improve the result because the remaining K2 layers dominate the residual damage.

Greedy row-wise-2bit upgrades from the all-K2 base selected layers 0, 6, 7, 5 first and reduced KL approximately 0.253 -> 0.071 -> 0.052 -> 0.039 -> 0.031, at rapidly increasing rate. This is direct evidence for representation- and layer-specific bit allocation.

Q-only and K-only all-layer K2 interventions each had KL ~0.28, while the joint Q+K intervention was ~0.29, far below the additive sum. Functional distortion is strongly non-additive in this regime.

## Actual function: MLP structured allocation

28M layer-7 MLP, same six holdouts:
- uniform K128/block32 on both FC and PROJ at ~0.21875 index bps: KL ~1.108.
- activation-effect channel selection plus higher-rate survivor VQ improved as sparsity increased but saturated around KL ~0.70.

Using calibration-loss activation×gradient Fisher saliency made channel selection substantially better. Pure structured pruning (no survivor quantization):
- drop 50%: KL ~0.177
- drop 75%: ~0.343
- drop 87.5%: ~0.496

But forcing the retained channels back to ~0.21875 average bps with VQ raised KL again:
- 50% retain / K128 block16: ~0.912
- 25% retain / block8: ~0.740
- 12.5% retain / block4: ~0.600

Sparse scalar alternatives at ~0.25 average weight bps also remained poor (best tested ~0.70 KL). Therefore the present MLP codec family does not support a quality-preserving 0.25-bps layer-7 representation.

## Actual function: V/O head allocation

28M layer-7 V/O head saliency was measured with calibration-loss activation×gradient Fisher at the pre-output-projection head context.

Structured head allocation was better than uniform V/O block-VQ, but 0.25 bps remained damaging:
- keep 8/16 heads exact (diagnostic, high rate): KL ~0.108
- keep 2 heads with 2-bit scalar weights (~0.25 average): KL ~0.696
- keep 1 head with 4-bit scalar weights (~0.25 average): KL ~0.367

Thus V/O also benefits from functional concentration but does not reach low damage at the 0.25-bps budget with the tested primitive.

## Actual embedding findings (previous local batch)

Across the scale family, uniform ~0.23–0.25 bps tied embedding PQ was catastrophic (KL roughly 4–6 across sizes). On 8M, separating input-only vs LM-head-only showed both roles matter, with LM-head/output geometry the stronger bottleneck. Weight-space PQ improvement alone did not fix functional quality.

Task-signature roots plus Fisher-selected private token exceptions were much better than uniform PQ, and at fixed 0.25 bps showed a genuine shared-root/private-exception optimum. On 28M, broader calibration improved exception selection substantially, but the 0.25-bps embedding KL remained roughly ~0.8 in multi-holdout evaluation. At 1–2 bps the same family improved sharply; broad calibration reduced 28M multi-holdout 2-bps KL to ~0.039 in the earlier batch. Thus the current embedding family remains a major 64x bottleneck rather than merely a calibration issue.

## Current boundary

These are real checkpoint and real forward-pass results, but they are fixed-probe experiments, not benchmark-quality certificates and not an emitted full-model codec. The strongest current conclusion is negative/architectural:

1. Q/K contain very cheap directions, but critical layers require a different representation family than ordinary block VQ.
2. V/O and especially MLP remain much harder at ~0.22–0.25 bps.
3. Embedding remains difficult at 0.25 bps even after task-aware root/exception allocation.
4. Functional distortion is highly non-additive; per-block reconstruction NMSE and individual KL cannot be summed into a model certificate.
5. The current 64x skeleton is therefore not quality-validated; its next progress requires new V/O, MLP, and embedding representations, not just larger K or better Euclidean reconstruction.

## Source ZIP SHA-256

- `1M.zip`: `6e95174c90a9a2a9a8e5f97d934e78e2863d6b88e9450c7b9a0ff4c25e0e1b16`
- `3M.zip`: `a6a7328ab03fa56a4e13de55ae8a6b9284f8166cdcb1c37574f82d0d685760ce`
- `8M.zip`: `d1a702e539bbb44393b092bbcbe3af9d3e16f9533284b55c1506a6196834c180`
- `28M.zip`: `d17e450912244aed8ee90933d8d91b3f8fd70e6a8f341eb3ce27b54fdfdf9958`
