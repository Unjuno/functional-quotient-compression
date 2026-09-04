# Mirror occupancy-control experiments T162-T166

These experiments are **synthetic controls** designed to interpret the existing TinyStories WQ real-model scaling result. They do not add a new real-model compression claim.

## Background

The existing TinyStories WQ experiment uses 32-scalar block VQ with K=128. As hidden size grows 64 -> 128 -> 256 -> 512, one WQ layer contains 128 -> 512 -> 2048 -> 8192 training blocks, so local blocks/codeword rise 1 -> 4 -> 16 -> 64. This creates a sample-sufficiency confound in the observed priced4/local8 holdout ratios 1.769 -> 1.276 -> 0.927 -> 0.909.

## T162: occupancy-only sign

With pair geometry fixed and only blocks/codeword increased 1 -> 64, the shared/local holdout distortion ratio increased for every tested private-mismatch level. Representative examples:

- private scale 0.0: 0.870 -> 0.975;
- 0.3: 0.911 -> 1.022;
- 0.5: 0.933 -> 1.073;
- 0.8: 1.016 -> 1.181.

Thus sample pooling helps the shared representation most at low occupancy. The occupancy-only effect has the **opposite sign** to the real TinyStories trend.

## T163: calibration-priced pair selection

An eight-layer synthetic control trained all 28 pair-shared codebooks, enumerated all 105 perfect matchings, selected the matching by calibration cost, and evaluated it on an independent holdout. All four tested private-scale/calibration conditions still had Spearman +1.0 between occupancy and selected shared/local ratio.

Pair-selection overfit therefore did not reverse the occupancy-only sign in this control.

## T164: randomized sign stress search

Twenty randomized settings varied K, block dimension, observation noise, private mismatch, and calibration size. Comparing occupancy 1 vs 64 after calibration-priced pair selection:

- decreasing cases: 0 / 20;
- minimum ratio change: +0.0758;
- median ratio change: +0.3042;
- maximum ratio change: +0.4694.

Within this tested VQ proxy family, occupancy/capacity effects alone did not reproduce the real decreasing scale trend.

## T165: geometry shrink is required in the proxy

Using the real occupancy schedule [1, 4, 16, 64], private mismatch was forced to scale as

`a * (width / 64)^(-beta)`.

On the tested grid, any monotone decreasing shared/local ratio first appeared around beta ~= 0.51. The best absolute fit was poor, so this beta is **not** a Transformer scaling exponent. The supported conclusion is only directional: private geometry must improve with scale strongly enough to overcome the occupancy-only trend in this proxy.

## T166: matched-sampling protocol

Three synthetic controls cleanly separated the effects:

1. full data + fixed geometry: shared/local 0.946 -> 1.070, Spearman +1;
2. matched training samples + fixed geometry: 0.946 -> 0.946, flat;
3. matched training samples + shrinking private geometry: 1.002 -> 0.893, Spearman -1.

This validates a direct real-model diagnostic: keep **K fixed and the number of codebook-training blocks per layer fixed across model sizes**. Then any remaining scale trend cannot be caused by codebook occupancy.

## Next real-model experiment

For TinyStories 1M / 3M / 8M / 28M WQ:

1. use the same K (initially K=128) for all four checkpoints;
2. sample exactly 128 training blocks per layer for every checkpoint (feasible for all four; 1M uses all its WQ blocks, larger models are subsampled);
3. train local and all pair-shared codebooks from this matched sample count;
4. choose pair matching on calibration split A;
5. estimate cross-fit private functional SNR from an independent split B;
6. evaluate the committed matching on a third holdout split;
7. repeat over multiple block subsamples / seeds.

### Falsification

- If the decreasing 1M -> 28M shared/local trend survives matched sampling and cross-fit, that is evidence for scale-dependent functional mirror geometry.
- If it disappears, the previous positive scaling result was primarily estimation/capacity/selection driven.

## Boundary

These controls strengthen the case that occupancy alone is not an adequate explanation in the tested proxy family, but they do **not** prove that the TinyStories real-model trend is intrinsic geometry. The matched-sampling real-model experiment remains required.
