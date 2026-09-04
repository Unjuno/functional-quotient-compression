# Mirror matched-support protocol update (T167–T176)

This note records a direction-changing **synthetic/protocol-validation** result. It is not new real-model evidence.

## Why the previous matched-occupancy plan was insufficient

The previous plan proposed fixing `K` and the number of codebook-training blocks per layer across TinyStories 1M/3M/8M/28M. T172 showed that this does **not** fully remove scale confounding.

For WQ with 32-scalar blocks, the 1M model has exactly 128 blocks/layer, whereas the 28M model has 8,192. Training all models from 128 blocks therefore means:

- 1M: the codebook sees the whole layer support;
- 28M: the codebook sees only 1/64 of the layer support, then is evaluated on a much larger population.

Synthetic fixed-geometry controls developed a false negative scale slope under this design. Therefore `fixed K + fixed training-block count` alone must not be treated as an intrinsic-mirror certificate.

## Matched-support correction

Intrinsic functional geometry and full-codec quality must be tested in separate lanes.

### Lane A — intrinsic mirror geometry diagnostic

Use a matched-size support at every scale and keep training, pair-pricing, and audit sets disjoint.

A practical WQ protocol that fits inside the 1M layer's 128 blocks is:

1. Select exactly 128 WQ blocks per layer.
2. Partition them into four folds of 32 blocks.
3. Use a small diagnostic codebook (`K8` or `K16`; K16 is preferable when compute permits).
4. For each rotation:
   - two folds train local and pair-shared codebooks;
   - one fold prices all 28 pair merges and selects among all 105 four-pair perfect matchings;
   - one fold is a disjoint audit set.
5. Rotate fold roles four times and average the audit shared/local ratio.
6. Use multiple block-support seeds where the model has more than 128 WQ blocks.
7. Separately estimate cross-fitted private functional SNR from independent passage splits.
8. Use a bootstrap confidence interval for the slope versus `log2(width)`; do not accept a negative Spearman sign alone.

Synthetic T176 validated the fold rotation: under width-invariant geometry the mean slope moved from about `-0.0122` for one arbitrary split to `+0.0041` after four-fold rotation, and slope SD fell from about `0.0321` to `0.0186`. Under synthetic shrinking-private-geometry conditions the rotated slope remained negative in all tested seeds.

### Lane B — actual codec quality

Use all real WQ blocks and the actual target codec/rate. Measure emitted bytes and full-model task metrics. A Lane-B shared-codebook win is operationally valuable but, by itself, must not be interpreted as proof that the intrinsic number of functional states is smaller, because regularization, finite-sample estimation, and optimization effects remain allowed.

## Power / replication guidance

Synthetic T167–T170 support the following starting policy:

- at least 8 independent block-support seeds for a modest scale trend when possible;
- at least 4 independent cross-fit passage split-pairs for the private-SNR diagnostic;
- pair/matching identity is not the main success metric near ties — use true/holdout regret and top-k stability;
- add pair-price splits adaptively when the best-vs-second matching margin is small rather than paying a fixed large split count everywhere.

## Claim boundary

The stronger Functional Mirror claim should require a conjunction:

1. matched-support geometry trend improves with scale;
2. cross-fitted private functional SNR decreases with scale;
3. independent full-model codec evaluation also benefits from sharing.

Only item 3 is currently supported by the existing TinyStories WQ real-model scaling result. Items 1 and 2 remain to be measured on the real checkpoints.
