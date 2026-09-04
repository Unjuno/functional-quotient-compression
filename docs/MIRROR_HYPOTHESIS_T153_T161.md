# Mirror hypothesis diagnostics: T153–T161

This note records high-value follow-up analysis and synthetic diagnostic experiments for the Functional Quotient Compression / Vector Mirror hypothesis.

## Evidence boundary

- `T153` and `T158` are reanalyses of existing TinyStories REAL_MODEL WQ results. No new model forward replay was executed.
- `T154–T157` and `T159–T161` are SYNTHETIC mechanism/diagnostic experiments.
- These results do **not** establish a new 28M functional compression result or a universal scaling law.

## 1. Four-point scaling does not identify an asymptotic law

The existing WQ task-priced-4/local-8 holdout KL ratios are:

- hidden 64: `1.769`
- hidden 128: `1.276`
- hidden 256: `0.927`
- hidden 512: `0.909`

A three-parameter plateau fit matches these four points much better in-sample than a pure power law, but leave-one-out error is worse because four observations do not robustly identify a three-parameter asymptote. The flattening from 8M to 28M is evidence against naively extrapolating the early power-law slope, not evidence for a known limiting ratio.

## 2. The useful mirror diagnostic is private functional SNR

Across 720 hierarchical synthetic conditions, the statistic most predictive of the minimal noninferior shared-state count was functional private energy divided by estimation/noise energy.

Spearman correlation with `log2(K*)`:

- pair-private SNR: `0.9335`
- total private SNR: `0.9328`
- pair-private energy: `0.8150`
- ambient width: `-0.0646`
- effective rank: `-0.0351`

Thus width or rank alone is not a sufficient mirror diagnostic in this family.

For true pair sharing, 2160 synthetic conditions gave approximately

`shared4/local8 distortion = 0.50035 + 0.999997 * pair_private_SNR`

with `R^2 = 0.9999949`. The fitted noninferiority threshold was pair-private SNR `≈0.49965`; the simple threshold `SNR <= 0.5` classified share/no-share correctly in `99.49%` of conditions.

This relation is specific to the tested estimator family, not a Transformer theorem, but it motivates a direct real-model measurement.

## 3. The previous gamma≈1/2 boundary is conditional

Synthetic phase experiments extended to width 2048 show that the private-growth exponent boundary depends on calibration/estimation scaling.

With fixed effective calibration, the transition concentrates near `gamma≈1/2` in the tested bias-variance model. When effective calibration grows with width as `d^alpha`, the asymptotic balance shifts according to the exponent combination `2 gamma + alpha - 1`.

Therefore `gamma=1/2` must not be promoted as a universal constant of Transformers.

## 4. Actual WQ scale results are confounded by fixed-codebook occupancy

The existing REAL_MODEL WQ experiment uses 32-scalar blocks and `K=128`.

Per layer:

| hidden | blocks/layer | local blocks/codeword | pair-shared blocks/codeword | priced4/local8 KL |
|---:|---:|---:|---:|---:|
| 64 | 128 | 1 | 2 | 1.769 |
| 128 | 512 | 4 | 8 | 1.276 |
| 256 | 2048 | 16 | 32 | 0.927 |
| 512 | 8192 | 64 | 128 | 0.909 |

Thus model scale and codebook occupancy are perfectly confounded in the four-point series.

More importantly, the local-8 representation class contains the shared-4 class: paired local codebooks can always be set equal. Under the same true objective and exact global optimization, local-8 therefore cannot have higher optimum distortion than shared-4. The observed holdout superiority of shared-4 at 8M/28M necessarily includes generalization, regularization, or optimization effects.

Consequently:

**holdout shared > local is evidence that sharing is useful, but it is not by itself proof that the intrinsic functional quotient contains fewer states.**

## 5. Synthetic nested-class control confirms the regularization mechanism

Across 1920 synthetic conditions:

- the local representation dominated calibration fit in `100%` of cases;
- the shared representation nevertheless won true/holdout distortion in `70.10%` of cases.

The shared win fraction decreased as calibration became more accurate, consistent with variance reduction / regularization rather than strict representational superiority.

## 6. Cross-fitting is required for an intrinsic mirror certificate

A two-split synthetic estimator used independent functional estimates `A` and `B`:

- pair-private energy from cross-split covariance;
- estimation/noise energy from half the split-difference energy.

Across 3840 conditions, this cross-fit private-SNR diagnostic predicted whether pair sharing improved holdout distortion with `98.31%` accuracy.

A direct ablation showed:

- cross-fit accuracy: `98.78%`
- naive same-split private-SNR accuracy: `50.91%`

Same-split private energy is strongly upward biased by calibration noise and should not be used to certify intrinsic mirror geometry.

## Updated mirror hypothesis

A defensible operational form is:

> A parameter family is functionally mirror-compressible when its **cross-fitted task-relevant private energy** is small relative to the estimation/noise energy saved by pooling into shared states. Apparent holdout gains from shared states must be separated from regularization and calibration overfit before claiming an intrinsically smaller functional quotient state count.

## Next REAL_MODEL experiment

For WQ on the TinyStories 1M/3M/8M/28M series:

1. use at least two disjoint calibration passage sets;
2. produce independent pair-merge / functional estimates from each split;
3. estimate pair-private functional energy by cross-split covariance;
4. estimate noise by split disagreement;
5. test whether private SNR predicts held-out pair-sharing noninferiority;
6. repeat with matched codebook occupancy across model sizes, by changing `K` or subsampling blocks, to separate model-scale geometry from fixed-codebook-capacity effects;
7. only after this separation estimate a scaling law for intrinsic minimal mirror-state count.

This experiment has higher information value than adding an unconstrained fifth model-size point.
