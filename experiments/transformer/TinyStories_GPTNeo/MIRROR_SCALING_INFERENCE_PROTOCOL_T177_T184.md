# Mirror scaling inference protocol update (T177–T184)

This note records methodology conclusions from local synthetic experiments T177–T184. These experiments are **not new REAL_MODEL compression results**. They refine how the existing one-checkpoint-per-scale TinyStories sharing trend should be interpreted and how a future population-level scaling claim must be tested.

## 1. Joint checkpoint-conditional certificate
A matched-support shared/local audit ratio and an independent geometry diagnostic should both point in the same direction. In T177, requiring the 95% upper bound of both slopes to be below zero was substantially more conservative than either metric alone.

However, T179 showed that **private SNR is not a geometry-specific scaling metric**: it can decrease solely because estimator noise increases with width. Therefore the roles are now separated:

- use cross-fit private SNR for **share/no-share decisions at a fixed scale**;
- use cross-fitted private fraction `P / (P + S)` (private functional energy over private plus shared functional energy), with raw private energy as a secondary check, for **scale-geometry claims**.

## 2. Ordinary A/B cross-fit can be biased by shared passage nuisance
T178 introduced a nuisance component shared by the A/B estimates from one passage group. Ordinary cross-fit then overestimated private SNR because the nuisance appears in the cross covariance but cancels from `A-B`.

A four-outer-group estimator removed this bias in the tested synthetic family by estimating nuisance from between-group differences after subtracting the inner measurement-noise contribution.

Practical implication: do not treat a single calibration pool split A/B as sufficient if systematic passage-group nuisance may be shared. Use independent **outer passage groups**.

## 3. Support-seed replication is not model-scale replication
T182 exposed a pseudoreplication failure mode. If there is one trained checkpoint at each width, repeatedly resampling blocks/passages estimates measurement uncertainty of those fixed checkpoints; it does **not** estimate between-training-run variance.

In the synthetic hierarchy used in T182, incorrectly treating block seeds as the inference unit inflated the null false-positive rate as the number of block seeds increased:

- 4 block seeds: ~5.1%
- 8: ~10.8%
- 16: ~17.9%
- 32: ~25.2%
- 64: ~31.0%

Using independently trained model replicas as the inference unit kept the null false-positive rate near the nominal level. With the tested variance scale and a true slope of `-0.03`, power was about 53% with 4 replicas/scale and 95% with 8 replicas/scale.

### Consequence for the existing TinyStories result
The current 1M / 3M / 8M / 28M WQ trend contains **one supplied checkpoint per scale**. It remains useful REAL_MODEL evidence that sharing becomes more favorable across those four particular checkpoints, but it is **not by itself a population scaling law**, regardless of how many block-subsample seeds are run afterward.

This is a stricter boundary than earlier project notes and should govern future claims.

## 4. Spend scaling-law budget on independent replicas
T183 kept a fixed total of 64 block-evaluation units per width and varied the allocation between model replicas and repeated block measurements. Under nonzero between-model variance, more independently trained replicas were much more valuable than repeatedly measuring the same checkpoint.

Representative planning result for a true slope of `-0.03`:

- `2 models × 32 block seeds`: ~14% power
- `4 × 16`: ~53%
- `8 × 8`: ~91%
- `16 × 4`: ~99%

Exact numbers are synthetic and must be re-estimated from observed real variance. The design conclusion is the durable part.

## 5. Paired training replicas can improve efficiency
T184 simulated a paired design in which replica index across widths shares some common training/data nuisance. As cross-width replica correlation increased, slope variance decreased and power increased while false-positive rate stayed controlled.

This suggests a future training study should, where technically valid, pair scale replicas by data/order/training seed and report the measured cross-width correlation. The correlation must not be assumed.

## Updated evidence hierarchy

### A. Checkpoint-conditional engineering claim
For the existing supplied checkpoints, a strong certificate can use:

1. matched-support, fold-rotated train/price/audit diagnostics;
2. independent outer passage groups;
3. fixed-scale cross-fit private SNR for share/no-share;
4. cross-fitted private fraction for checkpoint-specific geometry comparison;
5. actual full-support codec replay and emitted bytes.

This can support statements about **these checkpoints** and codec engineering decisions.

### B. Population mirror-scaling claim
To claim that functional mirror geometry generally improves with model scale:

1. train independent model replicas at every scale;
2. use model replica, not block seed, layer, or passage, as the inferential unit;
3. use hierarchical or paired scale analysis;
4. allocate additional budget to model replicas once within-model measurement is stable;
5. estimate required replica count from observed real between-model variance.

## Current scientific status

- Existing REAL_MODEL evidence still supports useful functional sharing in the tested TinyStories checkpoints, including WQ shared-codebook solutions that become quality-positive at 8M and 28M.
- The general Mirror/FQC hypothesis remains viable as a functional-sharing hypothesis.
- A **general positive scaling law is not yet demonstrated** because the current scale series lacks independent training replicas.
- Future claims must separate checkpoint-conditional engineering evidence from population-level scaling inference.
