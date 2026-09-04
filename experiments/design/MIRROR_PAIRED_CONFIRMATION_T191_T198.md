# Mirror scaling paired-confirmation design (T191-T198)

Evidence lane: **SYNTHETIC DESIGN / STATISTICAL IDENTIFICATION**. These results do not add new TinyStories real-model functional evidence.

## Main conclusions

1. Pairing training replicas across scale helps only for nuisance that is approximately common-mode across widths. In the synthetic variance decomposition used here, pairing data-composition and initialization components produced the largest power gains because those components were assigned the largest variance shares. This ranking is not an empirical claim about real training.
2. Scale-by-nuisance interactions can erase the benefit of pairing and systematic paired nuisance can bias the scale slope. Paired factors therefore need balancing/randomization across scale; they cannot simply be held fixed and assumed harmless.
3. Population inference should use one scale contrast/slope per independent training replica (or an equivalent clustered/mixed model). Layers, blocks, pairs, folds, and codebook seeds remain measurement repeats rather than population replicas.
4. Replica identities and pairings must be fixed before mirror diagnostics are observed. Random mispairing loses power; outcome-driven pairing can strongly inflate false-positive rates by manipulating the across-replica standard error.
5. With a paired-replica slope SD near 0.018 in the synthetic pilot, a Monte-Carlo-calibrated 4->6 sequential confirmation retained about 88.8% power at about 2.47% type-I error, close to fixed-six power (~90.0%), while using about 4.39 models/width under the null and 5.55 under the tested effect. These power numbers are conditional on the assumed variance.
6. The confirmatory scale-shape contrast must be predeclared. A linear contrast is broad; a 1M/3M-vs-8M/28M contrast is more powerful for a two-group crossover; a late-step contrast is best for a 28M-only change. Choosing among contrasts on confirmation outcomes inflated the tested null false-positive rate from 2.5% to 5.62%. Choosing the contrast on independent discovery checkpoints and testing only that contrast in new replicas preserved the nominal rate (~2.49%).
7. Four paired replicas are too few to estimate the paired slope SD precisely for power planning: in the synthetic experiment the sample SD was within +/-25% of truth only ~44.7% of the time. The confirmation test should retain unknown-variance t/Monte-Carlo calibration; pilot SD is a planning input, not a fixed known variance.

## Updated evidence boundary

The existing TinyStories 1M/3M/8M/28M checkpoints remain **discovery only**. They may define the candidate scientific contrast and diagnostic protocol. Population confirmation must use newly trained independent replicas with pairing metadata frozen before evaluation.

## Updated confirmatory workflow

- Predeclare one scale contrast from discovery data.
- Pair replica IDs across width before training.
- Prefer pairing high-variance common-mode factors, but randomize/balance any scale-specific interactions.
- Run matched-support / cross-fit geometry diagnostics independently from full-codec evaluation.
- Analyze independent replica-level contrasts/slopes.
- Start with four new replicas/width; only use a 4->6 or 4->8 sequential design after pilot variance is assessed without unblinding confirmation efficacy.

## Negative results preserved

- Pairing is not automatically bias-free.
- Post-hoc pairing is invalid.
- Naive IID regression is not the correct paired-replica population analysis.
- Confirmation-outcome-based contrast selection requires multiplicity correction.
- A four-replica pilot cannot precisely determine the paired variance or final power requirement.
