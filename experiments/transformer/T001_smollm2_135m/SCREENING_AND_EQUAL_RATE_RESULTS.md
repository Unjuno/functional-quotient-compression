# SmolLM2 screening and equal-rate selection results

This note records only the high-value conclusions from container-side re-analysis of the preserved T004 actual-model artifact. Intermediate screening heuristics and exploratory scripts remain outside the repository.

## Scope

- Model: `HuggingFaceTB/SmolLM2-135M`, pinned T001 checkpoint.
- Source actual experiment: T004 rank-3 GQA Q/K interventions.
- Candidate grid: 7 layers × 3 KV groups = 21 candidates.
- All candidates use the same structural intervention family and therefore have the same continuous scalar-count rate proxy.
- These are individual-intervention results. Sums of individual KL values are not joint-intervention distortion claims.

## 1. Equal nominal rate saving, highly unequal functional cost

Each candidate starts from four `576 × 64` family members, or 147,456 raw scalars. A rank-3 continuous factor-count proxy uses 110,608 scalars, for a nominal saving of 36,848 scalars (24.99%) before quantization and metadata.

Despite this equal nominal rate proxy:

- mean KL ranges from `0.00140412` to `0.09207353`: **65.57× spread**;
- relative logit RMS ranges by **25.49×**;
- top-1 flip fraction ranges from `0.00610` to `0.12805`.

Therefore equal structural rate saving does not imply comparable task cost. Task-aware allocation is necessary even before joint codec interactions are considered.

## 2. Local activation metric is much more informative than raw structural residual

For the 21 candidates, the fixed local metric

`m = attention_delta^0.75 × decoder_delta^0.25`

has Spearman correlation `0.9429` with mean KL on this fixed calibration probe. A layer-cluster bootstrap with 5,000 replicates gives a 95% interval `[0.8623, 0.9856]`.

To account for selecting the exponent from the grid `{0, 0.25, 0.5, 0.75, 1}`, an exact within-layer permutation test enumerated all `6^7 = 279,936` permutations. The best within-layer mean Spearman was `0.9286`; the familywise one-sided p-value after maximizing over the exponent grid was `0.000200`.

This supports local activation perturbation as a screening signal. It is not a replacement for exact task replay at commit time.

## 3. Simple depth interpolation of transfer gain fails

Transfer gain is strongly layer-specific but not smooth enough in layer index for naive neighboring-layer interpolation.

Leave-one-layer-out log interpolation produced:

- continuation-gain median relative error: **71.6%**;
- within-layer attenuation median relative error: **90.1%**.

Thus sparse depth interpolation should not be used as a certificate or commit metric.

## 4. Local-only held-layer screening remains useful

A leave-one-layer-out model using only local attention and decoder perturbations, with no held-layer full replay used in fitting, predicts final relative logit RMS with:

- median relative error: **27.4%**;
- Pearson: **0.866**;
- Spearman: **0.932**.

It identifies both the safest and most damaging KV group correctly in 6 of 7 held-out layers. The main exception is layer 5, where KV-specific continuation gain varies substantially.

## 5. Candidate-shortlist implication

On this 21-candidate dataset, to retain all true lowest-KL top-5 candidates:

- raw family residual requires retaining **18/21** candidates;
- shared-K reconstruction error requires **5/21**;
- the local activation metric requires **5/21**.

A layer-forward accounting proxy for a `K-error shortlist of 5 → local screening → exact replay of 5 finalists` is 264 layer-forward equivalents versus 630 for replaying all 21 candidates, a **58.1% reduction** while retaining the true lowest-KL top-5 on this dataset. This is not wall-clock evidence.

## Boundaries

Do not infer from these results that:

- individual KL costs add under simultaneous interventions;
- the local metric is a task-quality certificate;
- the layer-forward proxy is a measured runtime speedup;
- the rank-3 factor-count proxy is an emitted codec bit count;
- any of these results establish 64× feasibility.

Exact serializer accounting and exact task replay remain authoritative for commit decisions.
