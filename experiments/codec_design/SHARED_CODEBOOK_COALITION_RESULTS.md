# Shared-codebook coalition search: synthetic optimizer results

This note records the high-value results from container-side synthetic experiments T115–T123. These are **small synthetic optimizer experiments**, not scalability theorems.

## 1. Codebook count is a discrete rate–distortion variable

In a high task-heterogeneity family with 12 tensors, splitting tensors across multiple task-aware codebooks produced a smooth quality/metadata Pareto.

Representative T115 results:

| codebooks | codebook metadata vs local | task NMSE |
|---:|---:|---:|
| 1 | 8.3% | 0.3994 |
| 2 | 16.7% | 0.3898 |
| 4 | 33.3% | 0.3791 |
| 6 | 50.0% | 0.3729 |
| 8 | 66.7% | 0.3702 |
| 12 (local) | 100% | 0.3687 |

Four shared codebooks used about one third of local-codebook metadata for roughly a 2.8% task-NMSE penalty; six used half the metadata for roughly a 1.2% penalty.

## 2. Geometry-feature clustering is not a reliable coalition solver

Task-weight cosine tracked the *global* sharing penalty in a controlled sweep, but directly clustering tensors by task features was not reliably better than matched random partitions (T116).

T117 also tested:

- variance-geometry features;
- task-geometry features;
- concatenated/joint features;
- weighted joint features.

None consistently beat random partition controls across the tested seeds.

Therefore compatibility diagnostics can be useful evidence, but simple feature-space KMeans should not be treated as the coalition optimizer.

## 3. Pairwise merge pricing is stronger but not exact

T118 used the actual task-cost change from sharing a codebook between each tensor pair as a pairwise merge price. Pairwise-price clustering improved over task-feature clustering and random-average partitions in the tested instance.

T119 then used a six-tensor / three-codebook problem where all `90` set partitions were exhaustively evaluated. Across the first three seeds:

- pairwise pricing missed the exact optimum in `3/3` cases;
- mean regret was only about `0.77%`;
- task-feature clustering mean regret was about `2.54%`.

Thus pairwise pricing is a strong proposal mechanism, but is not an exact master solver.

A stronger D65-style pattern — every pair merge being harmful while a three-way merge is beneficial — was not found in five seeds / all 100 tested triplets (T120). The observed pairwise failures therefore appear to arise from global fixed-codebook-count partition competition rather than a simple triplet-synergy mechanism in this setup.

## 4. Replacement search recovered the exact small optimum

T121/T122 started from pairwise-price partitions and applied best-improvement:

- single-tensor moves between codebook groups;
- tensor swaps between groups.

Across six seeds, replacement search recovered the exact exhaustively enumerated optimum in `6/6` cases, usually with `0–3` operations.

This does not prove local replacement is globally sufficient in larger problems. It does support the search pattern already suggested by the historical root-pricing experiments: **price cheaply, then allow representation-changing replacement moves before commit.**

## 5. Lazy exact group evaluation

T123 repeated the same six-tensor / three-codebook problem with group costs evaluated only when required by the search.

- possible nonempty tensor subsets: `63`
- mean group-cost queries before exact verification: `40.2`
- mean query fraction: `63.8%`
- exact optimum recovered before verification: `6/6`
- mean replacement operations: `1.17`

This is a small result, but it suggests a practical QCO pattern:

`singleton/pair pricing -> candidate partition -> on-demand exact group costs -> move/swap replacement -> exact commit validation`

## Boundaries

Do not infer that:

- pairwise pricing plus replacement is globally exact in general;
- these subset counts predict Transformer-scale compute;
- group task costs are additive under simultaneous real-model interventions;
- random or feature clustering can be dismissed for all future codecs.

The scientific conclusion is narrower: **shared representation introduces a discrete coalition variable; pairwise prices are useful proposals, but exact or validated replacement moves remain necessary.**