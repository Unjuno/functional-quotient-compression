# MA-715 — RegMean statistics to Mirror merge coefficients

Status: **SCREENING — protocol frozen before data execution**
Branch: `research/ma-715-regmean-mirror-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Draw 11: 557 eligible candidates; see `source/random_draw.json` and complete pool `source/selection_pool.csv`.

## H — falsifiable hypothesis

Directly solving a small Mirror code from RegMean Gram/cross-moment statistics can preserve held-out task-mixture quality and lower final inference bytes, while outperforming byte-near task arithmetic and shared-basis controls on quality or merge compute.

## Prior-art delta

PA177 (RegMean) already uses layer-input Gram statistics to merge weights without retaining source examples. PA26 (Task Arithmetic) already combines task vectors. This screen tests whether those sufficient statistics can solve compact source-delta View codes directly, and whether that has any benefit over full RegMean followed by ordinary low-rank coding.

## Frozen design

The full dataset splits, source models, merge equations, rank sweep, controls and gates are in `PROTOCOL.json`. The screen uses the local sklearn digits dataset, four 90-degree image rotations, a shared multiclass ridge base and four rotation-specific source classifiers. It will compare full RegMean, task arithmetic, source-delta projection, output-merge SVD codes and direct statistics-to-Mirror codes.

Development uses the train and dev partitions only. The heldout test partition and new mixture coefficients remain locked unless the complete development gate passes in both worlds, including the Mirror-specific control gate. Actual inference payload, merge statistics, and total merge package are measured separately.

## D — pending

Protocol committed; no model fits or development scores have been inspected.

## C — strongest counter-hypothesis

Direct coefficient solving may be exactly ordinary low-rank regression in a different notation; output-SVD coding or task arithmetic may achieve the same quality and bytes, while full RegMean already provides the best merge.

## U — unknown

The rank needed to preserve classification quality, useful merges per stored byte, direct-solver compute advantage and any Mirror-specific gain remain unmeasured.
