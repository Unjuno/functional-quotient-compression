# GQA functional geometry: canonical real-model findings

This note records only the findings that survived local re-analysis of the preserved T003/T004 actual SmolLM2 results. Exploratory screening heuristics remain outside the canonical repository.

## Evidence source

- Model: `HuggingFaceTB/SmolLM2-135M`
- T004 grid: layers `0, 5, 10, 15, 20, 25, 29`, all three GQA KV groups, rank-3 joint Q/K family approximation (`21` candidates)
- Fixed calibration probe: the same 4 authored passages used by T003
- T004 was executed twice independently before the workflow policy changed; the two full result JSON files are byte-identical.
- Full-result SHA-256: `db35b182620a36c8330aac69afecf34b5ee28526ebba06eaed678d65a5b29b5a`

## Finding 1 — Raw family residual is not a task metric

Across the 21 T004 candidates:

- structural residual vs final relative logit RMS: Pearson `-0.0105`
- structural residual vs mean KL: Pearson `-0.0716`
- local attention relative RMS vs final relative logit RMS: Pearson `0.7815`

The raw normalized Frobenius family residual should therefore not be used as the QCO task cost.

## Finding 2 — Shared-K reconstruction error is the useful local structural coordinate

Within each audited layer, the three KV groups were compared while holding the layer fixed. An exact stratified permutation test enumerated all `6^7 = 279,936` within-layer KV permutations.

- K reconstruction error vs attention-output damage: mean Spearman `0.7143`, positive in `7/7` layers, one-sided exact `p = 0.00403`
- topology score using the GQA factor `3`: mean Spearman `0.6429`, `p = 0.00973`
- raw family residual: mean Spearman `0.0`, `p = 0.551`
- Q RMS reconstruction error alone: mean Spearman `-0.2143`, `p = 0.821`

After Bonferroni correction across these four precompared coordinates, K error remains below `0.05` (`p = 0.0161`) and the topology score also remains below `0.05` (`p = 0.0389`).

This is association evidence for the tested joint Q/K intervention family; it is not yet a K-only causal intervention result.

## Finding 3 — Transfer geometry is strongly layer-specific

Across the same 21 candidates, layer identity explains nearly all variation in the transfer ratios:

- log continuation ratio (`decoder output -> final logits`) eta-squared: `0.9756`
- log within-layer attenuation ratio (`attention output -> decoder output`) eta-squared: `0.9977`
- corresponding KV-head eta-squared values: `0.00113` and `0.000177`

A leave-one-KV-out calibration using the other two groups in the same layer predicts held-out final relative RMS from decoder relative RMS with median absolute percentage error `11.6%` and Pearson `0.794`.

This supports a staged screening geometry: local candidate perturbation plus a layer-specific transfer calibration, followed by exact full replay for finalists.

## Finding 4 — One calibration loss is not an admission metric

In T003, all three L15 Q/K rank interventions improved the tiny fixed calibration NLL while causing material KL and top-1 changes. Calibration NLL is therefore useful as one witness, but cannot be the sole commit criterion.

## Current operational hypothesis

For GQA candidate screening, separate:

1. reconstruction structure, especially shared-K error / topology leverage;
2. activation-conditioned local perturbation;
3. layer-specific continuation transfer;
4. exact full-model validation for commit decisions.

Do not infer codec-bit savings, benchmark quality, or 64x feasibility from these results.
