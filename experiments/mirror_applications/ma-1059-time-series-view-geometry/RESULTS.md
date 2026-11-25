# MA-1059 — Natural tourism-series View geometry

## H — Hypothesis

On held-out monthly tourism series, one frozen Chronos-T5-Tiny plus a shared rank-2 output LoRA basis and one support-inferred angle should recover useful forecasts with less adaptation state than independent per-series rank-2 LoRA. A support-selected scalar trend residual should be needed for at most 25% of series. A same-basis diagonal control must be beaten by at least 0.1%, and Mirror must improve by at least 2% over native and pooled LoRA.

## T — Trial

Pinned Chronos-T5-Tiny revision `29d808298f1a62493e7b9a5e08529d0d930fa189` (CPU float32) and Monash monthly Tourism benchmark `Monash-University/monash_tsf` revision `58aafbe2712ff481c014f562e42723f2820fd5d4`, archive SHA256 `6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c`, listed CC-BY-4.0. Context and horizon were both 12 months. Forty fit series trained the shared basis for 30 updates; 16 disjoint development series selected per-series updates and residual threshold. Series were assigned by deterministic SHA256 ordering of names before parsing numeric values. T1–T8 are excluded because an initial archive preview exposed their rows. The 24 fresh audit series were never numerically loaded.

Controls: native Chronos, pooled rank-2 LoRA, the shared basis with one-angle Mirror, a byte-near diagonal gain, two direct latent gains, independent per-series rank-2 LoRA, persistence, seasonal naive, and Mirror with the optional trend residual. Development explored 10/30/60 support updates. A residual is fit along a fixed centered 12-step horizon ramp from support forecasts only; its magnitude threshold was selected on a support-only held-fold procedure.

## Fact — development results

Lower MASE is better. These are development results, not fresh-audit results.

| Method | Macro MASE | CRPS | Actual complete payload bytes | Adaptation state bytes |
|---|---:|---:|---:|---:|
| Native Chronos | 4.657900 | 1,956.847 | 33,589,982 | 258 |
| Pooled rank-2 LoRA | 4.633805 | 1,952.533 | 33,624,807 | 35,083 |
| Mirror, 10 updates | 4.551984 | 1,947.048 | 33,624,862 | 35,138 |
| Diagonal gain, 30 updates | 4.563992 | 1,951.062 | 33,624,860 | 35,136 |
| Full two gains, 30 updates | 4.561471 | 1,950.645 | 33,624,924 | 35,200 |
| Independent rank-2 LoRA, 10 updates | 4.573154 | 1,935.135 | 34,147,043 | 557,319 |
| Mirror + trend residual, 10 updates | 4.595120 | not independently scored in summary | 33,625,123 | 35,399 |
| Persistence | 4.455252 | — | 258 | 258 |
| 12-month seasonal naive | 1.987944 | — | 258 | 258 |

The shared checkpoint files total 33,589,724 B. Actual method tensors, codes, residuals, IDs, and JSON metadata were serialized and their on-disk sizes measured. Mirror uses 6.30% of the independent adapter state, but the full payload is only 1.53% smaller than independent LoRA because the checkpoint dominates. Mirror improves development MASE by 1.77% over pooled LoRA, below the frozen 2% gate. The same-cost diagonal is 0.26% worse than Mirror. The seasonal naive is much better than every neural method here.

Selected support updates: Mirror 10; diagonal 30; full two gains 30; independent LoRA 10. The support-only residual threshold selected exactly `0.0`. Since the frozen inclusion rule is `abs(r_s) >= threshold`, that threshold allocates a private residual to all 24/24 fresh series and cannot satisfy the registered maximum 25% residual fraction. On development query windows, applying the residual to all series also worsened Mirror MASE from 4.551984 to 4.595120. The query comparison was not used to select the threshold.

## D — Decision

**FAIL at the development screen; no fresh audit run.** The frozen support-only selector chose a zero magnitude threshold, which deterministically allocates private residual state to every audit series and violates the preregistered sparsity gate. Mirror also falls short of the 2% development improvement over pooled LoRA (1.77%). The result does not establish fresh-series performance or capacity.

## C — Strongest counter-hypothesis

Chronos-T5-Tiny with a 12-month context is a poor baseline for monthly Tourism: 12-month seasonal naive MASE is 1.99 versus Mirror MASE 4.55. The measured adapter comparisons may reflect a weak base forecast family more than the usefulness of a series View.

## U — Unconfirmed

No audit targets were opened, so quality on the 24 fresh series, the realized residual fraction there, and fresh serialized forecast replay remain unmeasured. This is one benchmark and one deterministic development run. The first unseeded development run is preserved but invalidated; only the explicit-seed run informed the result. No capacity or broad compression claim follows.

## Interpretation

This is a negative development screen for the registered sparse-private-state criterion. The angle uses much less adapter state than independent LoRA, but simple diagonal gain is nearly tied and the residual selector collapses to all-private allocation. The model-free seasonal control dominates quality. No inference follows for other frequencies, datasets, or foundation models.

## Compute and provenance

The run used CPU float32 with one Torch thread; no GPU was available. Shared-basis training took 0.772 s for Mirror and 0.643 s for pooled LoRA, 30 optimizer updates each. Selected per-series adapter fitting across 16 dev series took 2.974 s for Mirror, 8.875 s for diagonal, 8.167 s for full two gains and 2.390 s for independent LoRA. Selected query p50 latency was 0.417/0.433/0.418/0.359 s respectively. These times are observed CPU wall-clock; FLOPs were not measured. The Mirror+trend residual's additional support forecast time was not isolated and is not included in the listed adaptation time.

The deterministic dev run used `torch.manual_seed(105900)`. An earlier unseeded development run is preserved under `runs/dev/invalidated_unseeded/` and did not set the frozen selection. Four protocol/data contract tests pass. The verifier checks archive/model/protocol hashes and recomputes all serialized dev state byte totals. Development forecast samples were not saved, so the aggregate dev metric table is not independently sample-replayed; fresh metric replay is inapplicable because fresh evaluation was not run.
