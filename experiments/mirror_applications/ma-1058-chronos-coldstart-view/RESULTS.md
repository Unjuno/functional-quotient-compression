# MA-1058 results — Chronos cold-start series View

## H — hypothesis
A shared rank-2 output LoRA basis with one support-inferred series angle improves future forecasts for entirely held-out series and uses under 10% of per-series rank-2 LoRA state.

## T — trial
Pinned `amazon/chronos-t5-tiny` (`29d808298f1a62493e7b9a5e08529d0d930fa189`; 8,394,496 parameters), CPU float32. PEMS-BAY was split by complete sensor identity: fit IDs 22–37; development IDs 38–45; fresh audit IDs 62–77. IDs 46–61 were discarded after an accidental finite-value-only screen; amendment AM-01 records that no forecast metrics were seen and the untouched IDs 62–77 were frozen before audit. The runner reads HDF5 columns by hyperslab.

For every series, fit/adaptation used six support windows in the first 60% of the time line; evaluation used three 12-step forecasts in the final 20%, with 288-step context and 32 sample paths. Every method used the same windows. Development selected 60 support updates for Mirror, diagonal, full and independent methods; the shared basis used 30 updates. Audit sampling seeds were 105803/105804; these are draws on the same series/time splits, not independent worlds.

## Fact — audit results
Lower MASE, WQL and CRPS are better.

| Method | MASE seed 105803 | MASE seed 105804 | CRPS | WQL | Complete bytes |
|---|---:|---:|---:|---:|---:|
| native | 3.321143 | 3.066054 | 2.433647 / 2.239611 | 0.036199 / 0.033448 | 33,590,013 |
| pooled_lora | 3.234800 | 3.066054 | 2.359718 / 2.239611 | 0.035141 / 0.033448 | 33,624,971 |
| mirror | 3.198259 | 3.123245 | 2.352945 / 2.250675 | 0.035476 / 0.033675 | 33,625,034 |
| diag | 3.183400 | 3.113521 | 2.341106 / 2.237134 | 0.035367 / 0.033321 | 33,625,033 |
| full | 3.240391 | 3.111031 | 2.372049 / 2.271458 | 0.035671 / 0.033916 | 33,625,109 |
| independent | 2.665245 | 2.627938 | 2.063263 / 2.021008 | 0.031398 / 0.029819 | 34,147,134 |
| persistence | 3.299206 | 3.299206 | 3.294618 / 3.294618 | 0.054542 / 0.054542 | 255 |
| seasonal_naive | 6.593916 | 6.593916 | 6.573958 / 6.573958 | 0.106031 / 0.106031 | 258 |

For Mirror, per-seed worst-quartile MASE was 5.1115 / 4.6810; for independent LoRA it was 3.1048 / 3.1115. Mirror query inference p50/p95 was 0.668/0.791 s and 0.590/0.633 s per 32-sample forecast on one CPU thread. Support adaptation took 47.94/43.89 s across 16 series (60 updates each); the shared Mirror basis took 1.49 s for 30 updates. Independent adaptation took 45.73/44.12 s; query p95 was 0.704/0.616 s.

### Actual serialized payload
Each Chronos method includes 33,589,724 B common model/config/generation files. Mirror includes a 34,893 B shared rank-2 basis, 64 B of 16 one-float series codes, and 353 B metadata: total 33,625,034 B. Diagonal control totals 33,625,033 B. Independent LoRA stores 557,056 B of per-series factors plus 354 B metadata: total 34,147,134 B. Mirror therefore uses 6.34% of independent adaptation state (35,310 vs 557,410 B), but the complete deployed payload is only 1.53% smaller because the shared Chronos checkpoint dominates. Pooled LoRA totals 33,624,971 B after removing one duplicated 4 B angle copy. Persistence and seasonal-naive controls require no Chronos checkpoint (255 B and 258 B respectively).

## D — decision
**FAIL (scoped).** Mirror passes the state-size ratio against independent rank-2 LoRA, but fails the registered quality gate: it misses the ≥2% improvement over native and pooled controls on both sampling draws, and it is 18.8–20.0% worse in MASE than independent per-series LoRA. The same-cost diagonal-gain control is better on both draws (0.47% and 0.31% lower MASE), well outside the preregistered 0.1% tolerance. Full two-coefficient and per-series LoRA controls also clarify the private-state frontier.

## C — strongest counter-hypothesis
One angular degree of freedom on a shared low-rank output basis cannot represent sensor-specific temporal behavior; independent two-factor LoRA helps substantially. The diagonal control also performs better than the angular View, so any small gain over Chronos is not Mirror-specific.

## U — limits
One PEMS-BAY network and one calendar range; Chronos pretraining overlap is unknown. The two sampling draws are not independent data worlds. A verifier regenerated 11 of 16 sampled first-query arrays exactly; five model-method arrays for seed 105804 were finite but differed on redraw. All 16 seed×method metric rows replay exactly from saved forecast samples and targets, and all payload byte counts/hash checks pass. Treat the saved forecast arrays as the authoritative audit outcomes; do not claim exact stochastic forecast regeneration for the five mismatched rows.

## Interpretation
The physical sharing reduces adapter state by 93.7% relative to independent LoRA, but does not preserve its quality, and saves only 1.53% in complete deployment bytes. This result marks a measured boundary: one low-dimensional series coordinate is too restrictive for this cold-start panel, while independent rank-2 residuals restore quality at higher bytes. No broad cold-start or capacity claim follows.

## Hypothesis for follow-up
If another dataset family shows stronger alignment across series, a support-selected private residual may recover quality. That selector was not tested here; it requires a new preregistered experiment and must count every residual byte.

Accounting amendment: pooled global angle was initially serialized both in the common basis and in each deployment code. The redundant basis copy was removed and complete payload totals corrected by 4 B; no fit, setting or prediction changed. See `source/accounting_correction.py`.
