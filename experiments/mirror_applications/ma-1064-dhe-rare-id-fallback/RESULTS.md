# MA-1064 — DHE rare-ID private fallback

## H — Hypothesis

Allocating per-ID Givens codes and private residuals only to items with low train-only frequency should improve fresh rare-item AUC by at least 0.005 over the strongest same-cutoff non-Mirror fallback on both seeds, keep overall logloss within 0.002 of shared DHE, use at most 25% of full-table bytes, and retain at least 80% of no-view throughput.

## T — Trial

GroupLens MovieLens 1M (archive SHA256 `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`, 5,917,549 B). Excluded all 99,718 pairs that formed the previously scored MA-1063 audit partition. On the remaining pairs, the registered new hash split produced 720,438 train, 90,076 development, and 89,977 fresh audit rows with no pair overlap. Fresh seeds: 106410/106411. Explicit rating >=4 is positive.

The shared k=128 DHE decoder and GMF head were trained for three epochs and frozen. Private-only adaptation compared shared DHE, ordinary per-ID residuals, private replacement vectors, per-ID Givens alone, Givens plus residual, and full user/item tables. Development selected cutoff 5 from {1, 5, 20}; each cutoff selects 364 item IDs and zero user IDs at this training size. AM-01 narrows the registered rare-slice target to items because the development partition has zero user-frequency<=5 rows and 137 item-frequency<=5 rows. AM-02 preserves a pre-audit development serialization failure and the unchanged successful rerun.

## Fact — frozen fresh audit

Means across the two fresh model seeds on the same 89,977-row audit partition. Payload bytes include serialized weights and exact inference metadata, including all selected train ID addresses. QPS is batch-1,024 CPU throughput measured over 40 repeats.

| Method | Overall logloss ↓ | Overall AUC ↑ | Item AUC, train freq <=5 ↑ | Payload bytes | QPS | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Shared DHE | 0.557089 | 0.774790 | 0.718505 | 106,972 | 252,335 | 4.674 |
| DHE + ordinary residual | 0.557066 | 0.774815 | 0.735417 | 156,099 | 223,178 | 5.639 |
| DHE + private replacement | 0.557071 | 0.774822 | 0.729412 | 156,327 | 227,260 | 5.430 |
| DHE + per-ID Givens | 0.557079 | 0.774801 | 0.725980 | 132,721 | 213,107 | 6.155 |
| DHE + Givens + residual | 0.557057 | 0.774824 | **0.740931** | 179,623 | 203,370 | 6.207 |
| Full tables | **0.556739** | **0.780700** | 0.763848 | 1,282,473 | 21,180,119 | 0.097 |

Mirror+residual improves rare-item AUC over ordinary residual by 0.004412 on seed 106410 and 0.006618 on seed 106411. The registered +0.005 margin is missed on one seed. Its overall logloss is better than shared DHE by 0.000027/0.000035 on the two seeds, so that gate passes. The complete payload is 14.0% of full tables, so the storage gate passes. Throughput is 81.5%/79.8% of DHE on the two seeds; it misses the 80% gate on seed 106411. Mean rare-item slice size is 133 audit ratings per seed. User-frequency<=5 has zero audit rows; zero-train-frequency items have 15 rows, too few to estimate AUC.

The selected payload stores 364 item IDs, 16 angles and 32 residual values per ID, and no private user state. The Givens coordinates add 23,524 B over ordinary residual state in the complete serialized bundle. The private ID list and other inference metadata are included in payload counts; the per-ID metadata/checkpoints/logits remain local and their hashes are recorded separately. All six methods trained 528 method-specific updates per seed; each DHE-backed method also shares 528 base updates. The Mirror+residual run takes an average 26.72 s for the base plus 13.64 s for private adaptation; FLOPs were not measured.

Serialized payload replay passed for 12/12 method/seed combinations. Maximum logit difference was 4.77e-7 and maximum metric difference was 2.54e-10. Audit labels and audit row IDs were not persisted. Selected training item IDs are present only in the local inference payload metadata and are charged by byte count.

## D — Decision

**FAIL** under the frozen gate. Storage and overall logloss pass, but the rare-item margin misses +0.005 on one seed and throughput misses 80% on the other. AUC is numerically higher than ordinary residual on both seeds, but this two-seed result does not meet the registered consistency and runtime thresholds. It is not evidence of increased capacity.

## C — Strongest counter-hypothesis

The rare-item slice contains only 133 ratings and shows substantial seed variation. The apparent Mirror advantage over residual may be seed noise or optimization variance, and the slight throughput regression may be measurement noise in a short CPU microbenchmark. The ordinary residual itself improves mean rare-item AUC over shared DHE by 0.016912, so most of the long-tail change is explained by adding private item state.

## U — Unconfirmed

One dataset and one new deterministic audit split; two model seeds are not independent data worlds. The user rare-ID slice is absent, and the 15 unseen-item rows are too few for AUC. The implementation tests frequency-selected private state, not per-ID selection by measured training error or marginal utility. No GPU serving result, production recommender result, near-convergence or capacity claim is made.

## Interpretation and follow-up

**Fact:** Private residual state improves the low-frequency item slice over table-free DHE while keeping the complete payload below 15% of full tables. Adding Givens angles and residuals improves mean rare-item AUC over residual-only by about 0.55 percentage points at an additional 23.5 kB, but the gain and throughput miss the frozen per-seed gates.

**Interpretation:** This screen shows where a small amount of private state begins to help: low-frequency item identities. It does not establish that per-ID rotations are the right private representation; residual-only is smaller and nearly matches the fresh result.

**Hypothesis:** Selecting private exceptions by train-only per-ID utility, rather than a global frequency cutoff, may improve quality per byte. That requires a separately frozen follow-up and fresh rows; it is not inferred from these audit scores.
