# MA-721 — SWAG posterior subspace with Mirror coordinates

Status: **FAIL (development gate)**
Branch: `research/ma-721-swag-mirror-posterior-20261008`
Baseline: `407ca7e2047326d1e4b753e55e05c4730f26f32b`
Draw 7: 528 eligible P0 candidates; seed `8c38748acbeec1315f8f38eee1e64a431a69cbd0e2fe53cb56458d350dd4e76c`, index 236. Pool/hash/row: `source/random_draw.json` and `source/selection_pool.csv`.

## H — hypothesis

For an SGD-derived rank-r posterior subspace over one shared MLP mean, Rademacher Mirror coordinates over the same covariance basis can improve corruption-shift NLL or calibration over Gaussian SWAG coordinates without degrading clean quality or increasing serialized inference bytes or K=8 compute.

## Prior art and controls

PA181 SWAG already stores a mean, low-rank covariance directions, and diagonal variance for posterior sampling. PA41 Rank-1 Bayesian Neural Networks use multiplicative rank-one factor distributions. The key control is Gaussian SWAG at the **same rank and covariance state**. Other controls are an SWA point model, Gaussian low-rank-only posterior, post-hoc Rank-1 factor perturbation, and an 8-checkpoint independent ensemble upper reference.

The Rank-1 factor control here is a post-hoc global-scale calibration screen, not a reproduction of the Rank-1 BNN training method. The task is a small sklearn digits benchmark, not large-model capacity evidence.

## T — execution

- MLP: 64→128→64→10, 17,226 parameters; SGD momentum 0.9, 80 epochs, snapshots from epochs 61–80.
- Development seeds: 7211 and 7212. Train/dev/fresh split counts are 1,150 / 287 / 360; the stratified fresh split is hash-locked and was not scored.
- Eight posterior members; ranks 4/8/16; 60 final development rows covering clean and Gaussian pixel corruption σ=0.25. Rank-1 global factor scale was chosen from the declared development grid only (σ=0.1 for seed 7211; σ=0.05 for 7212).
- SWAG fit used 720 updates per seed (about 0.34–1.04s); eight independent models used 5,760 updates per seed (about 2.71–2.72s). The 8-member forward proxy is 136,192 MACs/example, plus posterior weight reconstruction operations reported in `RESULTS_CORE.csv`.
- Actual `torch.save` payloads include mean, basis, scales, diagonal state and PRNG metadata as applicable. All 56 final payloads were reloaded, byte/hash checked, and their NLL/accuracy/ECE/Brier/disagreement replayed exactly. `source/metric_replay.json` records max metric difference 0. The final payload tensors clone rank slices so serialized bytes exclude unused backing storage. Per-condition payload captures total 29MB and are kept as local reproducibility artifacts rather than committed; each actual deployment payload size and SHA-256 are recorded in `source/development_raw.json`. Recreate the payloads by rerunning the frozen script before replay in a clean checkout.

## D — FAIL

No Mirror rank passed the preregistered development conjunction on both seeds. The corruption-shift improvement over same-rank Gaussian SWAG was far below the frozen threshold (≥0.02 NLL or ≥0.01 ECE), and the clean guard alone did not establish a Mirror-specific gain. The audit split remains unopened.

### Fact

Mean metrics over both training seeds:

| Method | Rank | Clean NLL | Clean ECE | Corrupt NLL | Corrupt ECE | Serialized bytes |
|---|---:|---:|---:|---:|---:|---:|
| SWA mean | — | 0.0969 | 0.018 | 0.3982 | 0.052 | 70,158 |
| Gaussian SWAG | 4 | 0.0970 | 0.018 | 0.3981 | 0.051 | 416,112 |
| Mirror Rademacher | 4 | 0.0970 | 0.020 | 0.3984 | 0.048 | 416,080 |
| Gaussian SWAG | 8 | 0.0969 | 0.018 | 0.3977 | 0.052 | 691,760 |
| Mirror Rademacher | 8 | 0.0970 | 0.020 | 0.3987 | 0.049 | 691,728 |
| Gaussian SWAG | 16 | 0.0969 | 0.018 | 0.3987 | 0.049 | 1,243,000 |
| Mirror Rademacher | 16 | 0.0969 | 0.017 | 0.3983 | 0.049 | 1,242,968 |
| Independent ensemble (8) | — | 0.0881 | 0.018 | 0.3540 | 0.045 | 554,112 |
| Rank-1 factors, dev-selected σ | 1 | 0.1039 | 0.024 | 0.3731 | 0.037 | ~70,500 |

At rank 4, the full SWAG/Mirror payload is smaller than eight independent checkpoints, but NLL is worse on corruption (about 0.398 vs 0.354). At rank 16, SWAG/Mirror state is more than twice the independent ensemble bytes. Mirror and SWAG payloads differ by only tens of bytes from metadata text and have essentially the same predictions. Mirror rank-4 disagreement was 0.001 on clean and 0.003 on corruption; the independent ensemble was 0.017 and 0.061.

The tuned post-hoc Rank-1 factor control improved corruption NLL/ECE over Mirror (0.373/0.037) with about 70.5KB, while its clean NLL rose to 0.104. This is a simple-control signal, not a Mirror result or a full Rank-1 BNN reproduction.

### Interpretation

The discrete code distribution did not add useful posterior-predictive behavior beyond Gaussian SWAG with the same covariance directions. The low disagreement shows that the sampled SGD subspace produced few distinct predictions on this dataset. At low rank it saves bytes versus eight separate checkpoints, but the independent ensemble retains better quality; at higher rank its covariance basis becomes larger than the ensemble.

### H / T / D / C / U

- **H:** A structured posterior coordinate can improve shifted predictive quality/calibration at matched SWAG rank, bytes and compute.
- **T:** Two MLP training seeds; 80 epochs; final 20 trajectory snapshots; 8-member posterior; ranks 4/8/16; Gaussian SWAG, Rademacher Mirror, rank-1 factor, SWA and 8 independent checkpoints; 60 clean/corrupted development rows.
- **D:** **FAIL.** No Mirror rank met the corruption improvement threshold over matched SWAG on both seeds. Fresh remained locked.
- **C:** Digits and one additive Gaussian corruption are narrow; the posterior trajectory itself may be underdispersed. The post-hoc Rank-1 control was not fully variationally trained.
- **U:** Fresh split, natural OOD shifts, full Rank-1 BNN training, larger datasets/networks, and near-convergence/fixed-byte capacity remain untested.

## Preserved implementation attempts

`source/development_attempt_01_duplicate_rank1_rows.json` retains the first run, where the rank-1 grid was accidentally repeated under each covariance rank. `source/development_attempt_02_tensor_view_bytes.json` retains a 60-row run whose tensor views serialized unused backing basis storage. Final metrics were rerun with the same frozen settings after correcting only row placement and cloning rank slices; no hyperparameter was selected from those attempts. See `source/attempt_log.json`.

**BOUNDARY:** fixed-budget posterior-predictive screen on a small digits MLP. No capacity claim, no conclusion on large Bayesian networks, and no Mirror-specific benefit established.
