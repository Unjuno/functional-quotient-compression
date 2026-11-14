# MA-1004 — off-orbit private sparse residual

Status: **FAIL — bounded development screen**
Branch: `research/ma-1004-off-orbit-private-residual-20261114`
Base checkpoint: MACE-MP small, 32,581,838 B, SHA-256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`.

## H — hypothesis

A rank-2 shared source-molecule basis and an 8-shot target code will be inadequate on uracil, while a small private sparse residual will recover force quality with a smaller actual payload than simpler controls. A Mirror-specific claim requires improvement over native low-rank factorization and simple gate/direct-update controls.

## T — what ran

**Fact:** Data are rMD17 CC0; the common prior is the official MACE-MP small checkpoint. The target split is fixed by `old_indices`: 8 support rows were used for adaptation and 128 disjoint development rows for selection. Source corrections used 8 selected rows each from benzene, toluene and naphthalene. Fresh rows were not used for metrics.

**Fact:** One CPU seed (100401) ran 24 force-only updates per method on the final 16-value nonlinear equivariant readout. The rank-2 source basis used centered SVD of the three source deltas. Methods were rank-2 target code, rank-2 plus four support-gradient-selected private coordinates, and a direct 16-coordinate readout update. A scalar force-scale gate was fit on the same 8 support rows. The earlier 32-shot frozen baseline is retained separately in `runs/frozen_development.json`.

**Fact:** The official PA300 wrapper was instantiated on the raw MACE-MP model and completed a force forward pass. It exposed 3,126,556 trainable parameters; no PA300 convergence or quality result was run. Full-model fine-tuning, energy quality for adapted methods, multi-seed replication, and fresh evaluation were not run. This was an explicitly bounded development screen, not the full registered control tournament.

## Development results

| Method | Support | Dev force component MAE (eV/Å) | Serialized adaptation payload (B) | Total with common prior (B) |
|---|---:|---:|---:|---:|
| Frozen MACE-MP | 8 | 0.254951 | 0 | 32,581,838 |
| Scalar force-scale gate | 8 | 0.241780 | 1,957 | 32,583,795 |
| Rank-2 source-SVD Mirror code | 8 | 0.254876 | 2,465 | 32,584,303 |
| Rank-2 + four private coordinates | 8 | 0.249430 | 3,097 | 32,584,935 |
| Direct 16-coordinate readout update | 8 | 0.243877 | 1,769 | 32,583,607 |

The payloads are actual `torch.save` serialized inference objects from the screen and include their format/model-hash metadata. The rank-2 objects include the basis, code, and materialized delta; no deduplication/compression of that redundant representation was applied. The common prior is charged in the total column. Inference timing is not compared: the runs did not use a controlled identical thread/runtime setup. The run JSONs preserve the payload hashes, row IDs, per-row metrics, and timing observations.

## D — decision

**FAIL at the development gate.** Rank-2 Mirror was effectively equal to frozen MACE-MP on force error. Adding four private coordinates improved it by 0.00545 eV/Å, but remained 3.2% worse than the same-budget scalar gate and 2.3% worse than the direct 16-coordinate update. Its serialized adaptation payload was 75% larger than direct-16 (3,097 vs 1,769 B). The scalar gate also had fewer learned degrees of freedom and a smaller actual payload than either Mirror variant. This screen therefore fails the predeclared requirement that Mirror-plus-private state recover quality at a more favorable byte/control frontier.

**Interpretation:** In this small readout-only screen, a source-derived low-rank view does not capture useful uracil force corrections. The private coordinates help slightly, while the simpler scalar/direct controls help more. The current Mirror encoding adds storage without a quality advantage.

**Claim boundary:** This is a development-gate result for one 16-parameter MACE readout slice, one seed, and force quality only. It does not establish bulk-material behavior, energy accuracy of adapted models, full PA300 behavior, or fresh generalization. The native rank-2 implementation is algebraically the same `B m` factorization, so this experiment provides no Mirror-specific advantage over that control.

## C — strongest counter-hypothesis

The final 16-value readout slice is too narrow to represent the molecular-domain force correction, and the three 8-example source tasks give a noisy basis. A broader equivariant sparse update could improve the target and change the byte/quality frontier. Against that, both cheap same-support controls already dominate this Mirror screen.

## U — what remains unknown

- Whether full PA300 EquiSparseDeltaSTR training or a broader equivariant sparse residual changes the result.
- Adapted energy quality and energy/force joint Pareto frontier.
- Multi-seed variability and fresh uracil quality.
- Whether a minimally encoded rank-2 payload, without storing its reconstructible materialized delta, can reduce bytes enough to matter; any such revision needs a new locked amendment and must still beat the simple controls.

## Fact / interpretation / hypothesis

- **Fact:** All measured values above come from support/development only; the fresh rows remain sealed. Actual serialized payload sizes are included, and the base checkpoint is counted in total bytes.
- **Interpretation:** The bounded Mirror/private route failed its early development gate against lower-byte simple controls.
- **Hypothesis:** The loss is structural (target correction is not aligned with the source basis), but the single readout slice cannot distinguish basis misalignment from insufficient adapter placement.
