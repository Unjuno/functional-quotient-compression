# MA-902 results — synthetic scene × time field screen

## H — hypothesis

A rank-4 multiplicative scene × time code over shared spatial atoms predicts held-out scene-time cells at lower serialized bytes than independent fields and more accurately than additive scene+time coding. A Mirror-specific gain must exceed native rank-4 tensor factorization.

## T — execution

- Synthetic field: 8 scene identities × 6 timestamps × 96 spatial samples; 4 shared smooth spatial atoms plus a small private pair residual.
- Each scene had 4/6 timestamps observed; 16 complete scene-time cells were held out. Every scene and timestamp appeared in training.
- Development seeds: 90201 and 90207; amended confirmatory fresh seeds: 90221, 90223, 90229.
- Adam, 3000 updates, learning rate .03, weight decay 1e-4. The weight decay was selected only on development seeds in Amendment 01. The first unregularized fresh set was opened during initial screening and is archived separately, excluded from confirmatory claims.
- Controls: shared rank-4 multiplicative Mirror code, algebraically identical native rank-4 factorization, additive scene+time code, independent dense full-cell oracle payload.
- Serialized NPZ inference payloads were saved for every fresh/development row; the 20 payloads were reopened, and field reconstruction was replayed.

## D — FAIL for the frozen synthetic quality gate and Mirror-specific attribution

### Facts

| Fresh seed | Mirror / native rank-4 heldout nRMSE | Additive nRMSE | Mirror payload | Independent oracle payload | Bytes saved | Query MAC proxy Mirror / dense |
|---|---:|---:|---:|---:|---:|---:|
| 90221 | 0.02277 | 1.37458 | 2,461 B | 17,682 B | 86.08% | 392 / 96 |
| 90223 | 1.52103 | 1.13711 | 2,462 B | 17,638 B | 86.04% | 392 / 96 |
| 90229 | 0.06961 | 1.37050 | 2,464 B | 17,651 B | 86.04% | 392 / 96 |

The native rank-4 control uses the same mathematical factorization, outputs, payload bytes, and MAC proxy as Mirror. Payload roundtrip and metric replay passed at max absolute reconstruction difference below 2e-5. The dense oracle includes held-out fields and is a storage/quality reference, not a predictor for unseen cells.

### Interpretation

The low-rank multiplicative structure has a large storage advantage over storing every complete scene-time field in this synthetic family. But the preregistered nRMSE ≤ .05 gate fails in two of three fresh seeds, and multiplicative Mirror exactly aliases ordinary native tensor factorization. Its 392 MAC proxy is 4.08× the dense control's 96. This is a scoped FAIL, not a Mirror-specific compression result for real neural fields.

### C — strongest counter-hypothesis

The observed compactness comes from native rank-4 separability in the generator. The severe fresh outlier suggests factor fitting/generalization is unstable for some factor configurations; adding private residual state may recover it but could consume the byte savings. The measured representation gives no Mirror-specific gain over native factorization.

### U — unresolved

Real C-NGP or K-Planes scenes, actual radiance rendering, view-dependent appearance, temporal continuity, and private-residual byte/quality frontier remain untested. No capacity claim follows from this synthetic finite grid.

## Protocol deviation retained

The initial unregularized execution and initially opened fresh seeds are preserved in `source/results/pre_amendment/`. Development seed 90201 exposed optimization instability. Amendment 01 selected weight decay 1e-4 using development only and replaced fresh seeds with 90221/90223/90229 before generating those data. The initial fresh data were not used for tuning or the confirmatory decision.
