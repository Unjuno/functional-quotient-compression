# MA-1053 Results — scoped FAIL

## H — Hypothesis

A shared rank-32 projection over the exact native MOMENT pre-head features, with three task-specific lightweight output operators, would preserve useful forecasting, anomaly-detection and classification functions with less adaptation state than three independent rank-32 projections. Eight paired-plane rotation angles per task (24 total learned scalar codes) before the shared projection were expected to add useful task function beyond hard sharing and equal-byte diagonal gains.

## T — Trial

**Facts.** Used the pinned MOMENT-1-base checkpoint (453,940,120-byte weights; checkpoint revision `5e44b0ea26376a176360f87831124e018f876d96`) and official MOMENT code commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e`. Forecasting used ETTm2 OT (12/4/4 month chronological split, context 512, horizon 96); classification used ECG5000 (stratified 400/100 fit/dev split and 4,500 official test cases); anomaly detection used the official tiltAPB2 series (45,000 normal fit points, 5,000 normal threshold-calibration points and a 74,985-point suffix containing one anomaly event). Train-only scaling and all feature masks followed the frozen protocol. Audit seed was 105302, after the protocol/runner/data split freeze. CPU only; batch 8; 30 updates per task for each learned method; rank 32; no audit tuning.

The post-audit reporting correction AM-02 clarified that the eight-value code is stored once per task role (24 scalars total for three roles) and corrected the serialized metadata; no model weights or quality outputs changed. The controls were pinned native heads (pretrained reconstruction; trained task-native forecasting/classification output heads), hard-shared projection, same-byte diagonal gains, Mirror rotations before the shared projection, independent rank-32 projections, seasonal-naive forecasting and a class-majority baseline. Task quality was scored separately, not combined into one score.

### Audit quality

| Task / metric | Native MOMENT head | Hard share | Diagonal | Mirror | Independent rank-32 |
|---|---:|---:|---:|---:|---:|
| Forecast MASE ↓ | 1.121936 | 0.917711 | 0.917809 | 0.920243 | **0.867829** |
| Forecast MSE ↓ | 0.136618 | 0.100113 | **0.099993** | 0.100624 | **0.087307** |
| ECG5000 accuracy ↑ | **0.867556** | 0.837556 | 0.850000 | 0.831778 | 0.784667 |
| ECG5000 macro-F1 ↑ | **0.354358** | 0.339569 | **0.345760** | 0.336637 | 0.324777 |
| tiltAPB2 pointwise F1 ↑ | 0.003317 | 0.044141 | 0.044901 | **0.046243** | 0.004711 |
| tiltAPB2 AUPRC ↑ | 0.010586 | 0.015396 | **0.015931** | 0.015390 | **0.018410** |

**Facts.** Mirror forecast MASE was 0.27% worse than diagonal and 0.28% worse than hard share. Classification accuracy was 1.82 percentage points below diagonal and 3.58 points below native, outside the preregistered 2-point tolerance. Anomaly pointwise F1 was 0.00134 above diagonal, while AUPRC was lower; at the frozen development threshold Mirror found 72 of 826 anomalous points and produced 2,216 false positives (precision 3.15%, recall 8.72%). The anomaly file contains one event, so these point metrics are descriptive and do not establish event-level replication.

The shared, diagonal and Mirror variants use one shared physical projection basis and three task-specific output operators. Diagonal and Mirror each store eight codes per task role (24 total scalar codes across the three roles). Their complete adaptation payloads were 887,940 B, 888,096 B and 888,101 B, respectively. Mirror costs 161 B more than hard sharing and 5 B more than diagonal gains. The independent rank-32 bundle is 1,084,711 B: Mirror uses 18.13% fewer adaptation bytes, but the complete bundle including one common MOMENT checkpoint/config is 454,829,170 B versus 455,025,780 B, a 0.043% difference. The three native task heads together require 18,891,020 B of adaptation state; the complete native bundle is 472,832,089 B. The reduced adaptation state comes from rank sharing, not a Mirror-specific advantage.

Forecast plus anomaly/classification task-side active MAC proxies and measured CPU timings are recorded in `RESULTS_CORE.csv` and `runs/{dev,audit}/run.json`. Audit extraction/training wall time was 286.04 s; four learned bundles each used 90 shared-method updates (30 per task), batch 8 and 240 examples per task. The frozen MOMENT encoder dominates the proxy at 5.51G MAC/example. The Mirror adds approximately 147,456 rotation operations/example, compared with 49,152 diagonal-gain operations/example.

Native pre-head outputs matched replay through the corresponding official heads exactly in the executed batches (maximum absolute difference 0.0). Reloaded serialized adaptation states reproduced their cached-feature predictions exactly (maximum difference 0.0). Three focused tests passed.

## D — Decision

**FAIL for the registered Mirror-specific gate.** The quality benefit expected from the task-specific rotations did not appear: hard sharing or the matched diagonal control equaled or beat Mirror on forecast and classification; both also beat it on anomaly AUPRC despite Mirror's small single-event point-F1 increase. The slight anomaly F1 increase over diagonal did not improve AUPRC and came with very low precision and recall. Shared low-rank projections reduced adaptation bytes relative to independent projections, but the improvement is attributable to hard low-rank sharing; complete-model savings against independent rank-32 projections are negligible because the common checkpoint dominates storage.

## C — Strongest counter-hypothesis

The three task heads already have enough private task-specific output freedom, and a shared bottleneck plus cheap diagonal gains captures the useful cross-task overlap. Rotating encoder features adds operations and a few bytes without exposing a better shared subspace. The independent rank-32 forecast head performed better but used more adaptation state; task-specific variation may require private projection capacity for forecasting, while the single-event anomaly result is dominated by threshold/score calibration.

## U — Unverified

No near-convergence or fixed-byte capacity frontier was measured. Results use one seed per split, one dataset per task and a single anomaly event; GPU latency and broad task transfer remain unknown. The original checkpoint only pretrains the reconstruction head; native forecast/classification heads are trained under the same small fixed budget. These results do not establish a general MOMENT multitask compression claim.

## Fact / Interpretation / Hypothesis

- **Fact:** In this fixed-update screen, the Mirror task code adds 160 B versus hard sharing and does not meet the per-task quality gates; diagonal gains match or beat it on the registered quality metrics.
- **Interpretation:** A shared rank-32 projection can reduce adaptation bytes, but the measured marginal Mirror coordinate does not improve the quality/storage/compute frontier.
- **Hypothesis:** The strongest remaining use for this placement may require a larger/richer private residual or substantially more task-specific data; this requires a separate registered experiment.
