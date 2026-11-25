# MA-1051 — iTransformer channel-conditioned Mirror residual

Decision: **FAIL (scoped fixed-budget screen)**

## H — hypothesis
Two per-sensor Givens coordinates on shared iTransformer variate tokens improve macro audit standardized MSE by at least 2% over hard sharing and matched FiLM, while retaining robustness under sensor-input ablation.

## T — experiment
- Data: `jimmygao3218/PEMS07`, pinned revision `e071fab0481c3db4eeb4193e79d821220b7b5bff`; `PEMS07.npz` SHA256 `a07887e610aa986276209bc9384d14ed7858fdcad1ded67f53dca19eb828600b`, 43,705,518 bytes, array `(28224,883,1)` float64. First 16 sensors only; no semantic units asserted.
- Chronological 60/20/20 split; context 96, horizon 24; 48 train origins, 12 development, 12 audit.
- iTransformer-style variate temporal projection and two layers of cross-sensor attention, d_model 32, 4 heads; scratch-trained, seed 105101, AdamW, 12 epochs, 36 updates per shared variant.
- Mirror: per-sensor 2-float angle/log-gain Givens transform on the first 2 token coordinates. Control: same-size diagonal FiLM. Hard sharing retains cross-sensor attention. Independent upper reference: sixteen separate univariate models.
- Sensor ablation zeros inputs for sensors 0,4,8,12 and keeps token slots.

## D — FAIL

| Method | Dev MSE | Audit MSE | Audit with sensor ablation | Payload bytes | Train seconds |
|---|---:|---:|---:|---:|---:|
| Shared | 0.254723 | 0.212165 | 0.249472 | 92,863 | 0.208 |
| Mirror | 0.254700 | 0.212196 | 0.249639 | 93,244 | 0.252 |
| FiLM | 0.254665 | 0.212180 | 0.249515 | 93,242 | 0.215 |
| Independent | 0.230859 | 0.239101 | n/a | 1,484,343 | 2.409 |

Mirror audit MSE is 0.0147% worse than hard sharing and 0.0075% worse than FiLM. Its complete serialized payload is 381 B larger than hard sharing; it is far smaller than the independent reference, which is also lower quality here. Input ablation degrades shared and Mirror similarly; Mirror is 0.0669% worse under ablation, inside the frozen 2% non-inferiority bound.

## C — strongest counter-hypothesis
One seed, 12 epochs, 16 sensors, and a reduced CPU architecture may not expose a useful channel-view regime. Mirror coordinates initialize at identity and make only small changes; a longer or different optimization schedule could change this screen.

## U — unresolved
No multi-seed replication, full official iTransformer pipeline, larger sensor set, private residual sweep, or accelerator timing. Fixed-budget evidence is not a capacity claim. The dataset README describes only the source/citation, so target units and physical interpretation are deliberately not claimed.

FACT: measured values and bytes appear above. INTERPRETATION: this Mirror insertion does not improve the tested storage/quality/compute frontier over hard sharing or FiLM. HYPOTHESIS: for this small iTransformer and dataset slice, shared cross-variate attention captures the useful structure without per-sensor functional codes.
