# MA-1056 — DLinear periodic residual View

## H — falsifiable hypothesis

An annual phase coordinate for each currency, applied to the first Fourier component of the DLinear seasonal residual, would improve final-period macro MASE by at least 2% over shared DLinear, the equal-state per-series gain, and the dev-tuned naive blend in both audit seeds. It would also cost fewer bytes than free independent Fourier coefficients and keep inference within 1.25× DLinear latency.

## T — execution

**Fact.** Pinned LTSF-Linear `DLinear.py` from commit `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` (SHA-256 `0893b53c…1451cc`), under its Apache-2.0 license. The public Exchange-Rate panel is pinned to `laiguokun/multivariate-time-series-data` commit `7f402f185cc2435b5e66aed13a3b560ed142e023`, compressed asset SHA-256 `6a4fbb56…1c2e9d`, 7,588 rows × 8 daily currency series. Fit and dev target rows are 0–4,551 and 4,552–6,069; audit targets are rows 6,070–7,587. Train-only per-series mean/std and lag-252 MASE scales are frozen. The dev loader stops at row 6,070; it does not load audit values before the selection freeze.

Development chose context 252 and learning rate .005 from `{252,504} × {.001,.005}` at 100 updates. The dev-tuned seasonal-naive blend selected α=0 annual component, i.e. last-value naive. Final models used seeds 105602 and 105603, 300 updates each, batch 32. Final audit uses 15 non-overlapping 96-step target windows per seed over all 8 currencies.

Controls: pinned shared DLinear; DLinear plus the same projected annual residual at fixed zero phase; same-size per-currency seasonal gain; Mirror per-currency phase; free per-currency sine/cosine coefficients; eight separate DLinear models; last-value, annual seasonal-naive and dev-tuned blend. Every learned model is stored in an actual `torch.save` inference payload. Audit metrics were independently replayed from these serialized payloads and their bytes/hashes verified.

### Fact — results

| Method | Seed 105602 MASE | Seed 105603 MASE | Mean MASE | Bytes | Mean inference ms/window |
|---|---:|---:|---:|---:|---:|
| Shared DLinear | 0.462692 | 0.428560 | 0.445626 | 197,469 | 0.0551 |
| DLinear + fixed periodic residual | 0.462791 | 0.428540 | 0.445665 | 197,469 | 0.1302 |
| DLinear + per-series gain | 0.462912 | 0.428494 | 0.445703 | 197,721 | 0.1844 |
| DLinear + Mirror phase | 0.462655 | 0.428594 | 0.445625 | 197,721 | 0.1391 |
| DLinear + free Fourier coefficients | 0.467156 | 0.434544 | 0.450850 | 197,721 | 0.1238 |
| Independent DLinear | 0.683783 | 0.627453 | 0.655618 | 1,568,515 | 0.4023 |
| Last-value / dev-tuned naive blend | 0.384308 | 0.384308 | 0.384308 | 0 | 0 |
| Annual seasonal naive | 0.956499 | 0.956499 | 0.956499 | 0 | 0 |

Mirror is effectively tied with shared DLinear (0.0004% lower mean MASE) and matched gain (0.02% lower), far inside the 1% alias threshold and well below the 2% success threshold. It loses to the last-value naive by about 16% MASE. Its full serialized payload is 252 B larger than DLinear; equal-state gain has the same size. Measured mean inference time is about 2.53× DLinear; this exceeds the 1.25× runtime gate. Independent models are 8× larger but underperform under the fixed per-model update budget.

The annual seasonal-naive control was poor, while last-value naive was strong. Learned Mirror phases were nonzero, but the quality changes were inconsistent across the two seeds and did not yield a useful storage/quality/runtime tradeoff. The first annual residual harmonic requires input projection work; this is reflected in measured inference latency.

## D — decision

**FAIL (scoped fixed-budget result).** Mirror does not meaningfully beat DLinear or the equal-state gain, and it is slower and larger than shared DLinear. The stronger last-value baseline wins clearly. This experiment establishes no Mirror-specific Pareto gain.

## C — strongest counter-hypothesis

Daily exchange-rate changes have little stable annual residual seasonality; random-walk persistence is a better forecast, and the phase code only rotates a weak input-derived harmonic. The comparable gain control explains its tiny error movement without any Mirror-specific benefit.

## U — remaining uncertainty

There are two optimization seeds but only one benchmark panel and 15 non-overlapping audit windows. Results are fixed-budget, not near-convergence capacity evidence. No GPU benchmark, longer horizons, or other DLinear benchmark families were measured. Runtime is measured on CPU with a small audit batch and should be treated as a local screen.

## Verification

Three tests pass: pinned DLinear output shape, zero-phase parity, and development/audit target split checks (development loader does not load audit rows). `source/verify_report.py` reloads all 12 learned audit payloads, checks actual file sizes and SHA-256 values, and reproduces each audit MASE exactly. See `VERIFICATION.json`.
