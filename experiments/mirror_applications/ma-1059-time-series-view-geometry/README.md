# MA-1059 — Natural tourism-series View geometry

**Mirror insertion:** this experiment adds one angle `m_s` to the shared rank-2 output LoRA latent of a frozen Chronos-T5-Tiny forecaster so held-out monthly tourism series can express distinct forecast functions without storing a full adapter per series. A separate one-scalar horizon-trend residual tests where private state becomes necessary.

The source is Monash Time Series Repository monthly Tourism (366 series; pinned revision/archive hash in `PROTOCOL.json`). The GroupLens archive readme permits research use and requests citation; raw data is not redistributed. T1–T8 were exposed by an initial archive preview and are excluded. Remaining series were split by SHA256 ordering of IDs before parsing numeric values. Forty fit series train the shared basis, 16 select settings, and 24 are held for fresh audit.

Controls include native Chronos, pooled LoRA, same-basis diagonal gain, two direct latent gains, independent per-series rank-2 LoRA, persistence and seasonal naive. The optional residual is selected by support-fold relative MASE improvement, with a null/no-residual option and a 25% development allocation cap. Actual serialized payload bytes include model, adapter factors, coordinates, residuals, IDs and metadata.

**AM-01 correction:** post-development review found a residual-evaluation hook leak and a denominator error. Those initial results are preserved as invalidated. The corrected development run selected a 2.5% support-fold relative-improvement threshold and residuals on 2/16 development series. The fresh audit configuration is now frozen; its numeric values remain unopened.
