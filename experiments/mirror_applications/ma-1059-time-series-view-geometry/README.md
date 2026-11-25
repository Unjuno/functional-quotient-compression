# MA-1059 — Natural tourism-series View geometry

**Mirror insertion:** this experiment adds one angle `m_s` to the shared rank-2 output LoRA latent of a frozen Chronos-T5-Tiny forecaster so held-out monthly tourism series can express distinct forecast functions without storing a full adapter per series. A separate one-scalar horizon-trend residual tests the boundary where private state becomes necessary.

The source is Monash Time Series Repository monthly Tourism (366 series; pinned revision/archive hash in `PROTOCOL.json`, CC-BY-4.0 in the HF data card). T1–T8 were exposed by an initial archive preview and are excluded. The other series were split by SHA256 ordering of IDs before values were read. Fit and dev evidence is in `runs/dev`; the 24 registered audit series were not opened.

Controls include native Chronos, pooled LoRA, same-basis diagonal gain, two direct latent gains, independent per-series rank-2 LoRA, persistence and seasonal naive. The optional residual is chosen with support-only dev folds. Actual serialized payload bytes include model, adapter factors, coordinates, residuals, IDs and metadata.

**Outcome:** FAIL at the development screen. The frozen support-only selection chose a zero residual threshold, which allocates private state to all audit series and cannot satisfy the registered <=25% residual gate. Mirror also improves pooled LoRA by only 1.77% on dev, below the 2% quality gate. No audit or capacity claim is made; details are in `RESULTS.md`.
