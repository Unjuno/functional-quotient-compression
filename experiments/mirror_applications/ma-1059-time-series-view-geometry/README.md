# MA-1059 — Natural tourism-series View geometry

**Mirror insertion:** this experiment adds one angle `m_s` to the shared rank-2 output LoRA latent of a frozen Chronos-T5-Tiny forecaster so that held-out monthly tourism series can express distinct forecast functions without storing one full adapter per series. A separate one-scalar horizon-trend residual tests where private state becomes necessary.

The fixed data source is the Monash Time Series Repository's monthly Tourism competition dataset (366 series; HF revision and archive hash in `PROTOCOL.json`). T1–T8 were exposed by an initial metadata preview and are excluded. Remaining series are split by a stable SHA256 ordering of series names before numeric values are parsed. Fit, development and audit IDs are frozen in the protocol.

Controls preserve the native Chronos model, pooled LoRA, byte-near diagonal gain, direct two-coordinate gain and independent per-series rank-2 LoRA. The trend residual is selected from support-only evidence using a development-frozen threshold. Report actual serialized bytes, adaptation state separately from the full checkpoint, forecast quality and residual selection fraction. This is a single-family screening result, not a capacity claim.
