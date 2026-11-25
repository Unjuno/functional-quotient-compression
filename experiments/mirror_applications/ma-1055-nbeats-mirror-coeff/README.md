# MA-1055 — N-BEATS shared seasonal basis with factorized Mirror phase address

## H — hypothesis

On a shared official N-BEATS interpretable seasonality basis, a small additive series-family × calendar-regime phase address will improve held-out-role forecast quality over hard sharing and equal-state additive-gain/common-shift controls while storing fewer role parameters than a flat per-role phase bank. A basis coefficient vector alone is already the native N-BEATS mechanism and will not count as Mirror-specific evidence.

## T — protocol

The experiment uses the pinned ServiceNow/N-BEATS implementation, with a documented NumPy compatibility patch in an isolated copy. Forecast- and backcast-coefficient chunks retain the upstream layout; only forecast cosine/sine pairs above DC are rotated together. Trend, DC, and backcast are unchanged. Fit and development use ETTm1; frozen audit uses ETTm2 non-OT channels because ETTm1 audit output was accidentally computed during development. The affected run is preserved as invalidated evidence.

Controls: official native N-BEATS per-channel training (independent upper control), N-BEATS shared across roles, equal-size additive gain, common phase, factorized series+regime phase, and flat role phase bank. Storage is actual serialized inference payload bytes including the official weights, all addresses, metadata, and compatibility/reconstruction state. Quality is MASE with seasonality-aware audit and a role ablation. Any Mirror-like phase is a coefficient-coordinate change exactly representable by N-BEATS native harmonic coefficients, so evidence can support factorized address compression only, never extra per-role functional capacity beyond those controls.

## Fact / interpretation / hypothesis

Facts are reported in `RESULTS.md` after the locked audit. Interpretation is limited to this synthetic small-model screen. The hypothesis concerns functional multiplicity under shared seasonality coefficients; it is not a general N-BEATS capacity claim.

## Reproduction

See `PROTOCOL.json`, `source/`, `tests/`, and `VERIFICATION.json`.
