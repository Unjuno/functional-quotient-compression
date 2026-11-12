# MA-882 — layer-alignment × model-pair cache transport

Status: SCREENING. Prior art: PA236 cross-model KV transfer and PA237 CacheBridge.

The experiment fixes target injection to all 30 SmolLM2 layers and fixes per-layer top-2 source-anchor routes before comparing full, PCA-coded, and CP-factorized translators. The experiment asks whether layer-map coefficients can be shared across model-pair identity, source depth, target depth, and KV role. It uses ordinary unweighted ridge and does not claim to reproduce CacheBridge.

See [PROTOCOL.json](PROTOCOL.json) for the frozen splits, controls, gates, actual-byte contract, and compute measurements.
