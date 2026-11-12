# MA-882 — layer-alignment × model-pair cache transport

Status: FAIL. Prior art: PA236 cross-model KV transfer and PA237 CacheBridge.

The experiment fixes target injection to all 30 SmolLM2 layers and fixes per-layer top-2 source-anchor routes before comparing full, PCA-coded, and CP-factorized translators. The experiment asks whether layer-map coefficients can be shared across model-pair identity, source depth, target depth, and KV role. It uses ordinary unweighted ridge and does not claim to reproduce CacheBridge.

Development froze PCA rank 256 and CP rank 8, neither passing quality selection. In fresh evaluation all 6 pair×seed quality gates failed while all 6 byte gates passed; CP payload was 4.29 MB versus 11.98 MB full and 5.01 MB PCA. See [RESULTS.md](RESULTS.md) for the report and replay verification.

See [PROTOCOL.json](PROTOCOL.json) for frozen splits, controls, gates, actual-byte contract, and compute measurements.
