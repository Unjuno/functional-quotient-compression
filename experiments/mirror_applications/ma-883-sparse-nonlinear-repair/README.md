# MA-883 — shared linear cache transfer with sparse nonlinear repair

Status: SCREENING. Prior art: PA236 cross-model KV transfer and PA238 nonlinear translator/context correction.

This candidate tests a pooled per-role linear translator with ordinary pair-specific FiLM coordinates, then adds a private nonlinear residual only for pair IDs marked hard on development. Fresh prompts come from whole WikiText-103-train article buckets absent from WikiText-2, avoiding the reused small WikiText-2 audit rows. The repair mask is static and frozen before audit.

See [PROTOCOL.json](PROTOCOL.json) for the article split, controls, quality/byte gates, and compute accounting.
