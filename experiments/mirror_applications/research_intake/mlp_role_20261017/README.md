# MLP-ROLE-001 — How does Mirror change ordinary MLP applications?

**Container CPU verified 2026-10-17.** [Full usage decision](../../../../docs/phase2/MLP_MIRROR_USE_DECISION_2026-10-17.md). This is **not** a trained Transformer or an adoption recommendation.

Four conditions: related handwritten-digit attributes (BCE), four related diabetes target transforms (MSE), masked/noisy digit patch reconstruction (MSE), and four future steps of a synthetic nonlinear autoregressive process (MSE). The diabetes condition is ONE real target transformed four ways; the forecasting condition is NOT natural time series.

- Development: 4 roles × 7 methods × 2 learning rates = 56 models; LR .012 chosen before fresh.
- Fresh: 4 roles × 3 new data worlds/splits × 2 seeds × 7 methods = **168 models**, 550 AdamW updates, batch 64.
- New-world long: 4 roles × 3 OTHER worlds × 2 seeds × 6 methods = **144 models**, 1400 updates. Designed and SHA-frozen after primary; not retroactive preregistration.
- Native controls include shared FFN, same-parameter paired gains, per-pair Mirror, native task readout, factorized LoRA and independent FFNs (as applicable).
- Attributes: Mirror test BCE .11308 vs gain .13772 (3/3 worlds), but task-specific readout .10321 with fewer bytes and faster CPU forward. Long: Mirror .11782 vs gain .13389, native readout .07371.
- Digit denoise: Mirror MSE .03676 vs gain .03570, pair readout .03292, so no Mirror-specific quality advantage. Tabular and forecasting MSE gains were negligible; other controls matched or exceeded Mirror.
- Raw payloads: fresh 168 final+best payload pairs => 336, independently decoded and FP64 replayed. Long 144×2 = 288 independently checked. The two source locks match, **9 tests pass**, full retraining records 8 exact matches.

All four tasks share the same input per view in each example, thus this is a dense four-view evaluation. No top-1 expert switching, language NLL, accelerator execution or runtime advantage is claimed. Natural datasets appear in digit/diabetes inputs; their repeated splits are not independent datasets.

Read `PROTOCOL.json`, `LONG_PROTOCOL.json`, `PRE_FRESH_LOCK.json`, `PRE_LONG_LOCK.json`, `results/fresh/{SUMMARY.csv,ANALYSIS.json,AUDIT.json}` and `results/long/{SUMMARY.csv,ANALYSIS.json,AUDIT.json}`. Source files in the same directory run the exact study. The conversation full-evidence ZIP includes raw histories/model payloads.