# MA-1170 status — 2026-10-20 JST

**COMPLETED: FROZEN PILOT FAIL.**

- Frozen local JSON protocol SHA256: `86da7956cfac5eab05eb93be4b15fabeb3c9f46cab4807eb7edf1f3ae431557f`.
- CPU 12/12 fits (4 variants x 3 seeds), 500 updates each; source/tests/weights supplied in full archive.
- 9/9 source tests PASS; all model packet replay checks PASS; separate actual PCM16 roundtrip FLAC exact.
- Fresh NMSE mirror median 0.342928, affine 0.333077, sparse 0.347384, native 0.340326; 5% Mirror-specific gate FAIL.
- posthoc rank16 PCA=0.229338 Fresh NMSE, materially stronger control but never used to retune frozen experiments.
- Model is *not* a waveform codec; 8-bit 16-dimensional Mel-feature codes discard information.
- Existing `FROZEN_PROTOCOL_SUMMARY.md` predates model execution; the original JSON corresponding to its cited SHA was not available. Local protocol was committed to on disk before any training and verified by SHA in all checkpoints.
- See REPORT_JA.md, RESULTS_CORE.csv, VERIFICATION.json and full source/checkpoints archive.
