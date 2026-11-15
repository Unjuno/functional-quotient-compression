# MA-1048 status

- Status: FAIL
- Branch: `research/ma-1048-moirai-frequency-variate-view-20261115`
- Fresh PEMS-BAY audit completed on 12 held-out frequency×sensor pairs.
- Candidate CRPS is 13.994/85.062 vs zero-shot Moirai 4.074/4.039; additive control matches candidate within 0.02%.
- Factorized coefficients use 128 B vs 120 B additive and 224 B full-table reference; complete checkpoint payload is about 55.3 MB.
- CPU inference and postprocess measured; no accelerator claim. Saved prediction replay and verification pending.
