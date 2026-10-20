# MA-1171 STATUS

- **2026-10-20: COMPLETED FAIL** under frozen protocol (`PROTOCOL.json` SHA-256 `27f914fe06676bf46b8a786667df644859da3a48e0c707b172514d8ca39fee1a`).
- 30 CPU trainings, 500 updates each; 5 pre-train test groups PASS, fresh opened only after completing training, 30/30 inference-packet byte exact audit PASS.
- Small benefit vs matched affine: 6/6 fresh OOD F1 positive but <0.05 absolute in all pairs; strict criterion vs affine AND dense_loop 0/6, no practical winner. Dense onepass clearly strongest.
- Source and checkpoints retained in this report bundle. Research branch only; no merge to main.
