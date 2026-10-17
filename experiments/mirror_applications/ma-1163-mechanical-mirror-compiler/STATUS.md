# MA-1163 — 2026-10-18

- Current status: **SCREENING; tested fixed nonlinear Mirror construction shows M0 / no Mirror-specific quality-per-byte advantage**.
- Training first: FP64 native 64→16→10 NN, 1210 independent weights, all 3 seed development accuracy >=90%; PASS.
- Mechanically fit pretrained native weight vectors: 30 first-m fresh rows including native, Mirror and no-Mirror baseline; m=1210 exact logits equivalence and original accuracy PASS. m<=256 retention FAIL.
- Trained Mirror directly from labels: 3 seeds × m16/256/1210. Full-dimensional Mirror learns; underparameterized variants fail at acceptable quality.
- Post-hoc development-only selected-directions pilot: 48 rows. Never mix its dev metrics into fresh claims.
- Independent precommitted selected-directions fresh replication: 3 new stratified partition worlds × 2 teacher seeds × m512/768 × 3 controls = 36 rows. Retention m768 3/3 world PASS; Mirror-specific dominance 0/3 FAIL.
- m768 sparse tensor + index payload 9573 B vs native 11997 B, but including standalone nonlinear-program source 17217 B; no one-model code-accounted compression advantage.
- No claim of general-purpose Mirror compression, FP800+ training, GPU acceleration, or 1-scalar functional equivalence.
- Prior MA-1162 and MA-597..604 statuses remain unchanged. Main not modified.
