# MA-292 status

- Status: FAIL for Mirror-specific value; shared-basis compression signal reproduced.
- Branch: `research/ma-292-task-vector-mirror-basis-20261009`
- Frozen protocol/source commit: `5cfa9d2b`
- Fresh: 108 rows (54 held-out task + 54 held-out composition), worlds 29210–29212 × seeds 0–2.
- Mirror/generic/PCA held-out NRMSE ~2e-7 at ~4.36 KB; Mirror does not beat PCA/generic.
- See `VERIFICATION.json`.
- Verification rerun: 3 tests pass; all 108 fresh rows and 180 payload size/SHA-256 entries checked.
