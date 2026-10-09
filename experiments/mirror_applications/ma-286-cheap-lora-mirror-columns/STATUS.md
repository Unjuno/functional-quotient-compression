# MA-286 status

- Status: FAIL for Mirror-specific advantage; aligned shared-subspace result replicated.
- Branch: `research/ma-286-cheap-lora-mirror-columns-20261009`
- Frozen protocol/source commit: `352f9ce5`
- Fresh: 108 rows, worlds 28610–28612 × seeds 0–2.
- Aligned Mirror/generic: NRMSE 2.99e-8 at ~5.34 KB; Cheap-LoRA 0.0125 at 12.38 KB.
- Independent tasks: shared code near no-adapter; full LoRA near exact.
- See `VERIFICATION.json`.
- Verification rerun: 2 tests pass; all 108 fresh rows and 180 payload size/SHA-256 entries checked.
