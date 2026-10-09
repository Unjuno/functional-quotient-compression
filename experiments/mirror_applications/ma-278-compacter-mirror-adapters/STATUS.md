# MA-278 status

- Status: FAIL (registered storage gate missed; no Mirror-specific advantage).
- Branch: `research/ma-278-compacter-mirror-adapters-20261009`
- Frozen protocol/source commit: `146a4fa2`
- Fresh: 90 rows, worlds 27810–27812 × seeds 0–2.
- Aligned Mirror/generic: 1,643 B, NRMSE 5.79e-9; Compacter 1,737 B, 8.06e-8.
- Independent LoRA: 3,137 B, NRMSE 8.62e-5; shared codes near no-adapter.
- See VERIFICATION.json.
- Verification rerun: 3 tests pass; 90 fresh rows and 150 payload size/SHA-256 entries checked.
