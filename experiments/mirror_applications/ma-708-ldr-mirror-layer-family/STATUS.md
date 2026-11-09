# MA-708 status

- Status: FAIL
- Branch: `research/ma-708-ldr-mirror-layer-family-20261109`
- Base commit: `8538db5a`
- Protocol frozen before audit: yes
- Fresh/audit opened: yes
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Result

Across three synthetic audit seeds, ranks 1/2/4 used mean 13.08%/25.58%/50.58% of independent dense payload; rank 8 used 100.57%. Reconstruction plus apply arithmetic was 1.0156x/1.0312x/1.0625x/1.125x dense apply. Serialized replay held-out operator NRMSE was <=6.53e-8. Native LDR shared atom coefficients exactly match candidate payload hashes and functions, so this is a scoped FAIL for Mirror-specific value.

## Next action

Advance to MA-709; retain MA-708 as synthetic LDR feasibility evidence only.
