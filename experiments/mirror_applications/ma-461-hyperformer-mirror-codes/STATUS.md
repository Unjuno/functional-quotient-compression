# MA-461 status

- Status: FAIL
- Branch: `research/ma-461-hyperformer-mirror-codes-20261009`
- Base commit: `7665f84a`
- Protocol freeze: `682f9588`
- Development selected: 1,000 updates (from 200/500/1,000)
- Fresh worlds: 46110–46112 × seeds 0–2
- Tests/results/registry: pending final commit

## H / T / D / C / U

- H: HyperFormer-generated Mirror codes reduce generator bytes while preserving held-out adapter quality.
- T: Synthetic affine-in-context 2×2 adapters; full linear HyperFormer, two-angle Mirror decoder, generic rank-2 basis, independent teacher-matrix upper control. 64 training and 64 held-out contexts per world-seed. Actual package bytes measured at N=1/20/64.
- D (Fact): N20 NRMSE: HyperFormer 5.51e-7, Mirror 0.908, rank-2 basis 0.364, independent oracle 0. Bytes/context: 174.25B, 180.45B, 180.45B, 222.25B.
- D (Interpretation): Mirror misses quality and byte gates; generic rank-2 basis matches bytes and has much better quality.
- C: The task family's affine matrix variation is not aligned with a shared-matrix two-angle Mirror orbit.
- U: Natural adapters and Transformer fine-tuning remain untested.

## Next action

Commit checked results, update registry/ledger, push branch, and proceed to MA-462.

## Blockers

None.
