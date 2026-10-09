# MA-486 status

- Status: FAIL
- Branch: `research/ma-486-sparse-dictionary-mirror-20261009`
- Base commit: `7754a91f`
- Protocol frozen: yes; development/fresh completed
- Fresh: 3 worlds × 3 seeds
- Verification: yes (`2e8c9202`); artifact replay test passed (1/1)

H: Sparse shared-dictionary codes should reconstruct useful function vectors with substantially fewer actual bytes than dense functions.

T: 32D synthetic vectors, 64 charged atoms, 128 functions/bank; dense, dense coefficient, signed sparse and FP32 sparse controls; top-k 2/4/8/16.

D: FAIL. Top-4 was 12,769B (70.8% dense) but NRMSE .1093; top-8 NRMSE .0258 but 15,329B (85.0% dense). No point met <=.05 error and <=50% bytes.

C: Dictionary and support-index overhead dominate at this bank size.

U: Larger banks, learned atoms, entropy coding and trained model functions.

Next: MA-487 — LISTA router for Mirror atom coefficients, on its dedicated branch.
