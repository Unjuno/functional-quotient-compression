# MA-821 status

- Status: FAIL for Mirror-specific attribution; task-adaptive rank frontier observed synthetically.
- Branch: `research/ma-821-rank-adaptive-matrix-memory-20261111`
- Base: `63a54114`
- Development: seed 82101; eight tasks; prescribed ranks 1–8; 2,048 heldout vectors.
- Fresh/audit: sealed after exact native SVD rank-allocation alias.
- Verification: focused test passed; matrix replay exact.

## H / T / D / C / U

- **H:** Task-matched Mirror rank beats fixed-width storage/compute and native rank truncation.
- **T:** Synthetic group-composition memory; independent, fixed rank-8, adaptive Mirror and native SVD controls.
- **D:** **FAIL attribution.** Adaptive code uses 4,471 B, reaches nRMSE 3.16e-8 and composition error 1.32e-8 at 368 MAC proxy; native SVD is 4,473 B with identical metrics. Full matrices use 7,967 B and 512 MAC proxy.
- **C:** Native taskwise SVD truncation explains the whole frontier gain; only 2 B differs by serialization labels.
- **U:** Learned memory, unknown task rank, real tasks and fresh seeds.
