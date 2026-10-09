# MA-601 status

- Status: **FAIL (development)**
- Branch: `research/ma-601-hash-compressed-attention-heads-20261009`
- Worlds 60101/60102 complete; fresh 60111–60113 sealed.
- Mirror CE 3.236/3.282 at 5,006 B; independent CE 2.138/2.120 at 10,421 B; rank1 hash CE 3.141/3.135 at 6,058 B; salted hash 4,627 B.
- Verification passed: 16/16 payloads replay byte-exact; 3 tests passed.

## H / T / D / C / U

- **H:** Givens Views over shared hashed Q/K/V would recover useful attention-head multiplicity at lower bytes.
- **T:** Four-head synthetic teacher, 8-token sequences, 32-dimensional inputs; eight student controls; 1,500 updates; two dev worlds; payloads serialized.
- **D:** **FAIL.** Mirror fails quality by >1 nat vs independent heads, loses to MQA and cheap activation/low-rank controls, and exceeds the 1.05x salted byte cap. Head output diversity increases relative to tied hash but does not preserve function quality.
- **C:** Independent teacher projections require private head state; rank-one residuals perform better than a pure View.
- **U:** No natural LM, scale, near-convergence or optimized GPU evidence.

## Facts / interpretation / hypothesis

- **Fact:** Details in `RESULTS_CORE.csv`, `runs/dev_*/metrics.json`, and `VERIFICATION.json`.
- **Interpretation:** More logical heads do not guarantee useful head capacity.
- **Hypothesis:** Attention quality requires head-private Q/K/V directions that this View cannot encode.
