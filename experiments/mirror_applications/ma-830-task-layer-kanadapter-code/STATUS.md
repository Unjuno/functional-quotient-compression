# MA-830 status

- Status: FAIL for Mirror-specific attribution.
- Branch: `research/ma-830-task-layer-kanadapter-code-20261111`
- Base: `a2ebf46e`
- Development: synthetic 8 task × 6 layer bank; 24 checkerboard pairs held out.
- Fresh/audit: sealed after native factor control matched exactly.
- Verification: focused test passed; factor payload reconstruction exact.

## H / T / D / C / U

- **H:** Task×layer Mirror factors beat independent banks and native HyperFormer-style factors.
- **T:** Analytic separable KAN adapter bank; independent function tables, native factors, Mirror views, marginal controls.
- **D:** **FAIL attribution.** Mirror 1,253 B/zero error; native factors 1,233 B/zero error; independent tables 11,277 B/.000545. Task-only/layer-only fail.
- **C:** Ordinary task/layer factorization is smaller and exact.
- **U:** Trained adapters, unseen factor inference, natural quality and fresh data.
