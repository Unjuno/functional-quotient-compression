# MA-736 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-736-shared-private-operator-20261109`
- Base commit: `7f7e3319`
- Development: complete; seed 73601, eight operators, 34 result rows.
- Fresh/audit: unopened after gate failure and exact native aliases.
- Tests: 3 passed; deterministic metric replay recorded in `VERIFICATION.json`.

## H / T / D / C / U

- **H:** Shared operator plus low-description View and sparse private residual cover common and rare diffusion regimes under a byte budget.
- **T:** Analytic 1D variable-diffusion Green matrices, N=24, rank-4 basis from six regimes and two heldout; independent, shared mean, PCA, simplified random shared basis and top-k private controls.
- **D:** **FAIL.** k=8 first passes both heldout quality thresholds but saves only 20.7% against independent, below 25%; Mirror matches native PCA/private payload and outputs exactly.
- **C:** Oracle-selected PCA coefficients and private support explain performance. Native methods already contain all Mirror functionality tested here.
- **U:** Learned neural operator, realistic adaptation, natural physics regimes, experimental PDE and accelerator serving.
