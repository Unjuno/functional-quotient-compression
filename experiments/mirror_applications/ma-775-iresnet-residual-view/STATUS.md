# MA-775 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-775-iresnet-residual-view-20261109`
- Base commit: `6fb05964`
- Development: complete, seed 77501; five controls.
- Fresh/audit: unopened after exact native basis alias.
- Verification: pending.

## H / T / D / C / U

- **H:** Compact residual Views remain contractive/invertible and reduce bytes vs independent blocks.
- **T:** 24 4D matrices; rank 2; 16 train / 8 heldout; inverse fixed-point convergence measured.
- **D:** **FAIL** for Mirror attribution. 49.7% byte savings, error 1.86e-8 and all inverses converge, but compute doubles and native factorization aliases exactly.
- **C:** The generated maps lie in a shared rank-2 matrix subspace; ordinary factorization suffices.
- **U:** Nonlinear trained i-ResNet and real data.

## Next action

Run focused tests, deterministic replay and registry integrity check, then save the scoped FAIL.

## Decisions / rulings

- Paired fixed heldout inputs used for final scoring; initial unpaired screen is retained in `RESULTS_INITIAL_UNPAIRED_INPUTS.csv`.
