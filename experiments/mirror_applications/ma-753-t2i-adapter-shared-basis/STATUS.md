# MA-753 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-753-t2i-adapter-shared-basis-20261109`
- Base commit: `3bf30111`
- Development: complete, seed 75301; four controls.
- Fresh/audit: unopened after exact native PCA alias.
- Verification: pending.

## H / T / D / C / U

- **H:** Shared basis condition Views compress the adapter bank and preserve heldout control/composition.
- **T:** 12 synthetic matrix adapters; 8 train conditions; 4 heldout conditions; rank-2 shared basis; independent full and low-rank plus native PCA controls.
- **D:** **FAIL.** 2,006 B (77.2% below full bank), but 2× MAC proxy and exact native PCA alias.
- **C:** Shared function-subspace structure is built into the synthetic task; standard PCA completely explains the result.
- **U:** Natural adapter/diffusion measurements.

## Next action

Run focused tests, deterministic replay and registry integrity check, then save the scoped FAIL to this branch and cumulative evidence branch.

## Blockers

None.

## Decisions / rulings

- Amendment A01 corrects an evaluation design-matrix axis order and consistent compute accounting; the initial invalid output remains in `RESULTS_INITIAL_BUGGY.csv` and is excluded.
