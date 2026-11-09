# MA-771 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-771-realnvp-functional-view-20261109`
- Base commit: `e8a6d107`
- Development: complete, seed 77101; five controls.
- Fresh/audit: unopened after native alias and byte miss.
- Verification: pending.

## H / T / D / C / U

- **H:** Invertible coupling views encode nonlinear task transformations with compact codes.
- **T:** 2D affine coupling; 12 train / 4 heldout codes; linear, equal-size MLP, independent flow and native conditional-flow controls.
- **D:** **FAIL** for Mirror attribution and storage. Cycle error `4.77e-7`, but shared payload 701 B vs 573 B independent; native RealNVP is an exact alias.
- **C:** Standard conditional RealNVP explains the behavior, while independent NPZ compression is unusually effective here.
- **U:** Natural density estimation and higher-dimensional deployment.

## Next action

Run focused tests, deterministic replay and registry integrity check, then save this scoped FAIL.

## Blockers

None.
