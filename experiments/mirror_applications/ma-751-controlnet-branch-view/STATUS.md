# MA-751 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-751-controlnet-branch-view-20261109`
- Base commit: `e90b5056`
- Development: complete, seed 75101; five controls.
- Fresh/audit: unopened after exact native Givens alias and byte miss.
- Verification: three tests passed; deterministic non-timing replay checked.

## H / T / D / C / U

- **H:** Shared spatial control branch plus Views lowers bytes and preserves control quality.
- **T:** Synthetic two-layer conv branch, 8 condition IDs, 32 fit/16 heldout spatial maps each; copied, tied, FiLM, direct Givens and Mirror controls.
- **D:** **FAIL.** Mirror output error is zero but payload is larger than copied branches and MACs increase 23%; exact native alias.
- **C:** Ordinary Givens role conditioning explains the result; actual NPZ compression rewards the copied branch.
- **U:** Pretrained ControlNet, real image quality, learned adaptation and deployed latency.
