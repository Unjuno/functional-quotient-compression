# MA-511 status

- Status: PROMISING (held-out synthetic condition × behavior composition only)
- Branch: `research/ma-511-hierarchical-condition-behavior-mirror-20261009`
- Base commit: `25c4f935`
- Protocol frozen before development/fresh: yes (`5fcf1006`)
- Development complete: yes; 800 Adam steps selected
- Fresh/audit opened: yes; 3 worlds × 3 seeds × 2 rho × 5 methods
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Decision

**H:** Factorized condition and behavior coordinates extrapolate to held-out combinations with less storage than pair-specific interventions.

**T:** Synthetic 12×10 condition-behavior bank in a shared rank-4 64D basis; explicit pair table, tied code, diagonal factor, generic full matrix, Mirror Givens; rho=0/.1; 3 fresh worlds × 3 seeds.

**D:** PROMISING at rho=0: Mirror held-out NRMSE mean 9e-6 (max 7.9e-5), 3,297B vs exact pair table 4,773B and generic full matrices 4,001B (NRMSE mean 2.8e-5). Decode wall was ~0.063ms vs generic ~0.021ms. At rho=.1 Mirror error rose to .216.

**C:** Teacher functions were constructed from the same Givens family; generic matrices also fit nearly exactly. Runtime regressed on CPU.

**U:** Natural/pretrained CAST interventions and optimized deployment kernels remain untested.

## Next action

Commit result and verification, update registry/claim/board, run integrity checks, then continue to MA-516/517.

## Blockers

None.

## Decisions / rulings

Fresh worlds were evaluated once at the development-selected 800-step setting. No fresh tuning was performed.
