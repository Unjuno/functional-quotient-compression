# MA-464 status

- Status: FAIL for routed Mirror mixture; merged outputs equivalent
- Branch: `research/ma-464-adamix-mirror-adaptations-20261009`
- Base commit: `a904b900`
- Protocol freeze: `7b0839ca`
- Fresh worlds: 46410–46412 × seeds 0–2
- Results/registry/ledger: pending final commit

## H / T / D / C / U

- H: Stochastic mixtures of Mirror views provide AdaMix-like routed quality at lower bytes and survive merge.
- T: Four 2D skill matrices, three rotation-aligned plus one off-orbit; 1,000 balanced stochastic updates; routed and merged AdaMix/Mirror, shared, and private controls.
- D (Fact): N4 routed NRMSE: AdaMix 3.04e-6 / 1,957B, Mirror 0.2235 / 2,209B. Merged NRMSE: AdaMix 0.5931 / 1,705B, Mirror 0.5966 / 1,705B. Shared-only 0.5962 / 1,641B.
- D (Interpretation): Routed Mirror loses quality and bytes; merged Mirror is equivalent to merged AdaMix and adds no value over shared-only.
- C: The off-orbit fourth skill needs private residual parameters beyond a shared rotated basis.
- U: Higher-dimensional and natural mixture tasks remain unknown.

## Next action

Commit evidence, update registry/ledger, push branch, and proceed to MA-466.

## Blockers

None.
