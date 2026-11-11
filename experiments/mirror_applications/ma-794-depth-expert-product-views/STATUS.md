# MA-794 status

- Status: FAIL (scoped synthetic screen; Mirror attribution)
- Branch: `research/ma-794-depth-expert-product-views-20261111`
- Base commit: `8817f6c7`
- Development: seed 79401; eight layers, eight experts; one checkerboard holdout world.
- Fresh/audit: sealed and unopened after exact native bilinear alias and failed development quality gate.
- Verification: focused test passes; serialized payload reconstruction roundtrip checked.

## H / T / D / C / U

- **H:** Rank-2 depth×expert product coordinates over a shared basis recover heldout roles at ≤.02 nRMSE with ≥25% fewer complete bytes than independent roles and balanced routing.
- **T:** Synthetic 8×8 role grid, dimension 8, rank-2 shared matrix basis, checkerboard holdout; independent, hard-tied, additive, product-coordinate and native bilinear controls; actual compressed NPZ payload, holdout/all-role error, balanced identity-router loads, MAC proxies, example count, CPU operator-evaluation wall time, zero optimizer updates.
- **D:** **FAIL** for Mirror-specific value: product and native bilinear have identical payload hash/functions; heldout nRMSE .0521 fails the quality gate; the additive control is 7 B smaller and has a weaker heldout error, so no strict frontier improvement.
- **C:** The task is a rank-2 bilinear teacher, so ordinary bilinear factorization explains all recovered variation. Single-seed ALS and basis/intercept estimation may contribute to residual error.
- **U:** Fresh combinations, trained experts, language loss, learned routers/balancing, optimizer effects, broad seeds, private-residual crossover and GPU runtime.

## Next action

Proceed to the next unpaused P0 candidate, MA-795, after recording this scoped negative.

## Decisions

Fresh remained sealed because exact native-method alias and failure of the frozen quality gate already reject the Mirror-specific development hypothesis. The fixed identity router establishes only load arithmetic, not learned routing.
