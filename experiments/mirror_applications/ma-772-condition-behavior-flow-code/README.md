# MA-772 — Factorized condition × behavior codes inside affine coupling

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On four heldout condition-behavior pairs, product-factorized codes reach transform nRMSE `6.08e-9` and inverse-cycle error `4.77e-7` after fitting 2,500 updates. The product factor state uses 1,071 B. Direct per-pair codes achieve exact output with a smaller actual NPZ payload of 715 B and no fit updates. An additive condition+behavior code uses 1,071 B but heldout code RMSE `.543` and transform error `.0948`, above the `.02` gate. Native product factorization and Mirror have identical hashes, bytes, outputs and cycle errors.

## Interpretation

Separating a context gate from a behavior code supports heldout composition in this rank-1 synthetic family; the additive control misses the interaction. However, actual serialized direct pair codes are smaller than the product factorization, and ordinary conditional RealNVP/low-rank product state exactly explains the Mirror behavior. Parameter factorization did not produce an inference payload win here.

## Hypothesis result

**FAIL** for Mirror-specific value and the storage gate. Fresh/audit was not opened after exact native factorization alias and direct-code byte advantage.

## Frozen protocol / scope

PA199 and PA101 motivate coupling transforms and condition-triggered activation views. A 4×4 grid of condition and behavior factors produces a scalar affine-coupling code via their product. Twelve pairs fit the factors; four combinations are held out. Compare direct per-pair codes, additive condition+behavior factors, native product factors and Mirror. Evaluate 256 random 2D inputs/pair, actual NPZ bytes, transform nRMSE, inverse-cycle error, MAC proxy, fit updates and wall time.

The teacher is analytically rank-1 and the native control is the same product factorization. This is not a trained conditional flow, realistic behavior control, or higher-dimensional likelihood result.

## H / T / D / C / U

- **H:** Separating when-to-act from what-behavior predicts unseen condition/behavior combinations with fewer bytes while preserving invertibility.
- **T:** Seed 77201; 4 conditions × 4 behaviors; 12 train pairs / 4 heldout; 2D coupling; 2,500 factor updates; direct and additive controls.
- **D:** **FAIL.** Product state error `6.08e-9` and cycle error `4.77e-7`, but uses 1,071 B vs 715 B for direct pair codes; exact native factorization alias.
- **C:** Native product factorization captures the apparent View; compressed direct task codes are smaller than factored state.
- **U:** Learned factors, noisy/incomplete conditions, natural flows, interaction rank >1 and deployment cost.
