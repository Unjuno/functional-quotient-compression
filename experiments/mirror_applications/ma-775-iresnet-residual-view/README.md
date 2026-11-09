# MA-775 — Contractive i-ResNet residual task views

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

Across eight heldout 4D residual blocks, a rank-2 shared basis uses 970 B vs 1,927 B for independent matrices (49.7% fewer) and yields output nRMSE `1.86e-8`. The fixed-point inverse converges for all heldout blocks, with max recovery error `4.77e-7` in a mean 8.75 iterations. Maximum teacher Lipschitz norm is `.421`; the minimum sampled Jacobian singular value is `.793`. Basis application costs 16,384 MACs vs 8,192 independent (2×), plus 256 precomposition MACs. Native basis and Mirror have identical payload hash, bytes, quality, inversion and operations. Hard tying and scalar FiLM have heldout output errors `.138` and `.092`.

## Interpretation

The synthetic contractive family permits nearly exact invertible residual functions from a shared rank-2 basis, with a useful actual-byte reduction and convergent inverse. Computing both basis projections doubles the inverse/forward MAC proxy. Ordinary native low-rank factorization exactly explains the function and storage; no Mirror-specific gain is established.

## Hypothesis result

**FAIL** for Mirror-specific value and strict compute/storage Pareto improvement. Fresh/audit was not opened after exact native alias. Initial unpaired-input metrics are preserved; final comparisons use paired fixed inputs.

## Frozen protocol / scope

PA200/PA48 motivate contractive residual invertibility and reversible computation. Twenty-four linear residual maps in 4D are generated from two shared matrices. SVD basis is fit on 16; codes on eight heldout rules use 32 calibration samples/rule. The fixed-point inverse solves `x = y - F(x)` to tolerance `1e-6` for 500 iterations max. Compare independent matrices, hard tie, scalar FiLM, native rank-2 basis and Mirror. Record complete NPZ bytes, output error, inverse errors/iterations, Lipschitz and Jacobian margins, MACs and time.

This is a synthetic linear contractive operator screen. It does not test trained i-ResNets, image models, reversible-memory training or accelerator runtime.

## H / T / D / C / U

- **H:** A small residual code family yields multiple invertible blocks without storing independent residual matrices.
- **T:** Seed 77501; 24 blocks; 16 basis-fit / 8 heldout; dimension 4; rank 2; 32 calibration examples; fixed-point inversion.
- **D:** **FAIL** for Mirror-specific value. Actual payload saves 49.7% and all inverses converge, but compute doubles and native factorization exactly aliases Mirror.
- **C:** Standard rank-2 matrix factorization is sufficient; the teacher is generated in that subspace.
- **U:** Learned residual factors, nonlinear i-ResNet, stochastic inverse stability, real tasks and deployed inverse runtime.
