# MA-785 — Residual re-specialization around merged experts

Status: **FAIL** for Mirror-specific value.

## Fact

A synthetic 16-expert bank is grouped into four routing clusters, with a rank-2 shared residual plus private heterogeneity `σ∈{0,.05,.20}`. At σ=0, merge centers plus a rank-2 basis reconstruct heldout experts at nRMSE `5.16e-8` using 4,462 B vs 9,189 B independent (51.4% fewer); merge-only is 2,812 B but has nRMSE .0779. Mirror and native rank-2 decompression have identical payload hashes and metrics. At σ=.05, shared rank-2 error rises to .217. Private rank-8 reduces this to .0377 at 17,012 B, above the independent 9,176 B; private rank-12 restores error to `3.33e-8` but costs 22,933 B. At σ=.20 the shared rank-2 error is .588; private rank-8 remains .0944 and rank-12 costs 22,917 B vs 9,195 B independent for exact recovery.

## Interpretation

Merged centers and shared coordinates efficiently recover experts from a deliberately aligned orbit. The gain disappears quickly with private variation: residual capacity must grow until it erases the storage benefit. Standard low-rank decompression is exactly the same function/state as Mirror. Inference remains one dense matrix application after paid one-time reconstruction, while reconstruction MACs and bytes grow with private rank.

## H / T / D / C / U

- **H:** Routing-guided merged centers plus compact Views recover member experts using fewer bytes; private residual rank reveals where shared state stops sufficing.
- **T:** Seed 78501; 16 synthetic 12×12 experts in four known clusters; shared basis rank 2; heterogeneity σ=0/.05/.20; private ranks 0/2/4/8/12; every fourth expert held out for error. Independent, merge-only, Mirror/native shared basis, and private low-rank controls. Actual compressed NPZ bytes, output nRMSE and reconstruction/inference MACs recorded; no optimizer updates.
- **D:** **FAIL** for Mirror-specific value. Aligned σ=0 saves 51.4% bytes with near-exact recovery, but native factorization exactly aliases. At σ=.05, the private rank needed for ≤.02 error exceeds independent bytes.
- **C:** The positive case is generated from a rank-2 shared residual; the private component is unstructured and not compressible by that basis.
- **U:** Actual M-SMoE routing-derived merges, trained nonlinear experts, expert imitation/NLL, fresh checkpoints, and methods for selecting private rank online.

## Scope

This is an oracle linear checkpoint-compression screen, not a trained M-SMoE/MC-SMoE result. The heldout split measures operator reconstruction; all checkpoint weights are available to the compressor. Fresh data stayed sealed after exact native low-rank alias.
