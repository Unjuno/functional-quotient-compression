# MA-786 — Merged expert clusters to logical member code bank

Status: **FAIL** for Mirror-specific value.

## Fact

Thirty-two synthetic 16×16 experts are organized into eight known clusters with Zipf-like member frequencies. At heterogeneity σ=0, routing-weighted centers plus a rank-2 shared residual basis recover heldout members at nRMSE `6.68e-8` using 11,106 B vs 31,260 B independent (64.5% fewer). Merge-only uses 8,499 B but error is .1039. Mirror exactly matches native PCA payload and metrics. At σ=.10, shared rank-2 error rises to .535. Private rank-8 lowers it to .171 at 42,967 B; private rank-16 restores error to `5.96e-8` but costs 74,214 B vs 31,177 B independent. Reconstruction MAC grows from 4,096 for shared rank-2 to 20,480 at private rank-16; dense inference MAC per expert remains 256 after reconstruction.

## Interpretation

Routing-weighted merging and shared residual coordinates are effective only when member differences align with a small shared basis. Moderate unstructured variation requires enough private factors to exceed the independent archive. The measured shared-basis code is exactly standard PCA/SVD decompression, so Mirror-specific value is not established.

## H / T / D / C / U

- **H:** Routing-weighted merged centers plus compact member Views recover cluster members with fewer bytes; private residual demand reveals the point where sharing stops helping.
- **T:** Seed 78601; 8 clusters × 4 members; dimension 16; frequency weights `[8,4,2,1]`; shared basis rank 2; heterogeneity σ=0/.10; private ranks 4/8/16; last member of each cluster held out for reconstruction error. Independent, weighted merge-only, Mirror/native PCA and private SVD controls. Actual compressed NPZ bytes and reconstruction/inference MACs recorded; no optimizer updates.
- **D:** **FAIL** for Mirror-specific value. Aligned residuals save 64.5% bytes with near-exact reconstruction, but native PCA exactly aliases. At σ=.10, rank-8 error remains .171 and rank-16 uses 2.38× independent bytes.
- **C:** The positive case is generated from a rank-2 residual subspace; unstructured private residuals defeat shared coding.
- **U:** Actual routing-derived clusters/frequencies, trained MoE end-task quality, online code selection, and natural expert checkpoint distributions.

## Scope

This is an oracle synthetic checkpoint screen, not M-SMoE/MC-SMoE training or downstream MoE evaluation. The compressor observes all expert matrices; heldout members evaluate reconstruction. Fresh stayed sealed after exact native PCA attribution.
