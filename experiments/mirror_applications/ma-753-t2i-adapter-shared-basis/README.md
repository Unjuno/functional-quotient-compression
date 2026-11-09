# MA-753 — T2I-Adapter bank to shared adapter basis

Status: **FAIL** (scoped synthetic Mirror-specific claim)

## Fact

On 12 synthetic adapters, a shared rank-2 basis preserves heldout condition maps (nRMSE `9.76e-8`) and pairwise compositions (`1.08e-7`) using 2,006 B versus 8,811 B for the independent full bank, a 77.2% reduction. Its MAC proxy is 196,608 versus 98,304 for evaluating the same individual and pair requests with dense adapters. An independent rank-2 SVD control is 3,338 B but has heldout nRMSE `.763`. The native shared PCA control and Mirror have identical 2,006 B payload hashes, errors and MACs.

## Interpretation

This designed low-dimensional adapter family supports substantial bank compression and exact pairwise composition, at the cost of doubling the matrix-multiplication proxy. The function and payload are fully explained by ordinary shared PCA plus condition coefficients. No Mirror-specific gain is established; the synthetic subspace is known to be shared by construction.

## Hypothesis result

**FAIL** for Mirror-specific value and the strict Pareto gate, despite the useful generic shared-basis compression signal. Fresh/audit was not opened after exact native alias.

## Frozen protocol / scope

PA191 motivates a bank of plug-and-play spatial guidance adapters. We generate 12 matrix-valued adapters from a shared rank-2 function subspace. The basis is estimated using 8 training conditions; codes for 4 heldout conditions are fit from 8 calibration examples each. We compare independent full matrices, independent rank-2 per-adapter SVD, native shared PCA and Mirror shared views. Complete NPZ bytes, heldout operator error, pairwise-add composition error, active MAC proxy and wall time are reported. The amendment correcting the calibration design-matrix axis order and consistent MAC accounting is disclosed; the initial invalid run is preserved separately.

This does not evaluate trained T2I-Adapters, a frozen diffusion backbone, generated images, prompt adherence or aesthetic quality. No accelerator runtime or capacity claim is made.

## H / T / D / C / U

- **H:** Shared adapter basis plus compact condition codes preserves heldout controls and composition with at least 25% fewer bytes and a better Pareto point than independent rank-2 adapters.
- **T:** Seed 75301; 12 adapters; 8 basis-training conditions; 4 heldout conditions × 8 calibration examples; rank 2; full, independent low-rank, native PCA and Mirror controls; zero optimizer updates.
- **D:** **FAIL.** The shared PCA/Mirror payload saves 77.2% vs full adapters but uses 2× MACs, and exactly aliases native PCA. Independent rank-2 is smaller than full but loses quality.
- **C:** The generated adapter bank lies in a shared rank-2 function subspace by construction; ordinary PCA is sufficient to compress and compose it.
- **U:** Natural adapter updates, basis estimation from realistic training, additional controls, diffusion image quality and deployed accelerator costs.
