# M-SMoE merged expert shared-code diagnostic — 2026-11-09

## Evidence

- MA-785: four synthetic routing clusters, 16 experts, rank-2 shared residual. Aligned heldout nRMSE `5.2e-8` at 4,462 B vs 9,189 B independent; exact native low-rank alias. At heterogeneity .05, rank-8 private residual remains above the .02 error gate and rank-12 costs 22,933 B vs 9,176 B independent.
- MA-786: eight frequency-weighted clusters, 32 experts, rank-2 residual. Aligned heldout nRMSE `6.7e-8` at 11,106 B vs 31,260 B independent; exact native PCA alias. At heterogeneity .10, rank-8 error is .171 and rank-16 costs 74,214 B vs 31,177 B independent.

## Interpretation and ruling

The positive compression signal is produced by designed low-rank residual structure and exactly matches ordinary low-rank/PCA decompression in two consecutive merged-expert candidates. Private residuals recover heterogeneity only after exceeding independent storage. Pause further M-SMoE merged-center plus shared-basis member-code variants until they propose a mechanism distinct from PCA/low-rank decompression or use actual routing-derived trained checkpoints with an end-task advantage. This pause does not cover Expert Upcycling or router compression.

## H / T / D / C / U

- **H:** A merged center plus small member codes recover expert distinctions.
- **T:** MA-785 and MA-786 synthetic linear operator screens with explicit private rank sweeps and actual NPZ payload bytes.
- **D:** **FAIL** for Mirror specificity; repeated exact native low-rank aliases; heterogeneous private state erases storage savings.
- **C:** Standard PCA/SVD explains the full effect; the synthetic teachers are built to have low-rank residual structure.
- **U:** Trained MoE checkpoints, actual M-SMoE routing groups/frequencies, and downstream routed quality.
