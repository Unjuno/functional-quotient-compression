# MA-790 — Factorized Views for SIMoE sparse interpolation codes

Status: **FAIL** for the frozen quality/byte gate and Mirror-specific value.

## Fact

The screen uses 128 synthetic experts interpolated from 16 shared 8×8 anchors with two sparse coefficients per task. Raw SIMoE indices plus values use 6,011 B with exact output. Rank-1 factorization uses 5,408 B but heldout output nRMSE is .970 and coefficient-generation MAC rises from 128 to 1,040 per expert. Rank-2 uses 5,970 B with error .934; rank-8 uses 9,243 B with error .764. Rank-16 restores output error to `1.09e-7` but costs 13,564 B, 2.26× raw sparse state. Independent full experts use 31,167 B. Every factorized Mirror row exactly matches its native PCA coefficient-code control by payload hash and output.

## Interpretation

Sparse task-specific support is already a compact representation. Dense low-rank factorization blurs the sparse supports and fails quality; rank sufficient to reproduce the coefficient bank exceeds raw sparse bytes and costs substantially more decode compute. The native PCA control fully explains every Mirror factorized row.

## H / T / D / C / U

- **H:** Factorizing many sparse interpolation coefficient vectors reduces complete SIMoE expert-bank bytes while preserving heldout outputs at acceptable decode compute.
- **T:** Seed 79001; 128 experts, 16 anchors, dimension 8, top-2 sparse coefficients; 96 train / 32 heldout task IDs; SVD ranks 1/2/4/8/16. Compare raw sparse codes, native PCA, Mirror and independent full experts. Actual compressed NPZ payload bytes, heldout coefficient/output nRMSE and decode MAC recorded; zero optimizer updates.
- **D:** **FAIL.** No factor rank meets both ≤.02 heldout output error and ≥25% fewer complete bytes than sparse SIMoE. Exact reconstruction needs rank16 and costs more than twice the raw sparse payload; native PCA aliases every Mirror row.
- **C:** SIMoE's explicit sparse supports are already efficient; task coefficient vectors are deliberately sparse and largely unaligned for global PCA.
- **U:** Trained SIMoE anchors, natural task correlations, learned sparse-support prediction, end-task language quality and accelerator decode fusion.

## Scope

This is a synthetic linear anchor screen, not a trained SIMoE model or LM evaluation. Fresh remained sealed after exact native PCA attribution.
