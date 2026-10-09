# MA-319 — Tucker matrix-bank Mirror layer coefficients

Status: SCREENING. Prior art: PA35 Tucker/matrix-bank parameterization.

## H

A shared matrix bank plus free per-layer coefficients already compresses layer matrices. Test whether Givens Mirror addresses reduce the coefficient state further and improve total storage-quality frontier beyond that native control.

## T

Synthetic bank: K=8 matrices of 32×32, shared by L=64 logical layers. Compare full independent matrices, Tucker bank with free coefficients, shared Mirror vector plus per-layer Givens angles, and generic rank-2 PCA coefficients. Test aligned orbit and independent coefficients. Development worlds 31900–31901; fresh 31910–31912; three seeds each. Serialize fp16 and evaluate after deterministic decode; all shared-bank state is charged.

## D

Pending development/fresh runs.

## C

The aligned coefficients are generated from the same Givens orbit; generic PCA may match the result, and the shared bank may dominate total bytes.

## U

Synthetic matrices only; no natural Transformer layer weights or language-model quality.

Development: on aligned orbit coefficients, Mirror uses 16,699 B / functional NRMSE 5.2e-4 vs free Tucker 17,559 B / 2.8e-4 (4.9% fewer bytes, below the 10% gate) and generic PCA 16,887 B / 3.6e-4. The common bank dominates payload. Mirror code fitting takes ~3 ms vs PCA ~0.3 ms. Independent coefficients are represented accurately by free Tucker (NRMSE ~2.9e-4), while Mirror (1.07) and PCA (~0.79) fail. Fresh tests frozen conditions.
