# MA-319 — Tucker matrix-bank Mirror layer coefficients

Status: FAIL. Prior art: PA35 Tucker/matrix-bank parameterization.

## H

A shared matrix bank plus free per-layer coefficients already compresses layer matrices. Test whether Givens Mirror addresses reduce the coefficient state further and improve total storage-quality frontier beyond that native control.

## T

Synthetic bank: K=8 matrices of 32×32, shared by L=64 logical layers. Compare full independent matrices, Tucker bank with free coefficients, shared Mirror vector plus per-layer Givens angles, and generic rank-2 PCA coefficients. Test aligned orbit and independent coefficients. Development worlds 31900–31901; fresh 31910–31912; three seeds each. Serialize fp16 and evaluate after deterministic decode; all shared-bank state is charged.

## D

FAIL. Fresh aligned Mirror is 16,699 B / functional NRMSE 5.03e-4 vs free Tucker 17,559 B / 2.93e-4, a 4.9% saving that misses the 10% gate. Generic PCA is 16,887 B / 3.56e-4. Independent coefficients require free Tucker; Mirror NRMSE is 1.036.

## C

The aligned coefficients are generated from the same Givens orbit; generic PCA may match the result, and the shared bank may dominate total bytes.

## U

Synthetic matrices only; no natural Transformer layer weights or language-model quality.

Development: on aligned orbit coefficients, Mirror uses 16,699 B / functional NRMSE 5.2e-4 vs free Tucker 17,559 B / 2.8e-4 (4.9% fewer bytes, below the 10% gate) and generic PCA 16,887 B / 3.6e-4. The common bank dominates payload. Mirror code fitting takes ~3 ms vs PCA ~0.3 ms. Independent coefficients are represented accurately by free Tucker (NRMSE ~2.9e-4), while Mirror (1.07) and PCA (~0.79) fail. Fresh tests frozen conditions.

## Fresh result / decision

FAIL. Fresh aligned results reproduce a 4.9% byte saving versus free Tucker, below the 10% gate; generic PCA is slightly larger but more accurate. Independent coefficients need free Tucker state. The shared matrix bank dominates total serialized bytes.
