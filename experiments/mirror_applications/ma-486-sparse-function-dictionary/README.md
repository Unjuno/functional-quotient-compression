# MA-486 — Sparse dictionary Mirror function representation

Status: SCREENING  
Branch: `research/ma-486-sparse-dictionary-functions-20261008`  
Prior art: PA93 (LISTA / learned sparse coding)

## H — Hypothesis

With a shared functional dictionary of 32 matrix atoms, three-sparse coordinates can represent heldout functions with lower complete payload and active decode work than dense codes, while retaining exact function outputs relative to the shared-atom target.

## T — Frozen protocol

Synthetic linear function bank: D=16, dictionary size 32, 3 active atoms per function, 128 training and 64 heldout functions. Compare independent matrices, dense float/int8 coefficients, sparse float/int8 OMP codes, and exact native sparse-coding controls. Every method pays shared dictionary bytes. OMP operates over the full set of generated functions; report encoding operations separately from decode and query operations.

## D — Pending frozen development runs

Fresh seeds 48611–48613 remain sealed unless frozen gates pass. Native OMP is an attribution control; sparse coefficient coding alone is not considered Mirror-specific if its bytes/output exactly alias this standard method.
