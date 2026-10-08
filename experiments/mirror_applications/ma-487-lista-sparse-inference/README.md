# MA-487 — LISTA inference for sparse function coordinates

Status: SCREENING  
Branch: `research/ma-487-lista-sparse-function-inference-20261008`  
Prior art: PA93 (LISTA)

## H — Hypothesis

A trained fixed-depth LISTA predictor can infer sparse functional coordinates from a presented function matrix with fewer inference operations than exact three-step OMP while retaining heldout RMSE <=0.01 and at most four active atoms.

## T — Frozen protocol

Use the MA-486 rank-32 orthonormal dictionary and 3-sparse function family. Compare direct top-3 projection, exact OMP and learned LISTA at depths 1/2/4. Fit only on the 128 training functions. Report model payload bytes, steps, active atoms, heldout reconstruction error, encoder/decode/query compute and wall times. Function matrices are encoder inputs; these are not included in codebook payload claims.

## D — Pending frozen development runs

Fresh seeds 48711–48713 remain sealed unless frozen gates pass. A successful LISTA result is an inference-method result; it is not attributed to Mirror if native sparse inference controls explain it.
