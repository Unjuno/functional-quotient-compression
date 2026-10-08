# MA-488 — Shared/private function dictionary

Status: SCREENING  
Branch: `research/ma-488-shared-private-function-dictionary-20261008`  
Prior art: PA94 (shared/private dictionary learning)

## H — Hypothesis

A small shared functional basis plus private residual vectors selected only for functions outside that basis will preserve heldout function behavior while reducing complete payload versus storing all matrices, and expose the heterogeneity level where private state erases the byte benefit.

## T — Frozen protocol

Use a 4-atom shared matrix dictionary and function banks at heterogeneity p=0, .125, .25, .5. Compare shared-only, thresholded shared/private residual storage, a byte-identical native shared/private control, full float matrices and int8 matrices. Private residual threshold is fixed at L2>0.10. Measure actual NPZ bytes, overall and private-only heldout RMSE, fallback count and compute. Every method pays its shared dictionary where applicable.

## D — Pending development runs

Fresh seeds 48811/48812/48813 remain sealed unless the frozen gates pass. Native shared/private identity is an attribution control; this design tests when private parameters are needed, not a new sparse coding method.
