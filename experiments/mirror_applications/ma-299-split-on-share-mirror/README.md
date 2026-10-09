# MA-299 — Split-on-Share Mirror code allocation

Status: SCREENING. Prior art PA30 SETA.

## H

A shared rank-3 task basis can host incoming skills as small Mirror codes; allocate private sparse residuals only after a fixed relative error threshold is crossed, delaying storage growth while preserving novel tasks.

## T

Synthetic sequential linear tasks, D=256, ten tasks, six initially shared and four later novel sparse tasks. Fixed split threshold 0.15. Compare independent sparse vectors, never-split basis, adaptive Mirror split, and generic PCA split. Dev worlds 29900–29901; fresh 29910–29912, three seeds each. Actual serialized bytes charge basis, codes, private indices/values, and metadata.

## D

Pending development/fresh run.

## C

This may only reproduce generic low-rank compression plus an ordinary residual threshold; novel sparse task detection may be too easy.

## U

Synthetic linear task storage screen only; no faithful SETA training, routing, or retention.

Development: the fixed threshold split all four later novel tasks and no initial shared tasks. Adaptive Mirror payload was 11,543 B vs 20,590 B for independent sparse vectors (about 44% lower), with query NRMSE ~1.9e-7. Generic PCA split was effectively identical at 11,539 B. Never-split was smaller but query NRMSE ~0.89. Sparse residual indices use int16 and values float32; numerical residuals below 1e-6 are omitted as reconstruction tolerance.
