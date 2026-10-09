# MA-299 status

- Status: FAIL for Mirror-specific value; adaptive allocation signal observed
- Branch: `research/ma-299-split-on-share-mirror-20261009`
- Frozen protocol: `64695862`
- Fresh: complete, 3 worlds × 3 seeds; 36 rows
- Verification: complete

## H
A shared basis plus task codes can delay physical split until a task exceeds a fixed reconstruction threshold, saving bytes while retaining quality.

## T
Synthetic D=256 task stream: six shared low-rank tasks then four tasks with sparse novel directions. Compared independent sparse tasks, never-split, thresholded Mirror and generic PCA with same threshold. Fresh worlds 29910–29912 × seeds 0–2; payload charges indices, values, factors and metadata.

## D
FAIL for Mirror specificity. Adaptive split triggered on all four novel tasks; payload ~11,542 B versus independent 20,590 B (~44% reduction) at query NRMSE ~2.09e-7. Never-split used 3,295 B but NRMSE ~0.88. Generic PCA matched Mirror.

## C
The constructed low-rank-then-sparse stream makes split decisions unusually easy; generic PCA plus residual threshold reproduces the behavior.

## U
No learned continual task stream, optimizer, routing, SupSup, SETA, forgetting, or natural-task evidence.
