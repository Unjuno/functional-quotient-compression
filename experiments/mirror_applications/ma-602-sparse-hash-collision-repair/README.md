# MA-602 — Sparse private collision repair

Status: **FAIL (development; fresh sealed)**. Prior art PA120 HashedNets. This tests where private parameters become necessary to undo hash collisions.

## H — Hypothesis

Private residuals only at duplicate logical connections, optionally combined with a Givens View, would recover hash4096 accuracy at or below its bytes and beat a same-count random exception control.

## T — Execution

64→128→10 MLP on sklearn digits; hash2048 and hash4096 native controls; Givens-only, collision-selected private residual with/without Givens, same-count random residual+Givens, rank-one residual, and dense upper. Dev seeds 60201/60202; 800 AdamW updates. Each exception index (uint16) and value (FP16) is serialized and charged. Fresh 60211–60213 stayed sealed.

## D — Decision

**FAIL.** The 2,048-bucket maps contain 6,181/6,180 duplicate logical connections out of 8,192 (75.4%). Collision residual+Givens therefore costs 34,995/34,991 B versus 13,791 B for native hash4096. Accuracy is .9822/.9800, missing the required +1pp over hash2048 and same-count random exceptions; no-view collision residual is .9778/.9800, within .5pp of the Givens version. Random residual+Givens is better on seed 60201 (.9844) and tied on seed 60202 (.9800). The private state is too dense to count as a sparse repair, and Givens has no demonstrated incremental value.

## C — Strongest counter-hypothesis

At 2,048 buckets, most logical connections collide, so exceptionizing all duplicates effectively restores most of the dense matrix and loses the storage advantage. Native 4,096 buckets costs far less and accuracy is similar.

## U — Boundaries

One small image task and fixed-update training. No natural-language or near-convergence capacity result.

## Facts / interpretation / hypothesis

- **Fact:** Hash2048 uses 9,695 B at .9800/.9800 accuracy; hash4096 uses 13,791 B at .9800/.9778. Collision+Givens uses 34,995/34,991 B at .9822/.9800. Exception indices and values each consume about 12.36 KB. Fresh stayed sealed.
- **Interpretation:** Sparse private repair is only compact if harmful collisions are a small minority. At this bucket ratio, more than three quarters of connections need exceptions, which is not sparse.
- **Hypothesis:** Increasing the bucket count is more efficient than paying private state for each duplicate connection; a future repair should identify a much smaller set of task-harmful collisions rather than all duplicate addresses.
