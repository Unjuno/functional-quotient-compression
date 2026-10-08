# MA-951 — Task-conditioned Mirror exceptions to C4 equivariance

Status: SCREENING
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Prior art: PA275 (G-CNN), PA276 (steerable CNNs)

## H — falsifiable hypothesis

A shared exact C4-equivariant operator plus a low-description task-specific Mirror exception can recover useful non-equivariant task functions with fewer actual bytes than independent maps, while preserving an explicit measure of equivariance error. Compare against an unrestricted rank-2 residual basis.

## T — frozen scope

Finite-group linear operator screen: four cyclic coordinates, four task IDs, aligned Givens residual teachers plus an independent residual boundary, three development/audit worlds and three model seeds. The methods are equivariant-only, Mirror exception, rank-2 basis exception, hard-shared residual and independent maps. Draw31 replay is saved at `source/draw31_exclusions.json`. Full seeds/gates are frozen in `PROTOCOL.json`.

## D — decision

Pending development and fresh audit.

## C — strongest counter-hypothesis

A plain rank-2 basis can represent the same aligned residual family with fewer runtime operations. Independent maps may be required for unrelated task functions. Exact equivariance alone may already solve tasks that do not need symmetry breaking.

## U — boundaries

This is not an image CNN experiment. C4 transformed copies are measured as symmetry checks, never counted as additional learned functions.
