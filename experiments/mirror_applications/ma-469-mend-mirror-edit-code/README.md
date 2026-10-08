# MA-469 — MEND generates Mirror edit codes

Status: SCREENING (protocol/source frozen; development pending)  
Evidence lane: MECHANISM/STORAGE/EDIT_SUCCESS/LOCALITY/RUNTIME  
Branch: `research/ma-469-mend-mirror-edit-code-20261008`  
Base commit: `b92524d`  
Prior art: PA86 (MEND), PA87 (ROME)

## H — Hypothesis

A low-rank support-gradient signal can be mapped to two shared edit coordinates and decoded into useful per-edit weight changes with lower actual bytes than a MEND-style full-update editor, while preserving heldout edit success and locality. Direct native low-rank and analytic ROME controls isolate the contribution of the Mirror code.

## T — Frozen setup

A frozen 6×6 linear model receives 20 sequential edit requests. Each request has 24 support examples and 128 edit-query examples; locality uses 256 unrelated inputs. The target update lies in a seeded shared two-matrix subspace. Sixteen requests train the editor; four heldout requests are generated from support gradients only. Models train 2,200 Adam updates, batch size eight. Development seeds: 46901 and 46902; fresh seeds 46911–46913 remain sealed. The complete edit-system payload includes the base, editor, persistent per-edit update/code state, and key vectors. The operation proxy charges a 6×6 model application plus the editor input/output linear maps and (for Mirror) the shared-basis decode; query wall time and edit-generation wall time are timed separately.

## Prior-art delta

PA86's MEND maps low-rank edit-gradient decompositions into parameter updates. MA-469 compares direct full-matrix generation to compact code generation over a shared update basis. PA87's analytic rank-one update and an independent support-fit upper bound are included.

## Results

Pending frozen development runs.
