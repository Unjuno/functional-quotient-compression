# MA-951 — Task-conditioned Mirror exceptions to C4 equivariance

Status: FAIL
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Prior art: PA275 (G-CNN), PA276 (steerable CNNs)

## H — falsifiable hypothesis

A shared exact C4-equivariant operator plus a low-description task-specific Mirror exception can recover useful non-equivariant task functions with fewer actual bytes than independent maps, while preserving an explicit measure of equivariance error. Compare against an unrestricted rank-2 residual basis.

## T — frozen scope

Finite-group linear operator screen: four cyclic coordinates, four task IDs, aligned Givens residual teachers plus an independent residual boundary, three development/audit worlds and three model seeds. The methods are equivariant-only, Mirror exception, rank-2 basis exception, hard-shared residual and independent maps. Draw31 replay is saved at `source/draw31_exclusions.json`. Full seeds/gates are frozen in `PROTOCOL.json`.

## D — decision

**FAIL overall for the frozen gate because the storage ratio narrowly misses its 60% cutoff; the aligned quality gate passes.** On 3 fresh audit worlds, Mirror task accuracy was 0.99996 vs 0.99973 for independent maps, and Mirror query MSE was below `1e-6`. The actual adapter payload was 272 B vs 424 B independent (64.2%, vs the required <=60%). The unrestricted rank-2 basis used 360 B with accuracy 0.98969 and MSE 0.000759. On independent residual worlds, Mirror accuracy fell to 0.748 and MSE rose to 0.429; independent maps stayed at 0.99959 accuracy and `4.3e-5` MSE. This supports a boundary: aligned task views are compact, unrelated task functions need private state.

### Fact

The equivariant-only C4 model had exactly zero C4 equivariance error but only 0.515 aligned task accuracy. Mirror's aligned equivariance error was 0.788 while accuracy approached 1.0, showing the task exceptions recover symmetry-breaking behavior. Mirror used 24 parameters and 272 serialized bytes; independent maps used 64 parameters and 424 bytes. Eager CPU batch-1 latency was 0.161 ms for Mirror, 0.032 ms for rank-2 basis and 0.0065 ms for independent maps. Mirror's implementation materializes every task view before selecting one, so this timing is unoptimized.

### Interpretation

The aligned task structure permits large logical variation with a small residual code, while the exact equivariant base alone loses task accuracy. The rank-2 basis reaches slightly lower accuracy than Mirror but is also smaller than the independent bank; the Mirror payload is smaller still, yet misses the preregistered 60% byte gate. The large CPU latency is a real property of this eager implementation.

### Hypothesis

The aligned teacher was constructed from the same Givens family, so this does not establish a general G-CNN improvement. A fused selected-view kernel might reduce latency, and an image-domain equivariant benchmark could change the frontier; both remain untested. No capacity claim is made from fixed-budget learning.

All 360 task-level audit rows and actual payload hashes are preserved. A second audit execution exactly reproduced query MSE, accuracy, equivariance error and payload bytes.

## C — strongest counter-hypothesis

A plain rank-2 basis represents the aligned family compactly with lower runtime; independent maps are required for unrelated residuals. Exact equivariance alone can miss tasks that need symmetry breaking.

## U — boundaries

This is not an image CNN experiment. C4 transformed copies are measured as symmetry checks, never counted as additional learned functions.
