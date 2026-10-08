# MA-297 — Mirror codes inside a SETA-style shared sparse subspace

Status: SCREENING  
Evidence lane: MECHANISM / CONTINUAL / STORAGE  
Base commit: e259f27 (worker-ready baseline)

## H — falsifiable hypothesis

After one fixed shared-subspace discovery step, a task-specific Mirror coordinate over its retained sparse atoms can preserve aligned task functions with lower actual inference bytes than task-private sparse residuals, without increasing earlier-task error by more than 0.01. Independently generated task functions should reveal when private sparse parameters are required. Mirror-specific value requires beating the matched shared-basis coefficient control at comparable bytes and compute.

## T — protocol

Candidate selection was randomized over the then-UNTESTED P0 second-expansion queue: `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299]`. Python `secrets.randbelow(10)` returned index `8`; selected MA-297. UTC date: 2026-10-08. No live MA-297 branch existed after `git fetch origin --prune`.

The harness uses NumPy/SciPy CPU linear maps. One shared physical matrix and a discovered sparse feature basis are frozen before sequential acquisition of four known task IDs. Conditions are (a) aligned coefficient combinations inside the discovered shared subspace and (b) independent task teachers. Compare SETA-style shared codebook/no task code, Mirror rotation coordinate over paired shared atoms, ordinary per-task sparse coefficients on the same basis (non-Mirror coefficient control), matched low-rank residual, and independent full task weights. Development worlds select learning rate and sparse support size only. Fresh worlds are locked independently.

This is a screening abstraction of SETA's discover-shared/private-then-compress decomposition, not a reproduction of the full SETA model or routing method.

## Mirror insertion

The coordinate is one learned angle per task, applied as a 2D rotation to a pair of frozen shared sparse atoms. Shared atoms are paid once. Each task also pays its task ID and angle record. The matched ordinary control stores two unconstrained coefficients for that same atom pair. Sparse private residual entries, indices, and values are all charged.

## Gates

PASS for the narrow aligned mechanism only if all three fresh worlds have mean seen-task MSE <=1.10x the matched ordinary coefficient control, earlier-task MSE increase <=0.01, and <=0.5x its incremental task bytes; Mirror must also improve the byte/quality frontier over that control or result is not Mirror-specific. FAIL if Mirror misses quality/retention or does not beat the coefficient control. Independent-task results locate the private-parameter boundary. These fixed-update results cannot establish capacity.

## Fact / interpretation / hypothesis

**Fact:** see locked result tables and the final decision below.  
**Interpretation:** limited to this synthetic sparse-linear screen.  
**Hypothesis:** if a shared sparse subspace contains task variation as a low-dimensional orbit, a compact code can delay private allocation; arbitrary task maps will not.

## C — strongest counter-hypothesis

The teacher may align too closely with the selected two-atom rotation, while the ordinary coefficient representation is already the exact native coordinate system. In that case it can match or beat Mirror and disprove a Mirror-specific benefit.

## U — unresolved

No language model, learned router, full SETA reproduction, near-convergence capacity frontier, or GPU runtime was tested. Synthetic known-task linear regression does not establish general continual-learning performance.
