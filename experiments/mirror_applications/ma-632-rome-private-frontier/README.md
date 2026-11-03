# MA-632 — ROME aligned/private edit frontier

Status: SCREENING; prior art PA135. Dedicated branch: `research/ma-632-rome-private-frontier-20261103`.

## H

As rank-one edit deltas depart from a shared low-rank subspace, private residual dimensions are needed to retain efficacy. This screen measures the actual bytes/quality frontier as true private rank and stored residual rank increase. Mirror-specific credit requires beating a standard concatenated low-rank update basis with per-edit coefficients.

## T

Synthetic 32D linear layer, 16 edits; 12 training edits and four held-out diagonal factor pairs. Shared component has rank 2; private component has true ranks 0, 2, 4, 8, 16. Sweep stored private rank 0, 1, 2, 4, 8, 16. Compare independent rank-one updates, shared plus private Mirror bases, and ordinary joint low-rank basis/control. Measure efficacy, one-hot key locality, complete payload bytes and compute. Fresh seeds 63211–63213 remain sealed on native low-rank alias.

## D

Pending.

## C / U

Counter-hypothesis: rank-split Mirror state is just ordinary low-rank update compression; one-hot keys make locality trivial. Pretrained factual editing and non-orthogonal key interference remain untested.
