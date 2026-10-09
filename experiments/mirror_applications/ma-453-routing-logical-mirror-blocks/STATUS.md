# MA-453 status

**FAIL for independent-block compression; PROMISING Pareto result against finite prototype routing.**

## H / T

Tested whether one shared canonical tanh block plus continuous (scale, shift) task coordinates can outperform an 8-prototype Routing Network on fresh continuous nonlinear tasks. Three fresh worlds × three seeds, with 64 tasks per seed. Development selected K=8 prototypes.

## D — Fact

At N=20, mean NRMSE: shared-only 0.253; Routing K=8 0.084; Mirror 0.0022; independent per-task block 0.0022. Actual bytes/task: Routing 97.85B; Mirror 91.65B; independent 91.65B. Mirror exactly matches independent block quality and bytes. N=64 shows the same equality. Mirror adaptation uses 5,120 support-update MAC/task vs K=8 router's 256 prototype-evaluation MAC/task in this implementation.

## Interpretation

Mirror improves the quality-storage point over finite K=8 prototype routing, but it does not compress the two scalar task parameters relative to independent per-task blocks. The apparent logical block multiplicity comes from the same (a,b) values that the private baseline stores. This does not establish Mirror-specific capacity or storage reduction.

## C — Strongest counter-hypothesis

The canonical tanh block is deterministic and the task-specific scale/shift are ordinary weights. Mirror only changes their label and does not reduce state.

## U — Unknown

Larger neural blocks, learned routers, natural task distributions, and training cost of generating reusable modules remain untested.
