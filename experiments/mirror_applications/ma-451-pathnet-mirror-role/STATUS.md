# MA-451 status

**FAIL at the registered N=20 storage gate; a storage crossover appears by N=64.**

## H / T

Tested per-task Givens roles on PathNet-selected physical modules, comparing PathNet with development-selected path defaults, support-inferred Mirror roles, independent full task matrices, and one shared route. Three fresh worlds × three seeds, 64 task IDs/world/seed, four held-out path identities.

## D — Fact

At N=20, mean NRMSE: PathNet 0.186; Mirror 0.000; independent 1.88e-07. Mirror recovered every teacher angle grid value. Actual bytes/task at N=20: PathNet 155.05B; Mirror 158.25B; independent 146.05B. At N=64, Mirror was 63.45B/task vs independent 89.64B/task. Both PathNet and Mirror use 8 physical modules. Query timing at N=64 averaged 0.105ms PathNet and 0.393ms Mirror; six-candidate support search costs 3,840 MAC/task in addition to 8,192 query module MAC.

## Interpretation

The role coordinate recovers exact held-out functions beyond a fixed PathNet path role, but task-specific codes do not amortize enough by N=20 under serialized accounting. They beat independent-module bytes by N=64. Support-time search raises compute and wall time. This is only an aligned synthetic mechanism test.

## C — Strongest counter-hypothesis

Independent modules are cheaper at moderate multiplicity, and ordinary PathNet is faster because it avoids support-based role search. The large-N crossover may depend on the deliberately low-dimensional Givens family.

## U — Unknown

Learned path routing, nonlinear modules, and natural task transfer remain untested. Path IDs were supplied and charged; route discovery was not evaluated.
