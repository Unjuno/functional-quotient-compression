# MA-571 — Deep Equilibrium Mirror coordinate

Status: **FAIL for storage/Mirror-specific gates**  
Branch: `research/ma-571-deq-equilibrium-mirror-20261009`  
Base commit: `68dc82f2`  
Prior art: PA124 Deep Equilibrium Models

## H

A compact task coordinate inside a shared contractive equilibrium operator yields multiple stable fixed points with fewer bytes and no material increase in solver iterations versus direct task forcing or independent equilibria.

## Frozen mechanism screen

Two-dimensional contractive affine DEQ with eight task equilibria placed on a fixed rotation orbit. Compare one shared hard-tied equilibrium, a per-task Mirror angle that rotates a shared anchor state, direct coefficient/bias coding, and independent operator/fixed-point storage. Use fixed-point iteration from zero at tolerance 1e-6 and report equilibrium nMSE, convergence iterations, contraction/stability, actual NPZ bytes, function MACs and wall time. Two development seeds; no training updates. The direct coefficient basis is the specificity control.

PASS requires all tasks stable, nMSE ≤1e-6, at least 20% fewer bytes than independent storage, no >10% increase in iterations vs direct controls, and ≥10% byte improvement over direct coefficient code. FAIL on direct alias, instability or quality miss. Fresh sealed on alias.

## H / T / D / C / U

- **H:** One equilibrium map plus a small View creates many stable task equilibria.
- **T:** 2D contractive affine operator, eight task views, two development seeds; eight rows.
- **D:** FAIL. All task equilibria are stable and converge in mean 22 iterations; Mirror uses 970B, independent fixed-point storage 907B, direct coefficients are byte/function-identical. Hard tying is 713B but nMSE ~2.0.
- **C:** Direct task-specific bias, direct coefficients, and independent fixed points/operators.
- **U:** Nonlinear learned DEQ, implicit-gradient training, natural tasks and accelerator solver costs.
