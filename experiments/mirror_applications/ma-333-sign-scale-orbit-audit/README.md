# MA-333 — Mirror sign/scale orbit audit

Status: SCREENING. Prior art: PA47 monomial weight-space symmetries.

## H

Activation and normalization determine which monomial hidden-unit Views are exact gauges: positive scaling with inverse outgoing scaling should preserve ReLU, sign flips should preserve tanh, and those transformations should generally fail for GELU and normalized preactivations.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-hidden-unit signed scale coordinate `m` before the activation and compensates the outgoing weights, testing when the coordinate is a pure gauge and when it changes the function.

One fixed two-layer MLP is the shared object. The cheapest control is parameter reindexing/ordinary rescaling; independent weights are the storage reference. This is a symmetry audit, not a compression or task-quality claim.

## T

16-24-8 MLPs with ReLU, GELU, tanh, and LayerNorm+ReLU. Test hidden permutation, positive nonuniform scaling with inverse outgoing compensation, and signed flip with inverse compensation. Use development worlds 33321-33322 and fresh worlds 33331-33333; 4,096 deterministic Gaussian inputs/world. Serialize full base weights plus each view code; report output NRMSE, max absolute difference, bytes and isolated time. No training.

## Gates

PASS: permutation exact in all four conditions; positive scale exact for ReLU only; sign flip exact for tanh; nonmatching activation/normalization cases show NRMSE >1e-3. FAIL if these predictions fail. This only maps exact equivalence classes for the tested functions.

## Boundary

Fixed random synthetic networks; no learned tasks, utility, model merging or deployment. Approximate invariance and trainability are not assessed.
