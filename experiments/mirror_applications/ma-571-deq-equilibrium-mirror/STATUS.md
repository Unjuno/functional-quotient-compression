# MA-571 status

- Status: **FAIL for Mirror-specific benefit and frozen storage gate**
- Branch: `research/ma-571-deq-equilibrium-mirror-20261102`
- Protocol freeze: `a2f612e8`; A1 inference-state/MAC correction preserved.
- Fresh seeds: 57111–57113.

## H / T / D / C / U

**H:** A compact role-conditioned view inside a shared contractive fixed-point operator recovers distinct equilibria with lower bytes than independent role operators, while solver cost remains controlled.

**T:** 8D contractive equilibrium, 4D inputs, two balanced roles, tolerance `1e-7`, max 100 solver iterations. Compared no-view, Mirror, direct angle, rank-2 input correction and independent role input maps. Three fresh worlds; zero optimizer updates. A1 loads stored solver settings, role codes and basis indices and charges input projection, each solve iteration, output projection and view MACs.

**D — FAIL.** Fresh Mirror mean/max NRMSE was `1.1e-08/1.14e-08`; direct-angle was identical. Mirror mean payload was `2834 B` vs `2987 B` independent role maps, about `94.9%` (frozen gate <=50%). The direct angle code exactly aliases Mirror in outputs, bytes and solver MACs. Mean solver iterations were `7.04`; maximum residual stayed below `1e-7`.

**C:** The teacher inserts exactly the same affine Givens role angles as the candidate. The direct-angle control is therefore a full function/byte/compute alias, while the storage gate also misses by a wide margin.

**U:** Trained DEQ models, natural tasks, gradient/implicit-differentiation training, GPU, energy, and whether more varied equilibria can share this operator.

## Facts

- Fresh NRMSE mean: native `0.1351`, Mirror/direct `1.1e-08`, rank-2 `0`, independent `1.22e-16`.
- Actual serialized bytes mean: native `2427 B`, Mirror/direct `2834 B`, rank-2 `3047 B`, independent `2987 B`.
- Mirror/direct mean active MAC proxy `66368`; it includes input/output projections, fixed-point iterations and view. End-to-end CPU mean `0.7771 ms`; Python implementation specific.
- Solver residuals, bytes/hashes and direct-control alias replay verified; zero optimizer updates.

## Interpretation

A role view can select an equilibrium function on this deliberately matched contractive task, but the simple direct angle code is the same operation and the serialized payload is 94.9% of the independent-role map state. This does not support a Mirror-specific DEQ benefit or a trained-model/capacity claim.
