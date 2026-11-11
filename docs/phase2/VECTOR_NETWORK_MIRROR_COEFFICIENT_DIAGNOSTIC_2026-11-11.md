# Vector-Network coefficient-View lane diagnostic (2026-11-11)

## Decision

Pause MA-801 and further direct Mirror coefficient-table variants in the MA-800/801 compositional weight-atom lane pending a mechanism distinct from native coefficient/tensor factorization. Continue to the separate Koopman regime-View family at MA-802.

## Evidence

| Experiment | Scoped fact | Attribution / frontier |
|---|---|---|
| MA-800 | Synthetic K=8 rank-1 atoms with per-input top-2 local selection; two heldout task compositions reach nRMSE 5.08e-8 and 28 atom subsets are active. | Actual factorized payload is 2,076 B vs 1,851 B independent (+12.2%); native low-rank task coefficient factorization aliases exactly. |
| MA-801 | Synthetic task×state CP coefficient teacher; checkerboard heldout pairs reach nRMSE 1.91e-8 with input-local top-2 inference and 28 active subsets. | Factorized payload 2,269 B vs 2,275 B independent (0.26% savings); native CP aliases exactly. |

Both screens provide only synthetic coefficient-family evidence. Coefficients/task-state factors are oracle generated; no trained Vector Network or learned fast-weight inference was used. Local top-2 atom selection is executed per example. Fresh combinations stayed sealed after exact native alias/storage-gate failures.

## Interpretation

Input-local atom selection does not by itself create Mirror-specific capacity. The direct coefficient coordinate and task×state tensor coordinate reproduce ordinary low-rank/CP coefficient parameterizations. Actual compressed payloads erase theoretical parameter-count savings because the independent coefficient tables are already low-entropy and factor metadata is costly. Neither experiment improves the registered storage frontier.

Resume only after a design beats native sparse atom inference and tensor/CP coefficient factorization at actual bytes, while preserving local inference quality and charging fast-weight adaptation state. The pause is limited to the direct coefficient-code lane; it does not cover other dynamic weight-generation families.
