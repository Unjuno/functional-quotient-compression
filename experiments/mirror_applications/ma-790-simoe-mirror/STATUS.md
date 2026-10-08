# MA-790 status

Status: FAIL — Draw28, development selection and audit protocol were frozen before audit query generation.

## H

Factorized sparse coefficient coordinates can recover unseen combinations with fewer actual coefficient-state bytes than 16 free SIMoE rows; any Mirror-specific advantage must beat equal-rank direct low-rank coefficients.

## T

Three coefficient worlds × three model initializations; 12 fixed linear anchors, 16 factor-pair tasks, even-parity training/development and odd-parity audit. Each method was trained for up to 2,000 updates using frozen development checkpoint selection. Audit has 8 held-out tasks per world/init. Native sparse SIMoE adapts the independent held-out coefficient row on 64 support examples; factorized methods are evaluated zero-shot. Full row-level data are in `artifacts/metrics_audit.csv`; settings and development digest are in `source/AUDIT_FREEZE.json`.

## D

**FAIL for the frozen aligned composition gate.** Mean aligned held-out query MSE: Mirror 0.250522, ordinary low-rank 0.217821, native SIMoE support-300 0.000064. By world, Mirror met the ≤1.05× low-rank criterion in only 1/3 worlds. Mirror coefficient state: 720 B / 928 B free-row reference (77.6%, storage subgate passes); ordinary low-rank state is 680 B. Full payload including shared anchors: 7,424 B Mirror, 7,384 B low-rank and 7,632 B native. Off-orbit MSE: Mirror 0.696693, low-rank 0.702974, native support-300 0.000096. Audit training wall sum 1,098.634 s; coefficient-only CPU batch-1 latency means: Mirror 0.074 ms, low-rank 0.039 ms, native 0.024 ms.

## C

Sparse interpolation is already a compact expert address; ordinary low-rank factorization is a simpler, smaller control and has lower pooled aligned error in this run.

## U

This controlled synthetic mechanism screen is not an LLM or full SIMoE result. Native SIMoE uses 64-shot adaptation while Mirror is zero-shot; do not interpret that comparison as equal inference budgets. No full-network, natural routing, GPU or near-convergence capacity claim is established.
