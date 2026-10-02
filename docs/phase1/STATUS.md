# Phase I — FQC post-training compression (paused)

Decision date: 2026-10-03. The research owner chose to preserve Phase I and move active effort to Phase II (Mirror-native conditional architectures).

This is a priority change, **not a claim that compression was solved or proved impossible**. Existing `src/`, `tests/`, `claims/`, `experiments/`, provenance and Git history are not removed or relicensed. Legacy roadmap documents remain historical unless explicitly superseded by the phase index.

The main-branch baseline is commit `41e084440e5c8525c1e1adeefd172be7f05acb10`. Its README is retained byte-for-byte as `README_before_phase2_20261003.md` (Git blob `c0faa891d11d0d198a0089a1240e524b077afa18`). Relative links in that verbatim snapshot are historical; use the original root README at that commit for live navigation.

## Closing ledger

- Functional/task-sensitive selection, nonadditive compression interactions and exact serialization remain the preserved core findings.
- The baseline main README records real-checkpoint codec engineering through T282. This phase transition does not independently rerun or recertify that historical lane.
- A quality-preserving 64x real-Transformer result was not demonstrated in that ledger; the tested simple family failed quality.
- Superiority of FQC-specific sharing over a strong non-sharing control at equal final bytes remains unestablished in the baseline ledger.
- Synthetic prototype results are not retroactively converted to real-model evidence.
- Codebook count is not an intrinsic functional-state count. Partial exact restoration is not a certified optimum; component KL cannot be added to certify joint quality.

## Resume conditions

A future resumption should start from the preserved evidence contract, verified checkpoints, exact decoded artifacts, held-out data and strong matched-byte non-sharing controls. Do not restart broad synthetic sweeps merely because they produce favorable deltas.

Current active lane: [Phase II](../phase2/README.md).
