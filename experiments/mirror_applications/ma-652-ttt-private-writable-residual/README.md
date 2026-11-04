# MA-652 — TTT private writable residual frontier

Status: FAIL for Mirror-specific attribution; scoped positive result for low-rank state plus private residual.

## H

How much private writable state is required to preserve sequence-specific online behavior on top of shared TTT modes?

## T

Sixteen 32D streams, four shared rank-4 output modes, 128 online updates, two development seeds. Sweep orthogonal private residual rank 0/2/4/8. Compare full TTT-Linear, shared-only, shared+private, independent-private, and an identical native dictionary-ID control. Actual serialized state bytes, NLL/MSE, and update MACs measured.

## D

Shared+private state is 22–69% of full TTT bytes across ranks 0–8 and its tail MSE is below full TTT in this task. Shared-only quality degrades as private target rank grows; matching private residuals restore it. Rank8 exceeds the 60% state gate. Shared dictionary saves only about .55–.75KB versus independent private bases at ranks 2/4/8, and native IDs exactly match the Mirror function and state. Fresh stayed sealed.

## Fact / Interpretation / Hypothesis

- **Fact:** 40 rows; all tests and replay pass.
- **Interpretation:** Private residuals recover sequence-specific directions at a measurable state cost; Mirror codes add no benefit over native IDs.
- **Hypothesis:** This frontier may generalize to real TTT if hidden-state directions are sufficiently structured.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
