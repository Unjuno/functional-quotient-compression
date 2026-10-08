# MA-674 — Sparse top-k Mirror LoRA atoms

## H — hypothesis

A shared low-rank atom bank plus sparse per-task codes may reduce actual bytes and active adapter work versus independent task deltas. It must beat a direct MoLE-style mixture using the same atoms to establish Mirror-specific value.

## T — planned test

Frozen conditions are in `PROTOCOL.json`. Draw43 selected MA-674 uniformly from the eligible P0 pool. PA149 motivates the direct mixture control. Functional delta matrices, not raw LoRA factors, are compared to avoid gauge-only claims. This is a synthetic oracle coding screen, not trained-adapter transfer.

## D — status

SCREENING; protocol frozen.

## C — strongest counter-hypothesis

Sparse atom mixtures are already a direct mixture-of-LoRA-experts parameterization; Mirror naming adds no function or compute benefit. A fitted shared basis may also fail on private fresh tasks.

## U — unconfirmed

Implementation, heldout reconstruction, bytes, active atoms and residual costs.

## Fact / Interpretation / Hypothesis

- Fact: MoLE directly composes trained LoRA modules (PA149).
- Interpretation: comparison must use gauge-invariant full updates and matched sparse mixing.
- Hypothesis: sparse shared atoms may lower bank storage for aligned compositions.
