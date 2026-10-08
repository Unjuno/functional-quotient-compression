# MA-447 — Mirror-conditioned learned optimizer

## H — Hypothesis

A compact domain Mirror code conditions one shared update policy to outperform an unconditioned schedule by 10%, match separate per-domain schedules within 5%, and reduce policy bytes.

## T — Test

Two 2D linear-regression support-design families, three fresh worlds, three seeds, 10 tasks per domain, and 1/2/4/8 updates. Compare tuned Adam, unconditioned learned schedule, independent schedules, and conditioned schedule. A1 charges the complete shared basis; quality measurements were unchanged.

## D — FAIL

Step-4 mean NRMSE: Adam 0.5008, unconditioned 0.7708, conditioned 0.6020, separate 0.6370. The conditioned policy loses to Adam and misses the separate-schedule quality gate in domain 0. Exact N=20 serialized bytes/task: conditioned 151.65B, separate 139.05B, Adam 113.85B.

## C — Strongest counter-hypothesis

The task family does not need learned conditional optimization; tuned Adam performs better with fewer bytes.

## U — Unknown

No natural-task or nonlinear evidence.
