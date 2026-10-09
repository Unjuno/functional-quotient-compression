# MA-301 — Continuous Mirror supermask

Status: SCREENING. Prior art: PA31 SupSup and PA32 Piggyback/PackNet.

## H

A shared continuous task-view mask basis may express many subnetworks with fewer serialized bytes than separate binary masks. Generic PCA may explain any gain.

## T

Synthetic fixed-backbone mask family: D=4096, 64 tasks, masks thresholded from a shared rank-2 score basis. Compare exact packed binary task masks, fp16 Mirror basis plus codes, and generic rank-2 PCA of the binary masks. Development worlds 30100–30101; fresh worlds 30110–30112; three seeds each. Measure actual serialized payload bytes, mask agreement and output NRMSE on random inputs.

## D

Pending development and fresh runs.

## C

The task family is generated from the Mirror basis itself, so a favorable result establishes only aligned representational feasibility; binary masks or PCA may match it.

## U

No trained SupSup/Piggyback learning, natural task, language quality or continual retention evidence.

Development screen: packed binary masks use 32,847 B exactly. Continuous Mirror uses 16,750 B with query NRMSE ~0.0081. Generic PCA uses 24,964 B and NRMSE ~0.171. The stronger fitted rank-2 logistic factorization uses 16,755 B and NRMSE ~0.0483, at about 0.74 s fit time per run. This generic factorization is near the quality gate at effectively the same bytes, so fresh worlds determine whether Mirror's aligned encoding advantage persists. The teacher family is explicitly aligned to the Mirror score basis.
