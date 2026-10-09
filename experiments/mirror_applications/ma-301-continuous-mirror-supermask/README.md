# MA-301 — Continuous Mirror supermask

Status: PROMISING, limited to the aligned synthetic mask family. Prior art: PA31 SupSup and PA32 Piggyback/PackNet.

## H

A shared continuous task-view mask basis may express many subnetworks with fewer serialized bytes than separate binary masks. Generic PCA may explain any gain.

## T

Synthetic fixed-backbone mask family: D=4096, 64 tasks, masks thresholded from a shared rank-2 score basis. Compare exact packed binary task masks, fp16 Mirror basis plus codes, and generic rank-2 PCA of the binary masks. Development worlds 30100–30101; fresh worlds 30110–30112; three seeds each. Measure actual serialized payload bytes, mask agreement and output NRMSE on random inputs.

## D

PROMISING, narrowly scoped. Fresh Mirror masks use 16,750 B vs packed binary masks 32,847 B (~49% smaller) with query NRMSE 0.00854. Generic rank-2 logistic factorization uses 16,755 B and has NRMSE 0.04798; PCA uses more bytes and has NRMSE 0.158. Teacher tasks are generated from the same rank-2 score basis Mirror stores, so no general task-mask or learning claim follows.

## C

The task family is generated from the Mirror basis itself, so a favorable result establishes only aligned representational feasibility; binary masks or PCA may match it.

## U

No trained SupSup/Piggyback learning, natural task, language quality or continual retention evidence.

Development screen: packed binary masks use 32,847 B exactly. Continuous Mirror uses 16,750 B with query NRMSE ~0.0081. Generic PCA uses 24,964 B and NRMSE ~0.171. The stronger fitted rank-2 logistic factorization uses 16,755 B and NRMSE ~0.0483, at about 0.74 s fit time per run. This generic factorization is near the quality gate at effectively the same bytes, so fresh worlds determine whether Mirror's aligned encoding advantage persists. The teacher family is explicitly aligned to the Mirror score basis.

## Fresh result / scope

Fresh results support a storage-quality frontier improvement only for the deliberately rank-2-aligned mask family. The packed binary baseline remains exact; Mirror trades a small function error for about half the bytes. A generic logistic factorization is at the same byte budget but has higher error under its frozen 250-step fit. Longer generic convergence, trained SupSup/Piggyback, natural tasks and inference latency remain untested.
