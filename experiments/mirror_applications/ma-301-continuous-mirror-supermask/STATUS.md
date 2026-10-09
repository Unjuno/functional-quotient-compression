# MA-301 status

- Status: PROMISING (aligned synthetic mechanism only)
- Branch: `research/ma-301-continuous-mirror-supermask-20261009`
- Frozen protocol commit: `6f70b52c`
- Fresh: complete, 3 worlds × 3 seeds; 36 rows
- Verification: complete

## H
A continuous task-view mask code can preserve subnetwork function quality at lower payload than independent packed binary masks.

## T
D=4096, 64 synthetic tasks whose binary masks are thresholded rank-2 shared scores. Controls: packed binary masks, Mirror basis+codes, PCA, and fixed-step generic logistic factorization. Fresh worlds 30110–30112 × seeds 0–2. Actual serialized bytes include all stored factors/codes or packed bits.

## D
PROMISING, narrowly. Mirror uses 16,750 B vs binary masks 32,847 B (~49% reduction), with mask agreement 0.999963 and query NRMSE 0.00854. Byte-matched logistic factorization (16,755 B) has NRMSE 0.04798; PCA uses more bytes and has NRMSE 0.158.

## C
The teacher is directly generated from the same rank-2 score view Mirror stores. The generic logistic fit may be optimization-limited at 250 steps.

## U
No trained SupSup/Piggyback, natural tasks, generalization across task families, learned Mirror coordinates, or inference-latency evidence. Treat this as an aligned representation feasibility result only.
