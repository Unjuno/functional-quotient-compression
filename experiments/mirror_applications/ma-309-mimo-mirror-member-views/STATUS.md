# MA-309 status

- Status: FAIL — corrected FP16 inference payload missed accuracy and diversity gates.
- Branch: `research/ma-309-mimo-mirror-member-views-20261008`
- Base commit: `086e974`
- Initial fresh seeds 30911–30913 quarantined after FP32 evaluation bug. Corrected fresh seeds 30921–30923: Mirror 0.809 accuracy / 0.189 disagreement at 4,652B; standard MIMO 0.986 / 0.497 at 6,256B.
- Fifteen corrected payloads hash/byte checked and metric-replayed exactly; five tests pass.
