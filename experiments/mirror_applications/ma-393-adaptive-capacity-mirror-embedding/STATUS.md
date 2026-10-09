# MA-393 status

- Status: **FAIL** at frozen complete-payload gate; strong aligned rare-band mechanism result; fresh remains sealed.
- Branch: `research/ma-393-adaptive-capacity-mirror-embedding-20261009`
- Protocol freeze commit: `4a06a30b`
- Implementation commit: `d07256bd` (correctness amendment; original implementation `19e041c7`)
- Prior art: PA60 Adaptive Input Representations.
- Corrected development seeds: 39301, 39302.
- Fresh seeds: 39311, 39312, 39313 (sealed).
- Corrected payloads: eight; sizes, SHA-256 and all quality metrics replay exactly.
- Tests: four passed.
- Amendment: first probe used non-nested random projections in the byte-matched control; outputs are retained in an excluded folder and are not analyzed. Nested control fixed before corrected development rerun.

## Result

Mirror weighted NRMSE was .00301/.00407 and rare-band NRMSE .0124/.0154, against .150/.145 weighted and .537/.442 rare-band for standard adaptive embeddings. At matched bytes, the +1-coordinate control had .128/.133 weighted NRMSE. Mirror used 15,427/15,404B (60.46%/60.43% of full, narrowly missing the ≤60% limit) and incurred slower training/inference. Fresh stayed sealed. See README and RESULTS_CORE.csv.
