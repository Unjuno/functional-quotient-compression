# MA-395 status

- Status: **FAIL** at frozen dense-comparator total-payload gate; held-out functional recovery was near exact; fresh remains sealed.
- Branch: `research/ma-395-albert-factorized-embedding-mirror-20261009`
- Base commit: `eae1017b`
- Prior art: PA61 ALBERT factorized embedding.
- Development seeds: 39501, 39502.
- Fresh seeds: 39511, 39512, 39513 (sealed).
- Observed/held-out pairs: 820/204 and 854/170.
- Serialized payloads: ten; sizes, SHA-256 and metrics replay exactly.
- Tests: four passed.

## Result

Mirror held-out NRMSE was 0/3.1e-8, outperforming FiLM and hard sharing and matching dense transforms. Mirror used 7,293/7,268B, or 81.3% of dense payloads, narrowly missing the ≤80% limit; fresh stayed sealed. See README and RESULTS_CORE.csv.
