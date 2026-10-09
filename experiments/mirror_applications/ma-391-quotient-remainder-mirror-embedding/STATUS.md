# MA-391 status

- Status: **FAIL** at frozen development embedding-fidelity gate; fresh remains sealed.
- Branch: `research/ma-391-quotient-remainder-mirror-embedding-20261009`
- Protocol freeze commit: `0028f612`
- Implementation commit: `71715b83`
- Prior art: PA59 compositional / quotient-remainder embeddings.
- Development seeds: 39101, 39102.
- Fresh seeds: 39111, 39112, 39113 (sealed).
- Address uniqueness: 1,024/1,024 in both worlds.
- Serialized payloads: twelve; sizes, SHA-256 and metrics replay exactly.
- Tests: four passed.

## Result

Mirror NRMSE was .12475/.13911 against the frozen .03 limit, while direct coefficients reached .00943/.01505. Mirror decoder NLL/top-1 remained close to direct coefficients, and Mirror used 10,461B (16.5% of full, 73.6% of direct-coefficient payload). Fixed addition/product/concat controls were much worse. Strict development gate failed, so fresh stayed sealed. See README and RESULTS_CORE.csv for fact/interpretation/hypothesis separation.
