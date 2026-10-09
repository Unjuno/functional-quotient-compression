# MA-389 status

- Status: **FAIL** at frozen development quality gate; fresh remains sealed.
- Branch: `research/ma-389-hash-embedding-mirror-importance-20261009`
- Protocol freeze commit: `56174c62`
- Implementation commit: `78424904`
- Prior art: PA58 Hash Embeddings
- Development seeds: 38901, 38902
- Fresh seeds: 38911, 38912, 38913 (sealed)
- Payloads: ten actual compressed NPZ inference payloads; all size/hash/metric replay checks exact.
- Tests: four passed.

## Result

Mirror NRMSE was .094917/.063938 against the frozen .03 limit; native Hash was .005776/.005399. Mirror used 7,662/7,648B, 43.4% of the full table and 88.8% of native Hash. Decoder top-1 remained 98.44%/98.05%, but this does not satisfy the required vector-fidelity gate. Fresh remained sealed. See README and RESULTS_CORE.csv for the fact/interpretation/hypothesis report.
