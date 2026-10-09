# MA-392 status

- Status: **FAIL** at frozen complete-payload gate; fresh remains sealed.
- Branch: `research/ma-392-token-domain-mirror-embedding-20261009`
- Protocol freeze commit: `5c028efa`
- Implementation commit: `5bfc0b68`
- Prior art: PA59 compositional / quotient-remainder embeddings.
- Development seeds: 39201, 39202.
- Fresh seeds: 39211, 39212, 39213 (sealed).
- Observed/held-out pairs: 819/205 and 815/209.
- Serialized payloads: ten; sizes, SHA-256 and metrics replay exactly.
- Tests: four passed.

## Result

Mirror held-out NRMSE was 1.41e-7/3.33e-8, matching or beating dense transforms (.0162/.0143) and outperforming FiLM/hard sharing. Actual payload was 10,291/10,275B, or 60.4%/60.3% of dense—above the frozen ≤50% limit—so fresh remained sealed. See README and RESULTS_CORE.csv for fact/interpretation/hypothesis separation.
