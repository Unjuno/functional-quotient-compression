# MA-1042 status

- Status: FAIL (Mirror-specific attribution); shared-index synthetic mechanism passes its scoped exactness/storage/runtime gates
- Branch: research/ma-1042-pgm-domain-residual-view-20261115
- Official PGM-index source commit: c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407 (Apache-2.0)
- Final implementation confirmatory fresh seeds: 1050–1052, epsilon 8
- Superseded audits 1044–1046 and 1047–1049 are preserved and excluded from confirmation
- 120,000/120,000 exact predecessor answers for Mirror, ordinary affine and independent controls; all updates verified
- Complete serialized payload: 133,720 B shared vs 534,260 B independent (−74.97%); ordinary affine control is functionally identical
- Four verification tests pass; report: RESULTS.md

## Interpretation boundary

This is a synthetic exact-affine-domain shared-index mechanism result, not evidence on natural databases. Pinned DynamicPGMIndex updates are checked, then its final key set is used to rebuild a static PGMIndex for predecessor queries. Direct dynamic predecessor serving, concurrency, SSD behavior and loaded tail latency remain untested.
