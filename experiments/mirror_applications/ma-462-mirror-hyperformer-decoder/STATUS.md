# MA-462 status

- Status: FAIL
- Branch: `research/ma-462-mirror-hyperformer-decoder-20261008`
- Base commit: `d12d62a1902c702e57b09cfaba49d0862d485226`
- Development complete: yes (final deterministic run: 120 fits, 2 worlds, 5 alpha values)
- Fresh/audit opened: no (development gate failed)
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Decision

No rank met the full gate on both worlds. Rank 8 met all clauses on world 4622 but missed the seen-quality tolerance on world 4621. The initial non-deterministic backbone-bias setup was corrected; final 120-row replay matched exactly on all semantic metrics and byte lengths. Fresh worlds remain unopened. See `README.md` and `source/development_summary.json`.
