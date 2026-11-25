# MA-1053 status

- Status: FAIL (scoped to preregistered multi-task Mirror quality/storage gate)
- Branch: `research/ma-1053-moment-task-codes-20261125`
- Base commit: `9aee6aacf`
- Development complete: yes (seed 105301)
- Fresh/audit complete: yes (seed 105302)
- Results committed: yes (`cc56f59a7937d6e4984f2bb15d1d1fed24c5a9df`)
- Verification committed: yes (`cc56f59a7937d6e4984f2bb15d1d1fed24c5a9df`)
- Registry row updated: yes

## Audit summary

Mirror MASE 0.920243 vs diagonal 0.917809 for forecast; ECG5000 accuracy 0.831778 vs diagonal 0.850000 and native 0.867556 (misses the 2 pp tolerance). On the single tiltAPB2 event, pointwise F1 is 0.046243 vs diagonal 0.044901, but AUPRC is 0.015390 vs diagonal 0.015931; precision 3.15%, recall 8.72%. The all-task preregistered gate fails.

Mirror adaptation payload is 888,101 B after correcting metadata to 8 codes per role (24 total; 161 B above hard sharing and 5 B above diagonal); independent rank-32 projections use 1,084,711 B, but full-bundle savings against them are only 0.043%. See `RESULTS.md` and `RESULTS_CORE.csv`.

## Verification

Three focused tests pass. `source/verify_report.py` independently rechecks data/protocol hashes, frozen source hashes, task-native pre-head parity, actual serialized bytes, prediction reload and all three task metrics from saved predictions. The anomaly benchmark has one event and is descriptive only.

## Next action

Update the registry, claim ledger, status board and worker queue, then continue at next eligible P0 MA-1055 (N-BEATS shared basis + Mirror coefficient; PA335). MA-1054 is P1 and remains queued after current P0 candidates.
