# MA-534 status

- Status: FAIL
- Branch: `research/ma-534-transcoder-mirror-mlps-20261009`
- Base commit: `2386e1e03d8590d95590c2bce90a5373dece5bd3`
- Development complete: yes (seeds 53401, 53402)
- Fresh/audit opened: no (frozen gate failed)
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Run the repository MA registry/experiment verification, then push the checked FAIL evidence.

## Blockers

None. The quality tolerance to explicit mean deltas passed both development seeds, but the 2x payload gate and both required native-control margins failed.

## Decisions / rulings

- Amendment 1 resolved an impossible deployment claim: selected 16 paid encoder rows are evaluated directly with ReLU; no full-dictionary global top-32 mask is inferred for free.
- Amendment 2 adds per-method and total wall-clock reporting only.
- Two implementation errors before any persisted metrics are recorded in `execution_log/attempts.jsonl`; same frozen development splits were rerun.
- Deterministic replay matched both splits, all six methods' metrics and payload sizes exactly. Fresh seeds remain sealed.
