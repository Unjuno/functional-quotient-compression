# MA-945 status

- Status: FAIL (development gate)
- Branch: `research/ma-945-coanerv-mirror-20261008`
- Base commit: 379a9417cb32c9f96c68c779315f90381151eed1
- Last verified commit: pending-result-commit
- Development complete: yes (44 fits, 2 seeds, 2 videos)
- Fresh/audit opened: no (no product-Mirror rank passed)
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Decision

The product-Mirror token code passed storage and CPU query-throughput clauses but missed the native full-token quality gate on `container` for every tested rank. Rank 8 beat byte-matched simple controls, while the rank-2 private residual recovered part of the content-dependent gap. Fresh clips remain unopened. See `README.md` and `source/development_summary.json`.

## Next action

Commit checked result and branch-local registry/claim/status records, push this research branch, then re-fetch baseline for a new randomized draw.

## Blockers / boundary

No GPU was available; the protocol is CPU-scaled and makes no full-paper GPU runtime claim. This is not a full CoANeRV reproduction or conventional coded-video bitrate result.
