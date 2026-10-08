# MA-446 status

- Status: FAIL (development gate)
- Branch: `research/ma-446-learned-optimizer-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Protocol frozen: `699061f0684d22ced8985974938ff0f836c8d3ee` with pre-run schedule amendment `26ff1d90425cdeb3902882ad5c4b3682f85310e5`
- Development evidence and replay: verified, pending evidence commit
- Development complete: yes (two worlds, 64 held-out tasks)
- Fresh/audit opened: no
- Registry row: FAIL pending bookkeeping commit
- Tests: 5 passed, 0 failed
- Actual serialized payloads: 28 checked; exact prediction replay

## Decision

The restartable-byte gate passed, but the learned Mirror optimizer did not beat the best same-code SGD/Adam/Meta-SGD control by the required margin in either development world. It also missed the full-weight learned-optimizer quality bound in world 44601. Mirror Adam was the strongest simple method on this aligned synthetic task family.

## Next action

Commit the checked negative result and branch-local registry/status/claim updates. Keep fresh worlds unopened.

## Boundaries

Synthetic rank-3 linear tasks; fixed four-update screen; no capacity or natural-task claim.
