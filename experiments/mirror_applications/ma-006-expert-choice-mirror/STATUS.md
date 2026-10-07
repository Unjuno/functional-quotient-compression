# MA-006 status

- Status: SCREENING
- Branch: `research/ma-006-expert-choice-mirror-20261007`
- Base commit: `eca36e2` (verified MA-004 parent)
- Protocol frozen before development; data not opened
- Registry: UNTESTED

## Routing contract

Expert-choice dispatch gives each expert exactly batch_size/4 candidate tokens; overlaps are allowed and unselected tokens receive zero MoE output. Dense token-choice top-1 is the routing comparator. No oracle fallback is supplied.

## Next action

Implement and test capacity routing and the expert controls; freeze source hashes before development. Fresh remains sealed unless the registered development gate passes.
