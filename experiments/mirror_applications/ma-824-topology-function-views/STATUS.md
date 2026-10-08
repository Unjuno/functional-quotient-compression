# MA-824 status

- Status: SCREENING — amendment frozen; amended fresh run pending
- Branch: `research/ma-824-topology-function-views-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 22; uniform from 536 eligible P0/UNTESTED rows, index 289; amendment 1 adds fresh seeds 82406–82408
- Last verified commit: pre-data protocol freeze follows
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Run amended development/fresh payload screen; retain initial aliased-storage attempt as invalid evidence.

## Blockers

None identified. The exact two-node program executor is CPU-feasible.

## Decisions / rulings

- PA128 Neural Interpreters already use compact function signatures and shared execution; matching this control is not Mirror-specific.
- Draw22 pool and exclusions are preserved in `source/`.
- Program combinations are treated as exact cases tested, never as a capacity count.
- Initial serialization attempt was invalid because PyTorch deduplicated aliased node tensors in the independent control. Amendment 1 deep-copies those weights and uses unused fresh seeds 82406–82408; the earlier attempt is preserved and excluded from decisions.
