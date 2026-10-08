# MA-824 status

- Status: FAIL — amended runs exact, but native-interpreter Mirror-specific byte gate failed
- Branch: `research/ma-824-topology-function-views-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 22; uniform from 536 eligible P0/UNTESTED rows, index 289; amendment 1 adds fresh seeds 82406–82408
- Last verified commit: `cee0029`
- Development complete: yes (2 seeds)
- Fresh/audit opened: fresh yes; separate audit none
- Results committed: yes (this branch)
- Verification committed: yes (this branch)
- Registry row updated: yes (this branch)

## Next action

Commit amended evidence and continue with another uniform draw.

## Blockers

None identified. The exact two-node program executor is CPU-feasible.

## Decisions / rulings

- PA128 Neural Interpreters already use compact function signatures and shared execution; matching this control is not Mirror-specific.
- Draw22 pool and exclusions are preserved in `source/`.
- Program combinations are treated as exact cases tested, never as a capacity count.
- Initial serialization attempt was invalid because PyTorch deduplicated aliased node tensors in the independent control. Amendment 1 deep-copies those weights and uses unused fresh seeds 82406–82408; the earlier attempt is preserved and excluded from decisions.
- The amended Mirror compression versus independent duplication passed (64.9% lower bytes), but the ≥10% advantage over the native interpreter failed in all three fresh worlds. All four tested program outputs were exact; no learning or capacity conclusion follows.
