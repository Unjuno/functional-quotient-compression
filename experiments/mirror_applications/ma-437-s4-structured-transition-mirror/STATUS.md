# MA-437 status

- Status: FAIL
- Branch: `research/ma-437-s4-structured-transition-mirror-20261008`
- Protocol frozen: yes (`276e77d`), amendment A1 recorded after a teacher-generator mismatch was found
- Development complete: yes; selected LR 0.003 for shared/gate/rank1/independent and 0.01 for Mirror
- Fresh/audit opened: yes; corrected fresh worlds 43710/43711/43712, seeds 0/1/2
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Next action

Commit result artifacts and update registry, claim ledger, and status board; then proceed to MA-438.

## Blockers

None.

## Decisions / rulings

Amendment A1 invalidated the initial fresh set after a structural mismatch was found: the initial teacher rotated the full transition, including diagonal terms, instead of only the low-rank factors. Those initial results are not used. The corrected development and fresh runs retain the original worlds, seeds, controls, and gates.
