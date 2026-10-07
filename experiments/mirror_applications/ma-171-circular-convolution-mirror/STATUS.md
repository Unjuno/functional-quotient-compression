# MA-171 status

- Status: PROMISING, narrow HRR-aligned fixed-budget result
- Branch: `research/ma-171-compression-view-20261007`
- Base commit: `7f454be235b1d7ad5db14f75490e796a2b090110`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes; corrected A1 split passed, original split was accidentally run at wrong LR and excluded
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Complete full metric replay, commit this experiment and trackers, then continue to MA-173.

## Blockers

None. The wrong-LR access is disclosed and contained in `RESULTS_EXPLORATORY.csv`.

## Decisions / rulings

- Development aggregate metric selected LR 0.003. The original fresh invocation mistakenly used 0.01; seeds 17111–17113 are exploratory only.
- Amendment A1 fixed the already development-selected 0.003 and new seeds 17121–17123 before access; the corrected fresh gate passes in 3/3 HRR-aligned worlds.
- This is a fixed-update, deliberately aligned synthetic mechanism result. It is not capacity evidence or a general HRR superiority claim.
- Arbitrary independent role functions fail shared HRR and require private parameters.
