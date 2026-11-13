# MA-996 status

- Status: FAIL (scoped synthetic screen; official MACE-MP result not established)
- Branch: `research/ma-996-mace-material-mirror-20261113`
- Base commit: `156d437db`
- Last verified commit: pending
- Development complete: yes (seed 99601; selected width 0.65, rank 2)
- Fresh/audit opened: yes (99611, 99613, 99617)
- Results committed: pending
- Verification committed: pending

## Next action

Record the scoped FAIL, run the serializer/replay verification, then continue with MA-997.

## Blockers

Official MACE-MP and MPTrj are not locally available; GPU is unavailable. The scoped synthetic screen is explicitly not a substitute for the registered natural-material result.

## Decisions / rulings

- **FACT:** rank-2 Mirror and native low-rank produce identical heldout results. Actual four-target payload is 2,066 B versus 1,774 B independent; MAC proxy is 48% higher.
- **INTERPRETATION:** no Mirror-specific benefit is established; conservative-force consistency is achievable in this synthetic radial basis.
- **HYPOTHESIS:** natural MACE-MP transfer remains unknown; MA-997 tests sparse-support adaptation as a separate mechanism.
