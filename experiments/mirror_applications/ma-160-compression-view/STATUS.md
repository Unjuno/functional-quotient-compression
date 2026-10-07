# MA-160 status

- Status: PROMISING (preregistered fresh quality gate missed narrowly)
- Branch: `research/ma-160-compression-view-20261007`
- Base commit: `5ec66a61c6007b4bc5bb3e5902701880a47d3594`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes, only after both development seeds passed
- Results committed: no
- Verification committed: no
- Registry row updated: pending

## Next action

Run the frozen tests and replay, update experiment trackers, and commit the checked result on this research branch.

## Blockers

None.

## Decisions / rulings

- Development gate passed on both seeds. The fresh all-seed quality gate missed because seed 16011 was 1.106× independent-int4 activation MSE (limit 1.10×); bytes passed at 393B versus 607B (0.647×).
- Result is PROMISING rather than PASS because the near-boundary storage tradeoff repeated and residual-free Mirror was much worse, but the preregistered fresh quality condition did not pass.
- No settings changed after fresh access. No protocol deviations were made.
- This is synthetic matrix reconstruction evidence only; arbitrary independent roles require private state and show no Mirror compression benefit.
