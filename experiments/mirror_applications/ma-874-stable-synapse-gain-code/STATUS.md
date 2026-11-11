# MA-874 status

- Status: FAIL (Mirror-specific attribution); synthetic gain-state screen passes scoped quality/reset/storage gate
- Branch: `research/ma-874-stable-synapse-gain-code-20261111`
- Base commit: `54fee5d6`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes, after protocol freeze and development validation
- Results committed: pending
- Verification committed: pending

## Next action

Commit and push the checked MA-874 evidence, then integrate it into the cumulative research branch.

## Blockers

None.

## Decisions / rulings

- Rank-2 gain state passes the synthetic quality/reset/storage gates, but native PCA/context-gain exactly aliases it.
- The fast recurrent-decay control is a narrow, misspecified writable-weight comparison; independent task readouts also pass quality and remain the strongest simple function-space control.

- Synthetic PA235-inspired mechanism screen only; this does not reproduce the original biological or trained-RNN study.
- Native PCA/context-gain is an explicit exact-alias baseline; Mirror-specific success requires a distinct result beyond it.
