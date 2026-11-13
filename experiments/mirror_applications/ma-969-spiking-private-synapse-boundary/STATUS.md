# MA-969 status

- Status: NOT ESTABLISHED (invalid implementation screen; preserve raw run)
- Branch: `research/ma-969-spiking-private-synapse-boundary-20261113`
- Base commit: `6d463c54f`
- Protocol frozen: yes
- Development complete: yes (seed 96901)
- Fresh/audit opened: yes (optimizer seeds 96911/13/17 on one locked trial split)
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## Disposition

Do not interpret this run as evidence against Mirror/private synapse adaptation. Audit found that sparse weights are overwritten by zero-initialized deltas rather than formed as `shared_weight + delta`; at k=32 the initialized effective synapses are all zero. The implemented spike surrogate derivative and MAC proxy also differ from the frozen protocol. The independent control and D1 core are near chance, so the intended quality boundary is not identified. Preserve all outputs as an invalid implementation screen. Any corrected run requires a new protocol amendment/experiment ID and fresh audit data.

## Completed work

Dataset extraction, development run, three optimizer-seed fresh runs, actual serialization and result logging completed. Fresh optimizer seeds share one acquisition and fixed held-out trials; they are not independent data worlds.

## Blockers

None for the scoped CPU screen.
