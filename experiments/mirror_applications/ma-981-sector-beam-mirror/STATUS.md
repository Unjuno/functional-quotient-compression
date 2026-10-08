# MA-981 status

- Status: SCREENING — protocol frozen before channel generation
- Branch: `research/ma-981-sector-beam-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 23; uniform from 534 eligible P0/UNTESTED rows, index 401
- Last verified commit: pre-data protocol freeze follows
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Implement phase-only codebook models, channel generator, rate objective, actual payload serialization and tests; freeze before simulation.

## Blockers

None for the simulation hypothesis. Hardware outcomes are outside this experiment's evidence lane.

## Decisions / rulings

- PA288 learned beam codebooks are the mandatory native baseline; PA289 low-dimensional site feedback motivates the compact-control axis.
- Draw23 pool, exclusions and replay are stored under `source/`.
- Simulation results will not be presented as RF hardware performance.
