# MA-981 status

- Status: FAIL — Mirror quality passed; storage and scalar-control gates failed
- Branch: `research/ma-981-sector-beam-mirror-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 23; uniform from 534 eligible P0/UNTESTED rows, index 401
- Last verified commit: pre-data freeze `72cf06f`; result commit follows
- Development complete: yes (2 seeds)
- Fresh/audit opened: fresh yes; no separate audit
- Results committed: pending result commit
- Verification committed: pending result commit
- Registry row updated: pending result commit

## Next action

Commit the verified FAIL record and continue with another uniform draw.

## Blockers

None for the simulation hypothesis. Hardware outcomes are outside this experiment's evidence lane.

## Decisions / rulings

- PA288 learned beam codebooks are the mandatory native baseline; PA289 low-dimensional site feedback motivates the compact-control axis.
- Draw23 pool, exclusions and replay are stored under `source/`.
- Simulation results will not be presented as RF hardware performance.
- The frozen quality gate passed, but all fresh worlds missed the 25% actual-byte reduction; scalar phase control was smaller and essentially matched rate. Rank-2 payload did not meet the frozen byte-match tolerance.
- Post-run review corrected the phase-materialization operation proxy only; trained weights and rate measurements were unchanged.
