# MA-504 status

- Status: SCREENING — protocol frozen before generator/training implementation
- Branch: `research/ma-504-token-conditioned-reft-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 20; selected uniformly from 539 eligible P0/UNTESTED rows at index 97
- Last verified commit: pre-data protocol freeze commit follows
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Run the two frozen development seeds at both learning rates; select the common rate by paired Mirror-minus-FiLM dev MSE before any fresh seed.

## Blockers

None identified. CPU-only small mechanism experiments are feasible.

## Decisions / rulings

- PA96 motivates low-rank hidden-state intervention; PA63 requires conditional FiLM as the cheapest direct control.
- Draw20 pool/exclusions and replay are recorded in `source/random_draw.json` and `source/selection_pool.csv`.
- The candidate directory was created after selection and was not treated as a pre-selection exclusion.
