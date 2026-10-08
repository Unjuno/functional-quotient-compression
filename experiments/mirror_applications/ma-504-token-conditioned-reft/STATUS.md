# MA-504 status

- Status: FAIL — fresh payload gate missed in all three seeds; one seed also missed quality vs independent LoReFT
- Branch: `research/ma-504-token-conditioned-reft-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 20; selected uniformly from 539 eligible P0/UNTESTED rows at index 97
- Last verified commit: `0b40104` (pre-run frozen implementation); final verification is in this result commit
- Development complete: yes (2 seeds × 2 rates)
- Fresh/audit opened: fresh seeds yes; audit no
- Results committed: yes (this branch)
- Verification committed: yes (this branch)
- Registry row updated: yes (this branch)

## Next action

Commit and push the verified FAIL record; then continue with a fresh uniform draw.

## Blockers

None identified. CPU-only small mechanism experiments are feasible.

## Decisions / rulings

- PA96 motivates low-rank hidden-state intervention; PA63 requires conditional FiLM as the cheapest direct control.
- Draw20 pool/exclusions and replay are recorded in `source/random_draw.json` and `source/selection_pool.csv`.
- The candidate directory was created after selection and was not treated as a pre-selection exclusion.
- Post-run accounting review corrected the analytic active-MAC proxy to count A/B dimensions, generator layers and efficient rank-4 Givens updates; training artifacts and quality outcomes were unchanged. Wall time/throughput measure the implemented dense rotation construction.
