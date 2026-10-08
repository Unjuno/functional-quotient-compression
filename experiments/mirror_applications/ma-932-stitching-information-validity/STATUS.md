# MA-932 status

- Status: FAIL — information diagnostic passed; storage and Mirror-specific gates failed
- Branch: `research/ma-932-stitching-information-validity-20261008`
- Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
- Draw: 21; uniform from 537 eligible P0/UNTESTED rows, index 369
- Last verified commit: `80c0f2e`
- Development complete: yes (2 seeds)
- Fresh/audit opened: fresh yes; audit no
- Results committed: yes (this branch)
- Verification committed: yes (this branch)
- Registry row updated: yes (this branch)

## Next action

Record the result, update the registry and claim ledger, then continue with another uniform draw.

## Blockers

None identified. The controlled linear task is CPU-feasible.

## Decisions / rulings

- PA241 is a direct warning that stitching accuracy alone can coexist with substantially different information contents.
- Draw21 pool and exclusion snapshot are preserved under `source/`.
- The selected branch and directory were created after the draw and are excluded from its recorded pre-draw snapshot.
- One first launch raised a split-name `KeyError` before any optimizer update or metric evaluation; only the development/fresh generator split argument was corrected. The corrected frozen run completed all 60 model rows.
- All three fresh seeds passed the task0-match/task1-loss and linear-probe information diagnostic. All three failed the Mirror payload ratio. One also failed the byte-matched FiLM margin; audit remained unopened.
