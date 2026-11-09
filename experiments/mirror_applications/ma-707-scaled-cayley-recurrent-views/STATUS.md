# MA-707 status

- Status: FAIL
- Branch: `research/ma-707-scaled-cayley-recurrent-views-20261109`
- Base commit: `322e8039`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes
- Results committed: yes
- Verification committed: yes
- Registry row updated: pending

## Next action

Integrate scoped FAIL and advance to MA-708.

## Blockers

None.

## Decisions / rulings

Native low-rank Cayley is an exact attribution control; stability is measured by spectral radius and trajectory norms.


## Result

Rank1 state is 28.1% of independent and stable/exact, but reconstruction overhead is 43.2% and native Cayley exactly aliases. Initial stability/replay attempt preserved.
