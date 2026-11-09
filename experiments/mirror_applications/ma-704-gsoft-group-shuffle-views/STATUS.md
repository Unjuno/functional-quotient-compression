# MA-704 status

- Status: FAIL
- Branch: `research/ma-704-gsoft-group-shuffle-views-20261109`
- Base commit: `161cd9d4`
- Last verified commit: 16982ac8
- Development complete: yes
- Fresh/audit opened: yes
- Results committed: yes
- Verification committed: yes
- Registry row updated: pending

## Next action

Integrate the scoped FAIL and advance to next P0.

## Blockers

None.

## Decisions / rulings

Native GSOFT factors are an exact control; an exact alias is not Mirror-specific evidence.


## Result

Two-stage GSOFT code uses 37.8% of independent bytes and 1.25x MAC proxy, but native GSOFT exactly aliases it and the local NumPy implementation is about 28x slower than dense apply.
