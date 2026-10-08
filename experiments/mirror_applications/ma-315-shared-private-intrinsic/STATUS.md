# MA-315 status

- Status: FAIL (the 25% private-fraction development payload missed the 10% nearest-control margin)
- Branch: `research/ma-315-shared-private-intrinsic-20261008`
- Base commit: `9ccbb03b8a29f80ff489efacbef72d8a1b3921d1`
- Last verified commit: pending result commit
- Development: amended solver/layout run complete on 31501/31502
- Fresh/audit: sealed; development byte gate failed at 25% private fraction
- Results committed: pending
- Registry row updated: pending

## Next action

Commit the negative result and continue to MA-316.

## Blockers

None.

## Decisions / rulings

- Initial coordinate-descent development output is retained in `protocol_variants/initial_coordinate_descent/`.
- A variable-projection fit and deterministic packed code layout were recorded in `PROTOCOL_AMENDMENTS.md` before any fresh access; all development seeds were rerun.
- The 0% private point passed quality and byte gates, but both 25% development worlds saved only 6.9% versus direct FP16 codes. Do not open fresh.
