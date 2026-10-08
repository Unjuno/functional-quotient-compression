# MA-255 status

- Status: FAIL (registered development gates missed)
- Branch: `research/ma-255-context-superposition-20261008`
- Base commit: `cbd8140`
- Development complete: yes (seeds 25501, 25502)
- Fresh/audit opened: no (25511–25513 sealed)
- Results committed: yes
- Verification committed: yes
- Registry row updated: pending

## Next action

Record MA-255 failure in registry, ledger and status board; proceed to MA-257.

## Blockers

None.

## Decisions / rulings

Payload replay caught a PSP sign-bit serialization bug before results were finalized. The binding code was corrected to pack boolean signs, both registered development worlds were rerun, and all payload metrics were replayed from serialized inference state. No fresh data was observed.
