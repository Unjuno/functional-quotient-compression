# MA-255 status

- Status: FAIL (registered development gates missed)
- Branch: `research/ma-255-context-superposition-20261008`
- Base commit: `cbd8140`
- Development complete: yes (seeds 25501, 25502)
- Fresh/audit opened: no (25511–25513 sealed)
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes (FAIL, fresh remains sealed)

## Next action

Tests and serialized-payload replay verified locally; failure is present in registry/claim/status board. Continue to MA-257 after checking for a newer branch.

## Blockers

None.

## Decisions / rulings

Payload replay caught a PSP sign-bit serialization bug before results were finalized. The binding code was corrected to pack boolean signs, both registered development worlds were rerun, and all payload metrics were replayed from serialized inference state. No fresh data was observed.
