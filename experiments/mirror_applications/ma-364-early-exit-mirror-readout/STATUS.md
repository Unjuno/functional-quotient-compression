# MA-364 status

- Status: FAIL for Mirror-specific value
- Branch: `research/ma-364-early-exit-mirror-readout-20261008`
- Base commit: `c935a90`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no; 36411–36413 remain sealed
- Results committed: yes
- Verification committed: yes
- Registry row updated: no

## Decision summary

Mirror and direct coefficient controls had identical predictions and payload bytes on both development seeds. Shared coefficients saved bytes vs independent heads, but hard sharing was smaller with a small quality cost. No Mirror-specific benefit.

## Next action

Verify replay and tests, update records, then continue through the registry.
