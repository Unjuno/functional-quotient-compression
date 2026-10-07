# MA-005 status

- Status: SCREENING
- Branch: `research/ma-005-signed-mirror-mixture-20261007`
- Base commit: `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`
- Development complete: yes; v1 selected LR 0.01
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: SCREENING

## Next action

Source, tests, protocol, selected LR and dev-row hashes are frozen. Verify manifest, then open fresh worlds 50001–50003.

## Blockers

None.

## Decisions / rulings

The signed coefficient vector is supplied as a deterministic function of two input signs and is shared by all methods. This intentionally isolates signed expert composition from router learning, which was separately measured in MA-003.
