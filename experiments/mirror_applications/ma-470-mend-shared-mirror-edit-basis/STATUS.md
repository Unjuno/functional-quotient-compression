# MA-470 status

- Status: PROMISING, scoped synthetic shared-basis storage result; no Mirror-specific win over generic atoms
- Branch: `research/ma-470-mend-shared-mirror-edit-basis-20261009`
- Base commit: `ae59be94`
- Protocol freeze: `d3edd370`
- Development complete: yes
- Fresh/audit opened: yes, only after protocol freeze
- Results committed: yes (`a6bf0978`)
- Verification committed: yes (`a6bf0978`)
- Registry row updated: yes (branch tip `e5279c1a`)

## Next action

MA-470 is complete and pushed; proceed to MA-471 on its own branch.

## Blockers

None.

## Decisions / rulings

- Fixed missing edit-generation latency measurement after the fresh run; reran the exact frozen fresh worlds/seeds without changing model or gates.
- Least-squares coefficients use known target edit matrices and are an oracle upper bound, not a learned MEND coefficient generator.
- Generic independent atom basis is 628B smaller than Mirror+private at N=64 with matching quality; therefore no Mirror-specific gain is claimed.
- View-only misses edits that use the independent off-orbit atom; the private atom repairs this at a charged storage cost.
