# MA-567 status

- Status: FAIL (Mirror-specific and byte gate)
- Branch: `research/ma-567-mod-mirror-routing-20261009`
- Base commit: `a84cc98a`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Update registry and claim ledger, replay verifier, and commit the development-only FAIL.

## Blockers

None.

## Decisions / rulings

With identical fixed top-50% routing, Mirror and direct two-basis coefficients both used 1204B and had routed nMSE 0. Independent role blocks used 1235B, only 2.5% more than Mirror, below the 20% gate. Vanilla MoD used 959B at nMSE .24655. Router proxy (4096 MAC) and active block proxy (16512 MAC) matched across all methods. Fresh remains sealed.
