# MA-997 status

- Status: FAIL (scoped synthetic screen; official MACE adaptation not established)
- Branch: `research/ma-997-sparse-support-mirror-20261113`
- Base commit: `5862e2498`
- Last verified commit: `d97ec995aa0a5e59a49fd7b2beb2544e67389069`
- Development complete: yes (seed 99701; support indices [0, 4, 8, 12, 16, 20])
- Fresh/audit opened: yes (99711, 99713, 99717)
- Results committed: yes
- Verification committed: yes

## Next action

Record scoped FAIL and advance to MA-999; MA-998 is paused with the direct atomistic coefficient-code lane.

## Blockers

Official MACE-MP weights, MPTrj data, and GPU are unavailable; scoped synthetic mechanism screen is allowed by the research notes but does not establish natural-material outcomes.

## Decisions / rulings

- **FACT:** Mirror and native shared mask are identical at 1,732 B; independent masks are only 2.0% larger.
- **INTERPRETATION:** support sharing is real but does not establish a Mirror-specific advantage or meet the storage gate.
- **HYPOTHESIS:** condition-factor response may be a distinct locus; MA-999 tests it separately.
