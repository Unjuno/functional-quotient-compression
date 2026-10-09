# MA-372 status

- Status: FAIL (Mirror-specific value)
- Branch: `research/ma-372-matformer-mix-mirror-20261009`
- Base commit: `0eded323`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Update registry and claim ledger, verify metrics, and commit the development-only FAIL.

## Blockers

None.

## Decisions / rulings

Across two seeds, the factorized bank exactly recovered all six held-out compositions at 1401B; flat storage used 6017B and native width-only was 737B with held-out nMSE 0.10498 and only one distinct function. The ordinary direct factorized coefficient control is byte- and function-identical to Mirror at 1401B. Therefore Mirror-specific gate FAILS; fresh remains sealed. The factorization mechanism is aligned and synthetic only.
