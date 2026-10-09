# MA-565 status

- Status: FAIL
- Branch: `research/ma-565-layerdrop-surviving-layer-mirror-20261009`
- Base commit: `0ff3df2f`
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

Depth-2 direct gates improve nMSE only .17385→.16878 and depth-3 .08106→.07333. Independent shallow operators are 747B at zero error. Mirror-role and direct-gate payloads are identical. Per-depth packages charge the shared blocks each time; a serving bundle would charge them once but still require gate codes, and no storage/quality Pareto pass is established. Fresh remains sealed.
