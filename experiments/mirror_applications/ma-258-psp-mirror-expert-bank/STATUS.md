# MA-258 status

- Status: PROMISING (canonical output-channel Givens screen); supplemental flattened-vector rerun misses its stricter byte gate
- Canonical branch: `research/ma-258-psp-mirror-expert-bank-20261008` at `82263f49d4d0d5af96f4ee715ac7a6622b60da71`
- Supplemental branch: `research/ma-258-flat-orbit-rerun-20261008`
- Supplemental base commit: `3ea72d6`
- Canonical fresh opened: yes; canonical verification replayed 60 payload rows
- Supplemental clean development: yes (25821/25822; aligned and unrelated)
- Supplemental fresh opened: no (strict 0.50x byte cap missed at 0.512x)
- Pre-existing local artifact/cache files: preserved and excluded from supplemental run
- Results committed: yes
- Verification committed: yes

## Next action

Continue to MA-260.

## Decisions / rulings

The canonical remote branch had a complete, verified MA-258 result. This supplemental run uses a distinct flattened-parameter transform and stricter byte gate, so it is recorded as a sensitivity failure without changing the canonical PROMISING status.
