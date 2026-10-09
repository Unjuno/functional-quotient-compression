# MA-511 status

- Status: FROZEN_SCREENING
- Branch: `research/ma-511-hierarchical-condition-behavior-mirror-20261009`
- Base commit: `25c4f935`
- Protocol frozen: yes
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Implement held-out pair generator, generic matrix and Mirror factorized fits, then select common steps on development data.

## Blockers

None.

## Decisions / rulings

Unlike MA-510's noisy condition gate, this experiment conditions on explicit condition IDs and evaluates function composition on held-out condition-behavior pairs. Routing error is not mixed with representation error.
