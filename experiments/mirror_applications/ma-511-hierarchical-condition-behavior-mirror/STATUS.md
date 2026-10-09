# MA-511 status

- Status: FROZEN_SCREENING
- Branch: `research/ma-511-hierarchical-condition-behavior-mirror-20261009`
- Base commit: `25c4f935`
- Protocol frozen: yes
- Development complete: yes; selected 800 common Adam steps on held-out Mirror NRMSE
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## Next action

Freeze the 800-step choice and evaluate the locked fresh worlds/seeds.

## Blockers

None.

## Decisions / rulings

Unlike MA-510's noisy condition gate, this experiment conditions on explicit condition IDs and evaluates function composition on held-out condition-behavior pairs. Routing error is not mixed with representation error.
