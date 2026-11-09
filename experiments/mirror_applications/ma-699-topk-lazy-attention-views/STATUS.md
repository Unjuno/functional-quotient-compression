# MA-699 status

- Status: FAIL
- Branch: `research/ma-699-topk-lazy-attention-views-20261109`
- Base commit: `81c702ad`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: yes
- Results committed: yes
- Verification committed: yes
- Registry row updated: no

## Next action

Integrate the scoped FAIL and continue the registry P0 queue.

## Blockers

None.

## Decisions / rulings

Charge selected view IDs. Report active attention compute as k-dependent; do not infer capacity from combination count.


## Result

Top-k lazy output matched materialized controls within 1.86e-7; k8 shared payload was 16.06% of independent cache, while compute scaled to 8.25x one attention. Native phase control exactly aliases Mirror.
