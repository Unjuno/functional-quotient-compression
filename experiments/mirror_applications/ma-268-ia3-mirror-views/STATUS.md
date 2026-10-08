# MA-268 status

- Status: PROMISING (3/3 fresh worlds; synthetic rotation-aligned nonlinear task family)
- Result source branch: `research/ma-268-ia3-mirror-views-20261008`
- Evidence integration branch: `research/ma-268-ia3-reconciled-20261008`
- Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
- Protocol frozen before development: yes
- Fresh worlds opened: yes, only after dev indicated promise; LR 0.01 frozen
- Results committed: yes
- Verification committed: yes (2 source tests, exact metric replay)

## Next action

Trained nonlinear aligned mechanism PASS 3/3; independent-function boundary recorded. The separate post-fit linear variant missed its 20% byte threshold and is retained as a near-miss. Advance to MA-271.

## Blockers

No optimized GPU kernel available; eager CPU runtime is not competitive with IA3.
