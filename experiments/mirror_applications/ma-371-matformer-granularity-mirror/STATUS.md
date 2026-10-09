# MA-371 status

- Status: FAIL (Mirror-specific gate)
- Branch: `research/ma-371-matformer-granularity-mirror-20261009`
- Base commit: `bccbc91a`
- Last verified commit: pending
- Development complete: yes
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Update registry and claim ledger, verify integrity, and commit the development-only failure.

## Blockers

None.

## Decisions / rulings

Amendment A1 replaced an initial near-uniform/unlearnable label task with a learnable random linear teacher; no fresh worlds were opened. In two dev seeds, direct coefficients and Mirror have identical payload sizes (2084 B/world) and behavior. Shared prefix payload is 1670 B; the charged independent bank is 3942 B. Homogeneous 8x8 reaches mean accuracy about .77, while held-out mix quality is inconsistent and no frozen gate is passed. The result is FAIL for Mirror-specific value.
