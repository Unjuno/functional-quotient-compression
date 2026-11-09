# MA-761 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-761-goal-nca-mirror-code-20261109`
- Base commit: `c43ddfc0`
- Development: complete, seed 76101; five controls.
- Fresh/audit: unopened after native coordinate alias.
- Verification: pending.

## H / T / D / C / U

- **H:** Goal coordinates let one shared rule reach heldout goals at lower storage than independent per-goal rules.
- **T:** NCA-inspired 5×5 grid navigation; 20 train goals; 5 heldout goals; four starts each; shared and independent policy controls.
- **D:** **FAIL** for Mirror specificity. 100% heldout success at 5,631 B; native coordinate control exactly aliases Mirror.
- **C:** Ordinary coordinate conditioning is sufficient; this does not require Mirror-specific machinery.
- **U:** Multi-cell NCA pattern formation and real GoalNCA tasks.

## Next action

Run focused tests, deterministic replay and registry integrity check, then save the scoped FAIL.

## Blockers

None.
