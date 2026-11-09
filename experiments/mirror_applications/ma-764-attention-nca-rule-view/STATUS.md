# MA-764 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-764-attention-nca-rule-view-20261109`
- Base commit: `f667de88`
- Development: complete, seed 76401; five controls.
- Fresh/audit: unopened after native alias and feature-attention control.
- Verification: pending.

## H / T / D / C / U

- **H:** Attention-produced rule Views outperform feature-only attention for dynamic local behaviors.
- **T:** Two context-interpolated 3×3 rules; 20 training and 12 heldout contexts; 10-step rolls; independent, hard-tied, feature attention, native hypernetwork and Mirror controls.
- **D:** **FAIL.** Native hypernetwork exactly aliases Mirror; feature attention reaches `.00465` rollout error at equal bytes/MACs; View compute is 2× independent.
- **C:** Standard attention over nonlinear rule features explains the useful behavior nearly as well.
- **U:** Real trained Attention-NCA and emergent morphology.

## Next action

Run focused tests, deterministic replay and registry integrity check, then save the scoped FAIL.

## Decisions / rulings

- Amendment A01 fixes heldout one-step teacher comparison to use the same inputs; initial invalid metrics remain preserved in `RESULTS_INITIAL_INPUT_MISMATCH.csv`.
