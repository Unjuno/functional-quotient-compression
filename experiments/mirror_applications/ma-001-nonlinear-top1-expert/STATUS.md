# MA-001 status

- Status: SCREENING
- Branch: `research/ma-001-mirror-top1-expert-20261007`
- Base commit: `fa635d0f4b3f98528ac2f731b316e2a664289d3e`
- Protocol freeze: pending
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row: SCREENING

## Next action

Freeze the protocol and source, then run only the preregistered development world.

## Blockers

None. CPU PyTorch is available.

## Decisions / rulings

- MA-001 is narrowly scoped to nonlinear single-layer hard top-1 expert dispatch. It extends the already completed MA-003 linear screen and differs from MA-241's nonlinear cross-depth tying screen.
- Development is a gate: if selected Mirror quality exceeds 1.25x untied full-MoE MSE or Mirror bytes exceed 0.65x full-MoE bytes, do not open fresh worlds.
