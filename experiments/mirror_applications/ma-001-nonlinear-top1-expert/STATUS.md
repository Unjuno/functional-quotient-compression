# MA-001 status

- Status: SCREENING
- Branch: `research/ma-001-mirror-top1-expert-20261007`
- Base commit: `fa635d0f4b3f98528ac2f731b316e2a664289d3e`
- Protocol freeze: `58c82ae5e43ce135a6dbd48a041cefaf465a2645`
- Development complete: yes; selected LR 0.01 by preregistered mean MSE
- Fresh/audit opened: no; preregistered development access gate passed and fresh configuration is frozen
- Results committed: no
- Verification committed: no
- Registry row: SCREENING

## Next action

Run the three preregistered fresh worlds; do not tune after opening fresh data.

## Blockers

None. CPU PyTorch is available.

## Decisions / rulings

- MA-001 is narrowly scoped to nonlinear single-layer hard top-1 expert dispatch. It extends the already completed MA-003 linear screen and differs from MA-241's nonlinear cross-depth tying screen.
- Development is a gate: if selected Mirror quality exceeds 1.25x untied full-MoE MSE or Mirror bytes exceed 0.65x full-MoE bytes, do not open fresh worlds.
