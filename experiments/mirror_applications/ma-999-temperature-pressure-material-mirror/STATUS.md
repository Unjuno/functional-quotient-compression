# MA-999 status

- Status: FAIL (scoped synthetic condition-invariance screen; MatterSim not tested)
- Branch: `research/ma-999-temperature-pressure-material-mirror-20261113`
- Base commit: `d62e00892`
- Last verified commit: `a75194a96fefb8a793d42a67a309b39fbddefaf6`
- Development complete: yes (seed 99901; rank-1 code fixed in committed runner)
- Fresh/audit opened: yes (99911, 99913, 99917)
- Results committed: pending
- Verification committed: pending

## Next action

Record the scoped FAIL and advance to MA-1000 conservative-derivative audit.

## Blockers

Official MatterSim weights and thermodynamic reference data are unavailable; only a bounded synthetic falsification screen is claimed.

## Decisions / rulings

- **FACT:** shared-only energy/force law exactly matches all synthetic condition-shifted holdout ensembles; Mirror code increases actual payload by 208%.
- **INTERPRETATION:** thermodynamic sampling labels did not require different potential functions in this screen.
- **HYPOTHESIS:** explicit energy-dependent Mirror perturbations need to be audited separately before considering rollout value.
