# MA-740 status

- Status: FAIL (scoped analytic Mirror-specific claim)
- Branch: `research/ma-740-few-observation-operator-adaptation-20261109`
- Base commit: `4dea7a71`
- Development: complete; seed 74001, 15 method/observation rows.
- Amendment A01: generic CAVIA control uses development-selected ridge lambda 1e-4; initial unregularized run retained.
- Fresh/audit: seeds 74002–74004 sealed after quality/storage/compute gate miss.
- Verification: tests and deterministic non-timing replay recorded in `VERIFICATION.json`.

## H / T / D / C / U

- **H:** Four Mirror phase coordinates adapt an operator from fewer observations with lower state than a generic eight-value context.
- **T:** Analytic four-frequency operator; 16 tasks; 4/8/16 calibration observations; 256 held-out inputs/task; frozen base, phase code, direct native phase coordinate, CAVIA context and full coefficient control.
- **D:** **FAIL.** At n=4, Mirror test nRMSE is .2246; whole payload saves 4–5%, task-state ratio is .73, adaptation uses 7.7–15.4x function evaluations and direct phase code is an exact alias.
- **C:** Structured phase code is a useful inductive bias but not a distinct Mirror mechanism; nonlinear optimization cost erases its adaptation advantage.
- **U:** Fresh seeds, neural/PDE operators, natural tasks and realistic sensor observations.
