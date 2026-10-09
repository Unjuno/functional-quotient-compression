# MA-528 status

- Status: **FAIL**
- Branch: `research/ma-528-sae-behavior-codes-20261009`
- Protocol froze before implementation and development.
- Development seeds 52801/52802 completed; fresh seeds 52811–52813 remain locked/unopened.
- Deterministic replay exact across eight payload hashes, core metrics, splits and selected pools; three tests pass.
- A pre-development FP32 test tolerance correction is documented; no protocol or data-access deviation.

## H / T / D / C / U

- **H:** residual-selected shared SAE dictionary plus 16-term behavior codes beats activation-frequency pool selection while retaining explicit-FV quality and saving bytes against global OMP lists.
- **T:** pinned Pythia-70m+SAE; fit tasks 0–11; heldout 12–15; two dev seeds; no intervention, explicit FV, global OMP16, and aggregate-activation pool controls.
- **D:** FAIL. Residual pool gains +.359/+.670 nats over aggregate pool but loses 1.237/1.157 to explicit FV, loses .099/.090 to global OMP16, and costs 3,990 B vs 3,840 B. Full SAE makes standalone system 4,173,555 B larger than explicit FV deployment.
- **C:** better atom selection helps, but shared pool IDs erase savings; direct global sparse lists are smaller and higher quality.
- **U:** other checkpoints, task families, fresh data, private residual frontier and naturally supervised SAE steering remain unknown.

## Evidence classes

- **Facts:** replay max core-metric difference 0; all 8 payload hashes and both feature pools exact; fresh unopened.
- **Interpretation:** static shared SAE feature codes are not useful drop-in FV compression for these tasks.
- **Hypothesis:** conditional intervention tasks may behave differently; MA-529 is a separate conditional-control experiment.
