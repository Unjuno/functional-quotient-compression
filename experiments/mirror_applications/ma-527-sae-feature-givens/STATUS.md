# MA-527 status

- Status: **FAIL**
- Branch: `research/ma-527-sae-feature-givens-20261009`
- Protocol froze before implementation/development; hash is recorded in `freeze.json`.
- Development seeds 52701/52702 completed and deterministically replayed.
- Fresh seeds 52711–52713 remain sealed and unopened because explicit-FV quality failed.
- Five tests pass; 10 paid payload hashes, core metrics, splits and pool selection replay exactly.
- Implementation failures before metrics and per-method timing correction are preserved in `IMPLEMENTATION_AMENDMENT_1.json`; protocol unchanged.

## H / T / D / C / U

- **H:** one shared 16-feature SAE vector with eight Givens angles per task preserves FV quality and beats native pairwise gains and global OMP while using <=0.5x FV bytes.
- **T:** pinned Pythia-70m plus pinned 4x layer-3 SAE; fit tasks 0–11, held-out tasks 12–15, two development seeds, explicit FV and four non-Mirror controls.
- **D:** FAIL. Payload gate and both native-control margins pass, but Givens loses 2.333/2.607 gold-logprob nats to explicit FV; accuracy delta is −.0938/+.0313. Incremental payload is 3,432 B vs 34,826 B explicit, while standalone deployment is 4,172,997 B larger.
- **C:** the SAE feature span may be poorly aligned with these function vectors; the small gain over pairwise scalars may reflect parameterization/optimization.
- **U:** other SAE widths/checkpoints, task families, fresh worlds, near-convergence capacity and optimized runtime remain untested.

## Evidence classes

- **Facts:** deterministic replay exact; maximum core metric delta 0; fresh directory absent.
- **Interpretation:** Givens adds a small development-screen likelihood gain over same-size pairwise gains but fails causal-quality and full-system storage goals.
- **Hypothesis:** wider shared feature bases or task-specific feature selection may behave differently; MA-528 tests a separate behavior-code hypothesis.
