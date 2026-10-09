# MA-446 — learned optimizer restricted to Mirror coordinates

Status: **FAIL (development gate)**
Branch: `research/ma-446-learned-optimizer-mirror-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Draw 10: 558 eligible P0/UNTESTED candidates, seed `4d3f452738dfe7c524cb8f0be1b1f36560d6eae466ee36249a52ed0bb1ccb260`, index 78. Full ordered pool/hash: `source/random_draw.json`, `source/selection_pool.csv`.

## H — falsifiable hypothesis

A recurrent optimizer shared across tasks and restricted to a three-value Mirror code will improve four-update adaptation over tuned SGD, Adam, and Meta-SGD on the same code, stay within 5% of the same optimizer over full task weights, and use at least 30% fewer serialized restartable bytes across eight tasks.

## Prior-art delta

PA77 establishes recurrent learned optimizers based on gradient history. PA151 Meta-SGD is the simpler learned initialization and per-coordinate update control. This experiment asks whether a learned recurrent update rule is useful specifically when it changes only a compact Mirror code. It does not claim learned optimizers or low-dimensional task codes as new ideas.

## T — execution

- Synthetic few-shot linear tasks: input 4, output 3, 12 full weight values, one shared matrix plus a fixed orthonormal rank-3 matrix basis, and three task coefficients with standard deviations 0.2, 0.5 and 1.0.
- Two development worlds (44601, 44602), 32 held-out tasks each. Each task receives four updates of 16 support examples and is scored on 128 query examples. Fresh worlds stayed locked.
- Controls: hard sharing; Mirror-code SGD and Adam with learning rate selected from `{0.03, 0.1, 0.3, 1.0}` across both worlds; Meta-SGD on the same code; full-weight Adam; and the same recurrent learned optimizer over all 12 weights. Each learned optimizer used 256 meta-updates over 16 tasks and four inner updates.
- Selected rates: SGD 1.0, Mirror Adam 0.3, full Adam 0.1. Total run wall time was 5.73 seconds. Recurrent optimizer meta-training took about 0.86–0.96 seconds per world and variant.
- 2,240 per-task/update rows are retained. All 28 serialized files (14 inference-only, 14 restartable) were reloaded and hash/byte checked; both payload forms reproduced final predictions exactly. Details are in `source/metric_replay.json`.

## D — FAIL

The Mirror-only recurrent optimizer missed both quality gates in world 44601 and missed the 15% simple-control gate in both worlds. No fresh evaluation was opened.

### Fact

Mean query MSE after four updates, pooled across the two development worlds:

| Method | Query MSE |
|---|---:|
| Hard shared | 0.3812 |
| Mirror + SGD | 0.3225 |
| Mirror + Adam | 0.0706 |
| Mirror + Meta-SGD | 0.3654 |
| Mirror + learned recurrent optimizer | 0.2568 |
| Full-weight Adam | 0.0896 |
| Full-weight learned recurrent optimizer | 0.2750 |

Mirror recurrent optimization used **11,957 B** for an eight-task restartable library, versus **30,908 B** for the full-weight recurrent optimizer (61.3% fewer bytes). Its deployment-only inference payload was **2,388 B** versus **2,908 B** (17.9% fewer bytes). Those savings did not preserve quality: the Mirror recurrent optimizer had 3.64× the query MSE of Mirror Adam. It beat Meta-SGD and SGD on mean query MSE, but Adam was the strongest simple control.

As a separate fact on this deliberately rank-3-aligned teacher, Mirror Adam reached 0.0706 MSE versus full-weight Adam's 0.0896, using 3,660 B versus 6,484 B restartable payloads. This is a synthetic fixed-update representation result; it does not rescue the learned-optimizer hypothesis or establish general capacity.

Compute proxy per task/update episode was 5,760 multiply-adds for Mirror recurrent optimization versus 1,944 for Mirror Adam and 17,280 for full-weight recurrent optimization. Measured CPU wall time was about 62 microseconds per task for Mirror recurrent optimization and 49 microseconds for Mirror Adam; this short vectorized timing is noisy and not a deployment benchmark.

### Interpretation

The recurrent learned rule did not make Mirror-code adaptation more efficient than ordinary Adam. Restricting the trainable state to the aligned rank-3 code reduced storage and compute relative to a full-weight recurrent optimizer, but the recurrent rule was poorly matched to this optimization problem. In this task family, Mirror Adam was both smaller and much more accurate than the learned recurrent Mirror optimizer.

### H / T / D / C / U

- **H:** A shared recurrent update rule on `m` beats tuned code-space SGD/Adam/Meta-SGD after four support updates, stays within 5% of full-weight learned optimization, and reduces restartable bytes by at least 30%.
- **T:** Two fixed development worlds, 32 tasks each, four support updates, seven method families, actual serialized inference and restartable payloads, 2,240 recorded task/update rows.
- **D:** **FAIL.** The byte gate passed, but the learned Mirror optimizer failed its simple-control quality gate in both worlds and its full-optimizer quality gate in one.
- **C:** Adam's per-coordinate moment normalization is already a strong low-dimensional optimizer; a recurrent learned rule adds state and compute without enough task-family diversity to learn a better update policy.
- **U:** New task families, near-convergence behavior, learned optimization on nonlinear neural models, natural tasks and production runtime remain untested.

## Protocol amendments and implementation attempts

A1 made the train/dev seed schedule and task count explicit before scores. A2, after the initial development run and before any audit/fresh access, added separate inference-only byte reporting and the Adam restart step counter; it changed no quality settings or gate. The pre-A2 metrics and payloads are preserved under `source/attempt_results/attempt_3/` and `source/attempt_payloads/attempt_3/`. All implementation and verification attempts are retained in `source/attempt_log.json`.

**BOUNDARY:** synthetic rank-3 linear tasks and a four-update learning-efficiency screen. No capacity, natural-language, hardware or production optimizer claim.
