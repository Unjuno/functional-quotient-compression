# MA-498 — Code-distance regularization for a logical function bank

Status: **FAIL for Mirror-specific attribution; frozen development screen**  
Branch: `research/ma-498-code-distance-mirror-bank-20261008`  
Prior art: PA95 (ECOC / metric learning)

## H — Hypothesis

For eight fixed logical linear functions, a 500-update minimum-distance regularizer will reduce address-interference error under Gaussian code noise at sigma .2 by at least 25% versus random orthogonal addresses in both development seeds, while keeping the complete serialized inference payload at or below 1.05x compact raw-ID bytes.

## T — Executed protocol

Seeds 49801 and 49802; eight fixed 16x16 linear experts; 10,000 requests at each sigma in {0,.05,.1,.2,.3,.5}; raw three-bit IDs, random orthogonal codes, random spherical codes, 500-step learned distance codes, and the exact same native metric-learning implementation. Expert matrices, explicit codebook and schema/decoder metadata were included in actual uncompressed NPZ payload sizes. The full per-sigma results and payloads are retained under `runs/dev_49801/` and `runs/dev_49802/`. Fresh seeds 49811–49813 were not opened.

At sigma .2, distance-regularized codes had route error 0.0004 in both seeds and output RMSE 0.02723 / 0.03397. Random orthogonal route error was 0.0010 / 0.0016; compact raw-ID error was 0.0052 / 0.0063. Every decoded method retained all eight functions. The learned code had minimum pair distance 1.4513 / 1.4527, versus 1.4142 / 1.4142 for orthogonal and 0.9291 / 0.6817 for random sphere.

Actual payload was 9,384 B for learned distance codes, 9,381 B for random orthogonal, 9,378 B for random sphere and 8,842 B for raw IDs. The learned bank is 1.0613x raw-ID size, above the frozen 1.05x maximum of 9,284 B. Its decoder plus query compute proxy is 3.20M per 10,000 requests versus raw ID's 2.64M; the learned route decoder alone uses 640k versus raw ID's 80k operations. Training took 0.272 / 0.317 seconds for 500 updates; timings are small synthetic CPU measurements and are not production latency claims.

## D — FAIL

The robustness gate against random orthogonal codes passed (60% and 75% lower route error), but the strict payload gate failed in both seeds. Moreover, `native_metric8` and `mirror_distance_reg8` have byte-identical payloads and exactly identical outputs/metrics in both seeds. Therefore this result is explained by ordinary minimum-distance metric learning and provides no Mirror-specific gain. Fresh seeds stayed sealed under the frozen rule.

## C — Strongest counter-hypothesis

The modest robustness improvement is entirely the effect of increasing minimum code distance through standard metric learning. The measured address code costs 542 extra bytes and 560k extra active-compute proxy operations relative to the raw-ID implementation.

## U — Still unconfirmed

This screen covers only eight independent synthetic linear functions and additive isotropic Gaussian address noise. It does not test learned routers, natural tasks, other code dimensions or deployment hardware. It provides no claim about general capacity or naturally occurring routing corruption.

## Evidence classification

**Facts:** frozen development metrics, serialized byte lengths and replay/alias checks are reported above and in `runs/`, `RESULTS_CORE.csv` and `VERIFICATION.json`.  
**Interpretation:** the strict quality/storage feasibility conjunction failed; native metric learning fully accounts for the robustness gain.  
**Hypothesis:** a task-aware native metric learner or naturally corrupted router address may show a different Pareto frontier; this experiment does not establish it.
