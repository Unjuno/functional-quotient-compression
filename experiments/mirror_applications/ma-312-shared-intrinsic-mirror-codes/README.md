# MA-312 — one shared intrinsic basis, many Mirror task coordinates

Status: PROMISING (aligned 256-task storage/quality point; higher fit compute)  
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE / RUNTIME  
Base commit: `3837e9bf9842d9aae4a0b83dd5a6f8d5e11ebdf4`  
Prior art: PA33 intrinsic-dimensional fine-tuning

## H — falsifiable hypothesis

Across a 256-task bank, sharing a paid random intrinsic basis and representing each aligned task with one Mirror angle will yield a useful serialized byte/quality advantage over independent eight-dimensional SAID coordinates and the nearest two-coefficient control. On unrelated task codes, the scalar view should fail and the full intrinsic coordinates or private weights should be required.

> **Mirror insertion:** this experiment adds one task angle `m_t` to a shared two-column slice of a random intrinsic update basis so many task updates can be represented without an independent intrinsic vector or full weight vector per task.

This extends MA-311 in task-bank scale and basis amortization: The earlier MA-311 branch tested 64 tasks and failed the actual NPZ byte gate. A separate MA-311 d=64 follow-up (`research/ma-311-intrinsic-mirror-coordinate-20261008`) measured three fresh worlds: 14.1–21.1% savings versus ordinary intrinsic vectors, but one world missed the all-world 0.85x byte gate and required one aligned private fallback. MA-312 tests a distinct 256-task bank and whether task-count amortization changes the total-payload frontier; it does not assume a reversal.

## T — frozen mechanism screen

A shared 32-parameter linear predictor has a paid random orthonormal 32×8 update basis. There are 256 task IDs. Aligned task updates occupy a fixed-radius circle in the first two basis coordinates. Unrelated task updates are independent eight-dimensional Gaussian vectors. Each task supplies 64 supervised support inputs for post-fit code estimation and 128 held-out test inputs.

Controls: hard tying; PA33 SAID-8 with one FP32 eight-coordinate vector per task; a direct FP16 two-coefficient code on the same shared plane; Mirror with one FP16 angle/task and shared FP16 radius; and independent full 32-dimensional task weights. Each method pays only the basis columns it uses, but every paid basis/core/code and NPZ header is included. Mirror uses a fixed 720-angle training search; coefficients and SAID vectors use least squares. Zero gradient optimizer updates. Task IDs are supplied.

Development seeds: 31201–31203. Fresh seeds 31211–31213 were locked before access. Dimensions, precision, solver, angle grid and gates were frozen before opening them. Fresh access followed because all three development worlds passed actual-byte and quality gates against the nearest direct coefficient control.

## Gates

**Promising / open fresh:** on all development aligned worlds, Mirror normalized test MSE ≤1e-4; actual serialized payload ≤90% of the direct two-coefficient control and ≤75% of SAID-8; Mirror remains within 1e-5 absolute normalized MSE of the direct control.  
**Fail:** any aligned quality/byte gate fails, or unrelated task vectors require richer state with no aligned actual-payload frontier improvement.  
**Not established:** deterministic replay, serialization, or split integrity fails.

## Storage and compute

Actual deterministic NPZ payload bytes after reload are authoritative. Charge shared core, basis columns used, radius, task codes, SAID coordinates, full task vectors, metadata and archive headers. Report examples, fit operation proxy, active inference operations, encoder wall-time and test throughput separately. The 720-point Mirror fitting search is charged; no optimizer updates are hidden. CPU only.

## C / U

**Strongest counter-hypothesis:** each Mirror code is a polar encoding of a two-value coefficient vector. The task bank is generated from the same circular chart; actual serialization and direct coefficients may eliminate Mirror's apparent code-width advantage. MA-311 showed archive overhead can reverse tensor-byte rankings.

**Unconfirmed:** learned deep task adapters, natural task distributions, efficient continuous angle solvers, router inference, continual retention, and fixed-byte near-convergence capacity.

## Decision

FACT: see result table and verification.  
INTERPRETATION: compare per-task code efficiency only after charging the shared basis and actual archive format.  
HYPOTHESIS: a larger task bank may amortize one shared basis while retaining a small per-task coordinate.  
BOUNDARY: synthetic supervised linear task bank; task count is not independent capacity.

## Results — fresh aligned storage/quality point

### H / T / D / C / U

**H:** across a large task bank, one angle per task should be a useful address over one shared intrinsic basis; unrelated coordinates should need richer per-task state.

**T:** 256 tasks, 32D linear predictor, paid shared random 32×8 basis, 64 support and 128 held-out examples/task. Development seeds 31201–31203 selected no settings; fresh seeds 31211–31213 ran the exact frozen 720-angle search and FP16 payload. Controls: hard tie, SAID-8, direct two-coefficient FP16 codes, independent full 32D task vectors. 0 optimizer updates. Every method reloaded its deterministic NPZ payload before scoring. Four tests and all 60 metric/byte/hash rows replayed exactly.

**D: PROMISING only for the aligned task-code storage/quality point.**

**Facts:** Fresh aligned worlds: Mirror actual payload **1,600B**, normalized test MSE **3.83e-7–5.30e-7**; direct two-coefficient control **2,100B**, normalized MSE **1.65e-9–2.32e-9**; SAID-8 **10,034B**, near-zero MSE; independent full **33,024B**, near-zero MSE. Thus Mirror used 23.8% fewer actual bytes than the direct coefficient control and 84.1% fewer than SAID-8, while meeting the frozen 1e-4 absolute quality gate and staying within 1e-5 absolute normalized MSE of direct coefficients. The coefficient method was roughly 200× lower relative error, but both absolute errors were very small; no claim of better quality is made.

This extends MA-311's actual-byte miss: with 256 tasks, the one-code-per-task savings amortized the shared basis and NPZ metadata, yielding a 500B payload reduction versus two FP16 coefficients/task. On unrelated maps, Mirror normalized MSE was **0.284–0.336**, direct coefficients 0.268–0.315, and SAID-8/independent were near zero. One angle did not recover arbitrary eight-dimensional task coordinates.

The fit-operation proxy was **70.78M** for the 720-point angle search, **67.6k** for direct coefficient least squares, and **1.18M** for SAID-8. Fresh aligned batched throughput was 11.85M examples/s Mirror, 14.96M coefficient, 17.44M SAID-8 and 19.97M independent. Active inference proxy was 102 operations/example Mirror vs 96 coefficient and 288 SAID-8. Thus storage improved; fitting and measured inference speed did not.

+**Interpretation:** many task addresses can reduce total serialized state when a shared low-dimensional basis is paid once and the target codes occupy a one-angle orbit. This is a storage/quality point with higher code-fitting cost and slower measured inference. It does not establish learning efficiency, general task multiplicity, or capacity.

**C — strongest counter-hypothesis:** the target codes are generated from the same circular chart. A packed or quantized two-coefficient control may narrow the byte margin, and the angle-search fit is deliberately inefficient relative to direct coefficients. The 256-task result is an aligned post-fit rate point, not evidence that task functions found from natural data will share this structure.

**U:** natural/deep tasks, efficient angle fitting, further quantization, router/task-ID cost, nonlinear basis interactions, quality at near-convergence, and runtime-optimized kernels remain untested.

+FACT: see `RESULTS_CORE.csv`, `FRESH_RESULTS.csv` and `source/replay_verification.json`.  
+INTERPRETATION: actual-byte storage/quality frontier improves for this aligned 256-task bank; compute and unrelated-function quality do not.  
+HYPOTHESIS: structured task codes may be useful when task deltas share a low-dimensional orbit.  
+BOUNDARY: synthetic supervised post-fit linear task family; no capacity claim.
