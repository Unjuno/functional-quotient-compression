# MA-311 — Mirror coordinates inside a random intrinsic subspace

Status: FAIL (actual-byte gate; fresh sealed)  
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE / RUNTIME  
Base commit: `574f074877ae7939d678404d37329f526ad9b92f`  
Prior art: PA33 intrinsic-dimensional fine-tuning

## H — falsifiable hypothesis

For tasks whose deltas lie on a one-dimensional orbit inside a paid four-dimensional random intrinsic subspace, fitting one Mirror angle per task from examples can preserve held-out predictions with fewer total serialized bytes than both ordinary SAID four-coordinate codes and the nearest ordinary two-coefficient subspace code. For arbitrary four-dimensional task codes, the one-angle View should fail and richer/private coordinates should be required.

> **Mirror insertion:** this experiment adds a scalar task angle `m_t` inside the already-small random intrinsic code `z_t` so that a task-specific weight update can be expressed without storing a four-value code or full weight vector per task.

## T — frozen mechanism screen

Use a fixed 16-parameter linear predictor, a shared 16×4 random orthonormal projection basis, and 64 task IDs. Aligned targets are `θ_t=θ_0 + P[r cos(m_t), r sin(m_t), 0, 0]`, with fixed radius `r=0.25`; unrelated targets use independent four-dimensional Gaussian intrinsic coordinates. Each task gets 64 labeled support inputs for code fitting and 256 held-out test inputs. Fitting uses closed-form least squares for SAID/two-coefficient/full controls and a fixed 720-angle search for Mirror. There are zero gradient optimizer updates; this is a supervised post-fit coordinate screen.

Controls: hard tie; PA33 random-subspace SAID with four FP32 coordinates/task; direct two-coordinate coefficient control in the same intrinsic plane (FP16/task); Mirror with one FP16 angle/task plus shared radius; independent full 16-dimensional task weights. The shared backbone and random projection basis are charged in all shared-state methods. The direct two-value control is the nearest simple control and is required for a Mirror-specific claim. Factor/task IDs are supplied; no router.

Development seeds: 31101–31103. Fresh seeds locked but unopened: 31111–31113. No hyperparameters will be selected from audit. Fresh is opened only if development meets the quality and byte gates.

## Gates

**Promising / open fresh:** all three development aligned worlds: Mirror test normalized MSE ≤1e-4; Mirror is within 1e-5 absolute normalized MSE of the two-coefficient control; Mirror total payload is ≤75% of the four-coordinate SAID payload and ≤90% of the direct two-coefficient control.  
**Fail:** Mirror misses quality, fails either byte gate, or unrelated codes require richer state without a useful aligned frontier point.  
**Not established:** payload reload, deterministic replay, or split integrity fails.

## Storage and compute

Actual reloaded deterministic NPZ bytes are authoritative. Charge `θ0`, the random projection basis `P`, radius, task codes, independent task vectors, shape/format metadata, and archive headers. Report support examples, optimizer updates, active projection/search operation proxies, encoder time and held-out throughput. Fixed search/closed-form fit work is not free; training compute and inference compute are separate. CPU only; no nanoGPT vendor edits.

## C / U

**Strongest counter-hypothesis:** an ordinary two-value polar coefficient code is functionally equivalent and may be nearly as small; the improvement can be only code precision/encoding, not a new functional degree of freedom. The task family deliberately lies on the Mirror chart.

**Unconfirmed:** trained deep models, natural tasks, learned task inference, optimizer efficiency, arbitrary task composition, fixed-byte near-convergence capacity and deployed runtime.

## Decision

FACT: see result table and verification.  
INTERPRETATION: separate random-subspace benefit from the extra Mirror structure.  
HYPOTHESIS: one scalar may encode a useful low-loss orbit inside an already-small intrinsic coordinate.  
BOUNDARY: synthetic linear functions only; no claim that 64 task IDs equal 64 independent functions.

## Results — development-only failure

### H / T / D / C / U

**H:** one angle per task should describe a low-dimensional orbit inside a random four-dimensional intrinsic parameter code more cheaply than ordinary SAID coordinates and direct polar coefficients; arbitrary codes should require richer state.

**T:** three development worlds (31101–31103), 64 tasks, 16-dimensional linear predictor, paid random 16×4 orthonormal basis, 64 supervised support inputs/task and 256 held-out test inputs/task. Fit methods: hard tie, SAID-4 least squares, direct two-coefficient least squares, 720-point Mirror angle search, and independent 16D least squares. Zero gradient optimizer updates. Fresh 31111–31113 were not opened because the actual-payload comparison failed its predeclared gate.

**D: FAIL for the registered actual-byte/quality gate.**

**Facts:** On aligned tasks, Mirror's mean test normalized MSE was **7.50e-7**, meeting its 1e-4 quality threshold and staying within 1e-5 of the two-coefficient control. Mirror total serialized payload was **1,248B**; the direct two-coefficient control was **1,140B**. Thus Mirror used 9.5% more actual bytes and failed the required ≤90% byte ratio. SAID-4 used **2,018B**, so Mirror did save 38.2% versus the four-FP32-coordinate baseline. Independent full vectors used **4,344B**. The Mirror raw tensor bytes were 322B versus 448B for coefficient codes, but its extra radius array and archive entry/header overhead reversed that advantage after serialization.

The fit-compute proxy was **17,694,720** operations for the 720-angle search, versus **16,896** for two-coefficient least squares and **69,632** for SAID-4. Aligned held-out throughput averaged 21.19M examples/s Mirror, 29.20M coefficient, 33.63M SAID-4 and 32.13M independent. These small CPU timings show no Mirror runtime advantage.

On unrelated tasks, Mirror mean normalized MSE was **0.210**; direct two-coordinate codes were 0.171; SAID-4 and independent full weights were numerical zero. A single angle cannot recover arbitrary four-dimensional coordinates. All result rows replayed with exact metrics/bytes/hashes; four tests passed.

**Interpretation:** the one-angle chart was much cheaper than unconstrained SAID fitting in inference-state tensors, but actual archive structure erased the expected byte gain over the nearest ordinary control. Its fixed search was also far more expensive. The screen therefore does not establish a Mirror-specific storage or compute win.

**C — strongest counter-hypothesis:** the paired sine/cosine coefficients are the direct ordinary-coordinate representation of the same circular function family. A packed format could reduce their archive overhead; a finer or continuous angle solver could reduce Mirror fit compute. Neither post-hoc possibility changes this frozen result.

**U:** Fresh replication, efficient angle fitting, packed byte formats, learned deep task adapters, natural tasks, and fixed-byte capacity remain untested. The unrelated stress test indicates a need for more than one scalar coordinate.

FACT: see `RESULTS_CORE.csv` and `source/replay_verification.json`.  
INTERPRETATION: SAID-4 vs Mirror is a narrow coordinate-compression signal; nearest-control byte and compute gates fail.  
HYPOTHESIS: packed formats or larger task banks might change total serialization economics, requiring a new preregistered protocol.  
BOUNDARY: synthetic supervised post-fit linear screen only; no fresh or capacity claim.
