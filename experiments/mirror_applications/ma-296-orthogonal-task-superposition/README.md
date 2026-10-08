# MA-296 — orthogonalized task-vector Mirror superposition

Status: SCREENING (protocol frozen before fresh access)
Evidence lane: MECHANISM / TASK-ARITHMETIC / STORAGE / RUNTIME
Base commit: `037b849`
Prior art: PA16 Parameter Superposition; PA26 Task Arithmetic.

## H — falsifiable hypothesis

A shared task delta can be bound by a compact orthogonal View, summed into one physical vector, and unbound at query time to recover multiple related logical task edits with less serialized state than independent task vectors and less interference than unbound addition or ordinary PSP. Independent task deltas should require private or richer state. A Mirror-specific claim also requires beating the direct shared-orbit generator that stores the same physical seed vector and task coordinates without the superposition step.

> **Mirror insertion:** this experiment adds a task-specific orthogonal coordinate `m_t` to a task delta before superposition, then applies its inverse at retrieval, so related task functions can share one stored delta vector rather than one full delta per task.

## T — fixed mechanism screen

Eight support task deltas define a shared physical state; eight new task views are evaluated per world. Each delta defines a linear function `f_t(x)=xᵀ(θ₀+Δ_t)` over held-out Gaussian inputs. The aligned family uses `Δ_t = Q(m_t) v*`, with blockwise Givens angles driven by one scalar `m_t`; the independent family uses unrelated isotropic task deltas. `Q` is orthogonal and its deterministic frequency vector is paid in the payload.

Controls: independent full task vectors; unbound arithmetic/single shared delta; standard Rademacher Parameter Superposition (PSP); rank-2/4/8 SVD basis fitted on support tasks; direct shared-orbit generator storing `v*` plus task angles; Mirror `Q(-m)` binding, vector superposition, and `Q(m)` unbinding. Signed pair sums/differences are checked for each method. No optimizer updates.

Development worlds 29601 and 29602 selected SVD rank 8 and PSP context seed 296103. Fresh worlds 29611–29613 and both settings are now frozen; see `DEV_SELECTION.json`. The development grid is preserved in `source/development_grid.json` and summarized in `DEV_RESULTS_CORE.csv`.

## Gates

**PASS:** In every fresh aligned world, Mirror task and signed-composition normalized MSE are each <=1.10x the direct shared-orbit generator, payload <=0.5x independent full vectors, and throughput >=0.5x the direct generator. Mirror-specific value additionally requires >=10% better quality/byte/compute frontier than both the byte-matched PSP control and direct shared-orbit generator. In the independent condition, any recovered functions and paid private residuals are reported separately.

**FAIL:** Mirror is matched or dominated by the direct shared-orbit generator at equal/lower payload, misses aligned quality/storage/runtime gates, or fails independent-task quality without a measured private-state recovery point.

**NOT ESTABLISHED:** serialization/replay mismatch, invalid splits, or incomplete fresh runs.

## Storage and compute

Actual deterministic inference payload bytes are authoritative. Charge base weights, superposed vector, task IDs and coordinates, PSP sign contexts, SVD bases and coefficients, frequency schedule, private residuals, tensor metadata, names and headers. Report support/task vectors seen, optimizer updates, active arithmetic proxy, construction and inference time, and throughput. Runtime workspace is separate from serialized bytes.

## Boundaries

Synthetic linear functions with known task identities and Gaussian inputs. This is not a learned task router, neural language model, or capacity result. The aligned teacher is deliberately generated from an orthogonal task orbit; independent tasks set the private-state boundary.
