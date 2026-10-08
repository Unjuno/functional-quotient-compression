# MA-315 — shared intrinsic Mirror with sparse private residual coordinates

Status: FAIL (25% private-fraction storage gate missed in both amended development worlds)
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE / RUNTIME
Base commit: `9ccbb03b8a29f80ff489efacbef72d8a1b3921d1`
Prior art: PA33 intrinsic-dimensional tuning; PA30 SETA shared/private subspaces; MA-311/312/314 intrinsic-coordinate screens

## H — falsifiable hypothesis

On a task bank where most updates share a fixed-radius 2D orbit but a controlled fraction need private coordinates, a shared Mirror angle plus validation-selected sparse residual dimensions will preserve held-out quality and use at least 10% fewer actual serialized bytes than the closest shared-basis direct-coefficient control when private fraction is at most 25%; at higher fractions, the byte frontier should approach or lose to global intrinsic coordinates.

> **Mirror insertion:** this experiment adds a per-task angle `m_t` to the common 2D subspace update and stores ordinary sparse coordinates only for residual directions, expressing a shared function plus rare private deviations without duplicating the whole task vector.

## Mirror vs prior art

SETA (PA30) already discovers shared and unique sparse subspaces. Here the synthetic decomposition is known by construction; the isolated question is whether `m` compresses the shared component after a residual is allowed. Adaptive direct coordinates on the identical paid random basis are the nearest control. This experiment does not claim to reproduce SETA's learned decomposition or continual retention.

## T — frozen mechanism screen

A shared 32D linear predictor and a paid random orthonormal 32×8 basis serve 256 task IDs. Every task has a fixed-radius 0.25 update on the first basis plane with a task-specific angle. At private fractions 0%, 25%, 50%, and 100%, respectively 0, 64, 128, and 256 tasks also receive a random six-coordinate residual in basis columns 3–8. Each task has 32 support, 32 validation, and 64 held-out test examples. Fit on support, select the smallest residual rank in `{0,2,4,6}` (or direct prefix dimension `{2,4,6,8}`) whose validation normalized MSE is at most `1e-4`, and score once on test. Task IDs are supplied; no optimizer updates.

Controls: hard tie; adaptive direct FP16 coefficients on the same shared basis; fixed SAID-8 FP32 coefficients; independent full 32D task vectors. Mirror uses one FP16 angle plus the same adaptive FP16 residual coordinates. For each residual rank, variable projection removes the candidate residual span before the 720-angle support search, then refits residual coordinates at the selected angle. Code offsets are derived from the stored rank tags and are not stored redundantly. Every payload pays shared predictor, used basis prefix, common radius, task rank tags, codes and fixed-timestamp ZIP/NPY headers. Development seeds 31501/31502; fresh seeds 31511–31513 stay sealed unless both private fractions 0% and 25% pass the preregistered byte/quality gate on both development seeds. An implementation-level solver/layout amendment was recorded after the first development run and before any fresh access; both development seeds are rerun under the amended implementation. See `PROTOCOL_AMENDMENTS.md`.

## Gates

**Promising / open fresh:** at private fractions 0% and 25% in both development worlds, Mirror mean normalized test MSE ≤`1e-4`, within `1e-5` absolute normalized MSE of adaptive direct, and actual payload ≤90% of adaptive direct. The 50% and 100% worlds map the frontier but do not open fresh by themselves.
**Fail:** either low-private development world misses quality or the 10% actual-byte margin. A storage win with >1.25× fitting operations or >1.25× active inference operations is storage-only and not a composite Pareto win. Record private-fraction points even when Mirror loses.
**Not established:** deterministic replay, serialized reload, or split-integrity checks fail.

## C / U

**Strongest counter-hypothesis:** one angle is a polar re-encoding of two direct coefficients; after adding even a few private coordinates, the fixed basis, dimension tags and archive headers can erase the saved coefficient bytes. The direct control should fit faster and more accurately.

**Unconfirmed:** learned shared/private decompositions, noisy and natural task distributions, gradient-trained models, task-agnostic routing, continual retention, and capacity near convergence.

## Decision

FACT: see `RESULTS_CORE.csv`, `VERIFICATION.json`, and the result-row replay.
INTERPRETATION: report quality, actual bytes, fit cost and active cost as a private-fraction frontier; do not pool with MA-314 protocol variants.
HYPOTHESIS: a shared low-description coordinate may postpone private state for rare residual functions.
BOUNDARY: synthetic post-fit linear task bank; no language, learned-decomposition, or capacity claim.

## Results — amended development screen

### H / T / D / C / U

**H:** a shared Mirror angle plus sparse private coordinates should beat adaptive direct coefficients by at least 10% actual bytes at 0% and 25% private-task fractions while meeting the same quality threshold.

**T:** 256 task IDs, 32D linear predictor, shared random 32×8 basis, 32 support / 32 validation / 64 test examples per task; private fractions 0/25/50/100%. Development seeds 31501 and 31502. Mirror stored an angle plus selected residual coordinates; controls were hard tie, adaptive direct FP16 coefficients, fixed SAID-8 FP32 and independent 32D vectors. All methods were reloaded from deterministic ZIP/NPY payloads. Zero optimizer updates. Fresh seeds remained sealed because the promotion gate failed.

**D: FAIL for the preregistered storage/quality Pareto claim.**

**FACT:** Under the amended implementation, Mirror test normalized MSE was 1.04e-6–1.29e-6 across worlds/fractions and every maximum task value stayed below 4.6e-5. It selected no residual coordinates for aligned tasks and six for private tasks. Actual Mirror/direct payloads were: 2,290/2,574B at 0%; 3,826/4,110B at 25%; 4,594/4,878B at 50%; and 6,130/6,414B at 100%. The byte reductions were 11.0%, 6.9%, 5.8%, and 4.4%. Thus the 0% point passed the 10% gate, but the 25% point missed it in both development worlds, leaving fresh sealed. Mirror remained within the registered 1e-5 absolute normalized-error margin of direct coefficients, which were much more numerically accurate.

Mirror fitting used 143,982,592 operation-proxy units versus 1,187,840 for adaptive direct (~121×). Active inference proxy was 104 vs 98 ops/example at 0%, 153.5 vs 147.5 at 25%, 203 vs 197 at 50%, and 302 vs 296 at 100%. Eager CPU throughput varied substantially on this short synthetic workload; no runtime win is established. Both development gates and the selector were deterministic; four tests and replay verification cover all 40 method/world rows and 10,240 task events.

**INTERPRETATION:** A single angle replaces one coefficient per task and saves storage when all task functions share the orbit. Once even 25% of tasks need residual coordinates, archive/model metadata dominates enough that the total payload falls short of the frozen 10% margin. Sparse private coordinates preserve quality, but they do not make this Mirror address a general shared/private compression win.

**C — strongest counter-hypothesis:** the fixed radius and first-plane orbit are favorable to Mirror; quantized or entropy-coded direct coefficients could narrow the 0% point, while a learned residual basis could reduce high-private costs. Conversely, a larger task bank could amortize array headers and change the 25% total-byte result.

**U:** fresh replication is sealed after the development byte miss; learned shared/private decomposition, quantized coefficient controls, larger task banks, task-agnostic routing, continual retention, and natural/deep models remain untested.

FACT: `DEV_RESULTS.csv`, `DEV_EVENTS.csv`, and `source/replay_verification.json` record the amended development run; the initial coordinate-descent result is retained under `protocol_variants/initial_coordinate_descent/`.
INTERPRETATION: the aligned storage point degrades quickly as private fraction grows; no all-frontier win was found.
HYPOTHESIS: explicit sparse residual state can preserve quality while tracing the boundary where shared Mirror state ceases to help.
BOUNDARY: two development worlds, synthetic post-fit linear tasks, no fresh results or capacity claim.
