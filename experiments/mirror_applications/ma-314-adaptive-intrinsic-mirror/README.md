# MA-314 — adaptive intrinsic dimension with Mirror views

Status: **FAIL for the preregistered Mirror storage/quality Pareto gate**
Branch: `research/ma-314-adaptive-intrinsic-mirror-20261008`
Prior art: PA33, with MA-311/312 as related intrinsic-coordinate screens.

## H — hypothesis

Across tasks with different intrinsic complexities, a validation-selected dimension plus a Mirror angle for the shared-radius first coordinate pair and private residuals will reduce actual serialized bytes by at least 10% versus validation-selected ordinary intrinsic coefficients at held-out normalized MSE <=1e-4. Functions outside the orbit should need private intrinsic coordinates.

> **Mirror insertion:** add one FP16 angle per task to a shared first pair of intrinsic basis columns; store higher-dimensional residual coefficients only when needed, and fall back to a private full intrinsic vector when the orbit misses validation.

## T — execution

A 32D linear predictor uses a deterministic random orthonormal 32x16 basis and 160 tasks: four aligned groups of 32 tasks at true dimensions 2, 4, 8 and 16, plus 32 unrelated 16D functions. Each task has 64 support, 64 validation and 128 test examples. Controls are hard tie, fixed direct dimensions 2/4/8/16, validation-selected adaptive direct coefficients, adaptive Mirror angle plus residual/private fallback, and independent full 32D weights. No gradient updates were made. The Mirror fit searches a fixed 720-angle grid.

Development seeds 31401/31402 selected threshold 0.0001 from {0.0001, 0.001, 0.01}. The threshold was frozen before fresh access. Fresh seeds 31411–31413 used the same code and threshold. The original task-generation and fitting code remained fixed; before fresh, the group-dimension selection wording was clarified to use group medians, and the active operation proxy was corrected to use selected dimensions. Both amendments and pre-correction development summaries are preserved in `PROTOCOL_AMENDMENTS.md` and `protocol_variants/`.

Payload bytes are measured from deterministic ZIP/NPY inference states after reload. They include the base, used basis prefix, task dimensions, offsets, angle/radius or coefficients, mode flags and archive headers. `RESULTS_CORE.csv` contains 72 rows; `ALLOCATION_EVENTS.csv` contains 2,880 adaptive-task decisions.

## D — decision

**FAIL.** All three fresh Mirror payloads exceeded the adaptive-direct payload by 5.2%, missing the preregistered requirement of at least 10% fewer bytes. Fresh bytes were 7,082/7,080/7,086B for Mirror and 6,730/6,734/6,734B for adaptive direct. Mirror mean normalized test MSE was 8.16e-7, 1.59e-6 and 5.59e-7, satisfying the 1e-4 quality limit; direct adaptive coefficients had lower error in every fresh world.

The separate adaptive-dimension control did reduce payload versus fixed 16D direct codes: 6,730–6,734B versus 8,910B, about 24.5% fewer bytes, with quality below 4.7e-7. That storage gain is explained by dimension selection and is not Mirror-specific. Mirror saves one FP16 coefficient for each aligned angle task, but radius/mode/header state and private fallbacks erase the savings. The unrelated group used private 16D fallbacks in all fresh worlds (29–32 of 32 tasks per world); aligned group medians matched 2/4/8/16.

Fit-compute proxy was 166.36–166.73M for Mirror versus 1.61M for adaptive direct coefficients (about 103x). Active inference proxy was 332–332.4 versus 326–326.4 operations/example. Measured batched CPU throughput was 14.4–16.4M examples/s for Mirror and 23.8–25.5M for adaptive direct, roughly 0.59–0.69x. The timing is a small NumPy mechanism diagnostic.

## Fact / interpretation / hypothesis

- **Fact:** adaptive direct dimension reached 6,730–6,734B in all fresh worlds, while Mirror reached 7,080–7,086B. Mirror passed the held-out quality threshold, but lost the actual-byte comparison in 3/3 fresh worlds. Fixed-16D direct state was 8,910B. Unrelated tasks used private full-intrinsic fallbacks.
- **Interpretation:** validation-selected intrinsic dimension compresses a mixed-complexity task bank relative to a fixed maximum dimension. The Mirror angle did not improve the frontier over direct adaptive coefficients, and brought substantially more fitting work and lower measured throughput. Unrelated task functions required private state.
- **Hypothesis:** a cheaper view code or a polar control with shared radius may reduce metadata cost, but that is untested and would require a new protocol.

## C — strongest counter-hypothesis

The tasks deliberately contain a shared-radius circular first pair, favorable to the angle view, yet the native adaptive coefficient representation is still smaller. The result is likely due to state/metadata overhead and the need to allocate full vectors to unrelated tasks; a different serialization codec could shift the small byte gap.

## U — unresolved

No pretrained model, nonlinear task family, learned task router, optimizer-driven tuning, near-convergence capacity, or deployment kernel was measured. The dimension adaptation point is synthetic post-fit evidence only.

Protocol: [PROTOCOL.json](PROTOCOL.json). Pre-fresh amendments: [PROTOCOL_AMENDMENTS.md](PROTOCOL_AMENDMENTS.md). Data: [RESULTS_CORE.csv](RESULTS_CORE.csv), [FRESH_RESULTS.csv](FRESH_RESULTS.csv), [ALLOCATION_EVENTS.csv](ALLOCATION_EVENTS.csv). Replay: [source/REPLAY_VERIFICATION.json](source/REPLAY_VERIFICATION.json).
