# MA-614 — Mirror code composition before a shared interpreter

Prior art: PA127 Routing Networks/PathNet; PA128 Neural Interpreters.

## Fact

Two deterministic phase grids each contained 48 available ordered pairs and 16 held-out pairs. Sequential evaluation of the exact nested sine composition had zero reconstruction error. The one-call ten-harmonic Mirror code had held-out mean NRMSE2 0.07624 and maximum 0.19942. Its complete serialized payload was 2,273 B vs 2,465 B for sequential source codes and 2,977 B for explicit composite codes. The reported active-compute proxy was 65,696 vs 160 for sequential evaluation. Measured CPU latency was approximately 2.8–2.9 ms for the Mirror path and 0.31 ms for sequential calls. No training updates or fresh/audit runs occurred. Six payload files passed exact hash and byte replay; two tests passed.

## Interpretation

The preregistered quality gate (NRMSE2 <=1e-6) failed by a wide margin. The small byte saving versus sequential codes comes with substantial approximation error and compute/runtime cost, so this screen does not support replacing sequential composition with this one-call code.

## Hypothesis and limits

**H:** small Mirror codes could encode composite functions for one shared executor call. **T:** deterministic held-out phase-pair screen against exact sequential execution and explicit Fourier coefficients. **D:** FAIL for this fixed harmonic executor. **C:** harmonic truncation and unoptimized CPU evaluation may dominate; a wider basis may trade more bytes and compute for lower error. **U:** learned code composition, task-level quality, richer bases, optimized kernels, and fresh replication.

This is an analytic mechanism result, not a learned Neural Interpreter or natural-task capacity claim.
