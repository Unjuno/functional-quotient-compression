# MA-273 — BOFT Mirror adapter bank

Status: SCREENING (protocol frozen)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA21 BOFT.

## H — falsifiable hypothesis

When task orthogonal adapters share one butterfly-angle direction, a shared BOFT physical atom bank plus a scalar Mirror task coordinate retains per-task quality with fewer actual bytes and lower transform cost than storing independent BOFT factors per task. If a simple low-rank residual matches, the result is not Mirror-specific.

## Mirror insertion

> **Mirror insertion:** this experiment adds one scalar task coordinate `m_t` over a shared 4-stage butterfly orthogonal atom bank, so many task-specific transforms can be recovered without storing a full butterfly factor table for each task.

- Shared physical object: one 16x16 base map and one 4x8 butterfly angle bank.
- Native BOFT: per-task independent 4x8 butterfly angles.
- Mirror code: scalar multiplier per task applied to shared angle bank.
- Controls: independent full maps, shared map, native per-task BOFT, rank-2 additive residual.

## Protocol

Four tasks, 16D in/out. Teacher task transforms use a shared random butterfly-angle atom scaled by task coefficient. Dev worlds 27300/27301, LR .003/.01, 1,000 AdamW updates, batch128. Fresh worlds 27302–27304 sealed unless dev passes.

**PASS:** Mirror MSE <=1.10x independent; payload <=60% independent; and beats native BOFT by >=10% MSE at byte-near cost or lower bytes. **FAIL:** misses gate or native BOFT/simple residual matches at equal/lower bytes.

## Boundaries

This is an intentionally factorized BOFT teacher screen. It does not claim general BOFT compression, arbitrary adapter capacity, language quality, or near-convergence fixed-byte capacity.


### Development initialization audit

The first dev pass is invalidated: code audit found target butterfly atom/codes copied into Mirror initialization and target angle tables copied into BOFT initialization. Before any fresh access, protocol is amended to train all method coordinates from neutral initialization, retaining first-pass rows separately as invalidated diagnostics.

## D — FAIL (corrected development screen)

The neutral-initialization development selector chose LR 0.01. Across the two dev worlds, Mirror mean task MSE was about 0.00060–0.00353, while native independent BOFT angles were about 0.00037–0.00170 and independent full maps were near 1e-9. Mirror's payload was 3,593 B versus 3,787 B for native per-task BOFT, only a 5% reduction, while task quality was worse and far from the independent upper. It failed the quality gate. Fresh worlds 27302–27304 were not opened.

The initial dev pass is invalidated because it copied teacher atom/codes into candidate initialization; its rows are preserved in `INVALIDATED_INITIAL_RESULTS.csv`. The corrected run removes that leakage and provides the decision.

### C — strongest counter-hypothesis

The shared factorized angle atom and scalar task codes are bilinear and poorly conditioned from neutral initialization; the 1,000-update optimizer may not discover the task factorization. This is an optimization failure signal, not proof of representational impossibility. The native BOFT task-specific table also has 32 coordinates per task and is closer to the teacher's target.

### U — not established

Fresh replication, near-convergence capacity, learned shared/private angle atoms, arbitrary task transforms, and whether initialization/optimization rescues Mirror remain untested.

### Fact / interpretation / hypothesis

- **Fact:** Corrected dev used two worlds, two LRs, five methods and 1,000 updates. Actual serialized bytes include task codes and bases. Fresh seeds stayed sealed.
- **Interpretation:** The scalar-code factorization did not deliver useful task functions from neutral initialization within this screen and saved little over BOFT.
- **Hypothesis:** Better atom initialization or a private angle residual could rescue quality, but would require a separately preregistered extension and may erase the small byte advantage.
