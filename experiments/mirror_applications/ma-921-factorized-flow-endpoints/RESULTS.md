# MA-921 result — scoped analytic endpoint screen

## H — hypothesis

Learned start and end codes composed through one shared flow map improve held-out interval-pair map and semigroup quality over native continuous `(s,t)` conditioning at a useful serialized-byte cost.

## T — what ran

- Frozen amendment 02: three fresh analytic 2D linear ODE systems, seeds 92121/92123/92129; 8 endpoint values and 7 held-out ordered endpoint pairs per system.
- Same 600 updates, batch size 128, 32-wide shared MLP and paired initialization/data across native, Mirror and ordinary additive endpoint-code controls.
- The ordinary endpoint-embedding control has the same parameterization as the Mirror code. Exact ODE maps were the targets.
- Actual state-dict bytes measured with a common serialization filename. Checkpoint reload max output difference was 0. Repeated Mirror/control metrics were identical because their implementations are equivalent.
- Amendment 01 results are preserved in `source/pre_amendment/results_init_rng01.json` and excluded: shared RNG consumption produced different initial shared-MLP weights and training minibatches across conditions. A first serialization pass also included condition-dependent archive filename overhead; final bytes use a common filename.

## D — FAIL for this additive endpoint-code mechanism

### Fact

| Mean over 3 fresh systems | Native `(s,t)` | Mirror endpoint codes | Change |
|---|---:|---:|---:|
| Held-out pair map MSE | 0.06330 | 0.10960 | +73.1% worse |
| Semigroup composition MSE | 0.07554 | 0.07212 | −4.5% |
| Serialized inference state | 7,893 B | 8,461 B | +7.2% |
| MLP MAC proxy | 1,216 | 1,216 | same before endpoint lookup/add |
| Training wall clock | 1.66 s | 1.84 s | +10.7% |

The endpoint-code Mirror and ordinary additive endpoint-embedding control matched exactly in all map/composition metrics and bytes. Native won the held-out map gate on all 3/3 fresh systems. The small average composition change did not meet the joint quality gate.

### Interpretation

The tested learned endpoint table hurts held-out map quality and adds state. The continuous native time values appear easier to interpolate in these smooth linear systems. The exact Mirror/control alias shows no Mirror-specific value for this construction. Endpoint lookup and addition add operations not included in the MLP MAC proxy, so compute is not improved.

### Hypothesis

Factorized endpoint codes may overfit observed endpoint pairs or discard useful continuous distance structure. This result is limited to additive low-dimensional codes and analytic ODE flows; nonlinear/image flow maps remain untested.

## C — strongest counter-hypothesis

An additive endpoint table is the wrong factorization for smooth flows; Fourier endpoint bases, higher-rank endpoint interactions, or nonlinear pair codes may generalize better, while ordinary time embeddings can likely supply the same gains.

## U — unresolved

PA260 image generation/FID, nonlinear flows, near-convergence, alternate endpoint bases, independent per-interval upper control, and whether a conventional structured time embedding can recover useful quality without the extra endpoint table.
