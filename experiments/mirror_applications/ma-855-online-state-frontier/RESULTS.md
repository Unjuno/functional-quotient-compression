# MA-855 — NOT ESTABLISHED for the broad frontier; scoped Mirror attribution FAIL

## H / T / D / C / U

- **H:** Across one shared-subspace linear task family, compact support-adapted state may preserve quality with fewer bytes than full Hebb/Delta states; Mirror must beat the native/CAVIA-style control at matched budgets.
- **T:** Hidden linear teachers d_in=8, d_out=4 with shared mean plus rank-2 variation and σ=.05 residual. A 128-task, 64-example-per-task train bank fit shared mean and rank-4 decoder basis from observed supports only (8,192 examples; ~1.147M MAC proxy). Evaluation used 16 tasks/world, 32 random Gaussian support and 128 independent queries/task. Dev seed 29 selected rates delta=1.0, projected code=.29, context SGD=3.2; fresh seeds 73 and 89. Compared no adaptation, Hebb, full Delta, CAVIA-style support-gradient context proxy, Mirror projected Delta, native low-rank Delta, and support-only ridge. No outer-loop meta-training.
- **D:** Broad MA-855 frontier is **NOT ESTABLISHED** because CAVIA was a linear context proxy without outer-loop meta-training and only one synthetic family was tested. In this scope, Mirror-specific attribution **FAILS**: Mirror and native low-rank Delta are exactly identical. Fresh query nRMSE: full Delta .0506/.0627; CAVIA proxy .2139/.2054; Mirror/native .2279/.2250; Hebb .2418/.2813; ridge ~2–3e-6; no adaptation .4546/.5366.
- **C:** The learned train-task low-rank decoder and direct native low-rank parameterization explain the Mirror state savings; the broad family comparison is incomplete without meta-trained CAVIA and more task strata.
- **U:** Meta-trained CAVIA, learned Mirror decoder, multiple ranks/state budgets, anisotropic/noisy input strata, longer/interfering streams, and hardware throughput.

## Facts

Complete actual payloads are full Delta 2,781 B and Mirror/native low-rank 1,747 B (37.2% fewer). Float64 live task-state banks are 4,096 B full and 512 B low-rank; float32 serialized state snapshots are 2,048 B and 256 B. Full Delta online MAC proxy is 36,864 plus 65,536 query MACs; Mirror/native is 106,496 plus 67,584 query MACs. CAVIA proxy uses the same 1,747 B payload and 512 B live context, but 3,145,728 online gradient MACs plus 67,584 query MACs. Ridge uses full state and 77,824 fit MACs plus 65,536 query MACs. Shared basis training cost is reported separately (8,192 examples; ~1.147M MAC proxy; measured wall time ~0.006 s in the container). Fresh state reset uses the same state object cleared in place; query labels do not alter adaptation; serialized reload is byte-exact.

## Interpretation

The compact state reduces actual payload and live writable state, but this screen does not show a quality/compute Pareto improvement: full Delta is more accurate at lower online+query MACs. CAVIA proxy is somewhat more accurate than Mirror at identical bytes but far more expensive to adapt. The Mirror/native exact alias means these results describe ordinary low-rank state, not Mirror-specific capacity. The broad cross-family frontier remains unestablished.

## Amendment 02 disclosure

After fresh access, serialized metadata was made method-independent, live float64 bytes were separated from float32 snapshot bytes, and MAC proxies were corrected to match the executed operations. Rates and method choices did not change; deterministic fresh replay was used only to correct accounting and reset checks.
