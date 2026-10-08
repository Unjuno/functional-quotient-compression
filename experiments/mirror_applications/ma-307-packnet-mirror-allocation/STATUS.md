# MA-307 status

**FAIL for the strict registered Pareto gate; aligned storage point retained.**

## H / T / D / C / U

- **H:** View-first allocation can defer private weights for aligned tasks in a sequential stream while preserving earlier tasks, reducing total bytes versus PackNet-style masks followed by physical allocation.
- **T:** Eight tasks over a fixed 16-to-32 ReLU feature encoder; five Givens-aligned tasks followed by three unrelated task vectors. Support/validation/test sizes were 512/256/512. Development worlds 30710/30711 selected threshold 0.001 from {0.0001, 0.001, 0.01}; fresh worlds 30712–30714 used the locked threshold. Controls: hard tie, greedy packed binary mask then private allocation, independent readouts.
- **D:** Mirror used one shared readout, four one-byte view codes, and three private readouts in all fresh worlds. The PackNet-style control used seven private readouts after its task masks failed. Mirror payload averaged 854 B vs 1,333 B (35.9% less); mean normalized test MSE was 2.08e-5 vs approximately 9.2e-16 for PackNet/independent. Prior-task test error changed by 0. The repeated allocation-search median exceeded the 1.25x runtime gate in one of three fresh worlds (ratios 1.45x, 1.10x, 1.22x), so the strict composite Pareto gate did not pass.
- **C:** The first five tasks were deliberately generated from the same Givens orbit used by Mirror; an ordinary task stream may have fewer compressible tasks. The runtime threshold is sensitive for this small CPU workload.
- **U:** Full PackNet, optimizer-driven continual learning, natural task streams, learned routing, and near-convergence capacity are untested.

## Fact / Interpretation / Hypothesis

- **Fact:** Four aligned task views delayed private allocation in 3/3 fresh worlds; the last three unrelated tasks received private readouts. Total serialized bytes fell 35.9%; no previous task's test error changed. The predeclared per-world runtime ceiling was missed in one world.
- **Interpretation:** This is an aligned storage/allocation-delay frontier point with additional search compute and a small test-quality gap against private readouts. It is not a general PackNet replacement result.
- **Hypothesis:** A fused or amortized View search might preserve the byte reduction at lower compute, but requires a new preregistered experiment.

The protocol's `promising` gate required runtime <=1.25x in all fresh worlds, while its listed `fail` clauses did not cover a runtime-only miss. The operational FAIL disposition is for the overall registered Pareto claim; the positive storage/quality subresult is retained separately rather than described as a representational failure.
