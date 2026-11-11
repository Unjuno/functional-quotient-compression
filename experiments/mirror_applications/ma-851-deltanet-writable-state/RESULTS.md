# MA-851 — FAIL for Mirror-specific attribution; positive native state-compression screen

## H / T / D / C / U

- **H:** When stored values lie in a shared rank-2 output subspace, a rank-2 writable DeltaNet state preserves recall with lower serialized bytes and update/query compute than a full matrix.
- **T:** Synthetic D=32, rank 2, 16 orthogonal-key delta updates, 24 queries (16 stored keys plus 8 random); dev seed 19, fresh 41 and 67. Compared full matrix delta rule, native low-rank state, and Mirror low-rank state. No optimizer training; 16 support examples and 16 updates/world.
- **D:** **FAIL** for Mirror-specific attribution. Fresh nRMSE is 7.9e-8 / 8.9e-8 for full, and 1.26e-7 / 1.23e-7 for native and Mirror. Actual payloads are 4,407 B full, 1,065 B native low-rank, 1,085 B Mirror (75.4% Mirror reduction vs full). Low-rank update MAC proxy is 4,096 vs 33,280 full; query MACs 3,072 vs 24,576 full. Native low-rank matches Mirror function exactly and is 20 B smaller because of serialized names.
- **C:** Ordinary output-subspace factorization explains all storage/compute gains; Mirror adds no functional degree beyond native low-rank state.
- **U:** Nonlinear/noisy key distributions, varying rank, long sequences, learned basis, trained DeltaNet, and measured accelerator throughput.

## Facts

Both fresh worlds meet nRMSE < 1e-6. The paid low-rank payload includes 256 B shared basis and 256 B writable coefficient state; full writable matrix is 4,096 B. All arrays are serialized and reloaded for exact equality. Delta updates use observed support key/value pairs; the teacher does not provide a target matrix to the learner.

## Interpretation

This is a positive synthetic storage and arithmetic result for a native low-rank fast state under an exactly rank-2 value stream. It is a Mirror-specific FAIL because native low-rank state reproduces the same behavior at fewer bytes. No DeltaNet model or language-model claim follows.
