# MA-292 — task-vector Mirror basis screen

Status: **FAIL**  
Evidence lane: MECHANISM / STORAGE / TASK-ARITHMETIC  
Base commit: `e259f27`

## H — hypothesis

For task deltas lying on a shared two-direction, fixed-radius orbit, two shared physical basis vectors plus one per-task angle can retain task functions and signed task-vector compositions with fewer incremental bytes than a two-coefficient SVD basis. Independent task deltas should expose the private-parameter boundary. A Mirror-specific claim requires quality/compute comparison against the native SVD coefficient control.

## Selection

Draw from `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-288, MA-292, MA-296, MA-299]` with Python `secrets.randbelow(9)`. Draw index 6 selected MA-292. Candidate pool, value, and no-branch check are recorded in `PROTOCOL.json`.

## T — protocol

The NumPy CPU harness creates a shared pretrained linear model and a task-vector stream. Eight support task deltas discover a shared rank-2 basis. Eight new task deltas are compressed and evaluated per world. In the aligned condition, fresh task deltas occupy a fixed-radius circle in the discovered two-dimensional subspace. In the independent condition, task deltas are unrelated isotropic vectors. Each delta defines a scalar linear task `f(x)=xᵀ(θ₀+Δ)`; isotropic held-out inputs make function-output MSE proportional to delta reconstruction error.

Controls: shared-only zero delta, one-angle Mirror in the SVD-derived rank-2 basis, ordinary two-coefficient rank-2 SVD, rank-1 SVD, matched-byte rank-2 SVD with two FP16 coefficients, and independent full task vectors. Signed pairwise task arithmetic is audited by comparing reconstructed pair sums and differences with the corresponding full-vector result. All basis, task codes, task IDs, tensor metadata, and base weights are charged in a deterministic binary payload. Inference timing executes the saved coordinate path.

Development seeds 29201/29202 selected eight support task deltas (mean task MSE 0.034922 vs 0.036428 for four). Fresh seeds 29211–29213 use the frozen setting.

## Gates

PASS for the narrow aligned storage claim if all three fresh worlds have task-output MSE <=1.10× rank-2 SVD, signed pair-composition MSE <=1.10×, incremental bytes/task <=0.90×, and coordinate-path inference throughput >=0.5× SVD. Mirror-specific value additionally requires a better quality/byte/compute frontier than both FP32 and matched-byte FP16 rank-2 SVD. Independent tasks locate the private-vector boundary. This is a fixed task-vector compression screen, not language-model capacity evidence.

## D — decision

**FAIL for a Mirror-specific Pareto claim.** The broad aligned quality/byte/runtime screen against FP32 SVD passed in 3/3 worlds. The same-basis FP16 SVD control dominates Mirror: same per-task code bytes, 23 B smaller total payload, slightly better quality in all fresh worlds, and roughly 1.55x the coordinate-path throughput.

## Fact / interpretation / hypothesis

**Fact.** Fresh aligned task MSE was 0.0010051 for Mirror, 0.0010018 for FP32 SVD, and 0.0010018 for matched-byte FP16 SVD. The signed pair-composition errors were 0.0020101, 0.0020036, and 0.0020036 respectively. Mirror used 26 incremental bytes/task vs 30 B for FP32 SVD, but 26 B also for FP16 SVD; total payloads were 1,875 B (Mirror), 1,879 B (FP32 SVD), and 1,852 B (FP16 SVD). Mirror inference throughput was 19.0M examples/s vs 29.9M for FP32 SVD and 29.4M for FP16 SVD. The Mirror/FP32 quality ratios were 1.0030, 1.0023, and 1.0043 across the three fresh worlds, satisfying the broad gate. Its ratio to FP16 quality was also slightly worse in all three worlds; FP16 was faster and used fewer total bytes. On independent deltas, Mirror MSE was 0.07375, FP32/FP16 rank-2 SVD was 0.06234, and independent full vectors were near zero.  
**Interpretation.** A one-angle coordinate represents the aligned two-dimensional task-vector orbit with small distortion and saves one 4-byte coefficient per task against FP32 rank-2 SVD. That is a real but small physical saving. The simple FP16 coefficient control has equal incremental bytes, lower total payload, slightly lower error, and higher throughput, so Mirror adds no Pareto improvement. Unrelated task vectors require private or richer coordinates.  
**Hypothesis.** Fixed-radius aligned deltas admit a scalar phase code; arbitrary directions need more task-private state.

## C — strongest counter-hypothesis

Ordinary rank-2 SVD already stores the exact coordinates in the learned subspace; a one-angle code may save only four bytes per task while introducing norm mismatch and extra runtime. Sparse/top-k task-vector compression could also dominate both.

## U — unresolved

No Transformer checkpoint, natural task arithmetic, TIES/DARE, optimizer dynamics, or near-converged capacity frontier is tested. The eight support deltas are training inputs for the per-world codebook and are not part of the evaluated task count; their learned information is paid through the serialized shared basis.
