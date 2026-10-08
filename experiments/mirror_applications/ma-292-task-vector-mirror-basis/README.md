# MA-292 — task-vector Mirror basis screen

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / TASK-ARITHMETIC  
Base commit: `e259f27`

## H — hypothesis

For task deltas lying on a shared two-direction, fixed-radius orbit, two shared physical basis vectors plus one per-task angle can retain task functions and signed task-vector compositions with fewer incremental bytes than a two-coefficient SVD basis. Independent task deltas should expose the private-parameter boundary. A Mirror-specific claim requires quality/compute comparison against the native SVD coefficient control.

## Selection

Draw from `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-288, MA-292, MA-296, MA-299]` with Python `secrets.randbelow(9)`. Draw index 6 selected MA-292. Candidate pool, value, and no-branch check are recorded in `PROTOCOL.json`.

## T — protocol

The NumPy CPU harness creates a shared pretrained linear model and a task-vector stream. Four support task deltas discover a shared rank-2 basis. Eight new task deltas are compressed and evaluated per world. In the aligned condition, fresh task deltas occupy a fixed-radius circle in the discovered two-dimensional subspace. In the independent condition, task deltas are unrelated isotropic vectors. Each delta defines a scalar linear task `f(x)=xᵀ(θ₀+Δ)`; isotropic held-out inputs make function-output MSE proportional to delta reconstruction error.

Controls: shared-only zero delta, one-angle Mirror in the SVD-derived rank-2 basis, ordinary two-coefficient rank-2 SVD, rank-1 SVD, matched-byte rank-2 SVD with two FP16 coefficients, and independent full task vectors. Signed pairwise task arithmetic is audited by comparing reconstructed pair sums and differences with the corresponding full-vector result. All basis, task codes, task IDs, tensor metadata, and base weights are charged in a deterministic binary payload. Inference timing executes the saved coordinate path.

Development seeds 29201/29202 selected eight support task deltas (mean task MSE 0.034922 vs 0.036428 for four). Fresh seeds 29211–29213 use the frozen setting.

## Gates

PASS for the narrow aligned storage claim if all three fresh worlds have task-output MSE <=1.10× rank-2 SVD, signed pair-composition MSE <=1.10×, incremental bytes/task <=0.5×, and coordinate-path inference throughput >=0.5× SVD. Mirror-specific value additionally requires a better quality/byte/compute frontier than both FP32 and matched-byte FP16 rank-2 SVD. Independent tasks locate the private-vector boundary. This is a fixed task-vector compression screen, not language-model capacity evidence.

## Fact / interpretation / hypothesis

**Fact:** pending frozen development and fresh results.  
**Interpretation:** limited to synthetic linear task vectors.  
**Hypothesis:** fixed-radius aligned deltas may admit a scalar phase code; arbitrary directions will need more task-private state.

## C — strongest counter-hypothesis

Ordinary rank-2 SVD already stores the exact coordinates in the learned subspace; a one-angle code may save only four bytes per task while introducing norm mismatch and extra runtime. Sparse/top-k task-vector compression could also dominate both.

## U — unresolved

No Transformer checkpoint, natural task arithmetic, TIES/DARE, optimizer dynamics, or near-converged capacity frontier is tested. The four support deltas are training inputs for the per-world codebook and are not part of the evaluated task count; their learned information is paid through the serialized shared basis.
