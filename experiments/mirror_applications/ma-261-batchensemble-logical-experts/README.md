# MA-261 — BatchEnsemble rank-one logical experts vs Mirror views

Status: SCREENING. Evidence lane: MECHANISM / STORAGE. Base: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: For routed expert functions, input-side Givens Mirror views recover an aligned multi-expert family with lower serialized state than BatchEnsemble rank-one member factors, while arbitrary independent expert functions require private parameters.

> **Mirror insertion:** this experiment adds per-role Givens coordinates `m_r` to one shared expert matrix so role-specific functions are expressed without duplicating the matrix.

PA17 requires rank-one fast-weight factors as direct control. MA-003 established the aligned Givens-expert case and independent experts as a private-parameter boundary. This experiment isolates multiplicative expert factors vs Mirror views.

## T

Four roles are assigned from signs of the first two input coordinates (fixed oracle router). Linear regression maps 8D inputs to 6D outputs. Aligned worlds use one shared matrix with role-specific Givens rotations; independent worlds use separate random matrices. Fit role maps by least squares on 512 examples per role and evaluate on 1024 held-out examples per role. Compare independent matrices, hard tying, BatchEnsemble rank-one outer-product factors, and Mirror input Givens views. Development seeds 17/31; fresh seeds 107/227/311/419. Post-fit screen; zero optimizer updates.

## Gates

PASS: at least 3/4 fresh aligned worlds have Mirror MSE <=1.10x independent, <=80% of independent bytes, and fewer bytes than BatchEnsemble. FAIL: Mirror aligned MSE >1.25x independent in >=3/4 worlds or it uses more bytes than independent without a quality gain. Oracle router, no learned routing.

Actual uncompressed NPZ payload bytes include all matrices, factors, biases and angles. Routed MAC proxy and evaluation time are reported; transform/runtime optimization is outside scope.

## Results / H-T-D-C-U

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
COUNTER-HYPOTHESIS: pending.  
UNCONFIRMED: pending.  
BOUNDARY: synthetic linear regression only; no language, learned routing, training-efficiency or capacity claim.

## Results — H / T / D / C / U

FACT: On fresh seeds 107, 227, 311, 419, aligned Givens worlds had mean routed MSE: Mirror 3.19e-25, independent 6.17e-30, BatchEnsemble rank-one 0.1722, hard tying 0.2259. Actual payloads were Mirror 1,006 B, BatchEnsemble 1,562 B, hard tying 642 B, independent 1,794 B. Mirror used 56.1% of independent bytes and 64.4% of BatchEnsemble bytes; the aligned PASS gate was met 4/4. In independent-expert worlds, mean MSE was Mirror 1.467, BatchEnsemble 0.893, hard tying 2.078, independent numerical zero. Training examples per seed/condition were 2,048; optimizer updates were zero. The role assignment was an oracle sign router. Serialization roundtrip reconstructed exact expert matrices.

INTERPRETATION: A low-dimensional role orbit can be recovered with a compact functional coordinate. For unrelated role matrices, the coordinate is insufficient; even the rank-one control leaves residual error, while independent experts fit exactly. The lower bytes vs BatchEnsemble are tied to the aligned teacher.

HYPOTHESIS: Structured expert coordinates can replace a portion of expert-specific state when expert differences lie on the chosen orbit; orthogonal unrelated variation needs private state.

COUNTER-HYPOTHESIS: The teacher was generated from the Givens family, so the result demonstrates matched-family compression. Standard learned routing and deep nonlinear experts could change the ordering.

UNCONFIRMED: learned router quality/cost, neural or LM quality, near-convergence at fixed bytes, top-k compositions, and optimized runtime.

Decision: PROMISING for aligned synthetic mechanism; independent-expert capacity boundary confirmed only in this linear oracle-routed screen.
