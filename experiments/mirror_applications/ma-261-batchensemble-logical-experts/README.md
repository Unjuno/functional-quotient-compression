# MA-261 — BatchEnsemble rank-one logical experts vs Mirror views

Status: **FAIL (frozen relative-MSE gate met in only 1/4 fresh worlds; a distinct fixed-update variant also failed)**. Evidence lane: MECHANISM / STORAGE. Base: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: For routed expert functions, input-side Givens Mirror views recover an aligned multi-expert family with lower serialized state than BatchEnsemble rank-one member factors, while arbitrary independent expert functions require private parameters.

> **Mirror insertion:** this experiment adds per-role Givens coordinates `m_r` to one shared expert matrix so role-specific functions are expressed without duplicating the matrix.

PA17 requires rank-one fast-weight factors as direct control. MA-003 established the aligned Givens-expert case and independent experts as a private-parameter boundary. This experiment isolates multiplicative expert factors vs Mirror views.

## T

Four roles are assigned from signs of the first two input coordinates (fixed oracle router). Linear regression maps 8D inputs to 6D outputs. Aligned worlds use one shared matrix with role-specific Givens rotations; independent worlds use separate random matrices. Fit role maps by least squares on 512 examples per role and evaluate on 1024 held-out examples per role. Compare independent matrices, hard tying, BatchEnsemble rank-one outer-product factors, and Mirror input Givens views. Development seeds 17/31; fresh seeds 107/227/311/419. Post-fit screen; zero optimizer updates.

## Gates

PASS: at least 3/4 fresh aligned worlds have Mirror MSE <=1.10x independent, <=80% of independent bytes, and fewer bytes than BatchEnsemble. FAIL: Mirror aligned MSE >1.25x independent in >=3/4 worlds or it uses more bytes than independent without a quality gain. Oracle router, no learned routing.

Actual uncompressed NPZ payload bytes include all matrices, factors, biases and angles. Routed MAC proxy and evaluation time are reported; transform/runtime optimization is outside scope.

## Results — H / T / D / C / U

FACT: On fresh seeds 107, 227, 311, 419, aligned Givens worlds had mean routed MSE: Mirror 3.19e-25, independent 6.17e-30, BatchEnsemble rank-one 0.1722, hard tying 0.2259. Actual payloads were Mirror 1,006 B, BatchEnsemble 1,562 B, hard tying 642 B, independent 1,794 B. Mirror used 56.1% of independent bytes and 64.4% of BatchEnsemble bytes. Recomputing the frozen per-world relative-MSE criterion from `RESULTS_CORE.csv` gives Mirror/independent ratios of 1,390x, 302,057x, 11.7x and 0.227x: the <=1.10x criterion passed only 1/4, not the reported 4/4. In independent-expert worlds, mean MSE was Mirror 1.467, BatchEnsemble 0.893, hard tying 2.078, independent numerical zero. Training examples per seed/condition were 2,048; optimizer updates were zero. The role assignment was an oracle sign router. Serialization roundtrip reconstructed exact expert matrices.

INTERPRETATION: The decoded Mirror functions have tiny absolute error on the aligned teacher and use fewer bytes than both independent and the implemented BatchEnsemble control. However, the preregistered relative-to-independent quality gate is ill-conditioned near numerical zero and passes only 1/4; the formal registered result is FAIL. For unrelated role matrices, the coordinate is insufficient; even the rank-one control leaves residual error, while independent experts fit exactly.

HYPOTHESIS: Structured expert coordinates can replace a portion of expert-specific state when expert differences lie on the chosen orbit; orthogonal unrelated variation needs private state.

COUNTER-HYPOTHESIS: The teacher was generated from the Givens family, so the result demonstrates matched-family compression. Standard learned routing and deep nonlinear experts could change the ordering.

UNCONFIRMED: learned router quality/cost, neural or LM quality, near-convergence at fixed bytes, top-k compositions, and optimized runtime.

Decision: FAIL the literal preregistered relative-MSE gate; preserve the tiny absolute aligned error and lower-byte observation as descriptive evidence, not a PASS. The independent-expert boundary is confirmed only in this linear oracle-routed screen.

## Protocol variant reconciliation

The separate folder `protocol_variants/fixed_update_failure/` retains a second frozen protocol under this stable MA ID. It trained a 16D-to-8D top-1 two-expert regression model for 1,200 updates on two development worlds and failed its quality gate: BatchEnsemble rank-one and independent experts reached approximately 1e-10 / <=1e-8 MSE, while Givens Mirror stayed near 1.8. Fresh worlds were not opened.

The two protocols test different structures and optimization regimes and are not pooled as replication data. The post-fit 8D-to-6D screen has tiny absolute aligned error and lower payload than its controls, but fails its literal frozen 1.10x relative-MSE threshold in 3/4 worlds. The fixed-update variant independently fails during development, demonstrating that another expert partition can be outside this View family.

Source branches: original fresh report `research/ma-261-batchensemble-logical-experts-20261008` (verified result commit `95ed3c9c8bfc15d241b5433bbd0466e1f2b86c46`); distinct fixed-update negative `research/ma-261-rankone-mirror-experts-20261008` (report commit `f3a22dacac2039c4a664038f65b299915f482f34`). The original fresh branch's “PASS 4/4” wording is retained in its source branch but corrected here against the actual frozen threshold and recorded per-seed values.
