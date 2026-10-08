# MA-271 — OFT task views vs shared Mirror orthogonal orbit

Status: **FAIL for Mirror-specific advantage (an exact simple rank-one control matches the Mirror state and function)**. Evidence lane: MECHANISM / STORAGE / RUNTIME. Base: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: For task transforms lying on a shared low-dimensional orthogonal orbit, one shared orthogonal generator family plus low-description task addresses can preserve function and geometry at lower bytes than independent OFT transforms. A simple rank-one task-code × transform-basis control may match Mirror exactly, limiting any Mirror-specific claim.

> **Mirror insertion:** this experiment adds scalar task coordinates `m_t` to a shared bank of plane-rotation angles `φ_j`, forming orthogonal task-specific views without storing each dense task matrix.

PA20 established orthogonal finetuning. Controls are full OFT matrices, blockwise OFT angles, and simple rank-one shared angle-template factorization. Development showed the Mirror parameterization is mathematically identical to that rank-one control. Before fresh access both use identical packed serialized state so container/header overhead cannot create a false advantage.

## T

8D frozen linear block, six tasks, four 2D rotation planes. Teacher task Q is block diagonal with plane angles `m_t * φ_j`. Fit/recover from 512 train and 1024 held-out examples per task. Development seeds 27/41; fresh 127/239/331/443. Post-fit screen, zero optimizer updates. Measure MSE, orthogonality error, inverse-cycle error, bytes and transform application time.

## Gates

PASS: at least 3/4 fresh seeds Mirror MSE <=1.10x full OFT, <=80% OFT bytes, orthogonality/cycle error <1e-10, and lower bytes than the simple rank-one control. FAIL: >1.25x OFT MSE in >=3/4 or no byte-quality gain vs the simple control.

Charge shared W, full/blockwise Q states, shared angle template, task addresses, and reconstruction state. Store actual NPZ payload bytes. Transform latency is a small CPU microbenchmark.

## Results — H / T / D / C / U

FACT: Across four fresh aligned worlds, Mirror and the simple rank-one angle factor control were exactly the same functions and stored exactly 1,086 bytes under identical serialization, with mean MSE 6.18e-31. Blockwise OFT stored 1,198 bytes and full dense OFT 4,078 bytes, each at numerical-zero MSE. All transforms had orthogonality errors <=3.2e-16 and inverse-cycle errors <=2.3e-16. On independent plane-angle tasks, Mirror and rank-one control were again identical at MSE 0.1601, while OFT controls remained numerical-zero. No optimizer updates; each task condition used 3,072 fit examples.

INTERPRETATION: A one-dimensional angle factorization compresses shared-orbit task transforms relative to independent OFT, but the same mechanism is ordinary rank-one factorization and has no Mirror-specific advantage. Arbitrary task transforms need more coordinates.

HYPOTHESIS: Shared low-dimensional orthogonal orbits may be an efficient parameterization, while factorization rank sets the task-diversity boundary.

COUNTER-HYPOTHESIS: This experiment chose a teacher exactly on the factorized orbit; larger generators or non-orthogonal updates may change the tradeoff.

UNCONFIRMED: learned fine-tuning, high-dimensional workloads, diffusion quality, throughput under optimized input-side kernels, and fixed-byte training.

Decision: FAIL for Mirror-specific advantage. The orthogonal-orbit compression mechanism works, but the simple non-Mirror rank-one control reproduces it exactly.

## Protocol variant reconciliation

The separate `protocol_variants/task_views_vs_dense_oft/` directory preserves a second frozen MA-271 study. That 16D trained task-map screen used eight task-specific Givens angles and compared Mirror to dense OFT, low-rank residuals and independent maps. It reported PROMISING: 3/3 fresh worlds passed its quality/storage gate at 3,309B vs 7,253B dense OFT, although eager CPU throughput was only 0.19–0.21M examples/s. This screen did not include the exact simple rank-one task-code factorization.

The root post-fit experiment added that strongest control. Across four fresh worlds, Mirror and ordinary rank-one angle factorization were exactly the same functions and both serialized to 1,086B. Therefore MA-271 is FAIL for a Mirror-specific contribution, while the aligned orthogonal-orbit compression result remains valid as a generic structured/rank-one factorization result. Do not pool their seeds or claim they are replications.

Source branches: exact-control audit `research/ma-271-oft-mirror-views-20261008` (result commit `eb9211c5830564363e10cdff71d1a5e34ac8f6e5`); trained dense-OFT cross-over `research/ma-271-oft-mirror-task-views-20261008` (report commit `21149d0505655d3a899d079f0302c16b72b9eb00`).
