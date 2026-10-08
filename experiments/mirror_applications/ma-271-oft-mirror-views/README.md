# MA-271 — OFT task views vs shared Mirror orthogonal orbit

Status: SCREENING. Evidence lane: MECHANISM / STORAGE / RUNTIME. Base: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: For task transforms lying on a shared low-dimensional orthogonal orbit, one shared orthogonal generator family plus low-description task addresses can preserve function and geometry at lower bytes than independent OFT transforms. A simple rank-one task-code × transform-basis control may match Mirror exactly, limiting any Mirror-specific claim.

> **Mirror insertion:** this experiment adds scalar task coordinates `m_t` to a shared bank of plane-rotation angles `φ_j`, forming orthogonal task-specific views without storing each dense task matrix.

PA20 established orthogonal finetuning. Controls are full OFT matrices, blockwise OFT angles, and simple rank-one shared angle-template factorization. Development showed the Mirror parameterization is mathematically identical to that rank-one control. Before fresh access both use identical packed serialized state so container/header overhead cannot create a false advantage.

## T

8D frozen linear block, six tasks, four 2D rotation planes. Teacher task Q is block diagonal with plane angles `m_t * φ_j`. Fit/recover from 512 train and 1024 held-out examples per task. Development seeds 27/41; fresh 127/239/331/443. Post-fit screen, zero optimizer updates. Measure MSE, orthogonality error, inverse-cycle error, bytes and transform application time.

## Gates

PASS: at least 3/4 fresh seeds Mirror MSE <=1.10x full OFT, <=80% OFT bytes, orthogonality/cycle error <1e-10, and lower bytes than the simple rank-one control. FAIL: >1.25x OFT MSE in >=3/4 or no byte-quality gain vs the simple control.

Charge shared W, full/blockwise Q states, shared angle template, task addresses, and reconstruction state. Store actual NPZ payload bytes. Transform latency is a small CPU microbenchmark.

## H / T / D / C / U

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
COUNTER-HYPOTHESIS: pending.  
UNCONFIRMED: pending.  
BOUNDARY: synthetic orthogonal linear orbit; no diffusion/LM/fine-tuning/capacity claim.

## Results — H / T / D / C / U

FACT: Across four fresh aligned worlds, Mirror and the simple rank-one angle factor control were exactly the same functions and stored exactly 1,086 bytes under identical serialization, with mean MSE 6.18e-31. Blockwise OFT stored 1,198 bytes and full dense OFT 4,078 bytes, each at numerical-zero MSE. All transforms had orthogonality errors <=3.2e-16 and inverse-cycle errors <=2.3e-16. On independent plane-angle tasks, Mirror and rank-one control were again identical at MSE 0.1601, while OFT controls remained numerical-zero. No optimizer updates; each task condition used 3,072 fit examples.

INTERPRETATION: A one-dimensional angle factorization compresses shared-orbit task transforms relative to independent OFT, but the same mechanism is ordinary rank-one factorization and has no Mirror-specific advantage. Arbitrary task transforms need more coordinates.

HYPOTHESIS: Shared low-dimensional orthogonal orbits may be an efficient parameterization, while factorization rank sets the task-diversity boundary.

COUNTER-HYPOTHESIS: This experiment chose a teacher exactly on the factorized orbit; larger generators or non-orthogonal updates may change the tradeoff.

UNCONFIRMED: learned fine-tuning, high-dimensional workloads, diffusion quality, throughput under optimized input-side kernels, and fixed-byte training.

Decision: FAIL for Mirror-specific advantage. The orthogonal-orbit compression mechanism works, but the simple non-Mirror rank-one control reproduces it exactly.
