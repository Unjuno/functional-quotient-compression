# MA-455 — Sequential Mirror program over one physical block

Status: **FAIL** (development screen; fresh seeds sealed)  
Evidence lane: MECHANISM/STORAGE/RUNTIME  
Branch: `research/ma-455-sequential-mirror-program-20261008`  
Base commit: `04554b5`  
Prior art: PA81 (Routing Networks)

## H — Hypothesis

One physical 3×3 matrix reused at three sequential steps, with a distinct three-angle Givens View at each step, could approximate an ordered non-commutative function with low error and at least 25% fewer actual bytes than a three-matrix stack. Reversing the learned View order should damage quality. A direct native Givens parameterization determines whether any gain is Mirror-specific.

## T — What ran

The frozen protocol is `PROTOCOL.json` (SHA-256 recorded in `STATUS.md`). The synthetic teacher is `M₂ M₁ M₀`, with `Mⱼ=Rⱼ W Rⱼᵀ`, dimension 3, Gaussian inputs and output noise σ=0.01. Each method trained for 600 Adam updates × 16 examples (9,600 examples) and was evaluated on the same 512 held-out examples for each seed. Development seeds: 45501 and 45502. Controls: ordinary tied block, three independent blocks, per-step rank-1 residual, direct native Givens conjugation, and reversed Mirror step order. Fresh seeds 45511–45513 remained sealed after the frozen gates failed.

Actual payload is the uncompressed serialized `.npz`, including tensors and schema metadata. Inference ops count three vector/matrix applications and amortize deterministic view-matrix reconstruction across 512 held-out examples; cold-start reconstruction is separately reported. The per-batch training proxy charges reconstruction per 16-example update. Tiny single-batch query wall times are noisy and are not reliable latency evidence.

## Results — facts

| Seed | Method | Held-out RMSE | Actual bytes | Ops/example | Train wall (s) | Reverse-order RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 45501 | tied shared block | 0.27708 | 695 | 27.000 | 0.318 | — |
| 45501 | Mirror | 0.16023 | 1,013 | 27.316 | 1.099 | 1.23167 |
| 45501 | native Givens | 0.16023 | 1,013 | 27.316 | 1.058 | — |
| 45501 | rank-1 residuals | 0.01064 | 1,284 | 54.000 | 0.511 | — |
| 45501 | independent blocks | 0.01073 | 770 | 27.000 | 0.349 | — |
| 45502 | tied shared block | 0.05944 | 695 | 27.000 | 0.309 | — |
| 45502 | Mirror | 0.01216 | 1,013 | 27.316 | 1.275 | 0.83760 |
| 45502 | native Givens | 0.01216 | 1,013 | 27.316 | 1.152 | — |
| 45502 | rank-1 residuals | 0.01017 | 1,284 | 54.000 | 0.513 | — |
| 45502 | independent blocks | 0.01123 | 770 | 27.000 | 0.313 | — |

The step-order probe shows that the fitted maps are order-sensitive: reversing Views raised RMSE on both seeds. However, the Mirror payload was larger than the independent stack (1,013 vs 770 bytes, +31.6%) in both seeds. Seed 45501 also missed the 0.05 quality gate. Native Givens has exactly the same payload hash, byte count, and query output as Mirror on both seeds. The independent and rank-1 controls reached similar or better fit. Full rows and checksums are in `RESULTS_CORE.csv`, `runs/dev_*/metrics.json`, and `verification_report.json`.

## D — Decision

**FAIL.** The fixed gates require both low error and <=75% of independent-stack bytes, plus no exact native Givens alias. The reversal probe passes, and the forward active compute proxy is comparable to the independent stack, but the actual payload is larger, one seed misses the target quality, and the representation is exactly a native Givens conditioner. This experiment does not establish a storage Pareto gain or Mirror-specific functionality.

## C — Strongest counter-hypothesis

Any useful change is explained by ordinary per-step Givens-conditioned linear maps. The native control exactly reproduces Mirror, and a three-matrix bank or rank-1 residual bank fits at least as well. The ordered task tests composition, but this coordinate code does not compress it.

## U — Unconfirmed

This is a small aligned synthetic mechanism screen. It does not test natural tasks, learned routing, near-convergence capacity, broader storage/compute frontiers, or stable deployment latency. Fresh seeds were not opened by the frozen decision rule.
