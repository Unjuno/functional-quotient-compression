# MA-1166 Phase E: synchronous parallel virtual-Mirror graph — verified CPU experiment

**Date:** 2026-10-20 JST. **Status:** 36/36 paired synthetic runs COMPLETE; 8/8 structural/gradient unit tests PASS; 36/36 independently regenerated checkpoints, serialized payloads and fresh/OOD outputs PASS. Prior phases A-D not changed.

## Research question
Can one physical learned tensor generate a large batch of virtual MLP functions and optimize all of them through one backward pass? Does synchronous Jacobi execution of the full virtual graph improve generalization versus depth-ordered serial computation at the **same physical coefficient count**? Does nonlinear Mirror decoding specifically outperform a matched direct affine decoder?

## Exact model / protocol
- **ONE** trainable physical tensor of **442 float32 scalars**, consisting of shared core 64, edge gates 126, orientation/address codes 252.
- Reconstruct **126 virtual native MLP weight sets** of 110 coefficients each = **13,860 logical coefficients**, *not* 13,860 independent learned scalars.
- Nonlinear virtual-weight Mirror uses two code-controlled Householder rotations: `z=core@projection + fixed_edge`; `w=z+0.65 Q^-1 tanh(2.2 Qz+fixed_bias)`. The network MLPs are 6→8→6 with tanh.
- **SYNC:** six graph state nodes, all 42 source-destination pairs (6 input anchors + 6×6 including backward/self/long skips) × three views = 126 edge candidates; vectorized edge MLP evaluation per update; one-step and three-step versions; all edges active.
- **SERIAL:** all 21 forward node-pairs × six views = 126 edge candidates, depth-ordered evaluation.
- **Matched controls:** 442-scalar direct-affine serial and SYNC3 variants plus 442-scalar native residual 6→33→6 MLP with 7 additional output scale coefficients. Both Mirror and affine modes store their 442 scalars in exactly one learned tensor.
- 2 teacher families (`sync_aligned`/same Mirror decoder architecture and `serial_independent`/native MLP edges) × 3 independent seeds (461,463,467) × 6 modes = **36**. Train 2048, development 1024, fresh 1024 and OOD 1024, Adam 300 updates, batch 128, lr0.015, CPU FP32 one thread. No pruning, no sparsity penalty, no architecture threshold. **The trained graph retains all 126 candidate edges at inference.**

## Results: fresh normalized MSE (lower better)
| Mode | Aligned teacher 3-seed median | Independent teacher 3-seed median | Global 6-case median | Edge-MLP evals per input | CPU batch1 latency median µs | Actual serialized inference B median |
|---|---:|---:|---:|---:|---:|---:|
| mirror_serial | 0.060888 | 0.254014 | 0.153093 | 126 | 2585.5 | 2013 |
| mirror_sync1 | 0.057165 | 0.298020 | 0.161857 | 126 | 2730.3 | 2010 |
| **mirror_sync3** | **0.048882** | **0.211729** | **0.135797** | **378** | **2952.9** | **2010** |
| affine_serial | 0.052721 | 0.221409 | 0.145318 | 126 | 2733.6 | 2013 |
| affine_sync3 | 0.042863 | 0.203843 | 0.120560 | 378 | 1427.5 | 2010 |
| native442 | 0.009209 | 0.041752 | 0.025186 | — | 19.1 | 2021 |

Within identical teacher+seed pairs, SYNC3 Mirror beats depth-ordered serial Mirror **5/6** (paired NMSE ratio median 0.816) and SYNC1 Mirror **6/6** (ratio median 0.798). However, SYNC3 Mirror beats affine SYNC3 **1/6** (ratio median 1.089), and native442 **0/6** (ratio median 5.253).

The favorable SYNC3/SERIAL comparison uses **378 vs 126 active virtual MLP evaluations**, so this is *not a same-MAC superiority result*. The native MLP uses only ~396 native layer MACs per example, vs ~36,288 virtual-edge MLP MACs for SYNC3, **excluding Mirror decoding**. CPU latency values are exploratory owing to shared host scheduling; **no GPU benchmark** was run.

## Actual gradient audit
- SYNC graph vectorized all-edge forward/gradient matches separate edge-by-edge differentiable reference in 8/8 passing unit tests.
- A trained representative SYNC3 checkpoint has nonzero direct task-loss gradient in **64/64 physical core coefficients, 126/126 edge gate coefficients, 252/252 Mirror orientation codes**, from a **single backward**.
- For each of 6 teacher/seed runs, decomposed the *actual task-loss gradient through virtual-weight decoder* into contributions from six destination nodes. Exactly **5 negative cosine pairs / 90 total pairs** at those checkpoints; core-gradient cancellation ratios span **0.651–0.904**. Reassembled core gradient differs from autograd’s core gradient by <=1.5e-8. Direct gate gradients excluded from the destination decoder decomposition.
- This proves gradient superposition is computationally executable, **not** that 13,860 virtual weights gain 13,860 independent learned degrees, and not that measured conflict identifies prune-worthy connections.

## Frozen decisions
- **Parallel-gradient mechanism:** PASS (forward and gradient reference agreement, all physical parameter groups trainable).
- **SYNC3 vs serial Mirror quality gate:** PASS 5/6, **with 3× logical-edge compute**.
- **Mirror-specific quality/byte/active-compute gate:** FAIL against equally priced affine control.
- **General model quality:** FAIL to outperform ordinary MLP at matched 442 learned coefficients.
- **Architectural compression:** UNESTABLISHED (full unpruned supergraph is slow; program bytes and runtime workspace separately relevant).
- **Grokking, GPU compute, KV-cache benefit, natural language quality:** NOT TESTED in Phase E.

## Reproducibility notes
Experiment sources and all 36 physical state_dict checkpoints, 36 serialized inference payloads, metrics CSV/JSON, hashes, CPU-benchmark records, and complete replay artifacts were created in the companion Phase-E container evidence package. The **GitHub branch currently records this frozen protocol and the summary only**; do not call the GitHub branch alone a full source/checkpoint reproduction until the evidence package is separately uploaded. Repo parent Phase A–D evidence and negative verdicts are unchanged.

## Next-failure-driven probe
The big remaining missing operation is **learnable physical-to-virtual address mapping**, not more fixed Mirror views. The current graph uses a fixed seed basis and learned per-edge two-value orientation; test learned low-description index assignment/coordinate groups against direct sparse, low-rank, and affine controls at actual stored bytes and active MACs. Also perform true same-MAC comparisons before claiming parallel execution efficiency.