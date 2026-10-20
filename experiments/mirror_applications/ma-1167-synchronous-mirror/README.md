# MA-1167 — batched synchronous Mirror virtual-weight networks

**2026-10-20 JST; preregistered before model implementation/training.**

**Mirror insertion:** introduce learned low-description per-virtual-edge Mirror codes in a single shared **virtual MLP WEIGHT decoder**, allowing many logical node-to-node functions from one physical 289-scalar tensor, while comparing batched synchronous execution against ordinary native MLP and an equal-size affine shared-basis control.

## H
H1 **mechanism:** all 75 (5 source node × 5 target node × 3 mirror direction) virtual MLPs can be generated and executed as one batched operation **per Loop step**, all 5 node states updated synchronously from the previous step with no hidden within-step dependency; gradient through every edge reaches the shared physical vector in one backward call.
H2 **quality:** on teacher-generated smooth mappings, the Mirror model with 3 recurrent synchronization steps reduces holdout NMSE >=5% vs equal-289-coordinate affine control in at least 4/6 paired world/seed cases.
H3 **efficiency:** same Mirror model using batched tensors is faster than mathematically-equivalent per-edge Python loop implementation at CPU single-thread batch size 1 and 64, holding outputs and derivatives numerically consistent. GPU not available in this container; **no CUDA result claims**.

## T
- N=5 nodes, d=6 input/output channels, K=3 view families, edges E=5×5×3=75, Loop T=3. Every source→target pairing allowed including self and long-distance links; no pruning or sampled edge computation.
- One trainable **289-float Parameter object**: core=64, per-edge sigmoid gates=75, learned mirror codes=2×75. Each edge generates a virtual **6→8→6 native tanh MLP** with 110 logical floats, by fixed seeded mixing of the shared 64-vector, two learned-direction Householder reflections, tanh, inverse reflections. Total logical 75×110=8250 floats. All decoder directions/program values produced deterministically from a single documented PRNG seed. Physical learnable DOFs do not equal 8250 independent DOFs.
- Shared constant seeded programs and dynamically materialized virtual weights **count as inference-time initialization bytes and runtime/workspace**, separately from serialized trainable payload; program code not assumed free.
- Synchronous state update h_v^{t+1}=h_v^t+(alpha/T) sum_{u,k} gate_{uvk}MLP_{uvk}(h_u^t;theta) / (N K); all nodes updated from same prior states. Final prediction x+ mean_v(h_v^T−h_v^0) (fixed readout).
- Contrast: (a) equally parameterized 289-scalar direct-affine virtual decoder, same 75 edges/3 loops, all functions batched; (b) 289-scalar ordinary native residual MLP 6→21→6 (279 floats) plus ten learned output residual gains (do not tune hidden width); (c) Mirror T=1 ablation retaining same learned parameter count, 75 edges, but less active compute.
- Two synthetic worlds: `aligned` teacher is another random instance of full batched Mirror T=3 with same procedure but different physical core/codes; `native` teacher is independent 6→25→6 native residual MLP (only teacher weights; no oracle revealed). Student sees only teacher x/y. Inputs isotropic Gaussian.
- Three seeds 557,563,569, 2 worlds, 4 methods =24 train runs. Train2048/dev1024/fresh2048/OOD2048; independent RNG streams, OOD input std=1.6 vs ID std=.8, never tune from fresh/OOD.
- CPU FP32 torch, 1 thread, AdamW lr=0.015 weight_decay=0.0001, 450 updates, mini-batch 128, same optimization schedule for all; record learning curves and dev-best checkpoint selected by dev only.
- No training-time or post-training pruning, no gate penalty, no warm-start teacher weights, no search over test output. Fresh once per selected checkpoint.
- Mechanism tests: exact edge count, one physical Parameter object, equivalence to per-edge slow reference at T=1/T=3, every physical group gradients nonzero, joint-grad-vs-sum autograd, deterministic seeded decoder replay, dense all-to-all causal synchronization, training metric replay.
- Benchmark isolated CPU inference b1 and b64 (thread1) for batched Mirror and per-edge same-weight Python implementation; log executed 75×T edge MLP transforms per example; actual tensor bytes, source-code size, retained deterministic buffer memory, active MAC proxy.
- Project historical negative baselines MA-1163, MA-1164, MA-1165, MA-1166 stay immutable. Branch only; no main change.

## D
PASS H1 only with tests/replay; FAIL H2 if paired win criterion missed. H3 passes only if batched CPU wall-time strictly lower than per-edge CPU; separate from quality/storage claims. A smaller teacher error alone is NOT a compression finding. No LLM, real language, GPU, full optimized correspondence graph, independent training of all 8250 virtual floats, global structural search, or depth-parallel work is claimed.

## C/U
Nonlinear self-coupling across Loop steps creates serial dependence even if all node-edge functions within each step are batched. Learned per-edge codes 225 of 289 total and matrix-decoder mixing may underfit unaligned teachers. Complexity of fixed seeded programs must be included in implementation. The native baseline has vastly fewer active MACs, so Mirror-specific Pareto requires CPU time too. Two worlds ×3 seeds are exploratory only.
