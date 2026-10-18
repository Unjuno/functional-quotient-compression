# Mirror-native Parallel/Recurrent Sequence Network — focused research target (2026-10-18)

**Status: ARCHITECTURE TARGET + L0 ANALYTICAL PILOT ONLY. NO trained language-model advantage.**

**Branch-scoped first-principles follow-up:** [minimal shared-core definition](MIRROR_NATIVE_MINIMAL_CORE_PRINCIPLE_2026-10-18.md) and [MA-1162 role-based experiment roadmap](../../experiments/mirror_applications/ma-1162-mirror-native-sequence/MINIMAL_CORE_EXECUTION_ROADMAP_20261018.md). The follow-up treats one principal physical parameter **vector/tensor** of variable independent dimension m as the research object, with K parallel View roles and R recurrent computations. It considers joint encoder/decoder/cross-attention/FFN from one shared core subject to actual data dependencies, cache provenance and paid code/compute. Later MA-1162 aligned multihead pilots and separate MA-1163/1164 evidence do not revise the scoped historical L0 statement in this document.
Branch: `research/mirror-native-sequence-focus-20261018`. The main 2026-10-18 direction is now this architecture; no main merge or promotion of historical MA statuses.

## Purpose — beyond attaching Mirrors to an ordinary Transformer

Use **one reusable physical learned core** with **multiple compact Mirror/View addresses**, an **explicit causal token mixer**, **nonlinear views**, **recurrent re-use**, and **readout at selected computation depths** to build a *sequence model*, not merely a compressed conventional head.

Target independent design variables:
- physical parameter bytes `B(theta)`;
- logical Mirror directions `K` evaluated in parallel for the SAME hidden state;
- recurrent passes `T` of one shared core on UPDATED hidden states;
- view code / activation / schedule bytes `B(m,psi,eta)`;
- number/placement `P` of output reads and optional halting;
- virtual feature width `D_virt` (separate from trainable independent function rank);
- generated/high-precision weight-seed encoding **only after** ordinary models establish a gain.

**The key conjecture:** when task/sequence operations share a compact coordinate orbit, small addresses let one physical network perform functionally distinct causal token computations at useful accuracy/byte/train-FLOP, possibly with multiple outputs.

## Actual token mixing is a REQUIRED primitive

A channel-only Mirror applied independently to each token **cannot communicate information between positions**. The first falsifiable architecture uses a causal attention-like mixer with shared base projections, while each logical direction changes the actual Q/K relation:

```text
H^(0) := embed(tokens)                     # [batch,length,d]
for recurrent pass t:                      # sequential because H^(t+1) depends on H^(t)
  Q0,K0,V0 := shared_projections(H^(t))     # computed once per pass, not once per View
  in parallel across j in {1..K}:
    Q_j := Q0 R^Q_(t,j)
    K_j := K0 R^K_(t,j)
    V_j := (R^V_(t,j))^-1 phi(R^V_(t,j) V0) # nonlinear Value View, matching source convention
    A_j := causal_softmax(Q_j K_j^T / sqrt(d))
    U_j := A_j V_j                          # genuinely mixes preceding tokens
  H^(t+1) := norm(H^(t) + alpha_t aggregate(U_1,...,U_K)
                    + beta_t mirror_nonlinear_ffn(H^(t),m_t))
  emit any configured outputs from H^(t+1) through paid readout Views
  optionally halt or continue (cost-aware controller)
```

A second, **independent**, candidate can later replace attention with a causal structured shift/scan/memory mixer. It is *not* established that a channel Mirror alone provides attention or long-range retrieval. Retention / SSM / causal convolution are strong non-Mirror controls.

## Exact degeneracies and unavoidable costs

1. For identical orthogonal `R`, `(Q0 R)(K0 R)^T = Q0 K0^T` **exactly**: K identical Q/K rotations are not K useful attention patterns. Distinct `R^Q R^{K,T}`, nonlinearity, value operations or distinct readout must change the function measurably.
2. A Mirror-sandwiched ReLU with the same fixed orthogonal Q, applied to its own output repeatedly with no additional mixing/residual, is **idempotent**. More steps need not mean more functional complexity.
3. Parallel K Views reuse shared `Q0/K0/V0` when the input hidden state is identical, but still require **K separate attention score/matrix/value operations**, K nonlinear Views, memory and any aggregate/readout.
4. Recurrent passes t have changed hidden states; compute shared base projections again. Repeated steps are usually **sequential**, even when multiple Views at each step are parallel.
5. Multiple outputs from the same prefix do **not** imply valid joint future trajectories when a hidden random branch is sampled after the prefix (TM001). For real speculative decoding, target verification, p/q correction, KV provenance/copy/rollback and end-to-end tokens/s are required.
6. Middle-block Mirror changes later hidden/K/V states. A single canonical KV cache is exact across Views **only** with the algebraic conditions of MA-691 or cache-safe placement of MA-253; generic reuse is not guaranteed.
7. `D_virt >> d`, huge numerical precision (e.g. proposed 25k/58k-bit scalar) or millions of generated weights do not independently imply millions of learnable degrees of freedom. Count physical stored bits, codebooks, numeric arithmetic cost, rank of functional Jacobian and hidden-state RAM.

## What exists and what was just measured

- Earlier **MA-041**: Givens logical attention heads; useful on aligned synthetic teachers, unrelated function failure.
- **MA-247** and long-budget recheck: repeated nonlinear block, conflicting fixed/long-budget conclusions; seed and optimizer costs matter.
- **MA-249/MTPM-001/TM001**: parallel draft output plausible in some settings; hidden packet-level uncertainty remains a barrier; no verified real decode speedup.
- **MA-691/MA-253**: cache-space right transforms vs placement; no arbitrary hidden-state cache translation.
- **MA-1156..1159**: nonlinear Mirror sandwich, repeated coordinate, learned activation, oriented gate; hypotheses, not established successes.
- **New MA-1162 L0 math/CPU pilot**: 10 tests passed (causality, gauge null, distinct Q/K, gradients, nonlinear function difference, repeated ReLU idempotence, state roundtrip). At d=32, L=32, B=2, K=1/2/4/8/16 eager CPU median forward 0.170/0.329/0.687/1.268/2.511 ms. **Not a trained model, quality or speed claim.** Full machine-readable result and source linked below.

## Narrow execution focus and stop/go gates

| Stage | Single question | Evidence required | Decision |
| --- | --- | --- | --- |
| **L0 DONE** | Can parallel Mirrors produce distinct *causal* token computations without false QK gauges? | exact/FP32 null controls, causal perturbation, gradients, bytes, one-thread runtime | mechanism PASS only |
| **L1 NEXT** | Does Mirror-native token mixing actually learn tasks requiring earlier tokens? | masked retrieval/copy + ordered state + hidden-random-branch negative; 3+ fresh worlds×2 seeds; strong causal controls | advance only if learning works |
| **L2** | Does increasing K or T improve useful quality per **actual byte and compute**? | K=1/2/4/8, T=1/2/4, batch/length sweeps, paired fair controls; train-FLOPs-to-target | advance only if Mirror-specific frontier |
| **L3** | Can selected intermediate output times provide net benefit? | correct sequence NLL, early-exit calibration, halt overhead, multi-token accepted-length vs real decode tok/s | advance only if useful Pareto |
| **L4** | Does it generalize to a small causal language model? | real token NLL, strict causal masking, held-out corpus, cache integrity, resource accounting | no LLM claims before this |
| **L5 optional** | Does virtual-width or high-precision seed outperform equal-bit traditional weight banks? | equal stored bits, operation cost, gradient/Jacobian rank, full native controls | separate experiment, not needed for L1 |

**Strong controls:** ordinary causal Transformer at matched inference bytes; tied/recurrent Universal Transformer; grouped-query/shared-QKV attention; native gain/FiLM/LoRA view heads; causal depthwise convolution / small SSM or retention; a K-independent-head quality ceiling. Compare at both matched **stored bytes** and independently **matched total training FLOPs / inference MACs**, not just updates. Output quality and mode collapse are measured for all K, not head count alone.

**Learning discipline:** select the LR and hyperparameters on development worlds; freeze a formal protocol before fresh; multiple seeds; record fixed step and equal training-FLOPs curves; actual torch.save inference bytes, tokens, CPU wall-time and batch/length scaling; negative conditions remain visible. L0 numerical pilot is not a pre-registered training study.

[Experiment folder](../../experiments/mirror_applications/ma-1162-mirror-native-sequence/README.md) | [Protocol](../../experiments/mirror_applications/ma-1162-mirror-native-sequence/PROTOCOL.json) | [L0 code](../../experiments/mirror_applications/ma-1162-mirror-native-sequence/source/mechanism.py)

## External direct baselines (existing, not Mirror novelty)

- Universal Transformer: https://arxiv.org/abs/1807.03819
- RetNet: https://arxiv.org/abs/2307.08621
- Mamba selective state spaces: https://arxiv.org/abs/2312.00752

**H/T/D/C/U:** H=one physical core + coordinate Views + causal mixing may produce multiple useful logical operations per stored byte; T=L0 mathematical checks only so far; D=causal and view-distinct primitive confirmed, cost rises with K; C=simple shared-QKV/gating/Universal Transformer may match or beat; U=no trained L1, no natural-language result, no accelerated fused kernels, no independence-capacity proof.
