# MA-1162 — Mirror-native parallel/recurrent causal sequence model

**Status: UNTESTED as a trained architecture; L0 causal algebra pilot PASS (2026-10-18).**
Research branch: `research/mirror-native-sequence-focus-20261018`.

> **Mirror insertion:** insert a low-description functional coordinate `m` into both causal token-mixing Q/K/V views and the repeatedly applied nonlinear FFN of **one physical shared core**, giving parallel logical token processors and selected intermediate output times without duplicating full per-head/per-depth weights.

## Core falsifiable hypothesis

At matched **inference payload bytes** and **training compute**, the K-direction/T-step Mirror-native sequence model achieves lower held-out sequence NLL / higher causal retrieval accuracy than ordinary shared QKV, scalar gain/FiLM and tied recurrent Transformer, while staying causal. It must not rely on identical QK gauge rotations, leaked future labels, learned oracle routers, uncharged output heads or free recurrent FLOPs.

### Distinguishing feature

Parallel `K` Mirror views operate on the **same hidden state at step t**, so base Q/K/V projections can be shared **once per t**. Separate view Q/K rotations must create different relative scores. Recurrent `T` updates operate on changed states and generally cannot share that projection computation across t. Any apparent improvement can be explained by a simpler known shared/tied method until ruled out by native controls.

The new concept is broader than MA-041 attention-head sharing, MA-247 recursive depth, MA-249 drafting or the independent MA-1156/1157 nonlinear view studies. Those are existing evidence and direct controls, not new claims of novelty.

## L0 mechanism pilot — actually executed, but not trained

PyTorch CPU 2.10.0, one thread, B=2, L=32, d=32; tests and 5 deterministic numeric seeds. K causal heads reuse 3 learned d×d physical matrices; view code = 3*K angles on the first 2D feature plane. The explicit nonlinear value view uses `R^-1 ReLU(RV)` and attention uses separate Q/K angles. No vocabulary embeddings or LM head yet.

- **10/10 tests pass**: exact score-gauge null, nontrivial relative Q/K rotation, causal no-future leak, output diversity, view gradients, nonlinear sandwich, same-ReLU sandwich idempotence, residual recurrence, serialize-roundtrip/byte count, recurrent-input dependency.
- Across 5 seeds: identical QK score rotations differ by at most `9.5367431640625e-7` (FP32); earlier outputs under future-token perturbation differ by `0`; distinct QK scores differ by mean at least `0.72566`; between first two View outputs mean difference at least `0.03720`.
- K=1/2/4/8/16 had **3075/3078/3084/3096/3120 learned scalar parameters**, serialized bytes **15061/15061/15061/15061/15125**, and eager CPU forward medians **0.170/0.329/0.687/1.268/2.511 ms** for the fixed B,L,d fixture.
- This is an **unoptimized Python loop**; the increasing time demonstrates that parallelizable does not mean free and is not a fused-GPU prediction.
- L0 result can be reproduced with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests && python source/summarize_l0.py` from this experiment folder.

## What is NOT yet established

No supervised training, retrieval/copy performance, ability to learn next tokens, equal-FLOP frontier, independent capacity multiplication, real speculative decoding speed, virtual scalar precision benefit, or Transformer replacement has been tested here. MA registry status stays UNTESTED until a separately preregistered L1 train/fresh run is completed.

## Evidence and next work

- [Research objective / architecture](../../../docs/phase2/MIRROR_NATIVE_SEQUENCE_FOCUS_2026-10-18.md)
- [Frozen L0 checks + draft L1 resource protocol](PROTOCOL.json)
- [L0 metrics](results/RESULTS_L0.json)
- [L0 verification](VERIFICATION_L0.json)
- [L1 experiment plan](EXPERIMENT_PLAN_JA.md)
- [Status](STATUS.md)

**H/T/D/C/U:** H=parallel Mirror Views yield useful per-byte causal logical operations; T=L0 10 checks and five seeds, no training; D=causality and distinct outputs PASS, not capacity; C=ordinary attention and recurrent controls could fully explain any later gain; U=language/corpus/optimized runtime and true useful functionality.
