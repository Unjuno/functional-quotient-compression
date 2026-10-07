# Mirror Application Prior-Art Map

Date: 2026-10-07 JST
Purpose: help experiment workers distinguish established adjacent ideas from the project-specific Mirror hypothesis.

This is a research navigation document, not a novelty claim. "Direct" means prior work attacks the same physical multiplicity (expert/head/layer/etc.); it does not mean the exact Mirror parameterization has already been tested.

## PA01 — Expert tying across depth

**Tying the Loop — Tied Expert Layers in Mixture-of-Experts Language Models**  
http://arxiv.org/abs/2606.16825

Shares expert FFN parameters across consecutive Transformer layers while keeping attention and layer-wise routing distinct. Reports almost 2x memory reduction with little quality loss in the tested pretraining setups.

**Mirror implication:** MA expert/depth experiments must beat simple expert tying. The interesting delta is not "share experts", but whether a low-description view makes tied experts behave more like distinct experts.

## PA02 — Path-constrained / shared routing in MoE

**Path-Constrained Mixture-of-Experts**  
https://arxiv.org/abs/2603.18297

Shares router parameters across blocks of layers and studies constrained expert paths. Also includes a low-rank shared-base router control.

**Mirror implication:** router sharing and path factorization are already strong controls. Mirror-routing work should test functional views, not merely shared routing weights.

## PA03 — Low-rank routing for very large expert pools

**MoRE: Scaling Mixture of Experts with Hardware-Aware Low-Rank Routing**  
https://arxiv.org/html/2609.36301

Factorizes the router so more experts can be addressed at lower routing cost.

**Mirror implication:** if Mirror increases logical expert count, router cost must be compared against low-rank routing rather than a dense router only.

## PA04 — Hypernetwork-generated LoRA across attention heads

**Hypernetwork-Driven Low-Rank Adaptation Across Attention Heads (HyRA)**  
https://arxiv.org/abs/2510.04295

Uses a shared hypernetwork to generate coordinated low-rank adapters across attention heads.

**Mirror implication:** MA-LoRA/head proposals need to show why a Mirror coordinate is cheaper, more stable, or more expressive than a shared generator.

## PA05 — Input-conditioned low-rank weight generation

**Infinite-Parameter LLMs: Generating and Adapting Weights from Live Data**  
https://arxiv.org/abs/2609.18842

Uses a compact hypernetwork to generate low-rank modulations of a shared base network from live data.

**Mirror implication:** continual-learning and virtual-weight proposals overlap conceptually. The project-specific question is whether structured Mirror views give a better byte/compute/retention frontier than generic weight generation.

## PA06 — Recursive depth with generated per-step modulation

**Ouroboros: Dynamic Weight Generation for Recursive Transformers via Input-Conditioned LoRA Modulation**  
https://arxiv.org/abs/2604.02051

Reuses a Transformer block across depth and generates per-step diagonal modulation of frozen LoRA bases.

**Mirror implication:** MA virtual-depth experiments must compare against static per-step LoRA and input-conditioned generated modulation, not only ordinary weight tying.

## PA07 — Projection sharing inside attention

**Do Transformers Need Three Projections? Systematic Study of QKV Variants**  
https://arxiv.org/abs/2606.04032

Studies Q/K/V projection tying. Q != K=V reduces KV cache and combines with GQA/MQA.

**Mirror implication:** attention-view proposals should ask whether Mirror recovers quality lost by hard projection equality while retaining most of the storage/cache saving.

## PA08 — KV sharing across layers

**MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding**  
https://aclanthology.org/2025.findings-naacl.305/

Shares KV heads not only within a layer but across layers.

**Mirror implication:** MA KV work should compare "one physical KV state + logical Mirror views" against MLKV/GQA/MQA, with cache bytes and bandwidth measured.

## PA09 — Multi-token prediction

**Better & Faster Large Language Models via Multi-token Prediction**  
https://arxiv.org/abs/2404.19737

Uses a shared trunk with several future-token heads and reports training/sample-efficiency and self-speculative decoding benefits.

**Mirror implication:** TM/packet experiments should test whether multiple logical future heads can be generated from fewer physical heads/views.

## PA10 — Parallel Token Prediction

**Parallel Token Prediction for Language Models**  
https://proceedings.iclr.cc/paper_files/paper/2026/file/0a43c96ee43ce2e3d05d393c6426c284-Paper-Conference.pdf

Moves sequence randomness into model inputs so multiple dependent tokens can be generated jointly in one model call.

**Mirror implication:** TM001's hidden packet-latent failure has a close conceptual solution family. Future packet work should compare packet-level Mirror/latent codes against PTP-style random-variable conditioning.

## PA11 — Vector-symbolic / holographic binding

**Practical Lessons on Vector-Symbolic Architectures in Deep Learning-Inspired Environments**  
https://proceedings.mlr.press/v284/carzaniga25a.html

Benchmarks MAP, HRR and Hadamard linear binding. Reports MAP/HLB substantially faster than FFT-HRR in its tested setting and explores hierarchical composition.

**Mirror implication:** holographic candidates should not default to HRR. MAP/sign-permute and Hadamard binding are required controls.

## PA12 — Holographic Reduced Representations

**Holographic Reduced Representations**  
https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf

Foundational circular-convolution binding and fixed-width superposition/unbinding.

**Mirror implication:** circular/FFT Mirror ideas have established algebraic ancestors; novelty must come from using binding as a low-description functional address for shared neural parameters.

## PA13 — Attention interpreted as binding

**Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning**  
https://arxiv.org/abs/2512.14709

Connects attention to role/filler binding and proposes explicit VSA-inspired binding/unbinding modules.

**Mirror implication:** attention-binding candidates should compare to explicit binding heads, not only ordinary MHA.

## PA14 — Decoupled MoE with cache-safe expert placement

**Decoupled Mixture-of-Experts for Parametric Knowledge Injection**  
https://arxiv.org/abs/2606.14243

Places independently updatable experts only at the final-layer FFN so dynamic expert activation does not invalidate earlier KV caches.

**Mirror implication:** expert placement is part of the compression/runtime hypothesis. A final-FFN Mirror-MoE may preserve cache reuse while testing whether one physical expert basis can replace many knowledge adapters.

## PA15 — Intra-model routed verifier for speculative decoding

**VIA-SD: Verification via Intra-Model Routing for Speculative Decoding**  
https://arxiv.org/abs/2606.12243

Builds a slim verifier from the full model via intra-model routing and uses hierarchical draft -> slim verifier -> full verifier decisions.

**Mirror implication:** Mirror speculative-decoding ideas should compare against routed slim-verifiers. The interesting delta is whether low-description views can create several verification strengths or candidate distributions without separate model copies.

## Research gaps that remain especially relevant here

1. **Logical expert multiplicity from one physical expert via a structured view** — adjacent to expert tying, but not equivalent.
2. **Logical attention/KV heads from fewer physical projections plus reversible/low-description views** — adjacent to GQA/MQA/MLKV and QKV sharing.
3. **Virtual depth from tied blocks plus nontrivial low-description view coordinates** — adjacent to recursive Transformers and generated LoRA modulation.
4. **One adapter basis -> many logical adapters via invertible/binding coordinates** — adjacent to HyRA/Zhyper/hypernetworks.
5. **Packet-level Mirror address as shared randomness/plan for parallel decoding** — adjacent to MTP/PTP, directly motivated by TM001.
6. **Holographic binding as a functional parameter address rather than only a representation** — closest to the project's "physical object -> logical multiplicity" framing.
7. **Factorized address products** such as expert x head x depth x time, measured against a fully independent Cartesian-product parameterization.

## Worker rule

Before implementing an MA candidate:
1. read its registry row;
2. read every PA item referenced by that row;
3. write down the exact delta from prior art in the experiment README;
4. include the closest non-Mirror prior as a control when feasible;
5. do not claim novelty from absence of a paper in this map.
