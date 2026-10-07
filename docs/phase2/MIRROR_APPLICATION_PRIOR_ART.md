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

## PA16 — Parameter Superposition

**Superposition of many models into one**  
https://arxiv.org/abs/1902.05522

Stores several task models in one parameter tensor and uses task-dependent context vectors/matrices to bind and retrieve them with low interference. The paper studies binary, complex/unitary and rotational contexts and composition of contexts.

**Mirror implication:** this is one of the closest conceptual predecessors to the project. Any claim that a context/View coordinate alone creates many logical models must compare against parameter superposition. The interesting delta is whether learned/structured Mirror coordinates improve useful multiplicity, composability, routing, or interference at the same stored bytes.

## PA17 — BatchEnsemble

**BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning**  
https://arxiv.org/abs/2002.06715

Each logical member uses a shared weight matrix modulated by member-specific rank-one fast weights. The formulation is vectorizable within a device and was also used for lifelong learning.

**Mirror implication:** rank-one multiplicative modulation is a mandatory control for ensemble/expert logical multiplicity. Mirror must improve quality-per-byte, composability, or functional diversity beyond rank-one member factors.

## PA18 — VeRA

**VeRA: Vector-based Random Matrix Adaptation**  
https://proceedings.iclr.cc/paper_files/paper/2024/hash/1b53ad08de383a049e9668a9d0b6a053-Abstract-Conference.html

Shares one pair of frozen low-rank random matrices across adapted layers and learns small scaling vectors. It targets the cost of storing many LoRA-like adaptations.

**Mirror implication:** many Mirror-LoRA ideas are close to VeRA. The relevant test is whether structured/invertible View coordinates over a common basis improve adaptation quality or logical multiplicity over simple scaling vectors.

## PA19 — IA3 / T-Few

**Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning**  
https://arxiv.org/abs/2205.05638

IA3 learns small vectors that multiplicatively rescale key, value and FFN intermediate activations.

**Mirror implication:** activation-side Mirror ideas must compare to simple per-channel rescaling. A rotation/shear/binding view is only useful if it beats IA3-like diagonal modulation at comparable bytes.

## PA20 — Orthogonal Finetuning

**Controlling Text-to-Image Diffusion by Orthogonal Finetuning**  
https://arxiv.org/abs/2306.07280

Introduces OFT, which adapts pretrained weights through orthogonal transformations designed to preserve geometric relationships.

**Mirror implication:** orthogonal Mirror/View transforms have a direct PEFT predecessor. Mirror-specific value must come from sharing/composing addresses across tasks, layers, experts or heads rather than merely using an orthogonal transform.

## PA21 — BOFT

**Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization**  
https://arxiv.org/abs/2311.06243

Parameterizes dense orthogonal transforms as products of sparse butterfly factors, reaching O(d log d) trainable-parameter scaling for a dense transform in the butterfly setting.

**Mirror implication:** butterfly orthogonal transforms are a strong structured-Mirror candidate and control. Dense rotations should not be used as the only orthogonal baseline.

## PA22 — OFTv2 / input-centric orthogonal adaptation

**Orthogonal Finetuning Made Scalable**  
https://arxiv.org/abs/2506.19847

Reformulates OFT by applying the orthogonal transform to input vectors rather than materializing transformed weights and introduces a Cayley-Neumann parameterization.

**Mirror implication:** runtime-efficient Mirror transforms should be implemented input-side when possible and compared against matrix-free OFT-style execution.

## PA23 — Compacter

**Compacter: Efficient Low-Rank Hypercomplex Adapter Layers**  
https://arxiv.org/abs/2106.04647

Constructs adapter matrices as sums of Kronecker products between shared slow weights and layer-specific low-rank/rank-one fast factors.

**Mirror implication:** this is a direct control for shared structured adapter bases plus small local coordinates. Mirror-Kronecker/hypercomplex variants must beat or complement Compacter rather than only LoRA.

## PA24 — Monarch structured matrices

**Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture**  
https://arxiv.org/abs/2310.12109

Uses products of block-diagonal matrices and permutations to build expressive structured transforms with sub-quadratic compute, including sequence- and model-dimension mixing.

**Mirror implication:** Monarch factors offer a hardware-friendly structured transform family for logical views; compare against butterfly/circulant/low-rank views rather than assuming dense Mirror matrices.

## PA25 — Fast Weight Programmers

**Linear Transformers Are Secretly Fast Weight Programmers**  
https://arxiv.org/abs/2102.11174

Interprets linear attention as context-dependent fast-weight memory programmed by outer-product updates and studies delta-rule-style corrections.

**Mirror implication:** a Mirror address can itself be dynamic state rather than a static task/expert code. Fast-weight baselines are required for input-dependent or session-dependent Mirror coordinates.

## PA26 — Task Arithmetic

**Editing Models with Task Arithmetic**  
https://arxiv.org/abs/2212.04089

Represents a task as a weight-space delta from a common pretrained model and combines task vectors by addition, negation and analogical arithmetic.

**Mirror implication:** functional coordinates may live in task-vector space. Mirror composition should be compared to ordinary task-vector arithmetic and must measure interference as the number of combined tasks grows.

## PA27 — TIES-Merging

**TIES-Merging: Resolving Interference When Merging Models**  
https://arxiv.org/abs/2306.01708

Trims weak task-vector entries, resolves sign conflicts, and merges only aligned parameter directions.

**Mirror implication:** when several Mirror/task coordinates are superposed, sign/direction interference is a known failure mode. TIES-style conflict resolution is a strong control and possible component.

## PA28 — Model Soups

**Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time**  
https://proceedings.mlr.press/v162/wortsman22a.html

Averages compatible fine-tuned checkpoints in a common basin to obtain one model with no ensemble inference cost.

**Mirror implication:** continuous/interpolated Mirror coordinates should be tested against simple weight averaging/interpolation. If averaging works equally well, a more elaborate View mechanism is unnecessary.

## PA29 — Structured sparse / circulant LoRA

**Beyond LoRA: Is Sparsity-Induced Adaptation Better?**  
https://arxiv.org/abs/2606.13767

Studies Cheap LoRA variants that fix one factor and introduces a chained circulant-style column-subspace adaptation family.

**Mirror implication:** structured/circulant Mirror-LoRA must compare against extremely cheap fixed-subspace and chained structured adapters, not only full LoRA.

## PA30 — SETA continual sparse experts

**Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning**  
https://arxiv.org/abs/2606.07500

Discovers high-utility sparse parameter subspaces and dynamically separates overlapping shared experts from task-unique experts while preserving routing behavior.

**Mirror implication:** continual-learning Mirror experiments need to separate "discover shared/private subspaces" from "compress those subspaces via a View". SETA provides a strong decomposition control.

## PA31 — Supermasks in Superposition

**Supermasks in Superposition (SupSup)**  
https://proceedings.neurips.cc/paper/2020/file/ad1f8bb9b51f023cdc80cf94bb615aa9-Paper.pdf

Uses one fixed random network and learns a task-specific supermask for each task. It can infer task identity by optimizing a superposition of learned masks and can store many masks in a fixed-size attractor reservoir.

**Mirror implication:** a binary/sparse mask is itself a low-description functional coordinate. Mirror proposals should test whether a continuous/invertible View can provide better quality-per-bit, smoother composition, or cheaper task inference than task masks.

## PA32 — Piggyback and PackNet

**Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights**  
https://arxiv.org/abs/1801.06519

**PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning**  
https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf

Piggyback keeps backbone weights fixed and learns a binary mask per task; PackNet allocates disjoint/free weight subsets sequentially.

**Mirror implication:** continual-learning Mirror methods need mask/subnetwork baselines. A useful Mirror code should either use fewer bits than masks, compose better, or delay physical parameter allocation.

## PA33 — Intrinsic-dimensional fine-tuning

**Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning**  
https://aclanthology.org/2021.acl-long.568/

Shows that pretrained language models can often be adapted by optimizing a very low-dimensional parameter vector projected into the full parameter space; hundreds of dimensions can recover substantial task performance in some settings.

**Mirror implication:** a Mirror address may be interpreted as an intrinsic task coordinate. Random projected subspace tuning is a mandatory control for claims about small task coordinates.

## PA34 — Tensorized embedding layers

**Tensorized Embedding Layers**  
https://aclanthology.org/2020.findings-emnlp.436/

Uses Tensor-Train decomposition for embedding/softmax matrices and studies compression-quality tradeoffs in NLP/Transformer models.

**Mirror implication:** embedding Mirror views should compare against tensorized shared cores rather than only dense/low-rank embeddings.

## PA35 — Tensor decomposition across Transformer layers

**Exploring extreme parameter compression for pre-trained language models**  
https://arxiv.org/abs/2205.10036

Compares matrix, Tensor-Train and Tucker decompositions. Its Tucker formulation uses a fixed matrix bank with layer-specific coefficients, making parameter scale nearly constant with layer count.

**Mirror implication:** this is extremely close to "shared basis + small logical layer address". Mirror-depth and projection-bank claims must compare against Tucker/matrix-bank decomposition.

## PA36 — LoRETTA

**LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models**  
https://arxiv.org/abs/2402.11417

Uses Tensor-Train factors for tensorized adapters and weight reparameterization, reporting large reductions in trainable parameters versus common PEFT methods.

**Mirror implication:** tensor-factor Mirror adapters need LoRETTA as a direct PEFT control.

## PA37 — Git Re-Basin

**Git Re-Basin: Merging Models Modulo Permutation Symmetries**  
https://arxiv.org/abs/2209.04836

Aligns independently trained networks by neuron permutation symmetries before interpolation/merging.

**Mirror implication:** some apparent functional diversity is pure parameter-coordinate symmetry. Before calling a View a new logical function, audit whether it is only a function-preserving reparameterization; conversely, symmetry alignment can make Mirror/task deltas more mergeable.

## PA38 — HyperLoRA federated personalization

**Amortizing Federated Adaptation: Hypernetwork Driven LoRA for Personalized Foundation Models**  
https://arxiv.org/abs/2606.06154

Uses a hypernetwork to generate client-conditioned LoRA initialization and a learned server-side synthesizer in low-rank product space.

**Mirror implication:** federated Mirror codes should compare against generated client-specific LoRA and product-space aggregation.

## PA39 — PreLort rank-heterogeneous federated LoRA

**PreLort: Prefix-Nested LoRA for Federated Fine-Tuning under Rank Heterogeneity**  
https://arxiv.org/abs/2606.15963

Organizes heterogeneous client ranks into nested prefixes and aggregates each rank segment only across clients that actually train that segment.

**Mirror implication:** client/rank Mirror coordinates can be factorized along nested rank segments, but must beat prefix-nested LoRA as the simple sharing baseline.

## PA40 — Personalized federated hypernetworks

**Personalized Federated Learning using Hypernetworks (pFedHN)**  
https://arxiv.org/abs/2103.04628

A central hypernetwork maps compact client embeddings to personalized client models.

**Mirror implication:** client-specific Mirror addresses are a structured, potentially cheaper alternative to generic model generation. Compare bytes, communication, unseen-client generalization and personalization.

## PA41 — Rank-1 Bayesian neural networks

**Efficient and Scalable Bayesian Neural Nets with Rank-1 Factors**  
https://proceedings.mlr.press/v119/dusenberry20a.html

Places distributions over rank-one factors multiplying shared weight matrices and supports low-overhead multimodal posterior mixtures.

**Mirror implication:** uncertainty/ensemble Mirror views need Bayesian rank-one factors as a strong probabilistic control, not only deterministic BatchEnsemble.

## PA42 — MIMO implicit ensemble subnetworks

**Training independent subnetworks for robust prediction**  
https://arxiv.org/abs/2010.06610

Trains multiple functionally distinct subnetworks inside one network using multi-input/multi-output training and evaluates them in one forward pass.

**Mirror implication:** "multiple logical models inside one physical model" can emerge without explicit per-member weight views. Mirror ensemble claims must measure actual member diversity against MIMO.

## PA43 — Product Key Memory

**Large Memory Layers with Product Keys**  
https://arxiv.org/abs/1907.05242

Factorizes keys as a Cartesian product of sub-key sets, enabling very large sparse memories with efficient exact lookup.

**Mirror implication:** factorized Mirror addresses for memory/expert selection can use product-key indexing so logical address count grows multiplicatively without dense routing cost.

## PA44 — Modern Hopfield networks

**Hopfield Networks is All You Need**  
https://arxiv.org/abs/2008.02217

Modern continuous Hopfield layers provide high-capacity associative retrieval and connect directly to attention.

**Mirror implication:** Mirror address banks can be stored/retrieved as associative attractors rather than explicit tables; retrieval interference and memory capacity must be measured.

## PA45 — ACDC structured transforms

**ACDC: A Structured Efficient Linear Layer**  
https://arxiv.org/abs/1511.05946

Uses alternating learned diagonal matrices and fixed cosine/Fourier-like transforms, reducing dense-linear parameterization from O(N^2) toward O(N) with O(N log N) transform cost.

**Mirror implication:** ACDC/AFDF is an especially cheap structured View family: diagonal code + fixed global mixing. It is a required control for FFT/Hadamard/structured Mirror transforms.

## PA46 — LieTransformer / group equivariance

**LieTransformer: Equivariant Self-Attention for Lie Groups**  
https://proceedings.mlr.press/v139/hutchinson21a.html

Builds self-attention equivariant to Lie-group actions through principled parameter sharing.

**Mirror implication:** a Mirror coordinate can be interpreted as a group element. Group-action Mirror experiments should distinguish exact equivariance from learned functional specialization.

## PA47 — Monomial weight-space symmetries

**Monomial Matrix Group Equivariant Neural Functional Networks**  
https://arxiv.org/abs/2409.11697

Extends weight-space symmetry analysis beyond permutations to scaling/sign-flip symmetries represented by monomial matrix groups.

**Mirror implication:** permutation/sign/scale Views may be exact reparameterization symmetries rather than new functions. This provides an explicit symmetry audit for Mirror geometry.

## PA48 — Reversible PEFT

**Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning**  
https://arxiv.org/abs/2306.00477

Uses adapters and reversible computation so activations can be reconstructed during backward, reducing training-memory requirements.

**Mirror implication:** reversible Mirror transforms can target training-memory compression in addition to weight storage, but compute/recomputation cost must be reported separately.

## PA49 — Mixture-of-Depths

**Mixture-of-Depths: Dynamically allocating compute in transformer-based language models**  
https://arxiv.org/abs/2404.02258

Routes only a fixed-capacity top-k subset of tokens through each Transformer block, making token-level depth dynamic while total per-layer compute remains bounded.

**Mirror implication:** the functional coordinate can control whether/where computation happens, not only how weights are transformed. Mirror compute-routing must compare against MoD.

## PA50 — Adaptive depth / early exit

**DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference**  
https://aclanthology.org/2020.acl-main.204/

**Depth-adaptive Transformer**  
https://inria.hal.science/hal-02422914

Selects input/token-specific exit depth to trade quality for computation.

**Mirror implication:** a depth View can encode logical computation stage or confidence, but any speed claim must compare to simple halting/early-exit policies.

## PA51 — Universally Slimmable Networks

**Universally Slimmable Networks and Improved Training Techniques**  
https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Universally_Slimmable_Networks_and_Improved_Training_Techniques_ICCV_2019_paper.html

Trains one shared network that operates at many widths, using techniques such as the sandwich rule and inplace distillation.

**Mirror implication:** width is itself a logical model coordinate. Mirror can be tested as a small width-specific correction that recovers quality lost by aggressive weight sharing.

## PA52 — Once-for-All supernet

**Once-for-All: Train One Network and Specialize it for Efficient Deployment**  
https://arxiv.org/abs/1908.09791

Trains a single supernet supporting many depth/width/kernel/resolution subnetworks through progressive shrinking.

**Mirror implication:** architecture choice can be treated as a factorized address. Mirror should be tested as a correction/view over shared supernet weights, not as a replacement for subnet selection itself.

## PA53 — MatFormer

**MatFormer: Nested Transformer for Elastic Inference**  
https://arxiv.org/abs/2310.07707

Jointly optimizes nested Transformer FFN widths and allows many untrained Mix'n'Match layer granularities to be extracted from one model.

**Mirror implication:** nested-width logical models already exist without separate weights. Mirror can test whether tiny granularity/layer codes reduce interference or improve arbitrary mix-and-match configurations.

## PA54 — AdapterFusion

**AdapterFusion: Non-Destructive Task Composition for Transfer Learning**  
https://arxiv.org/abs/2005.00247

Keeps independently trained task adapters frozen and learns a separate fusion mechanism to combine their representations.

**Mirror implication:** adapter-composition Mirror ideas must compare to fusion over independent adapters; a useful result should reduce adapter storage or fusion cost, not merely compose them.

## PA55 — LoraHub

**LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition**  
https://arxiv.org/abs/2307.13269

Combines multiple pre-trained LoRA modules with learned positive/negative scalar coefficients for few-shot unseen-task adaptation.

**Mirror implication:** Mirror-LoRA composition must beat or compress simple signed LoRA-module mixtures.

## PA56 — Learning to Prompt (L2P)

**Learning to Prompt for Continual Learning**  
https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html

Maintains a prompt pool and dynamically retrieves prompts without requiring task identity at test time.

**Mirror implication:** prompt/View banks are another physical-multiplicity target. Compare explicit prompt storage to generated/compressed Mirror prompts.

## PA57 — DualPrompt

**DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning**  
https://arxiv.org/abs/2204.04799

Separates task-invariant general prompts and task-specific expert prompts.

**Mirror implication:** this maps directly to shared basis + private residual. Mirror prompt experiments can ask whether many expert prompts collapse to small coordinates around general prompts.

## PA58 — Hash Embeddings

**Hash Embeddings for Efficient Word Representations**  
https://proceedings.neurips.cc/paper/2017/file/f0f6ba4b5e0000340312d33c212c3ae8-Paper.pdf

Represents tokens by combining a few component vectors from a shared pool selected by hash functions, plus token-specific importance weights.

**Mirror implication:** vocabulary-level logical multiplicity can be created from a small component pool; Mirror embeddings should compare against hashing plus learned importance weights.

## PA59 — Compositional / quotient-remainder embeddings

**Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems**  
https://arxiv.org/abs/1909.02107

Builds unique embeddings from several smaller tables using complementary partitions such as quotient/remainder indexing.

**Mirror implication:** factorized address products can create unique logical embeddings with sublinear table storage. This is a direct control for factorized Mirror addresses.

## PA60 — Adaptive input representations

**Adaptive Input Representations for Neural Language Modeling**  
https://arxiv.org/abs/1809.10853

Allocates different embedding capacity to frequency bands and can tie adaptive input/output embeddings.

**Mirror implication:** embedding Mirror storage must compare against simply allocating less physical capacity to rare tokens.

## PA61 — ALBERT parameter sharing

**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**  
https://arxiv.org/abs/1909.11942

Uses factorized embeddings and cross-layer parameter sharing, including ablations that separately share attention or FFN parameters.

**Mirror implication:** shared-layer Mirror variants should compare directly to ALBERT-style hard sharing. A View is valuable only if it restores quality/flexibility cheaply.

## PA62 — One-shot NAS / shared supernets

**Efficient Neural Architecture Search via Parameter Sharing (ENAS)**  
https://arxiv.org/abs/1802.03268

Represents many child architectures as subgraphs of one shared supernetwork.

**Mirror implication:** architecture configuration can be another low-description logical coordinate, but weight-sharing bias/interference is a known confound. Mirror corrections should be evaluated against the same child architectures trained independently where practical.

## PA63 — FiLM conditioning

**FiLM: Visual Reasoning with a General Conditioning Layer**  
https://arxiv.org/abs/1709.07871

Applies conditioning-dependent feature-wise affine transformations, with a generator producing per-channel scale and bias.

**Mirror implication:** conditional feature scaling is a very cheap functional coordinate. Richer Mirror activation/feature views must beat FiLM in quality-per-byte or compositional behavior.

## PA64 — SPADE / spatially adaptive normalization

**Semantic Image Synthesis with Spatially-Adaptive Normalization**  
https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html

Generates spatially varying affine normalization parameters from conditioning input.

**Mirror implication:** the View coordinate need not be global; it may vary over token/position. Token-wise or spatial Mirror modulation should compare against conditional affine normalization.

## PA65 — StyleGAN2 weight modulation

**Analyzing and Improving the Image Quality of StyleGAN**  
https://arxiv.org/abs/1912.04958

Modulates convolution weights per sample using a compact style vector and then demodulates them. Equivalent feature scaling can be folded into effective weights.

**Mirror implication:** this is one of the closest examples of a small latent changing effective weights at inference. Mirror-weight modulation must compare against simple channel-wise modulation/demodulation.

## PA66 — CondConv

**Conditionally Parameterized Convolutions for Efficient Inference**  
https://arxiv.org/abs/1904.04971

Builds an input-specific kernel as a learned linear combination of physical expert kernels and applies the resulting kernel once.

**Mirror implication:** dynamic effective weights can be created by coefficient mixing without running all experts. Mirror dynamic-weight methods should compare storage and runtime against conditional kernel/weight synthesis.

## PA67 — Dynamic Convolution

**Dynamic Convolution: Attention over Convolution Kernels**  
https://arxiv.org/abs/1912.03458

Aggregates several kernels by input-dependent attention weights before applying the convolution.

**Mirror implication:** input-dependent mixing coefficients are a strong baseline for dynamic logical functions. A View must add structure beyond ordinary convex/soft kernel mixtures.

## PA68 — DeepSDF latent-code decoder

**DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation**  
https://arxiv.org/abs/1901.05103

Represents many shapes with one shared continuous decoder and compact per-shape latent codes; new instance codes can be optimized while the decoder is shared.

**Mirror implication:** one shared function plus instance latent code is a direct physical-to-logical pattern. Mirror latent codes should be evaluated by code size, interpolation/generalization and reconstruction quality.

## PA69 — Modulated implicit neural representations

**Modulated Periodic Activations for Generalizable Local Functional Representations**  
https://openaccess.thecvf.com/content/ICCV2021/papers/Mehta_Modulated_Periodic_Activations_for_Generalizable_Local_Functional_Representations_ICCV_2021_paper.pdf

Uses latent codes and a modulation network to alter amplitude, phase and frequency of periodic activations in a shared synthesis MLP.

**Mirror implication:** nonlinear activation geometry itself can be the functional coordinate. Mirror-SIREN/activation views need this as a direct control.

## PA70 — Neural ODEs

**Neural Ordinary Differential Equations**  
https://arxiv.org/abs/1806.07366

Defines continuous-depth models by repeatedly integrating a shared learned vector field, trading explicit layer copies for a continuous computation trajectory.

**Mirror implication:** depth can be represented by state/time in a shared dynamical rule. Depth Mirror experiments should distinguish discrete role specialization from continuous shared dynamics.

## PA71 — Deep Equilibrium Models

**Deep Equilibrium Models**  
https://arxiv.org/abs/1909.01377

Uses a weight-tied transformation solved to a fixed point, representing effectively infinite depth with one physical layer and implicit differentiation.

**Mirror implication:** repeated logical depth does not require independent blocks. Mirror can test multiple equilibria/roles or condition the shared fixed-point map, but must compare solve cost and stability.

## PA72 — Universal Transformer

**Universal Transformers**  
https://arxiv.org/abs/1807.03819

Recurrently applies a shared transition function across depth with timestep/depth information and optional adaptive computation.

**Mirror implication:** a depth address already differentiates repeated shared computation. Mirror-depth claims should compare against simple timestep embeddings and recurrent sharing.

## PA73 — Mamba selective state spaces

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
https://arxiv.org/abs/2312.00752

Makes parts of the state-space dynamics input-dependent, allowing selective propagation/forgetting while retaining efficient recurrent execution.

**Mirror implication:** the functional coordinate can modulate state dynamics per token. Mirror-SSM ideas need selective-SSM controls.

## PA74 — S4 structured state spaces

**Efficiently Modeling Long Sequences with Structured State Spaces**  
https://arxiv.org/abs/2111.00396

Uses structured normal-plus-low-rank state matrices and efficient convolution/recurrent forms for long sequence modeling.

**Mirror implication:** state-transition Views can be applied in an already structured low-description parameterization; compare to S4's native structure rather than dense transition matrices.

## PA75 — MAML

**Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks**  
https://arxiv.org/abs/1703.03400

Learns a shared initialization designed so a small number of task-specific gradient steps produce effective specialized models.

**Mirror implication:** task-specific logical functions can arise from a shared point plus a short update trajectory. Mirror task codes should compare against few-step adaptation cost and resulting delta storage.

## PA76 — LEO

**Meta-Learning with Latent Embedding Optimization**  
https://arxiv.org/abs/1807.05960

Learns a low-dimensional task latent, optimizes it, and decodes it into high-dimensional task parameters.

**Mirror implication:** low-dimensional functional coordinates plus a decoder are a direct alternative to structured Mirror codes. Compare decoder size, adaptation steps and code size.

## PA77 — Learned optimizers

**Learning to learn by gradient descent by gradient descent**  
https://arxiv.org/abs/1606.04474

Learns a recurrent optimizer that maps gradients/history to parameter updates.

**Mirror implication:** the added freedom may live in the update rule rather than the forward model. Mirror-optimizer coordinates should compare against learned update policies and standard optimizers.

## PA78 — Doubly sparse explicitly conditioned transforms

**Learning Doubly Sparse Explicitly Conditioned Transforms**  
https://arxiv.org/abs/2606.10975

Models a transform as a fixed canonical matrix followed by a learned sparse refining factor with explicit conditioning/stability constraints.

**Mirror implication:** "fixed efficient transform + sparse adaptive refinement" is a direct structured-View family and control for FFT/DCT/Hadamard Mirror variants.

## PA79 — Concept Modulation Models

**Concept Modulation Models: A Unified Framework for Identifiability and Extrapolation**  
https://arxiv.org/abs/2606.18509

Formalizes attribute-indexed modulators that induce concept distributions under a shared generative mechanism and analyzes identifiability/extrapolation through attribute potentials.

**Mirror implication:** factorized Mirror attributes should be evaluated for extrapolation to unseen coordinate combinations, not only interpolation on trained addresses.

## PA80 — PathNet

**PathNet: Evolution Channels Gradient Descent in Super Neural Networks**  
https://arxiv.org/abs/1701.08734

Represents a task by a pathway selecting a subset of modules in each layer, allowing later tasks to reuse/freeze previously useful modules.

**Mirror implication:** module-path identity is a functional coordinate. Mirror path methods should ask whether one physical module can serve more path roles or whether path codes can be compressed/factorized.

## PA81 — Routing Networks

**Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning**  
https://arxiv.org/abs/1711.01239

Dynamically composes reusable function blocks per input/task using a learned router and recursive path selection.

**Mirror implication:** dynamic function composition is an established alternative to generating new weights. Mirror-routing needs to show value beyond selecting/reordering existing blocks.

## PA82 — HyperFormer

**Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks**  
https://arxiv.org/abs/2106.04489

Uses shared hypernetworks conditioned on task, layer and adapter position embeddings to generate task-specific adapters and LayerNorm parameters.

**Mirror implication:** small task/layer codes generating many logical modules is already a strong baseline. Structured Mirror generation must beat generic shared hypernetworks in storage, quality or extrapolation.

## PA83 — AdaMix

**AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning**  
https://arxiv.org/abs/2205.12410

Trains mixtures of adapters or LoRA modules with stochastic routing and merges adaptation modules for inference.

**Mirror implication:** mixture diversity can help even when inference collapses to one merged module. Mirror mixtures should compare against stochastic PEFT mixtures and merged inference.

## PA84 — UniPELT

**UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning**  
https://aclanthology.org/2022.acl-long.433/

Combines several PEFT mechanisms such as adapters, prefix tuning and LoRA with learned gates.

**Mirror implication:** combining multiple adaptation freedoms is already beneficial. A Mirror combination must beat gated composition of standard PEFT components.

## PA85 — Polytropon / latent modular skills

**Combining Modular Skills in Multitask Learning**  
https://arxiv.org/abs/2202.13914

Learns a discrete task-skill allocation matrix over a bank of parameter-efficient skills, allowing tasks to compose subsets of reusable modules.

**Mirror implication:** logical capability can be represented by sparse skill combinations. Mirror-skill composition should compare against explicit modular skill banks and task-skill allocation.

## PA86 — MEND

**Fast Model Editing at Scale**  
https://arxiv.org/abs/2110.11309

Learns small editor networks that transform low-rank decompositions of edit gradients into parameter updates.

**Mirror implication:** learned functional changes can be generated from a compact edit signal. Mirror edit codes should compare against learned low-rank gradient transforms.

## PA87 — ROME

**Locating and Editing Factual Associations in GPT**  
https://arxiv.org/abs/2202.05262

Applies targeted rank-one updates to selected MLP weights to edit factual associations.

**Mirror implication:** a small low-rank/private residual is a strong direct control for storing one fact/association. Mirror must show better multi-edit storage, reversibility or interference.

## PA88 — MEMIT

**Mass-Editing Memory in a Transformer**  
https://arxiv.org/abs/2210.07229

Distributes calculated weight updates across causal MLP layers to insert many factual associations.

**Mirror implication:** multi-edit compression should compare against directly integrating memories into shared weights, including specificity/generalization and interference.

## PA89 — SERAC

**Memory-Based Model Editing at Scale**  
https://arxiv.org/abs/2206.06520

Stores edits explicitly in a memory, routes matching inputs to a counterfactual model, and otherwise defers to the base model.

**Mirror implication:** not every new behavior needs to live in base weights. Mirror edit banks must beat external memory on bytes, latency or generalization for the intended use case.

## PA90 — GRACE

**Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors**  
https://arxiv.org/abs/2211.11031

Adds a discrete key-value codebook around a model layer, retrieving local edit values for nearby latent queries without changing base weights.

**Mirror implication:** a codebook of local edits is a direct alternative to per-edit View codes. Mirror can test codebook compression, binding and shared edit bases.

## PA91 — VQ-VAE

**Neural Discrete Representation Learning**  
https://arxiv.org/abs/1711.00937

Uses a learned codebook and discrete latent indices to condition a shared decoder.

**Mirror implication:** a functional coordinate may be discrete and dictionary-coded. Mirror addresses should compare continuous codes against codebook index bits and reconstruction quality.

## PA92 — Residual vector quantization

**SoundStream: An End-to-End Neural Audio Codec**  
https://arxiv.org/abs/2107.03312

**High Fidelity Neural Audio Compression (EnCodec)**  
https://arxiv.org/abs/2210.13438

Stacks residual vector quantizers so multiple codebooks progressively refine the latent representation; codebook count can trade bitrate for quality.

**Mirror implication:** multiple small discrete coordinates can compose residual refinements. This is a direct control for additive/compositional Mirror codes and adaptive code budgets.

## PA93 — LISTA / learned sparse coding

**Learning Fast Approximations of Sparse Coding**  
http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf

Learns a fixed-depth network approximating sparse-code inference over shared dictionary atoms.

**Mirror implication:** sparse coefficients over a shared basis are a fundamental competitor to Mirror rule/expert codes. Learned code inference is also a strong router baseline.

## PA94 — Shared/private dictionary learning

**Learning a Low-Rank Shared Dictionary for Object Classification**  
https://arxiv.org/abs/1602.00310

Separates a low-rank shared dictionary from class-specific dictionaries and sparse coefficients.

**Mirror implication:** the project's shared/private decomposition has a dictionary-learning analogue. Mirror should test whether class/task-private dictionaries can be reduced to Views over shared atoms before allocating new atoms.

## PA95 — Error-Correcting Output Codes

**Solving Multiclass Learning Problems via Error-Correcting Output Codes**  
https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume2/dietterich95a.pdf

Represents classes by separated codewords and predicts code bits, gaining robustness from redundancy and Hamming distance.

**Mirror implication:** logical addresses need not only be compact; they can be deliberately redundant/error-correcting. Robust Mirror routing/addressing can trade a few extra bits for lower misrouting/cross-talk.

## PA96 — ReFT / LoReFT

**ReFT: Representation Finetuning for Language Models**  
https://arxiv.org/abs/2404.03592

Freezes model weights and learns low-rank interventions on hidden representations. LoReFT edits a learned low-dimensional representation subspace.

**Mirror implication:** hidden-state View interventions can be substantially cheaper than weight edits. Any representation-space Mirror method should compare directly to LoReFT/DiReFT.

## PA97 — Representation Engineering

**Representation Engineering: A Top-Down Approach to AI Transparency**  
https://arxiv.org/abs/2310.01405

Treats high-level representation directions/subspaces as units for reading and controlling model behavior.

**Mirror implication:** the functional coordinate may live directly in residual-stream representation space. Behavior control must be separated from actual weight-storage capacity.

## PA98 — Activation Addition

**Steering Language Models With Activation Engineering / Activation Addition**  
https://arxiv.org/abs/2308.10248

Builds steering vectors from contrastive activation differences and adds them during forward passes without optimizing model weights.

**Mirror implication:** additive activation vectors are an extremely cheap control for hidden-state Mirror Views.

## PA99 — Function Vectors

**Function Vectors in Large Language Models**  
https://arxiv.org/abs/2310.15213

Finds compact activation vectors carried by attention heads that causally trigger learned input-output functions and reports partial algebraic composition of such vectors.

**Mirror implication:** this is direct evidence that function identity can be encoded as a hidden-state coordinate. Mirror research should test compression, composition and extraction of function vectors rather than only parameter views.

## PA100 — Contrastive Activation Addition

**Steering Llama 2 via Contrastive Activation Addition**  
https://arxiv.org/abs/2312.06681

Averages contrastive residual-stream activation differences to construct steering directions that can be scaled and added at inference.

**Mirror implication:** robust behavior vectors can be estimated from data without weight training. Compare learned View codes to contrastive activation vectors at equal storage.

## PA101 — Conditional Activation Steering

**Conditional Activation Steering**  
https://proceedings.iclr.cc/paper_files/paper/2025/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf

Uses condition vectors to decide whether behavior steering vectors should be applied, enabling context-dependent activation control.

**Mirror implication:** dynamic routing of Views can be implemented entirely in activation space. Compare Mirror routers against condition-vector similarity gates.

## PA102 — Sparse-autoencoder feature steering

**SAEs Are Good for Steering — If You Select the Right Features**  
https://aclanthology.org/2025.emnlp-main.519/

Studies steering language models with sparse-autoencoder features and shows feature choice and side effects matter strongly.

**Mirror implication:** sparse learned features provide a dictionary of representation-space functional atoms. Mirror feature banks should compare to SAE feature activation and sparse combinations.

## PA103 — Transcoders

**Transcoders Find Interpretable LLM Feature Circuits**  
https://arxiv.org/abs/2406.11944

Approximates dense MLP sublayers with wider sparsely activating feature layers and enables feature-level circuit decomposition.

**Mirror implication:** logical functions may be sparse combinations of feature-level computations rather than views of dense weights. Transcoder features are a strong atom-basis control.

## PA104 — Sparse low-rank representation/weight intervention

**RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning**  
https://aclanthology.org/2024.emnlp-main.57/

Constrains low-rank updates so the resulting weight change affects selected rows/columns more sparsely.

**Mirror implication:** when Mirror edits are claimed to be local or knowledge-preserving, sparse low-rank updates are a strong locality control.

## PA105 — Rotary Position Embedding

**RoFormer: Enhanced Transformer with Rotary Position Embedding**  
https://arxiv.org/abs/2104.09864

Encodes position through rotation matrices applied to query/key representations while inducing explicit relative-position structure in attention.

**Mirror implication:** positional rotation is an established structured coordinate transformation. Mirror-RoPE experiments should test additional logical positional/domain roles beyond standard RoPE rather than claim rotation itself as new.

## PA106 — FiLM and feature-wise conditional modulation

**FiLM: Visual Reasoning with a General Conditioning Layer**  
https://arxiv.org/abs/1709.07871

Applies conditioning-dependent feature-wise affine transformations `gamma * x + beta` throughout a shared network.

**Mirror implication:** diagonal affine modulation is a very strong cheap conditional-function baseline. Richer Mirror geometry must justify its extra bytes/compute over FiLM.

## PA107 — MatFormer

**MatFormer: Nested Transformer for Elastic Inference**  
https://arxiv.org/abs/2310.07707

Jointly trains nested FFN submodels so one Transformer contains multiple accurate granularities and many mix-and-match submodels.

**Mirror implication:** logical model multiplicity can come from nested parameter inclusion alone. Mirror elasticity should compare to MatFormer and test whether Views recover specialization at fixed physical width or improve mixed-granularity submodels.

## PA108 — Universally Slimmable Networks

**Universally Slimmable Networks and Improved Training Techniques**  
https://arxiv.org/abs/1903.05134

Trains one network to execute at arbitrary widths using sandwich-rule sampling and inplace distillation.

**Mirror implication:** width is already a cheap architecture coordinate. A Mirror code may differentiate shared channels across widths or make width selection task/context dependent.

## PA109 — Once-for-All networks

**Once-for-All: Train One Network and Specialize it for Efficient Deployment**  
https://arxiv.org/abs/1908.09791

One trained supernetwork supports elastic depth, width, kernel size and resolution through progressive shrinking and subnetwork selection.

**Mirror implication:** hardware/configuration coordinates are another form of logical multiplicity. Mirror should be tested as a specialization coordinate inside a supernetwork, not mistaken for the concept of one-model-many-subnets itself.

## PA110 — LayerDrop

**Reducing Transformer Depth on Demand with Structured Dropout**  
https://arxiv.org/abs/1909.11556

Trains Transformers to tolerate dropping entire layers, enabling shallower inference subnetworks from one model without extra finetuning.

**Mirror implication:** variable logical depth can be obtained by omission alone. Mirror depth views should compare against LayerDrop-trained subnetworks.

## PA111 — Mixture-of-Depths

**Mixture-of-Depths: Dynamically allocating compute in transformer-based language models**  
https://arxiv.org/abs/2404.02258

Uses token-wise top-k routing to decide which tokens receive attention/MLP computation at each layer under a fixed compute budget.

**Mirror implication:** Mirror coordinates can potentially choose both *what function* and *whether compute is spent*. Compute-routing gains must be compared to MoD, not only dense depth.

## PA112 — QuaRot

**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**  
https://arxiv.org/abs/2404.00456

Uses function-preserving randomized Hadamard rotations, folding them into weights where possible, to remove outliers and enable low-bit weight/activation/KV quantization.

**Mirror implication:** some rotations are exact gauge changes whose value appears only after quantization. Mirror experiments must distinguish functional multiplicity from representation conditioning and can exploit coordinate choice specifically for compression.

## PA113 — SpinQuant

**SpinQuant: LLM Quantization with Learned Rotations**  
https://arxiv.org/abs/2405.16406

Optimizes function-preserving rotation matrices for quantized-model accuracy and shows substantial variance among random rotations.

**Mirror implication:** the choice of a function-preserving Mirror coordinate can itself be optimized for downstream compression. Learned rotations are a mandatory control for quantization-focused Mirror geometry.

## PA114 — SmoothQuant

**SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**  
https://arxiv.org/abs/2211.10438

Uses mathematically equivalent per-channel scaling to migrate quantization difficulty from activations to weights.

**Mirror implication:** scaling symmetries can be exploited purely for numerical conditioning. Mirror quantization must compare against this cheaper equivalence transformation.

## PA115 — Multi-Head Latent Attention

**DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**  
https://arxiv.org/abs/2405.04434

MLA jointly compresses keys/values into a low-rank latent representation cached once and reconstructs attention roles, substantially reducing KV-cache storage.

**Mirror implication:** "one physical latent -> many logical KV/head roles" is already practical. Mirror-KV should test whether compact View codes improve reconstruction, personalization, or layer/head sharing beyond MLA.

## PA116 — Prefix-Tuning

**Prefix-Tuning: Optimizing Continuous Prompts for Generation**  
https://arxiv.org/abs/2101.00190

Freezes the language model and stores small trainable continuous prefix states that condition all layers for each task.

**Mirror implication:** functional diversity can live in hidden-state prefixes rather than weight views. Mirror prompt/prefix compression must compare bytes and context/cache cost.

## PA117 — Prompt Tuning

**The Power of Scale for Parameter-Efficient Prompt Tuning**  
https://arxiv.org/abs/2104.08691

Learns task-specific soft input embeddings while freezing model weights.

**Mirror implication:** small task coordinates at the input are a minimal functional-control baseline, especially for large pretrained models.

## PA118 — HyperFormer++

**Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks**  
https://arxiv.org/abs/2106.04489

A shared hypernetwork conditions on task, layer and adapter position embeddings to generate task-specific adapters and layer-normalization parameters.

**Mirror implication:** this is a direct baseline for factorized task x layer x position coordinates generating functions. Mirror codes must be cheaper/simpler or more compositional than a generic shared hypernetwork.

## PA119 — Prompt pools / L2P and DualPrompt

**Learning To Prompt for Continual Learning**  
https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.pdf

**DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning**  
https://arxiv.org/abs/2204.04799

Use pools of compact prompts, often with input-dependent key/query selection, to store shared and task-specific knowledge over a frozen backbone.

**Mirror implication:** prompt-code banks already implement task-agnostic conditional functionality. Mirror prompt banks should compare routing/storage/interference directly.

## PA120 — HashedNets

**Compressing Neural Networks with the Hashing Trick**  
https://arxiv.org/abs/1504.04788

Maps many virtual connections to a smaller physical parameter vector via deterministic hash buckets, allowing virtual expansion without proportional stored weights.

**Mirror implication:** this is a direct physical-vs-virtual parameter baseline. Mirror hashing/virtual experts should measure collision interference rather than treating virtual count as capacity.

## PA121 — CondConv

**CondConv: Conditionally Parameterized Convolutions for Efficient Inference**  
https://arxiv.org/abs/1904.04971

Computes an input-specific effective kernel as a linear combination of expert kernels before applying the expensive convolution.

**Mirror implication:** input-dependent coefficient mixing is a generic way to generate logical weights. Mirror dynamic-weight methods should compare to linear expert-basis synthesis.

## PA122 — Dynamic Filter Networks

**Dynamic Filter Networks**  
https://proceedings.neurips.cc/paper_files/paper/2016/file/8bf1211fd4b7b94528899de0a43b9fb3-Paper.pdf

A filter-generating network produces sample- or position-specific filters on the fly.

**Mirror implication:** dynamic Mirror addresses are a structured subset of generic dynamic weight generation. The structured constraint must buy storage, stability, or generalization.

## PA123 — Universal Transformer

**Universal Transformers**  
https://arxiv.org/abs/1807.03819

Recurrently applies a shared Transformer transition across depth/time, optionally with per-position adaptive halting.

**Mirror implication:** repeated use of one block is established; the Mirror question is how cheaply to differentiate iterations while retaining useful weight sharing.

## PA124 — Deep Equilibrium Models

**Deep Equilibrium Models**  
https://papers.nips.cc/paper/8358-deep-equilibrium-models

Represents effectively infinite weight-tied depth by solving for an equilibrium and uses implicit differentiation with constant activation-memory scaling in depth.

**Mirror implication:** Mirror depth coordinates can be tested inside fixed-point/recurrent shared-weight systems, but iteration count and solver compute must be explicit.

## PA125 — AdapterFusion

**AdapterFusion: Non-Destructive Task Composition for Transfer Learning**  
https://aclanthology.org/2021.eacl-main.39/

Freezes separately learned task adapters and learns contextual attention-like fusion over their representations.

**Mirror implication:** multi-View composition should compare against explicit adapter composition rather than only adapter selection.

## PA126 — AdapterDrop

**AdapterDrop: On the Efficiency of Adapters in Transformers**  
https://aclanthology.org/2021.emnlp-main.626/

Trains adapters to tolerate removal from lower layers and prunes low-contribution adapters from AdapterFusion for runtime efficiency.

**Mirror implication:** logical adapter multiplicity and composition should be evaluated with variable active adapter count; storage and active compute are separate axes.

## PA127 — Routing Networks and PathNet

**Routing Networks and the Challenges of Modular and Compositional Computation**  
https://arxiv.org/abs/1904.12774

**PathNet: Evolution Channels Gradient Descent in Super Neural Networks**  
https://arxiv.org/abs/1701.08734

Both study reuse of shared neural modules through task/input-specific computational paths.

**Mirror implication:** the function address may select a composition/order of shared modules rather than only transform one module. Mirror routing must account for routing/module co-adaptation, collapse and interference.

## PA128 — Neural Interpreters

**Dynamic Inference with Neural Interpreters**  
https://arxiv.org/abs/2110.06399

Represents reusable functions by compact signature and code vectors. A shared interpreter executes code-conditioned functions, and type matching dynamically routes inputs.

**Mirror implication:** this is a close direct baseline for "small function code + shared physical executor -> many logical functions". Mirror code should be compared on code bytes, new-function extensibility, composition and held-out systematic generalization.

## PA129 — CAVIA

**Fast Context Adaptation via Meta-Learning**  
https://proceedings.mlr.press/v97/zintgraf19a.html

Separates shared model parameters from a low-dimensional task context vector adapted in the inner loop.

**Mirror implication:** a low-dimensional task coordinate is established meta-learning. Mirror must add structure, composability or lower adaptation/storage cost beyond a generic context vector.

## PA130 — Conditional Neural Processes

**Conditional Neural Processes**  
https://arxiv.org/abs/1807.01613

Aggregates context observations into a fixed-dimensional representation that conditions a shared neural function approximator.

**Mirror implication:** a functional coordinate can be inferred from examples rather than stored by task ID. Mirror function-code inference should compare against generic context aggregation.

## PA131 — DeepSDF latent auto-decoder

**DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation**  
https://arxiv.org/abs/1901.05103

One shared decoder represents a family of continuous functions; each instance is represented by an optimized low-dimensional latent code.

**Mirror implication:** "shared physical decoder + per-function code" is a mature compression/generative paradigm. Mirror's contribution must be code geometry or efficiency, not the general concept.

## PA132 — Modulated implicit neural representations

**Modulated Periodic Activations for Generalizable Local Functional Representations**  
https://arxiv.org/abs/2104.03960

A shared synthesis network represents many signals; per-signal latent codes drive a modulation network that changes periodic activations.

**Mirror implication:** activation-phase/frequency modulation is a rich direct control for nonlinear Mirror/View families.

## PA133 — COIN++

**COIN++: Neural Compression Across Modalities**  
https://arxiv.org/abs/2201.12904

Compresses signals as quantized/entropy-coded modulation vectors over a meta-learned shared implicit network rather than storing a separate network per signal.

**Mirror implication:** this is a direct example of physical network sharing plus paid per-instance functional codes. Mirror compression experiments should copy its discipline: code quantization, entropy/storage accounting and fast code fitting.

## PA134 — Neural Stored-program Memory

**Neural Stored-program Memory**  
https://arxiv.org/abs/1906.08862

Stores basis controller weights as programs in key-value memory and retrieves/interpolates them dynamically to produce working network weights.

**Mirror implication:** program memory is a direct baseline for dynamic logical weights. Mirror can compress program values, use structured program coordinates, or replace full program retrieval with a small View code.

## PA135 — ROME

**Locating and Editing Factual Associations in GPT**  
https://arxiv.org/abs/2202.05262

ROME inserts a factual association through a targeted rank-one MLP weight update.

**Mirror implication:** rank-one edit vectors are a compact unit of functional change. Mirror editing should compare against rank-one direct edits and preserve specificity/generalization.

## PA136 — MEMIT

**Mass Editing Memory in a Transformer**  
https://arxiv.org/abs/2210.07229

Extends direct parameter editing to thousands of memories by distributing calculated updates across causal MLP layers.

**Mirror implication:** a large edit bank is a concrete compression target for shared basis + View coordinates; editing efficacy, paraphrase generalization and locality are mandatory metrics.

## PA137 — MEND

**Fast Model Editing at Scale**  
https://arxiv.org/abs/2110.11309

Learns editor networks that transform the rank-one factors of standard fine-tuning gradients into local weight updates.

**Mirror implication:** editing coordinates can be generated from gradient factors. Mirror should test whether a structured editor code is cheaper or more stable than a generic gradient-transform MLP.

## PA138 — SERAC

**Memory-Based Model Editing at Scale**  
https://arxiv.org/abs/2206.06520

Stores edits explicitly and uses a scope classifier plus counterfactual model to override the base model only when an edit is relevant.

**Mirror implication:** parametric compression is not automatically better than explicit edit memory. Mirror edit banks must compare storage, retrieval latency, edit scope and interference.

## PA139 — GRACE

**Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors**  
https://arxiv.org/abs/2211.11031

Stores sequential edits in a discrete latent-space key/value codebook with per-key deferral radii while leaving base weights unchanged.

**Mirror implication:** codebook growth, locality and lifelong retention are strong controls for Mirror edit/code memory.

## PA140 — LOKI

**LOKI: Memory-Free Null-Space Constrained Lifelong Knowledge Editing**  
https://arxiv.org/abs/2606.19679

Selects edit layers dynamically and projects edit gradients into weight null spaces to reduce interference without replay memory.

**Mirror implication:** private/edit residuals can be constrained to low-interference subspaces before compression. Mirror coding should not conflate storage compression with interference avoidance.

## PA141 — RRDA

**When to Write and When to Suppress: Route-Specialized Dual Adapters for Memory-Assisted Knowledge Editing**  
https://arxiv.org/abs/2606.14668

Uses explicit edit memory, a relevance router, an edit adapter for routed prompts and a locality adapter for unrouted prompts.

**Mirror implication:** write and suppress are distinct logical functions. A shared View basis can be tested against two independent adapters, but routing/locality must remain explicit.

## PA142 — GaLore

**GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**  
https://arxiv.org/abs/2403.03507

Projects full-parameter gradients into periodically refreshed low-rank subspaces, reducing optimizer-state memory while still accumulating full-rank weight changes.

**Mirror implication:** the low-description coordinate can live in optimizer/update space rather than inference weights. Mirror optimizer claims must report optimizer memory separately from inference storage.

## PA143 — ReLoRA

**ReLoRA: High-Rank Training Through Low-Rank Updates**  
https://arxiv.org/abs/2307.05695

Periodically merges low-rank updates into base weights and reinitializes LoRA factors/optimizer state so repeated low-rank phases accumulate a higher-rank final update.

**Mirror implication:** several small functional coordinates applied over time can span a richer final function than one static coordinate. This is a temporal-composition control.

## PA144 — Test-Time Training layers

**Learning to (Learn at Test Time): RNNs with Expressive Hidden States**  
https://arxiv.org/abs/2407.04620

Makes the recurrent hidden state itself a model whose weights are updated by self-supervised learning during inference.

**Mirror implication:** a dynamic functional coordinate can be the test-time learner state. Mirror can constrain/compress this writable state rather than keep it fixed.

## PA145 — Mamba selective state spaces

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
https://arxiv.org/abs/2312.00752

Makes SSM parameters B, C and step size input-dependent, producing content-selective recurrent dynamics.

**Mirror implication:** the View coordinate can modulate state dynamics rather than Transformer weights. Input-dependent selectivity is the mandatory control.

## PA146 — Hyena

**Hyena Hierarchy: Towards Larger Convolutional Language Models**  
https://arxiv.org/abs/2302.10866

Uses implicitly generated long-convolution filters interleaved with data-controlled gating, defining input-dependent structured operators without materializing dense matrices.

**Mirror implication:** implicit filter generators offer another route to logical functional multiplicity with low stored parameter cost.

## PA147 — Titans

**Titans: Learning to Memorize at Test Time**  
https://arxiv.org/abs/2501.00663

Uses a neural long-term memory whose parameters are updated at test time, alongside short-term attention and persistent learned memory.

**Mirror implication:** stored, persistent and writable memories should be separated. Mirror may compress persistent/task coordinates or constrain writable long-term memory updates.

## PA148 — LoRAHub

**LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition**  
https://openreview.net/pdf?id=w8eCnnq57m

Combines previously trained LoRA modules with learned scalar coefficients for a new few-shot task using gradient-free coefficient search.

**Mirror implication:** code-space composition must beat simple signed/weighted LoRA composition before claiming a special composition mechanism.

## PA149 — Mixture of LoRA Experts

**Mixture of LoRA Experts**  
https://arxiv.org/abs/2404.13628

Learns layer-wise gates over multiple trained LoRA modules to compose their outputs dynamically.

**Mirror implication:** multiple adapter execution is a strong functional-composition baseline. A Mirror basis is valuable if it reduces adapter storage or pre-composes codes so fewer adapter forwards are needed.

## PA150 — learned optimizers

**Learning to learn by gradient descent by gradient descent**  
https://arxiv.org/abs/1606.04474

Learns an RNN optimizer specialized to a distribution of optimization problems.

**Mirror implication:** optimizer behavior can itself be a reusable logical function. A small optimizer View/task code could specialize one learned optimizer across task families.

## PA151 — Meta-SGD

**Meta-SGD: Learning to Learn Quickly for Few-Shot Learning**  
https://arxiv.org/abs/1707.09835

Meta-learns initialization plus per-parameter update direction/learning-rate multipliers.

**Mirror implication:** small learned update geometry is a direct control for gradient-space Mirror coordinates.

## PA152 — complementary episodic adapters

**A complementary learning system for continual episodic memory in large language models**  
https://www.biorxiv.org/content/10.64898/2026.08.24.746712v1

Stores episodes in dedicated extremely sparse low-rank adapters with competitive gating, then optionally consolidates them into base weights.

**Mirror implication:** episodic adapter banks are a concrete high-cardinality private-memory target for shared View compression; recall and interference must not be sacrificed.

## PA153 — Multi-Stream LLMs

**Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs**  
https://arxiv.org/abs/2605.12460

Instruction-tunes language models to process/generate multiple causally evolving streams in parallel.

**Mirror implication:** stream identity is another logical-role coordinate. Shared projections/FFNs can be tested with compact stream Views while preserving stream-specific state and causal boundaries.

## PA154 — Activated LoRA cache-compatible adapter switching

**Activated LoRA: Fine-tuned LLMs for Intrinsics**  
https://arxiv.org/abs/2504.12397

aLoRA adapts Q/K/V and other weights only after an invocation point so prefix states before activation remain identical to the base model and its KV cache can be reused.

**Mirror implication:** aLoRA obtains cache compatibility by restricting *when* adaptation is active. A cache-compatible Mirror can instead ask whether a known transform lets an already-different logical view reuse the same physical cache.

## PA155 — Standard-LoRA shared-prefix KV reuse

**Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving Tradeoffs**  
https://arxiv.org/abs/2609.17109

Directly reuses a base-model prefix cache across already-trained standard LoRA specialists. It reports large warm-cache TTFT savings at long context but does not establish quality equivalence; its tested ridge translator did not beat direct reuse, and physical cache storage was copied rather than aliased.

**Mirror implication:** exact algebraic cache compatibility would be stronger than approximate direct reuse: compare quality, physical storage aliasing, TTFT and completion latency.

## PA156 — LazyAttention / deferred positional encoding

**LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding**  
https://arxiv.org/abs/2606.04302

Keeps a position-agnostic physical KV representation and applies positional encoding lazily inside attention kernels, so one cache can serve multiple logical placements without materializing position-specific copies.

**Mirror implication:** this is a direct systems analogue of deferred Mirror transforms. If a View can be moved from cached K/V to query/output algebra, one physical cache can serve multiple logical Views zero-copy.

## PA157 — Cross-model linear KV translation

**Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**  
https://arxiv.org/abs/2608.03893

Fits per-head linear/ridge mappings from source-model caches to target-model caches, strips RoPE before mapping, and reports 2.7–25x mapper-vs-prefill speedups with variable quality retention.

**Mirror implication:** generic models need an approximate learned mapper; Mirror architectures can be designed so the source->View map is known analytically and exact by construction.

## PA158 — CacheBridge

**CacheBridge: Efficient Cross-Model KV Cache Transfer**  
https://arxiv.org/abs/2609.00891

Improves cross-model affine cache transfer with architecture-indexed source support, attention-sensitive calibration and bounded mapper cost.

**Mirror implication:** attention-aligned error, not raw KV reconstruction, is the correct metric when exact algebra is unavailable. It is a strong fallback control for approximate Mirror cache translators.

## PA159 — Cache-to-Cache semantic projection

**Cache-to-Cache: Direct Semantic Communication Between Large Language Models**  
https://arxiv.org/abs/2510.03215

Learns to project and fuse one model's KV cache into another model's cache space, using KV state itself as a communication medium.

**Mirror implication:** learned cache projection is already viable across heterogeneous models. Mirror-specific work should exploit known shared geometry to make the translator much smaller, exact, or both.

## PA160 — KVEraser direct KV-state editing

**KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing**  
https://arxiv.org/abs/2606.17034

Learns steering KV states that locally edit a processed context while leaving the remaining cache intact, avoiding full suffix recomputation.

**Mirror implication:** cache-space edits can change downstream behavior without exact full-cache reconstruction. This supports a second approximate lane when exact View algebra is too restrictive.

## PA161 — Multi-head Latent Attention

**DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**  
https://arxiv.org/abs/2405.04434

MLA caches one low-dimensional joint KV latent and absorbs key/value up-projections into query/output computation, sharply reducing KV cache storage.

**Mirror implication:** a canonical latent cache plus View-conditioned up-projections is a natural Mirror generalization: cache one physical latent, derive many logical K/V functions on read.

## PA162 — YOCO shared global cache

**You Only Cache Once: Decoder-Decoder Architectures for Language Models**  
https://arxiv.org/abs/2405.05254

A self-decoder constructs one global KV cache and multiple cross-decoder layers reuse it, reducing cache complexity and prefill cost.

**Mirror implication:** instead of hard-reusing identical K/V across layers, layer-specific Mirror read transforms may recover logical layer diversity while preserving the one-cache property.

## PA163 — CLSA shared cache and shared routing

**You Only Index Once: Cross-Layer Sparse Attention with Shared Routing**  
https://arxiv.org/abs/2606.06467

Builds on YOCO: cross-decoder layers reuse both one shared KV cache and one token-routing index, amortizing sparse-attention routing overhead.

**Mirror implication:** cache and routing can be jointly shared. Mirror View codes should be tested on top of a single physical cache/index rather than reintroducing per-layer cache or routing state.

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
