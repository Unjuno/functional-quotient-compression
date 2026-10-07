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

## PA164 — S-LoRA unified paging for adapters and KV

**S-LoRA: Serving Thousands of Concurrent LoRA Adapters**  
https://arxiv.org/abs/2311.03285

Stores many LoRA adapters off-GPU and uses Unified Paging to manage adapter weights and KV-cache tensors in one paged memory pool, with specialized heterogeneous-batching kernels.

**Mirror implication:** cache-compatible Mirror serving needs a real memory layout, not only an algebraic identity. Compare physical pages, fragmentation, adapter/View-code residency and heterogeneous batching.

## PA165 — Punica multi-tenant LoRA kernels

**Punica: Multi-Tenant LoRA Serving**  
https://arxiv.org/abs/2310.18547

Uses Segmented Gather Matrix-Vector kernels to batch different LoRA adapters over one shared backbone and efficiently serve heterogeneous adapter requests.

**Mirror implication:** structured View transforms need fused/grouped kernels comparable to multi-adapter serving kernels; eager Python/PyTorch overhead is not a fair runtime endpoint.

## PA166 — PagedAttention / vLLM physical KV sharing

**Efficient Memory Management for Large Language Model Serving with PagedAttention**  
https://arxiv.org/abs/2309.06180

Pages KV state into fixed-size physical blocks, allows logical blocks to map to shared physical blocks, and uses copy-on-write to support sharing across decoding branches/requests.

**Mirror implication:** exact Mirror cache reuse should physically alias canonical KV pages across logical Views. Logical value reuse without page aliasing is not a memory-sharing result.

## PA167 — RadixAttention / SGLang

**SGLang: Efficient Execution of Structured Language Model Programs**  
https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf

Uses a radix tree over shared prefixes and paged KV tensors to retain, match and evict reusable cache across structured generation programs.

**Mirror implication:** Mirror View identity can become metadata over shared radix/paged prefixes rather than a reason to duplicate prefix state.

## PA168 — LoRA-Switch token-wise dynamic adapters

**LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design**  
https://arxiv.org/abs/2405.17741

Routes adapters token-wise, uses one pre-gating decision across layers, and fuses adapter merging/switching with custom SGMM kernels.

**Mirror implication:** token-wise View switching is a realistic workload. A canonical-cache design should avoid prefix conversion when the View changes every token and be compared with fused adapter switching.

## PA169 — MoLoRA per-token composable specialization

**MoLoRA: Composable Specialization via Per-Token Adapter Routing**  
https://arxiv.org/abs/2603.15965

Routes tokens to specialized LoRA adapters within one sequence and unifies adapter dispatch with MoE-style grouped computation.

**Mirror implication:** per-token logical specialist identity must be stored/used without duplicating historical KV. Grouped-by-View canonical-cache attention is a direct candidate architecture.

## PA170 — Hydragen shared-prefix exact attention

**Hydragen: High-Throughput LLM Inference with Shared Prefixes**  
https://arxiv.org/abs/2402.05099

Decomposes attention over shared prefixes and unique suffixes, then batches queries over one shared prefix to reduce redundant KV reads and turn memory-bound matrix-vector work into more efficient matrix-matrix work.

**Mirror implication:** multiple logical Views reading the same canonical cache can batch transformed queries over one physical prefix; cache sharing can reduce bandwidth as well as storage.

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

## PA171 — Householder / scaled-Cayley orthogonal parameterizations

**Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections**  
https://proceedings.mlr.press/v70/mhammedi17a/mhammedi17a.pdf

Parameterizes orthogonal matrices as products of Householder-style reflections with efficient matrix-vector application and gradient computation.

**Orthogonal Recurrent Neural Networks with Scaled Cayley Transform**  
https://proceedings.mlr.press/v80/helfrich18a.html

Parameterizes recurrent matrices through a scaled Cayley transform of skew-symmetric matrices to retain orthogonality while allowing the relevant sign/eigenvalue cases.

**Mirror implication:** reflection vectors or skew-symmetric coordinates are compact, stable transform families for task/expert/recurrent Views. Mirror experiments using them must compare expressivity, matrix-free runtime and stability against OFT/BOFT rather than claiming orthogonality itself as novel.

## PA172 — Group-and-Shuffle / GSOFT

**Group and Shuffle: Efficient Structured Orthogonal Parametrization**  
https://arxiv.org/abs/2406.10019

Introduces Group-and-Shuffle structured matrices and GSOFT, using alternating small block transforms and permutations to form dense orthogonal transformations with fewer stages than butterfly constructions in the tested setting.

**Mirror implication:** GS matrices are a direct hardware-oriented candidate for a low-description View. The open question is whether one shared physical object plus many GS codes yields useful logical multiplicity beyond simply training one GSOFT transform per task.

## PA173 — low-displacement-rank neural matrices

**Theoretical Properties for Neural Networks with Weight Matrices of Low Displacement Rank**  
https://proceedings.mlr.press/v70/zhao17b/zhao17b.pdf

Studies neural networks whose dense weight matrices are represented by low-displacement-rank structure, including Toeplitz-like families, and analyzes their approximation/expressivity properties.

**Mirror implication:** displacement generators can be treated as the shared physical bank while small task/layer codes select logical matrices. This is a stronger structured-matrix control than arbitrary dense Mirror transforms.

## PA174 — learned compressed transforms with low displacement rank

**Learning Compressed Transforms with Low Displacement Rank**  
https://proceedings.neurips.cc/paper_files/paper/2018/file/8e621619d71d0ae5ef4e631ad586334f-Paper.pdf

Learns fast compressed transforms in a low-displacement-rank family rather than restricting the model to a single fixed Toeplitz/circulant form.

**Mirror implication:** learnable displacement structure supplies a reusable atom bank for logical Views. Compare learned LDR atoms, sparse coefficient composition and physical pruning before introducing richer transforms.

## PA175 — DARE sparse task deltas

**Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch**  
https://arxiv.org/abs/2311.03099

Introduces DARE, which drops many parameters from fine-tuning deltas and rescales the retained entries before model merging, exploiting redundancy in task deltas.

**Mirror implication:** sparse task deltas are a direct preconditioner/control for task-vector Mirror storage. Any claimed compression from a Mirror delta basis must be measured after DARE-style sparsification as well as before it.

## PA176 — DELLA magnitude-aware merging

**DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling**  
https://arxiv.org/abs/2406.11617

Uses magnitude-aware sampling of task-vector entries to reduce destructive interference during model merging.

**Mirror implication:** magnitude is a principled signal for deciding which delta components deserve shared coordinate capacity. Mirror shared/private allocation must compare against DELLA rather than random sparsification alone.

## PA177 — RegMean / dataless regression merging

**Dataless Knowledge Fusion by Merging Weights of Language Models**  
https://arxiv.org/abs/2212.09849

Introduces Regression Mean (RegMean), which uses layer-input Gram statistics to derive closed-form merged linear weights without requiring the original training examples at merge time.

**Mirror implication:** sufficient statistics can determine merge coefficients. A compact View-space merger should test whether the same statistics can solve for Mirror coordinates with less stored model state while preserving the RegMean objective.

## PA178 — Fisher-weighted model merging

**Merging Models with Fisher-Weighted Averaging**  
https://arxiv.org/abs/2111.09832

Approximates each model posterior with a Laplace/Fisher precision and merges parameters using importance-weighted averaging rather than an isotropic mean.

**Mirror implication:** importance weighting belongs in coordinate-space composition too. Compare Fisher-weighted Mirror-code merging against both full-weight Fisher merging and unweighted View arithmetic.

## PA179 — Model Stock

**Model Stock: All we need is just a few fine-tuned models**  
https://arxiv.org/abs/2403.19522

Studies how a small number of fine-tuned models can define useful interpolation geometry for producing a stronger merged model.

**Mirror implication:** a small model stock can itself define a low-dimensional functional manifold. Mirror should test whether that geometry can be represented by smaller reusable coordinates rather than storing many explicit interpolation endpoints.

## PA180 — KnOTS aligned LoRA merging

**Model merging with SVD to tie the Knots**  
https://arxiv.org/abs/2410.19735

Uses a joint SVD-derived transformation to align heterogeneous LoRA update spaces before applying merging methods in the aligned representation.

**Mirror implication:** LoRA task coordinates should be learned after subspace alignment when possible. Otherwise a Mirror basis may waste capacity explaining arbitrary factorization orientation instead of functional differences.

## PA181 — SWAG posterior subspace

**A Simple Baseline for Bayesian Uncertainty in Deep Learning**  
https://arxiv.org/abs/1902.02476

Stochastic Weight Averaging-Gaussian (SWAG) fits a Gaussian approximation around an SGD trajectory using a mean plus low-rank and diagonal covariance structure, enabling posterior sampling without independent full models.

**Mirror implication:** posterior samples are already low-dimensional logical models around one physical mean. A Mirror posterior must beat or compress the SWAG covariance representation at matched calibration and predictive quality.

## PA182 — Subspace Inference

**Subspace Inference for Bayesian Deep Learning**  
https://arxiv.org/abs/1907.07504

Constructs low-dimensional weight subspaces from training trajectories and performs Bayesian inference inside those subspaces.

**Mirror implication:** low-dimensional posterior coordinates are a direct prior art analogue of functional coordinates. The differentiating question is whether structured/factorized Mirror geometry stores the subspace or task-conditioned uncertainty more efficiently.

## PA183 — Laplace Redux

**Laplace Redux -- Effortless Bayesian Deep Learning**  
https://arxiv.org/abs/2106.14806

Systematizes practical Laplace approximations for neural networks with scalable curvature structures and post-hoc Bayesian uncertainty estimation.

**Mirror implication:** when the functional state is low-dimensional, curvature can be estimated directly in Mirror-code space. Compare code-space Laplace with full/subnetwork Laplace and deterministic View ensembles.

## PA184 — Packed-Ensembles

**Packed-Ensembles for Efficient Uncertainty Estimation**  
https://arxiv.org/abs/2210.09184

Packs several smaller independent subnetworks into grouped operations so ensemble members can be trained and evaluated efficiently in one network-shaped computation.

**Mirror implication:** efficient ensemble multiplicity does not require weight sharing. Mirror member Views must preserve useful predictive diversity while reducing physical member storage relative to Packed-Ensembles.

## PA185 — Snapshot Ensembles

**Snapshot Ensembles: Train 1, get M for free**  
https://arxiv.org/abs/1704.00109

Obtains multiple ensemble members from different points of one cyclic learning-rate training trajectory instead of training independent models from scratch.

**Mirror implication:** checkpoint trajectory is another source of logical model multiplicity. Mirror can attempt to encode snapshot differences as compact coordinates, but must match the ensemble benefit of storing the actual snapshots.

## PA186 — Fourier Neural Operator

**Fourier Neural Operator for Parametric Partial Differential Equations**  
https://arxiv.org/abs/2010.08895

Learns mappings between function spaces using Fourier-domain kernel parameterization, producing one operator model that can solve families of parameterized PDE instances and transfer across discretization resolutions.

**Mirror implication:** neural operators make the project framing literal: one physical learned operator already represents a function family. Mirror-specific value is cheaper specialization, composition, adaptation or regime-specific Views on top of that universal operator.

## PA187 — DeepONet

**DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators**  
https://arxiv.org/abs/1910.03193

Uses separate branch and trunk networks to encode an input function and an output evaluation location, then combines them to represent nonlinear operators.

**Mirror implication:** branch/trunk roles, sensing layouts and physical regimes expose several natural functional coordinates. Mirror should be tested against ordinary operator conditioning and latent-context adaptation, not against separately trained PDE models only.

## PA188 — R-GCN relation-basis sharing

**Modeling Relational Data with Graph Convolutional Networks**  
https://arxiv.org/abs/1703.06103

R-GCN gives each relation a message transform and explicitly proposes basis decomposition, where relation-specific weight matrices are linear combinations of shared basis matrices, plus a block-diagonal alternative.

**Mirror implication:** relation identity is already a compact function address. Mirror relation Views must beat or factor the free R-GCN basis coefficients and should test unseen relations rather than only memorized relation IDs.

## PA189 — CompGCN relation composition

**Composition-based Multi-Relational Graph Convolutional Networks**  
https://arxiv.org/abs/1911.03082

CompGCN composes entity and relation embeddings during message passing so relation semantics participate directly in the transformation rather than requiring an entirely independent operator for each relation.

**Mirror implication:** composition is a strong control for relation-conditioned functional coordinates. Mirror residuals are useful only if they add quality, extrapolation or storage efficiency beyond the relation-composition baseline.

## PA190 — ControlNet

**Adding Conditional Control to Text-to-Image Diffusion Models**  
https://arxiv.org/abs/2302.05543

Adds spatial conditioning by freezing the pretrained diffusion model and training copied encoder blocks connected through zero-initialized convolutions; separate controls can require substantial condition-specific trainable branches.

**Mirror implication:** the condition-specific copied branch is a high-value physical-duplication target. A shared control basis plus View codes should be judged on control fidelity, generation quality, bytes and extra FLOPs.

## PA191 — T2I-Adapter

**T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models**  
https://arxiv.org/abs/2302.08453

Learns lightweight condition-specific adapters on top of a frozen text-to-image diffusion model and supports composing multiple condition adapters at inference.

**Mirror implication:** T2I-Adapter is the cheap direct control for a Mirror control bank. The key question is whether a shared basis/code can preserve condition diversity and multi-control composition with fewer stored adapter parameters or fewer active adapter passes.

## PA192 — IP-Adapter

**IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models**  
https://arxiv.org/abs/2308.06721

Introduces a lightweight image-prompt adapter with decoupled cross-attention for image and text conditions.

**Mirror implication:** image-prompt roles can be modeled as logical attention Views, but Mirror must preserve the decoupling that prevents image conditioning from destroying text control.

## PA193 — Ctrl-Adapter

**Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model**  
https://arxiv.org/abs/2404.09967

Reuses pretrained ControlNets by learning compact bridges into different image/video diffusion backbones and includes fine-grained multi-condition routing.

**Mirror implication:** source-control x target-backbone is naturally a factorized coordinate. Test whether bridge parameters can be shared across this Cartesian space without sacrificing zero-shot control transfer.

## PA194 — CtrLoRA

**CtrLoRA: An Extensible and Efficient Framework for Controllable Image Generation**  
https://arxiv.org/abs/2410.09400

Uses low-rank adaptation to build extensible controllable-generation modules more cheaply than full condition-specific control branches.

**Mirror implication:** any low-rank Mirror control proposal must compare against a bank of CtrLoRAs and VeRA-like shared low-rank bases; the value must come from additional sharing/composition, not LoRA itself.

## PA195 — Growing Neural Cellular Automata

**Growing Neural Cellular Automata**  
https://distill.pub/2020/growing-ca/

Learns one local, recurrently applied neural update rule whose repeated decentralized interactions grow and regenerate target patterns.

**Mirror implication:** one physical local rule already produces a complex global computation. Mirror can ask whether a small rule/goal coordinate yields many stable logical dynamics without duplicating the update network.

## PA196 — Goal-Guided Neural Cellular Automata

**Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems**  
https://arxiv.org/abs/2205.06806

Conditions a shared NCA update process on goal information so the same self-organising substrate can produce different target-directed behaviors.

**Mirror implication:** goal encoding is a direct dynamic functional coordinate. Mirror-specific work should structure, compose, compress or robustify that coordinate and test unseen goals rather than merely adding conditioning.

## PA197 — Attention-based Neural Cellular Automata

**Attention-based Neural Cellular Automata**  
https://arxiv.org/abs/2211.01233

Augments NCA-style local computation with attention mechanisms to improve information exchange and learned self-organising behavior.

**Mirror implication:** context can choose the local function dynamically. Compare attention that only changes features against attention that generates/selects a compact update-rule View.

## PA198 — Online Task Adaptation via Self-Organisation

**Online Task Adaptation via Self-Organisation**  
https://arxiv.org/abs/2609.29281

Studies online adaptation through self-organising neural cellular systems, making deployment-time task change part of the learned dynamics rather than requiring a static task-specific model.

**Mirror implication:** online adaptation can be restricted to a compact functional coordinate, giving a direct test of whether transient task state can replace larger updates while retaining prior behaviors.

## PA199 — Real NVP / invertible coupling transforms

**Density estimation using Real NVP**  
https://arxiv.org/abs/1605.08803

Builds expressive exactly invertible transformations from tractable affine coupling layers, enabling exact forward/inverse mappings and latent-space manipulation.

**Mirror implication:** an invertible coupling map can act as a nonlinear coordinate chart for functional Views. Composition and inverse-cycle error become measurable properties, while exact invertibility prevents silent information destruction from being confused with useful specialization.

## PA200 — Invertible Residual Networks

**Invertible Residual Networks**  
https://arxiv.org/abs/1811.00995

Shows that residual networks can be made invertible under suitable contraction/Lipschitz constraints while retaining standard residual-network structure.

**Mirror implication:** reversible residual View families can specialize a shared block without abandoning invertibility. Runtime must include inverse iterations and any normalization needed to maintain the contraction constraint.

## PA201 — INNSteer nonlinear invertible activation steering

**Beyond Linear Activation Steering: Invertible Latent Transformations for Controlling LLM Behavior**  
https://arxiv.org/abs/2606.08454

Learns a lightweight invertible neural map from original LLM activations into a latent space where simple linear steering is easier, then maps the steered state back with the exact inverse. The resulting intervention is nonlinear and input-dependent in the original activation space.

**Mirror implication:** this is a direct Level-3 representation-space prior: a fixed small latent displacement can become a context-dependent nonlinear functional View after a learned coordinate warp. Multi-behavior Mirror work must compare against INNSteer rather than linear steering alone.

## PA202 — invertible adapter for one-step flow-matching policies

**Invertible Neural Network Adapter for One-Step Flow Matching in Robot Manipulation**  
https://arxiv.org/abs/2606.19194

Places an invertible neural adapter around the action-generation process so a flow-matching manipulation policy can refine high-dimensional actions in one inference step while preserving latent information.

**Mirror implication:** robot/action policies provide a latency-sensitive domain for invertible functional coordinates. A Mirror policy code is useful only if it preserves one-step execution and reduces per-skill storage or adaptation cost.

## PA203 — M-SMoE / MC-SMoE expert merge-then-compress

**Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy**  
https://proceedings.iclr.cc/paper_files/paper/2024/file/3d09a88c3372cdb79401fde592ca4db8-Paper-Conference.pdf

Uses routing statistics to align, group and frequency-weight expert merging; the merged experts exhibit lower-dimensional weight structure that is then compressed further.

**Mirror implication:** empirically merged expert groups define a much stronger candidate shared physical basis than arbitrary tying. Fit Mirror Views around merged experts and test whether discarded expert distinctions can be recovered cheaply.

## PA204 — MoRE / Mixture of Reused Experts

**MoRE: Mixture of Reused Experts**  
https://arxiv.org/abs/2609.18176

Shares expert pools across groups of adjacent Transformer layers while retaining per-layer routers; lightweight depth embeddings condition reused experts so they can distinguish which layer is invoking them.

**Mirror implication:** MoRE is very close to Mirror depth-specialized experts. Mirror must beat or complement simple depth embeddings while preserving the parameter savings of expert reuse.

## PA205 — UniPool globally shared expert pool

**UniPool: A Globally Shared Expert Pool for Mixture-of-Experts**  
https://arxiv.org/abs/2605.06665

Replaces per-layer expert ownership with a single global expert pool accessed by independent layer routers, using pool-level balancing and scale-stable routing. Reduced pools can match or exceed larger layer-wise expert budgets in the reported experiments.

**Mirror implication:** physical expert count is itself a global resource. Mirror should test whether compact layer/expert Views can shrink the physical pool further without losing depth-specific functions.

## PA206 — Expert Upcycling

**Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts**  
https://arxiv.org/abs/2604.19835

Expands a trained MoE by duplicating experts and extending the router while keeping active top-k fixed; continued pretraining breaks symmetry among duplicated copies. Utility scores can decide which experts receive more physical copies.

**Mirror implication:** expert duplication creates an explicit fork where the system can either pay for a new physical expert or try a cheap logical View first. This gives a clean physical-growth versus functional-coordinate frontier.

## PA207 — Cluster-Aware Upcycling

**Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling**  
https://arxiv.org/abs/2604.13508

Clusters dense-model activations semantically, initializes experts from cluster-specific truncated-SVD subspaces, initializes routing from cluster centroids, and uses self-distillation to stabilize specialization.

**Mirror implication:** semantic clusters and their SVD subspaces are a principled expert-coordinate initialization. A Mirror basis should compare against storing full cluster-specific upcycled experts.

## PA208 — Sparse Interpolated Mixture-of-Experts

**Automatic Expert Discovery in LLM Upcycling via Sparse Interpolated Mixture-of-Experts**  
https://aclanthology.org/2025.acl-long.816/

Creates experts during upcycling through sparse interpolation over reusable parameter components rather than relying only on identical expert copies.

**Mirror implication:** sparse interpolation coefficients are already compact logical expert coordinates. Mirror-specific value must come from factorizing, composing, robustifying or dynamically generating those coefficients.

## PA209 — REAP Experts

**REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression**  
https://arxiv.org/abs/2510.13999

Analyzes one-shot expert compression and derives a reconstruction-aware expert-importance criterion, showing strong results from pruning experts that contribute least to layer output.

**Mirror implication:** do not spend View capacity reconstructing functionally irrelevant experts. First prune with a causal/reconstruction-aware criterion; then ask whether useful removed distinctions can be represented as logical Views.

## PA210 — Routing-Free Mixture-of-Experts

**Routing-Free Mixture-of-Experts**  
https://arxiv.org/abs/2604.00801

Eliminates centralized top-k/softmax routing and lets individual experts determine activation through continuous learned self-activation, together with adaptive balancing.

**Mirror implication:** routing can live inside the expert function itself. A dynamic Mirror expert may jointly decide activation and transformation, but must compare against routing-free experts rather than assuming a separate router is necessary.

## PA211 — Soft MoE

**From Sparse to Soft Mixtures of Experts**  
https://proceedings.iclr.cc/paper_files/paper/2024/file/79fea214543ba263952ac3f4e5452b14-Paper-Conference.pdf

Uses differentiable soft assignments between tokens and expert slots rather than hard token-choice routing, avoiding several routing pathologies while retaining sparse expert computation.

**Mirror implication:** continuous expert assignment is a strong baseline whenever Mirror synthesizes or mixes logical experts. Distinguish gains from View geometry from gains due only to soft routing.

## PA212 — shared latent components in low-rank recurrent computation

**Interpretable compositional computation with recurrent neural networks**  
https://www.biorxiv.org/content/10.64898/2026.06.23.733979v1

Develops a theory in which multi-task low-rank recurrent networks reuse shared dynamical structures in a low-dimensional latent space, while task dependence can enter at distinct computational loci.

**Mirror implication:** this is direct evidence for reusable functional components in dynamics. Mirror task coordinates should be tested at the input, recurrent and readout loci separately, with causal perturbations validating that recovered components are real.

## PA213 — Vector Networks

**Learning Compositional Latent Structure with Vector Networks**  
https://arxiv.org/abs/2605.28007

Replaces fixed dense layer weights with libraries of reusable rank-1 weight atoms. Per-input local inference selects sparse atoms and coefficients to synthesize an input-specific low-rank weight matrix, yielding strong compositional generalization in reported tests.

**Mirror implication:** this is a direct dynamic-weight prior for shared physical atoms plus fast functional coordinates. Mirror must improve coefficient storage, inference cost, factorization or cross-task reuse beyond the native sparse atom inference.

## PA214 — deep Koopman latent linearization

**Deep learning for universal linear embeddings of nonlinear dynamics**  
https://www.nature.com/articles/s41467-018-07210-0

Learns nonlinear encoders/decoders so complex nonlinear dynamics can be represented by approximately linear evolution in a learned latent coordinate system.

**Mirror implication:** the latent linear operator is an especially interpretable place for task/regime View coordinates. Spectral stability and long-horizon error are mandatory because small operator errors compound through recurrence.

## PA215 — Koopman Neural Operator

**Koopman neural operator as a mesh-free solver of non-linear partial differential equations**  
https://arxiv.org/abs/2301.10022

Combines operator learning with Koopman-style latent evolution to model nonlinear PDE dynamics without tying the learned solution operator to a single mesh.

**Mirror implication:** PDE/system identity can be encoded directly in latent evolution-operator Views, giving a bridge between the FNO/DeepONet lane and structured recurrent dynamics.

## PA216 — adapter-bank motor options

**Learning Options for Compositional Motor Control with Adapter Banks**  
https://arxiv.org/abs/2609.17042

Uses a shared recurrent motor core plus a bank of residual adapters selected by discrete option codes; learned adapters become low-rank perturbations and can be sequenced by a high-level policy to produce novel motor combinations.

**Mirror implication:** this is an unusually direct physical-adapter-bank target. Mirror can compress option storage or factor option identity from sequence position, but must preserve out-of-distribution sequence composition.

## PA217 — separating the what and how of computation

**Separating the what and how of compositional computation to enable reuse and continual learning**  
https://arxiv.org/abs/2510.20709

Separates representations of computational content from the mechanisms that operate on that content so components can be reused and extended more independently across tasks.

**Mirror implication:** factorized coordinates should distinguish the operand/content ("what") from the transformation/program ("how"). Held-out cross-products and asymmetric continual-learning tests can directly test whether the factorization is meaningful.

## PA218 — Equilibrium State Estimation

**Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation**  
https://arxiv.org/abs/2606.13285

Forecasts multiple interacting systems in one pass by estimating a shared equilibrium state and predicting relative to that equilibrium, rather than running a separate predictor independently for each system.

**Mirror implication:** interacting-system identity is another logical role around shared global state. Mirror Views can specialize system reads/writes while retaining the single-pass shared computation.

## PA219 — task-dependent rank law for matrix memories

**The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition**  
https://arxiv.org/abs/2609.12259

Provides causal evidence in controlled group-composition tasks that learned matrix memory recruits the effective rank demanded by the task's minimal faithful representation.

**Mirror implication:** rank is not merely a hyperparameter but can be a task-imposed information budget. Mirror memories should allocate coordinate/state rank adaptively and compare against the known minimal-rank requirement.

## PA220 — GrapNet programmable dynamic neural graph

**GrapNet: A Programmable Dynamic-Architecture Neural Graph Substrate**  
https://arxiv.org/abs/2606.18923

Treats a neural graph's mutable connectivity as executable program structure rather than only as input data, enabling dynamic architectural relations over a reusable neural substrate.

**Mirror implication:** the functional coordinate can include topology/program structure, not only weights or activations. Separate the address cost of graph connectivity from the cost of node-function Views and measure dynamic execution overhead.

## PA221 — Kolmogorov-Arnold Networks

**KAN: Kolmogorov-Arnold Networks**  
https://arxiv.org/abs/2404.19756

Replaces scalar linear edge weights plus fixed node activations with learnable univariate functions on edges, typically spline-parameterized. The network therefore represents computation through compositions of local functions rather than dense matrices alone.

**Mirror implication:** edge functions themselves are physical objects that can be shared, transformed and addressed. Mirror-KAN work must compare against native per-edge functions and count function-evaluation cost, not only coefficient bytes.

## PA222 — GS-KAN shared parent functions

**GS-KAN: Parameter-Efficient Kolmogorov-Arnold Networks via Sprecher-Type Shared Basis Functions**  
https://arxiv.org/abs/2512.09084

Constructs unique KAN edge functions by applying learnable linear transformations to a single learnable shared parent function per layer, reducing the parameter explosion of one independent function per edge.

**Mirror implication:** this is a direct prior for "one physical function -> many logical edge functions." Mirror-specific value must come from better factorization, composition, dynamic generation, robustness or shared/private allocation beyond the GS-KAN transform.

## PA223 — Kolmogorov-Arnold Reservoir Computing

**Kolmogorov-Arnold Reservoir Computing**  
https://arxiv.org/abs/2606.19984

Replaces a recurrent reservoir with explicit univariate basis-function expansions over delayed coordinates and trains only a linear readout in closed form. It connects KAN-style representation with lightweight reservoir training and allows Fourier, B-spline or Chebyshev features.

**Mirror implication:** explicit basis functions form a cheap physical function bank for dynamical-system Views. Compare code-only/readout adaptation against independent KARC readouts and count active feature evaluation.

## PA224 — KanAdapter

**KanAdapter: A Kolmogorov-Arnold Network-based Plug-and-Play Module for Efficient Fine-tuning of Foundation Speech Models**  
https://arxiv.org/abs/2609.05281

Uses compact Group-Rational KAN modules as parallel adapters around frozen Transformer encoder blocks, replacing conventional MLP bottleneck adapters with learnable localized nonlinear functions.

**Mirror implication:** KAN-based task adapters are already a PEFT mechanism. Mirror-KAN adapters should share functional parents/bases across tasks or layers rather than merely replacing an MLP with a KAN.

## PA225 — NTK-CL

**Parameter-Efficient Fine-Tuning for Continual Learning: A Neural Tangent Kernel Perspective**  
https://arxiv.org/abs/2407.17120

Analyzes PEFT continual learning with Neural Tangent Kernels and identifies task-level feature orthogonality, sample size and regularization as major factors in continual generalization. NTK-CL adaptively generates task-relevant features without storing task-specific parameters.

**Mirror implication:** functional-coordinate separation can be evaluated in the induced NTK geometry rather than Euclidean code space. Any task-code bank should measure inter-task versus intra-task tangent similarity.

## PA226 — linearization of LLM fine-tuning

**Linearization Explains Fine-Tuning in Large Language Models**  
https://proceedings.neurips.cc/paper_files/paper/2025/file/becc00fe2e0ade58213cff16a166fa25-Paper-Conference.pdf

Studies first-order linearization around pretrained LLM parameters and shows that substantial aspects of fine-tuning behavior can be explained through the model's local tangent/Jacobian representation.

**Mirror implication:** when task adaptation is approximately linearizable, the natural compact object is a coordinate in tangent-feature space. Mirror should measure linearization residual before adding nonlinear/private capacity.

## PA227 — AI Engram / Fisher-geometric memory traces

**AI Engram: In Search of Memory Traces in Artificial Intelligence**  
https://arxiv.org/abs/2606.14997

Derives a closed-form estimator for causal memory traces satisfying specificity, reactivation, sufficiency and necessity constraints. The resulting solution corresponds to a minimum-norm/natural-gradient direction under Fisher information geometry and supports linear composition or erasure of memories.

**Mirror implication:** this is a strong causal alternative to heuristic task/edit vectors. Mirror memory coordinates should preserve the engram causal tests and can use Fisher geometry for code distance and shared/private decomposition.

## PA228 — differentiable plasticity

**Differentiable plasticity: training plastic neural networks with backpropagation**  
https://proceedings.mlr.press/v80/miconi18a.html

Meta-learns synaptic plasticity coefficients alongside slow weights so recurrent networks can change effective connections online through Hebbian-style updates after deployment.

**Mirror implication:** the functional coordinate can be the learning rule/plasticity itself, not only a static weight View. Report writable state and online update cost separately from persistent model bytes.

## PA229 — Hebbian Fast Weights

**Metalearning with Hebbian Fast Weights**  
https://arxiv.org/abs/1807.05076

Combines slow weights learned across tasks with fast weights written by a Hebbian rule during each new task, enabling one-shot binding of labels/representations.

**Mirror implication:** persistent task identity and transient fast state are naturally separate coordinate factors. Mirror can compress the fast matrix or structure the write rule, but must compare against the full Hebbian state.

## PA230 — DeltaNet fast matrix state

**Parallelizing Linear Transformers with the Delta Rule over Sequence Length**  
https://arxiv.org/abs/2406.06484

Scales DeltaNet, whose matrix-valued recurrent state is updated with a delta-rule associative-memory update. A generalized Householder/WY reparameterization enables parallel, memory-efficient training while preserving constant-memory recurrent inference.

**Mirror implication:** a writable matrix state is a dynamic physical memory object. Mirror can restrict it to a low-description state family or condition the write/erase rule, but long-context recall and state-update throughput are mandatory controls.

## PA231 — Learning Neural Network Subspaces

**Learning Neural Network Subspaces**  
https://proceedings.mlr.press/v139/wortsman21a.html

Learns lines, curves and simplexes containing diverse high-accuracy neural-network solutions within one training run, enabling model sampling/ensembling from a low-dimensional weight subspace.

**Mirror implication:** a trained low-loss model manifold is a direct functional-coordinate space. Count the basis/endpoints/control points as physical storage and compare new-task projection against random intrinsic/task-vector bases.

## PA232 — Bezier-surface mode connectivity

**Revisiting Mode Connectivity in Neural Networks with Bezier Surface**  
https://proceedings.iclr.cc/paper_files/paper/2025/file/fc65418739d66d1d4fc464807f177c91-Paper-Conference.pdf

Extends low-loss mode connectivity from curves between two models to learned Bezier surfaces connecting multiple models, with applications to averaging and ensembling.

**Mirror implication:** multi-dimensional low-loss surfaces can serve as learned model charts. Mirror coordinates on these surfaces should be compared against model soups, task-vector bases and simple learned subspaces.

## PA233 — Mesh Inference

**Mesh Inference: A Formal Model of Collective Inference Without a Center**  
https://arxiv.org/abs/2606.19537

Models collective inference among independent agents that exchange only admitted typed observations: no weights, gradients or hidden states are shared. In the linear-Gaussian regime, exact centralized recovery occurs under a carrier-connectivity condition, with decentralization incurring network-diameter-dependent latency.

**Mirror implication:** protocol/admission state can be a functional coordinate even when physical models remain private. Any collective Mirror must count messages/topology/policy bytes and preserve the observation-only boundary.

## PA234 — Structural Composition

**Structural Composition Enables Very Fast Learning**  
https://www.biorxiv.org/content/10.64898/2026.07.14.738142v1

Shows that multi-task models can learn a low-dimensional representation encoding how reusable task subcomponents are recombined. Restricting new-task learning to this structural subspace can greatly reduce required experience and sometimes reduce adaptation to hypothesis testing over a few discrete candidate points.

**Mirror implication:** useful task coordinates may encode recombination rules rather than module identities. This provides a direct test for module x rule factorization and discrete low-bit new-task codes.

## PA235 — Adaptive Behavior with Stable Synapses

**Adaptive behavior with stable synapses**  
https://arxiv.org/abs/2404.07150

Shows rapid in-context behavioral adaptation in gain-modulated recurrent networks with stable synaptic weights, using input segregation and dendritic/gain modulation rather than changing model parameters online.

**Mirror implication:** Level-3 functional state need not be a writable weight matrix. Gain/context state is a lower-cost control for dynamic Mirror adaptation and should be compared against Hebbian/DeltaNet fast-weight mechanisms.
