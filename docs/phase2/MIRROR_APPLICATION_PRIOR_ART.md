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

## PA236 — Cross-Model KV Cache Transfer

**Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**  
https://arxiv.org/abs/2608.03893

Fits per-head closed-form ridge maps between source/target KV states using layer selection and RoPE removal. Reported transfer accuracy varies materially by model pair; some pairs fail sharply.

**Mirror implication:** The map is direct prior art for cache translation. Mirror must compress a family of maps or provide exact identity-orbit handling and cannot claim cross-model KV reuse itself.

## PA237 — CacheBridge

**CacheBridge: Efficient Cross-Model KV Cache Transfer**  
https://arxiv.org/abs/2609.00891

Uses matched head support, causal-attention-weighted calibration, and fused sufficient-statistic accumulation for compact closed-form cross-model cache transfer.

**Mirror implication:** Match mapper bytes, calibration budget, attention-weighted errors, construction cost and GPU application time before claiming an additional Mirror advantage.

## PA238 — Mixture-of-Translators

**Mixture-of-Translators: Translating KV Caches Across Heterogeneous Large Language Models**  
https://arxiv.org/abs/2607.28979

Uses multiple translation modules plus target-side context correction to bridge heterogeneous source/target LLM cache spaces, and reports early/late injection failure modes.

**Mirror implication:** Mirror translator codes must beat or complement MoT's native mixture; track target trajectory quality and all translator/router bytes.

## PA239 — Universal cross-model context layer

**A Universal Context-Reuse Layer for Cross-Model KV Sharing**  
https://arxiv.org/abs/2608.30963

Studies KV handoff even across different model scales, architectures, tokenizer settings and families, with end-to-end prefill/latency and quality measurements.

**Mirror implication:** Cross-family transfer and source-target token provenance are live comparators, not speculative benefits. Mirror must preserve token semantics and count source execution.

## PA240 — StitchLLM

**StitchLLM: Serving LLMs, One Block at a Time**  
https://aclanthology.org/2025.acl-long.1305/

Routes across pretrained LLM blocks using trainable stitching layers while choosing computation under a serving cost-quality tradeoff.

**Mirror implication:** Cross-model block connectors already exist. Test a shared connector basis plus tiny Mirror codes against independent stitching layers and native routing.

## PA241 — Stitching validity counterexample

**Functional Alignment Can Mislead: Examining Model Stitching**  
https://proceedings.mlr.press/v267/smith25a.html

Shows that successful functional stitching can occur between representations with substantially different information or even distinct task/data biases.

**Mirror implication:** Functional output matching is insufficient evidence of shared knowledge or semantic equivalence. Include information-preservation and counterfactual transfer diagnostics.

## PA242 — Cross-model residual feature transport

**Transferring Linear Features Across Language Models With Model Stitching**  
https://proceedings.neurips.cc/paper_files/paper/2025/file/4569a868e7aa891248832ec08445d071-Paper-Conference.pdf

Uses affine maps between residual streams of different language models to transfer SAE features, probes and steering directions; semantic and structural features transfer differently.

**Mirror implication:** Mirror cross-model transport must outperform a per-pair affine connector and audit feature-specific loss, not only aggregate task scores.

## PA243 — Continual multi-scene C-NGP

**Incremental Multi-Scene Modeling via Continual Neural Graphics Primitives**  
https://arxiv.org/abs/2411.19903

Uses a shared continually trained neural graphics primitive model and scene pseudo-labels with replay for multiple 3D scenes without one full model per scene.

**Mirror implication:** Scene conditioning alone is established. Mirror must improve code cost, compositional variation, retention, or shared/private scene capacity over C-NGP.

## PA244 — ReFiNe multi-scene field

**ReFiNe: Recursive Field Networks for Cross-modal Multi-scene Representation**  
https://arxiv.org/abs/2406.04309

Uses a shared recursive hierarchical field representation to pack multiple shapes/scenes into one network with compact latent features.

**Mirror implication:** Mirror hierarchical scene coordinates must beat native ReFiNe latent codes and count node/structure storage and ray-query runtime.

## PA245 — Instant-NGP hash-grid fields

**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**  
https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding

Uses trainable multiresolution spatial hash grids plus a small fused neural network for efficient high-fidelity neural graphics primitives.

**Mirror implication:** Hash tables are paid physical parameters. A Mirror multi-scene decoder must beat one-scene-per-grid and ordinary shared-decoder+scene-code controls.

## PA246 — TensoRF tensor field decomposition

**TensoRF: Tensorial Radiance Fields**  
https://arxiv.org/abs/2203.09517

Represents scene radiance features with low-rank CP or vector-matrix tensor factors, cutting field storage compared with dense voxel features.

**Mirror implication:** Tensor factors and their ranks are a strong baseline for Mirror modulation of scene/time/material fields; count all factors as bytes.

## PA247 — K-Planes space-time field

**K-Planes: Explicit Radiance Fields in Space, Time, and Appearance**  
https://arxiv.org/abs/2301.10241

Factorizes high-dimensional radiance fields into pairs of coordinate planes, including distinct spatial and space-time factors and linear decoder.

**Mirror implication:** Mirror time/appearance addresses must improve over native plane factorization or permit extra useful functions at low marginal bytes.

## PA248 — 4D Gaussian Splatting

**4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**  
https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html

Combines canonical 3D Gaussians, decomposed 4D neural voxels and a compact deformation decoder for real-time dynamic rendering.

**Mirror implication:** Native shared canonical geometry+deformation is mandatory. Mirror must share multiple logical motions or actors without merely restating 4DGS.

## PA249 — Anchor-driven dynamic Gaussian codec

**ADC-GS: Anchor-Driven Deformable and Compressed Gaussian Splatting for Dynamic Scene Reconstruction**  
https://www.ijcai.org/proceedings/2025/132

Uses anchors, hierarchical deformation and rate-distortion objectives to remove neighboring Gaussian motion redundancy.

**Mirror implication:** Mirror anchor/motion codes need to improve the native anchor-driven storage/rendering frontier and preserve temporal continuity.

## PA250 — CC-4DGS compressed deformation

**CC-4DGS: Computational Deformation and Point-Cloud Compression for Storage-Efficient Dynamic Gaussian Splatting**  
https://arxiv.org/abs/2609.02184

Uses computational deformation with compact decoders instead of large learned hash tables, plus conditional Gaussian attribute coding and quantized residual codebooks.

**Mirror implication:** Compare against this already-compressed baseline; count all decoders/codebooks/appearance streams and actual rendering FPS.

## PA251 — P-4DGS predictive Gaussian coding

**P-4DGS: Predictive 4D Gaussian Splatting with 90x Compression**  
https://arxiv.org/abs/2510.10030

Applies spatial-temporal anchor prediction and context-adaptive entropy coding to compress dynamic Gaussian sequences.

**Mirror implication:** Prediction and entropy coding are strong direct controls for temporal Mirror codes; record actual bitstream size and random-access latency.

## PA252 — Object-centric Gaussian world model

**Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects**  
https://arxiv.org/abs/2606.01950

Represents objects with canonical-frame Gaussians and predicts action-conditioned rigid transformations using a spatial-temporal transformer.

**Mirror implication:** Mirror object identity, SE(3) motion and action roles separately; preserve rigid geometry and real rollout/OOD interaction metrics.

## PA253 — NanoVoice shared speaker adaptation

**NanoVoice: Efficient Speaker-Adaptive Text-to-Speech for Multiple Speakers**  
https://arxiv.org/abs/2409.15760

Performs multi-speaker batch adaptation with shared parameters and a trainable scale matrix to limit per-speaker adapter growth.

**Mirror implication:** The learned speaker scale is strong direct prior for speaker Mirror coordinates. Compare speaker identity quality and actual bytes/voice, not only trainable counts.

## PA254 — HyperTTS generated adapters

**HyperTTS: Parameter Efficient Adaptation in Text to Speech using Hypernetworks**  
https://aclanthology.org/2024.lrec-main.747/

Conditions adapter-weight generation on speaker representations using a learned hypernetwork for dynamic TTS adaptation.

**Mirror implication:** Mirror codes must reduce generated adapter state/generator cost or improve compositional voice quality beyond native HyperTTS.

## PA255 — Lightweight TTS mixture of adapters

**Lightweight Zero-shot Text-to-Speech with Mixture of Adapters**  
https://arxiv.org/abs/2407.01291

Selects speaker-conditioned mixtures of lightweight adapters inside a TTS decoder and variance adapter.

**Mirror implication:** Speaker-conditioned selection already exists. Mirror must compress adapter bank or improve voice diversity per byte without additional routing cost.

## PA256 — Hyper-MoA multi-speaker adaptation

**Hyper-MoA: Achieving high quality and parameter efficiency in few-shot multi-speaker TTS**  
https://doi.org/10.1016/j.apacoust.2025.111118

Combines a hypernetwork and mixture of adapters to adapt multiple new TTS speakers within one module.

**Mirror implication:** This is an especially close shared-speaker adaptation prior; compare native shared Hyper-MoA rather than separate speaker LoRAs only.

## PA257 — Interventional speech disentanglement

**Learning task-specific subspaces via interventional post-training of speech foundation models**  
https://arxiv.org/abs/2606.17967

Learns transformations separating speaker and content subspaces in speech representations through interventional contrastive training.

**Mirror implication:** Speaker/content Mirror factorization must be causally checked via swaps, cross-speaker/phonetic held-outs and OOD speaker verification.

## PA258 — StableVC timbre-style factorization

**StableVC: Style Controllable Zero-Shot Voice Conversion with Conditional Flow Matching**  
https://arxiv.org/abs/2412.04724

Separates content, speaker timbre and style for zero-shot voice conversion via conditional flow matching and adaptive gated dual attention.

**Mirror implication:** Mirror must improve the content/timbre/style coordinate or adapter size while preserving prosody and linguistic fidelity.

## PA259 — HybridCodec semantic-acoustic streams

**HybridCodec: Fast Dual-Stream, Semantically Enhanced Neural Audio Codec**  
https://arxiv.org/abs/2606.06743

Combines separate semantic and acoustic branches, distilling speech SSL semantics into a dual-stream neural audio codec.

**Mirror implication:** A Mirror semantic/acoustic state must beat native stream-specific codebooks and report bitrate plus reconstruction, speaker and content accuracy.

## PA260 — Flow Map Matching

**Flow map matching with stochastic interpolants: A mathematical framework for consistency models**  
https://arxiv.org/abs/2406.07507

Learns a two-time flow map, unifying few-step consistency/diffusion trajectory learning and enabling variable sampling budgets.

**Mirror implication:** A Mirror time-interval/path code must outperform ordinary FMM time conditioning, including flow-map composition errors and end-to-end NFE.

## PA261 — Consistency Models

**Consistency Models**  
https://proceedings.mlr.press/v202/song23a.html

Learns mappings from noisy states to data, allowing one/few-step diffusion generation with different compute budgets.

**Mirror implication:** Mirror step/condition views need a quality-per-byte and NFE advantage over native consistency-model conditioning and distillation.

## PA262 — S4S learned diffusion solvers

**S4S: Solving for a Fast Diffusion Model Solver**  
https://proceedings.mlr.press/v267/frankel25a.html

Directly optimizes a lightweight diffusion solver and optionally its discretization schedule to match teacher-sampler outputs.

**Mirror implication:** A Mirror solver coordinate is only useful if it outperforms learned native solver coefficients/schedules and counts all NFEs.

## PA263 — LoRA.rar subject-style hypermerge

**LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style Conditioned Image Generation**  
https://arxiv.org/abs/2412.05148

Learns a hypernetwork that composes subject and style LoRAs for held-out pairs, avoiding expensive per-combination optimization.

**Mirror implication:** Factorized Mirror subject/style codes must beat native LoRA.rar composition at equal storage and generation fidelity.

## PA264 — EST-LoRA timestep style selection

**Subject or Style: Adaptive and Training-Free Mixture of LoRAs**  
https://arxiv.org/abs/2508.02165

Applies timestep-dependent adaptive layerwise selection of subject and style LoRAs via energy/style-discrepancy heuristics.

**Mirror implication:** Temporal Mirror style/subject mixing must beat this cheap training-free baseline before claiming a special dynamic code benefit.

## PA265 — GenSplatCodec one-step Gaussian decoding

**GenSplatCodec: Feed-Forward Gaussian Splatting Compression via One-Step Diffusion**  
https://arxiv.org/abs/2607.24403

Combines compact structural Gaussian and reference-appearance streams with geometry-guided one-step generative decoding.

**Mirror implication:** Mirror scene/appearance code must preserve cross-view consistency while improving bits and latency against this native codec.

## PA266 — NeRV video INR

**NeRV: Neural Representations for Videos**  
https://proceedings.neurips.cc/paper/2021/hash/b44182379bf9fae976e6ae5996e13cd8-Abstract.html

Represents frame sequence as a neural function queried by a frame/time index, storing video-specific information in decoder weights.

**Mirror implication:** A frame code/time address and implicit video decoding are prior art; Mirror must compress a bank of video/segment-specific decoders or add useful functional differences beyond simple embeddings.

## PA267 — HNeRV hybrid video INR

**HNeRV: A Hybrid Neural Representation for Videos**  
https://arxiv.org/abs/2304.02633

Uses content-adaptive encoded frame embeddings fed to a neural video decoder rather than only fixed frame indices.

**Mirror implication:** Content-aware per-frame latent is a direct control for Mirror temporal Views; count encoder and per-frame embeddings/bitstream, not just decoder weights.

## PA268 — NerVast segment parameter sharing

**NerVast: Compression-Efficient Scaling of Implicit Neural Video Representations via Scene-based Parameter-sharing**  
https://openaccess.thecvf.com/content/WACV2026/html/Lee_NerVast_Compression-Efficient_Scaling_of_Implicit_Neural_Video_Representations_via_Scene-based_WACV_2026_paper.html

Groups temporally related video chunks, selects parameter-sharing masks using a Fisher-like sensitivity proxy and jointly optimizes shared/nonshared INR weights.

**Mirror implication:** This is a very close direct prior for selecting shared physical video parameters. Mirror must replace part of the remaining chunk-private state with small m and improve actual bitrate/decoder FPS beyond NerVast.

## PA269 — DCVC-UF chunk-parallel neural video

**Ultra-Fast Neural Video Compression**  
https://openaccess.thecvf.com/content/CVPR2026/html/Li_Ultra-Fast_Neural_Video_Compression_CVPR_2026_paper.html

Encodes multiple adjacent video frames into one chunk latent and reconstructs frames in parallel using cross-frame interaction and frame-specific decoders with streamlined entropy coding.

**Mirror implication:** The chunk latent and frame-parallel decoder are native prior art. Test Mirror frame-offset Views as a lower-byte alternative to frame-specific decoder state; measure actual coded bits and decode FPS.

## PA270 — DCVC-RT practical neural video

**Towards Practical Real-Time Neural Video Compression**  
https://arxiv.org/abs/2502.20762

Optimizes real-time neural video coding by reducing function-call and memory-I/O costs, using implicit temporal modeling and rate-control module banks.

**Mirror implication:** A small Mirror transform is not useful if it increases kernel launches or memory traffic. Compare native real-time codec, rate-control bank, true latency and coding bitstream.

## PA271 — DCMVC temporal context modulation

**Neural Video Compression with Context Modulation**  
https://arxiv.org/abs/2505.14541

Uses flow orientation and reference-driven context compensation to modulate propagated temporal features in a conditional video codec.

**Mirror implication:** Mirror temporal context modulation must beat existing oriented context and compensation, not simply beat unmodulated propagation.

## PA272 — CoANeRV shared coordinate decoder

**CoANeRV: Coordinate-Aware Token-Space Neural Video Representation**  
https://arxiv.org/abs/2608.13938

Forms per-video tokens in a feed-forward pass and reconstructs spatiotemporal coordinates through a shared coordinate-conditioned decoder.

**Mirror implication:** This already uses one decoder plus compact video-specific representations. Mirror can target factorized time/video/region codes only if it improves native token rate-distortion and memory.

## PA273 — S-NVRC prefix video codec

**Scalable Neural Video Representation Compression**  
https://arxiv.org/abs/2609.04273

Uses prefix-nested feature grids and decoder layers for one embedded scalable video bitstream with adjustable bitrate and decoding complexity.

**Mirror implication:** Variable code/rank/width for video is prior art. Mirror m must improve the coded rate-distortion-complexity frontier over native nested prefixes.

## PA274 — SIEDD shared implicit video encoder

**SIEDD: Shared-Implicit Encoder with Discrete Decoders**  
https://arxiv.org/abs/2506.23382

Combines a coordinate-based encoder shared across frame groups with parallel lightweight discrete decoders, emphasizing encoding speed.

**Mirror implication:** Mirror can compress the remaining group-specific decoders into shared Views, but must account for shared-encoder training and coding/decoding latency.

## PA275 — G-CNN exact group equivariance

**Group Equivariant Convolutional Networks**  
https://proceedings.mlr.press/v48/cohenc16.html

Uses group convolution/parameter tying so transformed features follow known group actions.

**Mirror implication:** A group element as View m is not independently novel. Exact equivariant group orbits often add no independent functional information; compare group convolution and gauge invariance.

## PA276 — Steerable CNN irreducible feature types

**Steerable CNNs**  
https://arxiv.org/abs/1612.08498

Constructs steerable feature representations from group-theoretic elementary types and equivariant filter bases.

**Mirror implication:** The steerable basis and group action are direct controls. Mirror-specific value requires task-dependent useful symmetry breaking, code factorization or lower actual storage/compute.

## PA277 — EGNN geometric equivariance

**E(n) Equivariant Graph Neural Networks**  
https://proceedings.mlr.press/v139/satorras21a.html

Constructs geometric message passing equivariant to rotations/translations/reflections/permutations without expensive higher-order features.

**Mirror implication:** Mirror task/geometry code must beat native EGNN and demonstrate more than coordinate-equivalent outputs; geometric frame changes alone are not added logical capacity.

## PA278 — e3nn tensor products and irreps

**e3nn: Euclidean Neural Networks**  
https://arxiv.org/abs/2207.09453

Provides E(3)-equivariant tensor-product and spherical-harmonic operators over irreducible geometric feature types.

**Mirror implication:** Any irrep-channel Mirror code must count tensor-product basis and Wigner/CG computation and preserve exact or bounded equivariance error.

## PA279 — Learned symmetry stochastic weight sharing

**Learning Symmetries via Weight-Sharing with Doubly Stochastic Tensors**  
https://arxiv.org/abs/2412.04594

Learns soft data-dependent symmetry structures through trainable approximately weight-sharing doubly stochastic tensors.

**Mirror implication:** Learned soft equivariance is prior art. Mirror should compress task-specific learned symmetry choices beyond native stochastic sharing and measure off-orbit task benefit.

## PA280 — SEMoLA discovered symmetry

**Learning equivariant models by discovering symmetries with learnable augmentations**  
https://arxiv.org/abs/2506.03914

Discovers data-relevant symmetry transformations using learned augmentations jointly with equivariant modeling rather than requiring a fixed known group.

**Mirror implication:** Mirror cannot claim symmetry discovery itself. Test whether compact m encodes learned group selection per task or domain more efficiently than SEMoLA/native augmentation.

## PA281 — TACOS spiking continual learner

**TACOS: Task Agnostic Continual Learning in Spiking Neural Networks**  
https://arxiv.org/abs/2409.00021

Combines synaptic consolidation, neuromodulation and metaplasticity to mitigate continual interference without explicit task labels and without growing stored state.

**Mirror implication:** TACOS is a strong continual spiking control; Mirror task labels are unfair when unavailable natively. Report synaptic/plastic state, spike operations, energy and forgetting.

## PA282 — STL-SNN learned thresholds

**A Synapse-Threshold Synergistic Learning Approach for Spiking Neural Networks**  
https://arxiv.org/abs/2206.06129

Jointly learns synaptic weights and spike thresholds, showing neuronal threshold parameters can modify effective spiking computation.

**Mirror implication:** Threshold-based functional modulation is prior art. Mirror adds value only by sharing/modulating multiple task- or time-specific thresholds at lower cost or higher accuracy/energy efficiency.

## PA283 — TEBN temporal spiking normalization

**Temporal Effective Batch Normalization in Spiking Neural Networks**  
https://proceedings.neurips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html

Uses per-time-step normalization scaling to improve SNN training dynamics and low-time-step performance.

**Mirror implication:** A temporal gain table is not Mirror-specific novelty; structured m must beat per-step TEBN coefficients at equal state and spike throughput.

## PA284 — EAS-SNN event sampling

**EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks**  
https://arxiv.org/abs/2403.12574

Co-designs adaptive event sampling and recurrent spiking representations for event camera detection.

**Mirror implication:** Mirror time-bin or sensor-event coordinates must preserve event timing and detect quality beyond native adaptive sampling at actual event operations and energy.

## PA285 — Event temporal hyperacuity

**Temporal coding enables hyperacuity in event-based vision**  
https://www.nature.com/articles/s41467-026-76878-6

Shows that precise event timing under controlled sensor motion carries subpixel discrimination information often lost in frame aggregation.

**Mirror implication:** Do not let temporal Mirror token/phase grouping discard event timing essential for target decisions; test fine-timing ablations and comparable latency/energy.

## PA286 — LightPro programmable photonic couplers

**LightPro: a linear photonic processor with full programmability**  
https://www.nature.com/articles/s44172-026-00707-3

Develops programmable photonic matrix-vector multiplication using phase-change tunable directional couplers plus architecture search and pruning.

**Mirror implication:** Programmable optical hardware already exists. Mirror must reduce number of configured devices, programming bits, switch latency and total energy relative to full LightPro reconfiguration.

## PA287 — MDR-HDONN physically reconfigurable diffraction

**Multi-Dimensional Reconfigurable, Physically Composable Hybrid Diffractive Optical Neural Network**  
https://arxiv.org/abs/2411.05748

Reuses physically fabricated diffractive elements by differentiably adapting system variables and composing optical/photonic modules.

**Mirror implication:** Optical physical reuse via tunable geometry is prior art. Mirror should test smaller phase/program codes at preserved optical task quality and switching energy.

## PA288 — Neural network beam codebooks

**Neural Codebook Design for Network Beam Management**  
https://arxiv.org/abs/2403.03053

Learns joint access/beam/CSI codebooks across sectors for efficient wireless beam management.

**Mirror implication:** Beam codebooks are direct prior for physical antenna elements with logical beam roles; Mirror must improve limited-feedback bits, spectral efficiency or codebook state beyond a learned beam codebook.

## PA289 — Site-specific Type-II CSI subspace

**Bridging Standardized Codebook and Site-Specific Beamforming: A Unified Limited-Feedback Framework**  
https://arxiv.org/abs/2604.14524

Uses low-overhead site fingerprints to infer a dominant beam subspace then feeds back its low-dimensional effective channel coefficients.

**Mirror implication:** Small subspace feedback codes are already known. Mirror task/site m must reduce feedback and preserve spectral efficiency beyond this direct baseline.

## PA290 — RIS joint beamforming

**Low-Complexity Joint Beamforming for RIS-Assisted MU-MISO Systems Based on Model-Driven Deep Learning**  
https://arxiv.org/abs/2311.15313

Jointly optimizes configurable intelligent-surface phases and access-point beamforming via model-driven learning.

**Mirror implication:** RIS phase state is a controllable physical operator, not automatically a new Mirror parameter. Compare native WMMSE/model-driven beamforming, CSI signaling and switch costs.

## PA291 — CsiNet feedback codec

**Deep Learning for Massive MIMO CSI Feedback**  
https://arxiv.org/abs/1712.08919

Encodes/downsamples MIMO channel-state information to low-dimensional feedback codewords and reconstructs useful channel estimates.

**Mirror implication:** Mirror CSI codes must be evaluated against native CSI feedback compressed rate, beamforming quality and calibration, not only reconstruction MSE.

## PA292 — RANF spatial-audio personalization

**Retrieval-Augmented Neural Field for HRTF Upsampling and Personalization**  
https://arxiv.org/abs/2501.13017

Retrieves acoustically similar listeners and uses their head-related transfer functions with a neural field to reconstruct dense spatial HRTF maps from sparse directions.

**Mirror implication:** Listener-specific neural fields and retrieval are prior art. Mirror must reduce per-listener code/measurements while preserving spatial audio perception and frequency response.

## PA293 — Anthropometric HRTF latent personalization

**Head-Related Transfer Function Individualization Using Anthropometric Features and Spatially Independent Latent Representation**  
https://arxiv.org/abs/2508.16176

Predicts personalized head-related transfer-function latents using listener anthropometric features and direction-conditioned decoder.

**Mirror implication:** A person latent and direction coordinate already exist; Mirror should test compact factorized listener×direction function Views against the native latent decoder.

## PA294 — Parameter-free approximate equivariance

**Parameter-free approximate equivariance for tasks with finite group symmetry**  
https://arxiv.org/abs/2506.08244

Imposes approximate equivariance with no new learnable group parameter by using a latent-symmetry consistency constraint.

**Mirror implication:** A zero-parameter symmetry regularizer is a very strong cost control; Mirror symmetry m needs additional useful functionality or generalization beyond this constraint.

## PA295 — Coherent nanophotonic programmable MZI

**Deep learning with coherent nanophotonic circuits**  
https://www.nature.com/articles/nphoton.2017.93

Experimentally demonstrated a programmable silicon-photonic processor using a mesh of Mach-Zehnder interferometers as an optical matrix-computation substrate.

**Mirror implication:** Optical phase-programming is established hardware; count configuration bits, calibration drift, insertion loss and measured reconfiguration cost before identifying Mirror-specific benefits.

## PA296 — MACE equivariant interatomic potentials

**MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields**  
https://arxiv.org/abs/2206.07697

High-order E(3)-equivariant atomistic force-field network.

**Mirror implication:** Native equivariant energy/force potential is the control; added material-code m must retain force consistency and improve state/quality.

## PA297 — MACE-MP material foundation model

**A foundation model for atomistic materials chemistry**  
https://arxiv.org/abs/2401.00096

Pretrained shared foundation interatomic potential across material chemistries.

**Mirror implication:** A reusable foundation potential is prior art. Compare Mirror material Views against its native frozen and fine-tuned models.

## PA298 — NequIP equivariant force fields

**E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials**  
https://www.nature.com/articles/s41467-022-29939-5

E(3)-equivariant atomic environment neural potential for accurate forces.

**Mirror implication:** Compare native NequIP and independent force fields; preserve energy invariance, force equivariance and conservative derivatives.

## PA299 — MatterSim atomistic foundation model

**MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures**  
https://arxiv.org/abs/2405.04967

General atomistic model spanning diverse elements and thermodynamic conditions.

**Mirror implication:** Mirror temperature/pressure/composition codes must beat native condition response and out-of-distribution simulations.

## PA300 — Sparse fine-tuning materials models

**Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-promoting Fine-tuning**  
https://arxiv.org/abs/2606.18691

Selective sparse parameter updating of equivariant materials foundation models compared to full and equivariant low-rank tuning.

**Mirror implication:** Mandatory strong sparse/native control; count selected weights/mask metadata and energy/force/MD quality before crediting Mirror m.

## PA301 — Frozen-transfer interatomic adaptation

**Fine-tuning foundation models of materials interatomic potentials with frozen transfer learning**  
https://arxiv.org/abs/2502.15582

Partially frozen pretrained potentials adapted for reactions and alloys with fewer reference computations.

**Mirror implication:** Mirror task adaptation must improve the native frozen-transfer sample/parameter frontier and count ab-initio reference cost.

## PA302 — FastMRI VarNet

**End-to-End Variational Networks for Accelerated MRI Reconstruction**  
https://arxiv.org/abs/2004.06688

Unrolled multi-coil MRI neural reconstruction with sensitivity estimates and k-space consistency.

**Mirror implication:** Compare native VarNet and matched scan/cascade adapters; uphold physical measurement consistency and realistic reconstruction metrics.

## PA303 — MoDL unrolled inverse problems

**MoDL: Model Based Deep Learning Architecture for Inverse Problems**  
https://arxiv.org/abs/1712.02862

Uses a shared denoiser across data-consistent iterative inverse-problem blocks.

**Mirror implication:** Depth-tying is prior art. Mirror per-iteration m must beat shared MoDL and native learnable iteration coefficients.

## PA304 — DUNE representation-space inverse problems

**Deep Unrolled Networks in Representation Space Applied to MRI Reconstruction**  
https://arxiv.org/abs/2606.21602

Physics-aware MRI unrolling in learned representation space with data consistency.

**Mirror implication:** Representation-space View requires data-consistency and measured image quality, not only a low-dimensional latent reconstruction.

## PA305 — D2SA MRI test-time adaptation

**D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction**  
https://arxiv.org/abs/2503.20815

Distribution-level and slice-level MRI adaptation across scanners/protocols.

**Mirror implication:** A scan m is not new test-time adaptation; count online updates, per-slice storage and disjoint measurement splits.

## PA306 — Multi-contrast MRI INR

**INR meets Multi-Contrast MRI Reconstruction**  
https://arxiv.org/abs/2509.04888

Investigates implicit neural representation for undersampled multiple-contrast MR imaging.

**Mirror implication:** Compare native continuous contrast/scan conditioning, not only one independent model per sequence.

## PA307 — SSDU self-supervised MRI

**Self-Supervised Learning of Physics-Guided Reconstruction Neural Networks without Fully-Sampled Reference Data**  
https://pmc.ncbi.nlm.nih.gov/articles/PMC7811359/

Splits acquired k-space samples to train data-consistent reconstruction without fully sampled references.

**Mirror implication:** MRI Mirror validation must prevent self-supervised mask leakage and hallucinated anatomy; SSDU is a native training control.

## PA308 — Quantum data re-uploading

**Data re-uploading for a universal quantum classifier**  
https://quantum-journal.org/papers/q-2020-02-06-226/

Reuses input-dependent angle gates over variational circuit depth.

**Mirror implication:** Parameterized gate schedule is prior art; Mirror m requires multi-task functional gains beyond data re-uploading, with gate/shot costs.

## PA309 — TensorHyper-VQC TT quantum adapter

**TensorHyper-VQC: a tensor-train-guided hypernetwork for robust and scalable variational quantum computing**  
https://www.nature.com/articles/s41534-025-01157-z

Classical tensor-train hypernetwork generates trainable quantum circuit parameters and addresses noise/trainability.

**Mirror implication:** TT-hypernetwork is strong direct control for compact generated quantum parameters; count hardware depth, shots and TT storage.

## PA310 — Superposed parameterised quantum circuits

**Superposed parameterised quantum circuits**  
https://arxiv.org/abs/2506.08749

Uses quantum random-access and postselection to represent multiple parameterised circuit states.

**Mirror implication:** Virtual circuit multiplicity is not free: count qRAM, postselection success, repeated attempts and actual physical gates.

## PA311 — VQC hardware-aware compilation

**Hardware-Aware Compilation Reshapes Trainability in Variational Quantum Circuits**  
https://arxiv.org/abs/2604.16527

Shows transpilation changes gradient/trainability properties of variational quantum circuits.

**Mirror implication:** Evaluate Mirror parameterized ansatz only after architecture-matched hardware compilation, noise and shot budgets.

## PA312 — CoOp visual-language context learning

**Learning to Prompt for Vision-Language Models**  
https://arxiv.org/abs/2109.01134

Learns prompt context vectors on frozen CLIP-like vision-language models.

**Mirror implication:** Static learned prompt m is prior art; Mirror must beat CoOp at comparable bytes and novel-class transfer.

## PA313 — CoCoOp conditional CLIP prompts

**Conditional Prompt Learning for Vision-Language Models**  
https://arxiv.org/abs/2203.05557

Uses image-conditioned generated prompt token for base-to-novel class generalization.

**Mirror implication:** Dynamic m is not novelty alone; compare conditional prompt generator cost and unseen-class accuracy to CoCoOp.

## PA314 — MaPLe coupled multimodal prompts

**MaPLe: Multi-modal Prompt Learning**  
https://arxiv.org/abs/2210.03117

Learns coupled visual/text prompts across layers of frozen vision-language model.

**Mirror implication:** A vision×text factor code must beat the native coupled prompts, charging both branches and all layers.

## PA315 — SAM2 promptable temporal memory

**SAM 2: Segment Anything in Images and Videos**  
https://arxiv.org/abs/2408.00714

Video object segmentation from prompts using a streaming temporal memory bank.

**Mirror implication:** Object memory and interactive mask prompts are native; Mirror must improve object-specific memory per byte and segmentation runtime.

## PA316 — SAM2Long robust memory paths

**SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree**  
https://arxiv.org/abs/2410.16268

Uses fixed-width memory-path tree search to reduce long-video segmentation drift.

**Mirror implication:** Compare against memory tree, including all path-state bytes and cumulative errors; compressed View identity alone is insufficient.

## PA317 — MoPEFT SAM adapter mixture

**MoPEFT: A Mixture-of-PEFTs for the Segment Anything Model**  
https://arxiv.org/abs/2405.00293

Selects among multiple PEFT modules to adapt SAM across target domains.

**Mirror implication:** Compare native mixture of PEFTs, not just LoRA or full SAM, with mask quality and encoder/adaptation compute.

## PA318 — ScaNN anisotropic quantization

**Accelerating Large-Scale Inference with Anisotropic Vector Quantization**  
https://arxiv.org/abs/1908.10396

Optimizes anisotropic quantization for efficient maximum inner-product approximate search.

**Mirror implication:** Compare native ScaNN recall/latency/index bytes and SIMD lookups; an exact isometric View yields no new ranking information.

## PA319 — RaBitQ controlled-error ANN

**RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search**  
https://arxiv.org/abs/2405.12497

Compact random vector quantization with distance estimator bounds and SIMD/bitwise retrieval.

**Mirror implication:** A Mirror query/shard code must improve recall-latency-byte frontier against error-controlled RaBitQ.

## PA320 — Matryoshka nested embeddings

**Matryoshka Representation Learning**  
https://arxiv.org/abs/2205.13147

Nested embedding prefixes support variable representation dimension and inference budgets.

**Mirror implication:** Variable-rank feature View is already prior art; Mirror m must beat nested-prefix retrieval at fixed bit/latency budget.

## PA321 — ColBERTv2 compressed late interaction

**ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**  
https://aclanthology.org/2022.naacl-main.272/

Compresses token-level late-interaction document vectors with residual coding.

**Mirror implication:** Native residual document storage and MaxSim quality form the control; factorized task View cannot assume one vector per document.

## PA322 — PLAID centroid-pruned late interaction

**PLAID: An Efficient Engine for Late Interaction Retrieval**  
https://arxiv.org/abs/2205.09707

Uses centroid interaction/pruning to reduce ColBERT multivector search time.

**Mirror implication:** Mirror logical search Views must beat PLAID's actual index/traversal latency, not merely centroid reconstruction.

## PA323 — QINCo implicit residual codebooks

**Residual Quantization with Implicit Neural Codebooks**  
https://proceedings.mlr.press/v235/huijben24a.html

Generates stage-dependent residual codebooks conditioned on decoded vector.

**Mirror implication:** Implicitly shared codebooks are prior art; Mirror m must improve true vector search and generated-codebook storage/compute.

## PA324 — DiskANN SSD graph retrieval

**DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node**  
https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html

Uses SSD-friendly approximate-neighbor graph access at large scale.

**Mirror implication:** Mirror graph or query Views must beat the native SSD I/O, tail latency and recall tradeoff, not only embedding bytes.

## PA325 — PGM-index dynamic compressed learned index

**The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds**  
https://www.vldb.org/pvldb/vol13/p1162-ferragina.pdf

Piecewise learned index supports dynamic predecessor/range queries with strong correctness and worst-case time/space bounds.

**Mirror implication:** Mirror index m must preserve the original ordering/correctness guarantees under updates and compare with adaptive native PGM.

## PA326 — Chronos time-series modeling

**Chronos: Learning the Language of Time Series**  
https://arxiv.org/abs/2403.07815

Scales and tokenizes numeric time-series for pretrained probabilistic forecasting.

**Mirror implication:** Mirror m must beat zero-shot forecast quality and calibrated native token sampling.

## PA327 — TimesFM foundation forecasting

**A decoder-only foundation model for time-series forecasting**  
https://arxiv.org/abs/2310.10688

Forecast transformer pretrained over varying horizons and frequencies.

**Mirror implication:** Horizon m must improve native scale/horizon conditioning and cheap output heads.

## PA328 — Moirai universal time series

**Unified Training of Universal Time Series Forecasting Transformers**  
https://proceedings.mlr.press/v235/woo24a.html

Universal transformer handles many frequencies/variates/datasets from pretraining.

**Mirror implication:** One shared forecaster exists already; test added m beyond native input conditioning.

## PA329 — PatchTST channel-shared forecasting

**A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**  
https://arxiv.org/abs/2211.14730

Temporal patch transformer shares feature weights across variates.

**Mirror implication:** An m per variate must beat already-shared PatchTST and simple channel embeddings.

## PA330 — TimeMixer multi-scale seasonal decomposition

**TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**  
https://arxiv.org/abs/2405.14616

Mixes trend/seasonal temporal factors across scales.

**Mirror implication:** Compare multi-timescale m with native TimeMixer mixing, not single-scale baseline.

## PA331 — iTransformer variate tokens

**iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**  
https://arxiv.org/abs/2310.06625

Represents full variable histories as tokens for cross-variate attention.

**Mirror implication:** Varying task/variable token identity is established; measure useful additional m.

## PA332 — TRACE efficient time-series adaptation

**TRACE: Time SeRies PArameter EffiCient FinE-tuning**  
https://arxiv.org/abs/2503.16991

Forecast-specific module-selected LoRA and adapted output heads.

**Mirror implication:** Compare structured Mirror codes against TRACE's native LoRA module selection at matched state.

## PA333 — MOMENT multi-task temporal foundation

**MOMENT: A Family of Open Time-series Foundation Models**  
https://proceedings.mlr.press/v235/goswami24a.html

Open pretrained time-series features support forecasting and other tasks.

**Mirror implication:** Task code m must outperform native pretrained temporal heads, not count a task label as capacity.

## PA334 — Lag-Llama probabilistic time series

**Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting**  
https://arxiv.org/abs/2310.08278

Pretrained probabilistic temporal forecaster uses lagged covariates.

**Mirror implication:** Mirror risk/calibration m must beat Lag-Llama CRPS/coverage at equal history and steps.

## PA335 — N-BEATS interpretable basis forecaster

**N-BEATS: Neural basis expansion analysis for interpretable time series forecasting**  
https://arxiv.org/abs/1905.10437

Uses shared learned basis and residual expansions for forecast functions.

**Mirror implication:** Basis coefficient m is direct prior art; require higher marginal value at equal basis bytes.

## PA336 — DLinear inexpensive forecast baseline

**Are Transformers Effective for Time Series Forecasting?**  
https://arxiv.org/abs/2205.13504

Simple decomposition/linear forecast controls can beat larger neural forecasters.

**Mirror implication:** DLinear and seasonal naive are minimum-cost controls for complex Mirror temporal Views.

## PA337 — DLRM categorical embeddings

**Deep Learning Recommendation Model for Personalization and Recommendation Systems**  
https://arxiv.org/abs/1906.00091

Categorical embedding tables dominate production recommendation state and memory traffic.

**Mirror implication:** Any claimed Mirror embedding benefit must count lookup memory bandwidth and full table bytes.

## PA338 — DHE table-free recommendation embeddings

**Learning to Embed Categorical Features without Embedding Tables for Recommendation**  
https://arxiv.org/abs/2010.10784

Deep Hash Embedding generates ID embeddings from deterministic hash features with no table.

**Mirror implication:** Shared decoder plus ID code is known; m must beat DHE and charge per-ID state.

## PA339 — QR compositional embeddings

**Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems**  
https://arxiv.org/abs/1909.02107

Combines small complementary table partitions to form categorical vectors.

**Mirror implication:** Mirror factor addresses must beat native QR code storage and collisions.

## PA340 — TT-Rec table tensorization

**TT-Rec: Tensor Train Compression for Deep Learning Recommendation Models**  
https://arxiv.org/abs/2101.11714

TT-Rec uses tensor-train embedding cores and optimized lookup kernels.

**Mirror implication:** Mirror must beat this very strong TT storage/latency/CTR frontier.

## PA341 — VQ-Rec transferable item codes

**Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders**  
https://arxiv.org/abs/2210.12316

Item content is encoded to discrete quantized codes for domain transfer.

**Mirror implication:** Discrete item Mirror codes require gain beyond VQ-Rec, especially cold start.

## PA342 — HSTU generative recommendation

**Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations**  
https://proceedings.mlr.press/v235/zhai24a.html

Sequential transducer scales event/action recommender modeling.

**Mirror implication:** User-session m must earn value beyond pretrained HSTU and must count session KV/runtime.

## PA343 — MMoE multitask recommender

**Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts**  
https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/

Shares experts across recommendation goals with per-task gates.

**Mirror implication:** A Mirror task-expert code must beat native MMoE gate and task quality/interference.

## PA344 — SatMAE multi-spectral satellite pretraining

**SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery**  
https://arxiv.org/abs/2207.08051

MAE pretraining incorporates band grouping and temporal positions.

**Mirror implication:** Band/time m is not novel alone; beat native spectral/temporal embeddings.

## PA345 — DOFA wavelength-conditioned Earth hypernet

**Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation**  
https://arxiv.org/abs/2403.15356

Wavelength-dependent hypernetwork emits sensor-adaptive filters for one EO foundation model.

**Mirror implication:** This is direct physical-condition to weight prior art. m must beat dynamic filters and generation cost.

## PA346 — CROMA SAR-optical Earth fusion

**CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**  
https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html

Aligns optical and SAR encoders using masked contrastive/fusion learning.

**Mirror implication:** Cross-sensor m must preserve registered alignment and beat native paired fusion.

## PA347 — AnySat multisensor Earth foundation

**AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**  
https://arxiv.org/abs/2412.14123

Scale-adaptive Earth transformer spans sensor types/resolutions.

**Mirror implication:** Mirror sensor-scale m must beat already generalized AnySat and OOD sensor transfer.

## PA348 — Prithvi EO multi-temporal model

**Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications**  
https://arxiv.org/abs/2412.02732

Earth foundation model learns shared spectral and temporal representations.

**Mirror implication:** Region/sensor/season Mirror effects must exceed native temporal transfer.

## PA349 — TerraMind any-to-any EO generative model

**TerraMind: Large-Scale Generative Multimodality for Earth Observation**  
https://arxiv.org/abs/2504.11171

Generative multi-sensor EO foundation model uses pixel and token representations.

**Mirror implication:** Mirror modal m must beat native modality-conditioned generation, respecting missing sensors.

## PA350 — AlphaEarth embedding-field maps

**AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data**  
https://arxiv.org/abs/2507.22291

Produces general global space-time sensor-fused embedding fields.

**Mirror implication:** Mirror task/region codes must improve index storage and spatial holdout downstream quality.

## PA351 — Compress then Serve: multi-LoRA shared basis

**Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead**  
https://proceedings.mlr.press/v267/gabrielsson25a.html

ICML 2025 jointly compresses many independently trained LoRA adapters into shared bases and per-adapter scaling matrices; clusters less-related adapters and evaluates large multi-adapter serving workloads.

**Mirror implication:** This is a DIRECT precedent for one physical low-rank basis + small logical adapter codes. Mirror m needs a marginal quality/byte/throughput gain beyond this method, including clustering and optimized serving.

## PA352 — Compress then Merge in common LoRA subspace

**Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter**  
https://proceedings.mlr.press/v306/he26h.html

ICML 2026 projects multiple adapters into common left/right low-rank subspaces with small core coordinates, then merges in core space at guaranteed output rank.

**Mirror implication:** A common U C_t V^T code is already known; distinguish per-task retrieval from producing one merged model and compare the strongest relevant CtM operation.

## PA353 — EigenLoRAx reused adapter principal subspace

**EigenLoRAx: Recycling Adapters to Find Principal Subspaces for Resource-Efficient Adaptation and Inference**  
https://arxiv.org/abs/2502.04700

Fits principal subspaces of existing pretrained adapters and learns small coefficients for new tasks, augmenting basis if coverage is insufficient.

**Mirror implication:** Direct baseline for naturally trained LoRA-bank coordinate reuse and unseen task induction; mirror must beat principal coefficients and orthogonal/private expansion.

## PA354 — VB-LoRA shared vector bank

**VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks**  
https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html

Shares a bank of vectors to compactly parameterize low-rank adapter weights and specialized adaptation.

**Mirror implication:** Strong shared-vector parameter-efficient baseline; charge vector bank cost and task-selection/code metadata.

## PA355 — MetaTT global TT adapters

**MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning**  
https://arxiv.org/abs/2506.09105

Uses a global tensor-train adapter factorized across layer, matrix type and optionally head/task dimensions; includes rank-adaptive optimization.

**Mirror implication:** Factorized m across task×layer×matrix already has a global TT analogue. Mirror must beat TT core compression, optimizer storage and end-to-end NLL/runtime.

## PA356 — Stable pretrained spectral basis across tasks

**Pretraining Induces a Reusable Spectral Basis for Downstream Task Adaptation**  
https://arxiv.org/abs/2605.07302

Reports stability of leading singular vectors of pretrained matrices across fine-tuned vision/language models; tunes spectral coefficients with small parameter budgets.

**Mirror implication:** Pretrained W stability is not proof that task DELTAS share a low-rank orbit. Measure both independently and include the cost of computing/storing U,V.

## PA357 — LoRA B-space sharing and interference

**Crowded in B-Space: Calibrating Shared Directions for LoRA Merging**  
https://arxiv.org/abs/2604.16826

Finds output-side LoRA factor B directions may be shared across tasks more than A, and proposes Pico output-direction calibration for merging.

**Mirror implication:** Test shared-left, shared-right and shared-both candidate bases. Raw B factor comparisons are gauge-dependent; use canonicalized projectors and holdout task quality.

## PA358 — GLoRA gauge-aware federated subspaces

**Beyond Factor Aggregation: Gauge-Aware Low-Rank Server Representations for Federated LoRA**  
https://arxiv.org/abs/2605.06733

Shows raw low-rank factor aggregation depends on basis gauge and proposes projector-aligned consensus subspace plus low-rank rank-compatible clients.

**Mirror implication:** Gauge-invariant projector/control is mandatory. Invariance under B->BG, A->G^-1 A is a basic correctness requirement for an LoRA-orbit audit.

## PA359 — Learning on LoRAs and GL symmetry

**Learning on LoRAs: GL-Equivariant Processing of Low-Rank Weight Spaces for Large Finetuned Models**  
https://arxiv.org/abs/2410.04207

Processes large LoRA collections with low-rank decomposition symmetry-aware invariant/canonical features, including natural diffusion and language LoRAs.

**Mirror implication:** Provides natural adapter-bank testing and canonicalization controls; do not learn spurious m based on arbitrary internal factor coordinates.

## PA360 — W2T canonical LoRA weight tokenization

**W2T: LoRA Weights Already Know What They Can Do**  
https://arxiv.org/abs/2603.15990

Canonicalizes LoRA parameterizations via QR/SVD before learning representations of adapter function/performance from weights.

**Mirror implication:** QR/SVD canonicalization is a direct prerequisite for robust adapter-distance and task family discovery; raw factor retrieval can be spurious.

## PA361 — Share evolving LoRA subspace

**Shared LoRA Subspaces for almost Strict Continual Learning**  
https://arxiv.org/abs/2602.06043

Builds and updates a common low-rank subspace across continual downstream tasks rather than maintaining many separate adapters.

**Mirror implication:** Shared evolving subspaces and retention are native controls; measure online basis drift, prior-task quality and writable/optimizer state as m is added.

## PA362 — LoDA shared and private adaptation

**Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning**  
https://arxiv.org/abs/2603.00191

Separates task-general and task-specific LoRA subspaces by projection energy and uses a recalibration for shared directions.

**Mirror implication:** Direct shared/private frontier control; demonstrate where cheap m replaces task-specific directions without catastrophic interference.

## PA363 — Zhyper task conditioned LoRA generator

**Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning**  
https://arxiv.org/abs/2510.19733

Generates conditioned LoRA weights from textual context using a factorized hypernetwork, with explicit parameter-efficiency objectives.

**Mirror implication:** Dynamic m conditioned on task text is not conceptually unique; include native Zhyper generator and its compute/state costs.

## PA364 — LRAgent low-rank multi-LoRA KV sharing

**LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents**  
https://proceedings.mlr.press/v306/jeon26b.html

ICML 2026 decomposes multi-LoRA KV state into shared base and compact adapter-dependent terms, using low-rank cache sharing and Flash-LoRA-Attention.

**Mirror implication:** Extremely direct prior for shared physical cache + small adapter state. Mirror must prove additional benefit beyond LRAgent's fused materialization-free low-rank attention.

## PA365 — PReCache base cache and low-rank correction

**PReCache: Efficient KV Cache Sharing for Multi-LoRA Agents via Low-Rank Precomputation and Neutral Reconstruction**  
https://arxiv.org/abs/2609.34054

Proposes PreLRShared and ReBaseShared for training-free shared-base KV with low-rank adapter caches and base-state reconstruction.

**Mirror implication:** Direct current baseline for multi-agent cache switching and prefix reuse. Check temporal provenance, target NLL, LR-state bytes, precompute and reconstruction overhead.

## PA366 — LoRDBA low-bit adapters

**Signs Beat Floats: Low-Rank Double-Binary Adaptation for On-Device Fine-Tuning**  
https://arxiv.org/abs/2605.24058

Encodes LoRA factors using binary sign carriers plus small magnitude/channel scale vectors to reduce unmerged adapter footprint.

**Mirror implication:** Lower-byte sign plus scales is a cheap direct control; Mirror code must beat bit-packed binary factors, not float16 LoRA only.

## PA367 — LoRA-RITE invariant optimization

**LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization**  
https://arxiv.org/abs/2410.20625

Introduces transformation/gauge-invariant low-rank optimizer preconditioning for task adaptation.

**Mirror implication:** Even the optimizer trajectory can change under gauge reparameterization. Code discovery, alignment and online update must be tested invariantly.

## PA368 — HyperLoader task layer hypernetwork

**HyperLoader: Integrating Hypernetwork-Based LoRA and Adapter Layers into Multi-Task Transformers for Sequence Labelling**  
https://arxiv.org/abs/2407.01411

Conditions a common hypernetwork on task, transformer layer and internal adapter slot to generate task-specific parameter-efficient operators.

**Mirror implication:** Task×layer×matrix physical operator generation is a native hypernetwork prior; compare factorized m against HyperLoader generator cost and quality.

## PA369 — TC-LoRA clustered CP task delta bank

**Tensorized Clustered LoRA Merging for Multi-Task Interference**  
https://arxiv.org/abs/2508.03999

Clusters training examples and jointly CP-decomposes LoRA task banks to disentangle shared and private factors and mitigate merging interference.

**Mirror implication:** Clustered tensor decomposition is a strong existing physical shared/private model of learned adapter deltas; measure held-out tasks and byte/rank Pareto.

## PA370 — ThanoRA task heterogeneity subspace

**ThanoRA: Task Heterogeneity-Aware Multi-Task Low-Rank Adaptation**  
https://arxiv.org/abs/2505.18640

Allocates task-specific low-rank subspace dimensions by task heterogeneity while preserving diversity and limiting cross-task interference.

**Mirror implication:** A flat small m for all tasks may fail; benchmark heterogeneity-aware variable rank and private residual allocation against ThanoRA.

## PA371 — TT-LoRA MoE sparse adapter specialists

**TT-LoRA MoE: Unifying Parameter-Efficient Fine-Tuning and Sparse Mixture-of-Experts**  
https://arxiv.org/abs/2504.21190

Uses independently trained tensorized LoRA specialists with sparse routing and frozen per-expert updates to reduce shared multi-task complexity.

**Mirror implication:** Native sparse TT-LoRA expert routing is a strong non-Mirror baseline for logical expert multiplication. Count router and all frozen expert state.

## PA372 — BOLT reusable task spectral basis

**Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation**  
https://openaccess.thecvf.com/content/CVPR2026/html/Park_Basis-Oriented_Low-rank_Transfer_for_Few-Shot_and_Test-Time_Adaptation_CVPR_2026_paper.html

Collects principal spectral directions from pre-adapted tasks, builds a shared orthogonal basis, then optimizes only per-task diagonal coefficients for previously unseen tasks. This is direct published prior for a shared basis with a tiny functional task code.

**Mirror implication:** Diagonal source-task basis codes are already an efficient native adapter. Mirror m must outperform BOLT at matched downstream accuracy and actual bytes including source-basis construction/adaptation.

## PA373 — CG-LoRA function-space direction selection

**Curvature-Guided LoRA: Matching Full Fine-Tuning in Function Space**  
https://arxiv.org/abs/2603.29824

Selects low-rank fine-tuning directions from a curvature-aware objective that approximates downstream prediction changes rather than only matching parameter deltas.

**Mirror implication:** Prior for evaluating adapter geometry through function-space quality and curvature. Mirror delta reconstruction error alone is not a sufficient task-quality metric; match downstream output/latency vs CG-LoRA.

## PA374 — Fora activation-subspace capability protection

**Fora: From Weight-Space to Function-Space Protection in Capability-Preserving Fine-Tuning**  
https://arxiv.org/abs/2606.31092

Builds input-activation-derived projectors for protecting preexisting capabilities while admitting controlled residual adaptation; contrasts activation geometry with weight SVD geometry.

**Mirror implication:** A Mirror chart that looks compact in weights can harm functional directions. Compare function-space capability retention, activation projectors, and per-task private exceptions.

## PA375 — Task Vector Bases compressed arithmetic

**Task Vector Bases: A Unified and Scalable Framework for Compressed Task Arithmetic**  
https://arxiv.org/abs/2502.01015

Expresses many learned task vectors as structured combinations of fewer task-basis vectors and studies preserving task arithmetic and composition.

**Mirror implication:** Shared task basis with small coefficients and compositional addition is established. Mirror must beat task-vector basis storage/accuracy and hold out task combinations.

## PA376 — SVD and CUR geometry in LoRA merging

**On the Representation Geometry of LoRA Model Merging**  
https://aclanthology.org/2026.findings-acl.261/

Studies complementary geometry of global SVD shared components and localized CUR task-specific components in learned LoRA merges.

**Mirror implication:** Require SVD-plus-CUR native shared/private control and task-utility metrics; an optimal delta Frobenius approximation may discard localized functional information.

## PA377 — StructLoRA task-aware information bottleneck

**Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adaptation**  
https://aclanthology.org/2026.acl-long.97/

Uses task-aware information bottleneck filtering and a training-only interlayer coordinator for low-rank adaptation; reports no extra inference module cost.

**Mirror implication:** Mirror code rank/placement selection must beat zero-extra-inference information filtering and coordinated task-aware low-rank baselines.

## PA378 — Training Jacobian task subspace geometry

**Understanding Gradient Descent through the Training Jacobian**  
https://arxiv.org/abs/2412.07003

Investigates Jacobian of trained weights with respect to initialization and shows data-dependent low-dimensional structure, including directions with limited in-distribution but meaningful OOD output effect.

**Mirror implication:** A weight-space orbit can hide large OOD functional changes. Test input-distribution and OOD Jacobian probes before compressing meaningful directions into m.

## PA379 — NTK regime LoRA optimization theory

**LoRA Training in the NTK Regime has No Spurious Local Minima**  
https://proceedings.mlr.press/v235/jang24d.html

Studies optimization geometry of rank-constrained fine-tuning in an NTK regime and explains conditions under which low-rank optimization avoids spurious local minima.

**Mirror implication:** Distinguish fixed-update training failure from representation capacity. Do not claim every low-rank optimization miss is a proof the Mirror chart lacks capacity.

## PA380 — FuLA functional latent model stitching

**Model Stitching by Functional Latent Alignment**  
https://arxiv.org/abs/2505.20142

Defines a functional latent alignment criterion for model stitching beyond simple affine output agreement, with adversarial/shortcut/cross-layer diagnostic tests.

**Mirror implication:** Mirror stitching/transfer must include information-sensitive and counterfactual probes, not just a high aggregate output agreement.

## PA381 — NTK linearization of LLM fine-tuning

**Linearization Explains Fine-Tuning in Large Language Models**  
https://papers.nips.cc/paper_files/paper/2025/file/becc00fe2e0ade58213cff16a166fa25-Paper-Conference.pdf

Connects regularized fine-tuning dynamics to linearized neural tangent kernel regression, including spectral effects of layer selection and LoRA updates.

**Mirror implication:** For Mirror m learned near pretrained theta, estimate functional Jacobian/NTK spectrum and local linearization error as an additional test beyond weight-delta rank.

## PA382 — RotatE relation rotations

**RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space**  
https://arxiv.org/abs/1902.10197

Models each knowledge-graph relation as a complex-plane rotation of shared entity embeddings, supporting inversion and composition patterns.

**Mirror implication:** Relation rotation as tiny m is direct prior art, not Mirror novelty. Test extra multi-domain/time sharing beyond RotatE and compare link-ranking plus actual relation bytes.

## PA383 — ComplEx complex relation embeddings

**Complex Embeddings for Simple Link Prediction**  
https://arxiv.org/abs/1606.06357

Models asymmetric multi-relational scores using complex-valued entity/relation factors.

**Mirror implication:** Mirror relational m must beat the cheap ComplEx relation diagonals and use filtered ranking, not just triplet reconstruction.

## PA384 — TuckER tensor relation cores

**TuckER: Tensor Factorization for Knowledge Graph Completion**  
https://arxiv.org/abs/1901.09590

Factorizes entity-relation-entity score tensor with a learned shared Tucker core.

**Mirror implication:** A shared operator core plus relation code is long-established. Compare dense/factorized Mirror code vs native TuckER core+relation coefficients.

## PA385 — QuatE quaternion relation rotations

**Quaternion Knowledge Graph Embeddings**  
https://arxiv.org/abs/1904.10281

Uses hypercomplex quaternion-valued entity/relation embeddings with relation-dependent quaternion rotations.

**Mirror implication:** Mirror noncommutative relation composition must outperform native QuatE, count quaternion state and respect inverse/composition semantics.

## PA386 — PairRE paired relation scaling

**PairRE: Knowledge Graph Embeddings via Paired Relation Vectors**  
https://arxiv.org/abs/2011.03798

Uses head/tail-specific paired relation vectors to support many-to-many link patterns.

**Mirror implication:** Separate left and right relation modulation already exists. An m-based two-sided View must beat PairRE with matched parameter sizes.

## PA387 — CompGCN multi-relational composition

**Composition-based Multi-Relational Graph Convolutional Networks**  
https://arxiv.org/abs/1911.03082

Composes node/entity and edge-relation embeddings inside one multi-relational message-passing graph network.

**Mirror implication:** Mirror relation Views must beat native CompGCN relation composition at comparable active edge processing and memory.

## PA388 — TNTComplEx time-relation factorization

**Tensor Decompositions for Temporal Knowledge Base Completion**  
https://arxiv.org/abs/2004.04926

Uses tensor decompositions and temporal factors for relation facts that evolve over time.

**Mirror implication:** Time and relation factor m must improve over native temporal tensor factorization, controlling future-edge leakage.

## PA389 — 5starE projective relation functions

**5* Knowledge Graph Embeddings with Projective Transformations**  
https://arxiv.org/abs/2006.04986

Models KG relation operators with projective transformations including rotation, scaling, reflection and inversion.

**Mirror implication:** A projective Mirror transformation family must beat this strong native prior, not claim discovery of multi-transform relation operator.

## PA390 — KrausKGE relation channels

**Relations Are Channels: Knowledge Graph Embedding via Kraus Decompositions**  
https://arxiv.org/abs/2605.10317

2026 work proposes relation operators based on Kraus decompositions with relation-dependent channel complexity and fan-out handling.

**Mirror implication:** Compare Mirror multi-operator banks against native Kraus rank and completely positive constraints. Relational channel families already support more than simple rotations.

## PA391 — ParamISP camera-metadata conditioning

**ParamISP: Learned Forward and Inverse ISPs using Camera Parameters**  
https://arxiv.org/abs/2312.13313

Controls learned RAW/sRGB forward and inverse camera pipelines using EXIF exposure/ISO conditions.

**Mirror implication:** A camera control m derived from EXIF is direct prior. Mirror needs additional accuracy or lower conditioner/filter storage than ParamISP.

## PA392 — MetaISP multi-device color rendition

**MetaISP -- Exploiting Global Scene Structure for Accurate Multi-Device Color Rendition**  
https://arxiv.org/abs/2401.03220

Single model conditions scene-aware RAW-to-RGB color rendition on target device appearance.

**Mirror implication:** Per-camera style codes already exist. Test Mirror low-bit per-device Views beyond native device conditioning and perceptual color error.

## PA393 — Uni-ISP cross-camera sharing

**Uni-ISP: Toward Unifying the Learning of ISPs from Multiple Mobile Cameras**  
https://arxiv.org/abs/2406.01003

Uses device-aware embeddings in one joint forward/inverse ISP trained across real multi-camera RAW/sRGB pairs.

**Mirror implication:** This is close physical-one-ISP/many-logical-camera prior art. Compare per-camera incremental Mirror m to native learned device embeddings with held-out devices.

## PA394 — PQDynamicISP condition-controlled operators

**PQDynamicISP: Dynamically Controlled Image Signal Processor for Any Image Sensors Pursuing Perceptual Quality**  
https://arxiv.org/abs/2403.10091

Dynamically controls lightweight ISP operations by environment and local conditions to reuse one ISP across sensors.

**Mirror implication:** A small environment/region functional parameter already exists. Mirror m must beat dynamic local ISP control at quality, speed and control bytes.

## PA395 — Modular ISP configurable stage functions

**Modular Neural Image Signal Processing**  
https://arxiv.org/abs/2512.08564

Provides independently controllable neural ISP stages and style/post-edit options within one pipeline.

**Mirror implication:** Mirror stage composition must beat native stage knobs and count total pipeline graph, decoder state and runtime.

## PA396 — Camera-aware real denoising

**Towards Controllable Real Image Denoising with Camera Parameters**  
https://arxiv.org/abs/2507.01587

Conditions image denoising on camera/acquisition parameters for adaptable raw/sensor noise.

**Mirror implication:** Noise-level and EXIF conditioning already have a direct baseline. Compare structured Mirror m against native camera/noise control on held-out sensors.

## PA397 — OmniLens latent lens PSF

**OmniLens++: Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation**  
https://arxiv.org/abs/2511.17126

Learns latent representations of lens point spread functions from large pretraining to correct unknown optical aberrations.

**Mirror implication:** PSF latent m is strong prior art. Mirror must reduce latent/filter or improve held-out physical lens calibration beyond OmniLens++.

## PA398 — Neural lens modeling

**Neural Lens Modeling**  
https://openaccess.thecvf.com/content/CVPR2023/papers/Xian_Neural_Lens_Modeling_CVPR_2023_paper.pdf

Learns differentiable lens image formation to model spatially varying optics.

**Mirror implication:** A position-dependent lens neural field is established; Mirror m must improve per-lens transfer at fixed optical forward accuracy.

## PA399 — Physics-informed low-rank aberration

**A Physics-informed Low-rank Deep Neural Network for Blind and Universal Lens Aberration Correction**  
https://openaccess.thecvf.com/content/CVPR2024/papers/Gong_A_Physics-informed_Low-rank_Deep_Neural_Network_for_Blind_and_Universal_CVPR_2024_paper.pdf

Uses physics-informed low-rank parameterization to restore lens-degraded images across aberrations.

**Mirror implication:** A low-rank optics corrector is a strong direct control. Mirror View cannot call generic rank compression a novel optical capability.

## PA400 — UP-OSI universal control plus system ID

**Preparing for the Unknown: Learning a Universal Policy with Online System Identification**  
https://arxiv.org/abs/1702.02453

Trains one policy over a range of physics parameters with an online system identification component.

**Mirror implication:** One physical policy plus low-dimensional dynamics context m is prior art. Mirror must improve m-state/quality or robustness beyond UP-OSI.

## PA401 — RMA latent extrinsics robot control

**RMA: Rapid Motor Adaptation for Legged Robots**  
https://arxiv.org/abs/2107.04034

Uses a common legged-control policy conditioned on estimated environmental extrinsics from recent sensor history.

**Mirror implication:** Dynamic latent physics m is directly established. Preserve RMA adaptation latency/history and compare no-privileged-observation fairness.

## PA402 — CoRMA contact-aware adaptation

**CoRMA: Contrastive RMA for Contact-Rich Meta-Adaptation**  
https://arxiv.org/abs/2605.22082

2026 work modifies RMA-style context-based adaptation for contact-rich robotic tasks using a contrastive objective.

**Mirror implication:** Mirror dynamic friction/contact code must beat native contrastive RMA and handle ambiguous hidden physics.

## PA403 — A-NC implicit adaptive neural control

**A-NC: Adaptive Neural Control with implicit online inference of privileged parameters**  
https://proceedings.mlr.press/v283/paluch25a.html

Adapts recurrent neural control implicitly online when changes in environment/robot parameters are not directly measured.

**Mirror implication:** Explicit Mirror physics codes must outperform implicit recurrent inference at equal history and no oracle physical parameters.

## PA404 — Graph-operator morphology world models

**Graph-Operator World Models for Morphology-Parameter Generalization in Continuous Control**  
https://arxiv.org/abs/2608.20936

2026 work explores morphology and physical parameter generalization via graph-structured dynamics operators.

**Mirror implication:** Mirror morphology m must beat native structure-conditioned operators and preserve physical rollouts on unseen bodies.

## PA405 — Morphology-conditioned world model

**Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion**  
https://arxiv.org/abs/2604.08780

Conditions shared world dynamics on quadruped morphology for transfer across embodiments.

**Mirror implication:** A morphology code is already prior art. Mirror should compress useful embodiment-specific dynamics beyond the native condition.

## PA406 — CTS teacher-student legged control

**CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion**  
https://arxiv.org/abs/2405.10830

Uses teacher-student legged locomotion training to bridge privileged training and deployable observation conditions.

**Mirror implication:** Mirror m adaptation must not use teacher-only physics at test time. Compare valid deployable student state and rollouts.

## PA407 — Neural Acoustic Fields room RIR

**Learning Neural Acoustic Fields**  
https://arxiv.org/abs/2204.00628

Models source-receiver-position-conditioned room acoustic responses as a continuous neural field.

**Mirror implication:** Room/source/receiver coordinates already condition one neural field. Mirror must compress multi-room physical field parameters beyond native NAF.

## PA408 — Real Acoustic Fields RIR benchmark

**Real Acoustic Fields: An Audio-Visual Room Acoustics Dataset and Benchmark**  
https://arxiv.org/abs/2403.18821

Introduces measured audio-visual room acoustics benchmark for evaluating continuous room fields.

**Mirror implication:** Use real measured impulse responses and room-held-out tests; synthetic aligned room kernels alone are not physical deployment evidence.

## PA409 — Retrieval-augmented room acoustic adaptation

**Data Augmentation Using Neural Acoustic Fields With Retrieval-Augmented Pre-training**  
https://arxiv.org/abs/2504.14409

Pretrains room-geometry conditioned acoustic field and adapts to unseen rooms using geometry retrieval/measurement enrollment.

**Mirror implication:** Native few-room-shot field adaptation already exists. Mirror room m must improve measurements, storage or RIR fidelity beyond retrieved priors.

## PA410 — Topology-aware RIR neural modeling

**TA-RIR: Topology-Aware Neural Modeling of Acoustic Propagation for Room Impulse Response Synthesis**  
https://www.isca-archive.org/interspeech_2025/zhao25i_interspeech.html

Learns room topology-conditioned acoustic decoder using source-receiver geometry and reverberant observations.

**Mirror implication:** Geometry m must beat topology-aware native embeddings and track physical RIR measures such as RT60 and DRR.

## PA411 — Neural acoustic multipole splatting

**Neural acoustic multipole splatting for room impulse response synthesis**  
https://arxiv.org/abs/2509.17410

Builds a physical acoustic multipole field with learned directivity and pruneable sources for efficient RIR synthesis.

**Mirror implication:** Share/mirror multipole coefficients only if RIR fidelity and pruning/propagation runtime beat native NAMS. Count multipole positions and source-specific state.

## PA412 — Direction-aware acoustic fields

**Direction-Aware Neural Acoustic Fields for Few-Shot Interpolation of Ambisonic Impulse Responses**  
https://arxiv.org/abs/2505.13617

Fits directional ambisonic RIRs with neural fields and explores few-shot adaptation including low-rank updates.

**Mirror implication:** Per-direction code and few-shot LoRA are already prior. Mirror must preserve phase/directivity and beat native Ambisonic NAF/LoRA.

## PA413 — Few-shot acoustic flow synthesis

**Few-shot Acoustic Synthesis with Multimodal Flow Matching**  
https://openaccess.thecvf.com/content/CVPR2026/html/Brunetto_Few-shot_Acoustic_Synthesis_with_Multimodal_Flow_Matching_CVPR_2026_paper.html

2026 work learns scene-consistent room acoustic synthesis from limited audio-visual room examples using multimodal generative flow.

**Mirror implication:** Mirror environment code must beat few-shot flow prior at equal NFE and measured RIR fidelity; source-room identity is not a novel free functional parameter.

## PA414 — Complete Characterization of Gauge Symmetries in Transformer Architectures

**Complete Characterization of Gauge Symmetries in Transformer Architectures**  
Hong Wang, Kelly Wang. PMLR 282 (2026).  
https://proceedings.mlr.press/v282/wang26a.html

**Established prior:** Determines canonical MHA Q/K and V/output head-wise gauge groups, including RoPE commutant restrictions and head permutations. Exact gauge actions preserve function. The proposed Mirror insertion must not claim these invariances as new logical functions.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA415 — Gauge Fiber Bundle Geometry of Transformers

**Gauge Fiber Bundle Geometry of Transformers**  
Hong Wang, Kelly Wang. PMLR 282 (2026).  
https://proceedings.mlr.press/v282/wang26b.html

**Established prior:** Studies quotient geometry, Ehresmann connection, gauge gradient split, curvature and holonomy. Prior art for horizontal and path-dependent diagnostics; Mirror-specific claim must be incremental m utility.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA416 — Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants

**Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants**  
Hong Wang, Kelly Wang. PMLR 282 (2026).  
https://proceedings.mlr.press/v282/wang26c.html

**Established prior:** Relates gauge-aware geometry and bispectral invariant signatures for equivalence diagnosis. Native invariant comparison is a strong control, not a new Mirror operator.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA417 — Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction

**Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction**  
Vishesh Tripathi, Abhay Kumar, Ramsha Khan. arXiv:2609.13285 (2026 preprint).  
https://arxiv.org/abs/2609.13285

**Established prior:** Reconstructs head content keys from grouped values with an absorbable query-side map and separately cached position key. Mirror can only claim extra gain from compact per-head/readout m or safe role sharing; paper itself does not establish end-to-end kernel throughput gains.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA418 — Permutation Equivariant Neural Functionals

**Permutation Equivariant Neural Functionals**  
Allan Zhou et al.. NeurIPS 2023; arXiv:2302.14040.  
https://arxiv.org/abs/2302.14040

**Established prior:** Neural functionals processing weights/gradients equivariantly under neuron permutations. This symmetry-aware encoding is established; compare functional m output to native weight-space models.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA419 — Universal Neural Functionals

**Universal Neural Functionals**  
Allan Zhou, Chelsea Finn, James Harrison. NeurIPS 2024; DOI 10.52202/079017-3326.  
https://proceedings.neurips.cc/paper_files/paper/2024/hash/bd20595c8e5802ba40ed418f4ec116f0-Abstract-Conference.html

**Established prior:** Builds permutation-equivariant neural functionals for general weight spaces, including learned optimizers. m code generation must beat ordinary UNF + cheap native adapter at counted generator bytes.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA420 — The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order

**The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order**  
John Sweeney. ICML 2026; PMLR 306.  
https://proceedings.mlr.press/v306/sweeney26a.html

**Established prior:** Relates order-dependent sequential learning to computable gradient Lie brackets. Bracket-based sequencing is prior art; proposed Mirror insertion tests factorized code composition rather than claiming the Lie bracket.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA421 — First-Order Predictable but Pairwise Fragile: Local Task Adaptation in Trained Transformers

**First-Order Predictable but Pairwise Fragile: Local Task Adaptation in Trained Transformers**  
Irina Piontkovskaia, Sergey Nikolenko. arXiv:2607.16821 (2026 preprint).  
https://arxiv.org/abs/2607.16821

**Established prior:** Reports order-sensitive pairs and fragile local task arithmetic in trained Transformers. Requires heldout pair, step-size and gauge audits; an optimistic global composition radius is not assumed.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA422 — Platonic Task Arithmetic

**Platonic Task Arithmetic**  
Junghwan Park, Woojin Cho. arXiv:2610.00929 (2026 preprint).  
https://arxiv.org/abs/2610.00929

**Established prior:** Proposes architecture-agnostic task descriptors and cross-model arithmetic/transfer via functional probes. Multi-model descriptor sharing is native; the Mirror delta is extra code-bank compression or cheaper factorized task x model state.

**Worker-ready note:** The isolated plan README states the implementation and direct control; verify the above original source before any real-paper reproduction claim.

## PA423 — Low-Rank Key Value Attention (LRKV)

**Low-Rank Key Value Attention** — James O'Neill, Robert Clancy, Mariia Matskevichus, Fergal Reid. arXiv:2601.11471 (2026 preprint).  
https://arxiv.org/abs/2601.11471

Native LRKV uses a shared full-rank key/value projection with low-rank head-specific residuals, preserving head-operator diversity. It is a direct control for any Mirror-multiplexed head update or KV cache; compare storage and training compute at fixed model scale rather than claiming the native architecture's savings.

## PA424 — Positional encoding and functional equivalence in attention

**Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity** — Viet-Hoang Tran, Khanh Vinh Bui, Van-Hoan Trinh, Tan Lai Ngoc, Tan Minh Nguyen. ICML 2026, PMLR 306.  
https://proceedings.mlr.press/v306/tran26d.html

Analyzes equivalences under sinusoidal and rotary positional encodings and explores symmetry-aware alignment and mode connectivity. Mirror RoPE audit must not assume arbitrary GL key transforms survive RoPE; compare with exact commutant families and invalid-transformation counterexamples.

## PA425 — xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction

**xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction** — Chi-Chih Chang et al.. ICML 2026, PMLR 306:12758-12778.  
https://proceedings.mlr.press/v306/chang26d.html

Code: https://github.com/abdelfattah-lab/xKV

**Native prior/control:** Cross-layer shared token basis and selective reconstruction are native xKV. Mirror must beat xKV-SR on quality, actual physical cache bytes and real decode cost.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA426 — KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity

**KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity** — Damien Lesens, Beheshteh T. Rakhshan, Guillaume Rabusseau. AISTATS 2026, PMLR 300:3556-3564.  
https://proceedings.mlr.press/v300/lesens26a.html

**Native prior/control:** Native KQ-SVD uses query-key attention-score reconstruction rather than plain key weight error. Mirror must beat strong source-calibrated per-role projector banks.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA427 — Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs

**Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs** — Hao Ban, Kaiyi Ji. ACL Findings 2026, DOI 10.18653/v1/2026.findings-acl.625.  
https://aclanthology.org/2026.findings-acl.625/

Code: https://github.com/OptMN-Lab/ALoRA

**Native prior/control:** ALoRA shares output B while fitting per-task A; identical A initialization can create spurious apparent cross-task similarity. Strong mandatory control for existing MA-1097.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA428 — Scalable Multi-Task Low-Rank Model Adaptation

**Scalable Multi-Task Low-Rank Model Adaptation** — Zichen Tian, Antoine Ledent, Qianru Sun. ICLR 2026.  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/791de7c35bb49cfca56744e67f90eef4-Abstract-Conference.html

Code: https://github.com/doem97/ICLR26_mtLoRA

**Native prior/control:** Native mtLoRA uses spectral-aware regularization, block-level adaptation and fine-grained dimension-wise routing; code compresses only this existing native state.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA429 — One Adapter, Many Tasks: Task-Conditioned Feature Transformations for Continual Learning

**One Adapter, Many Tasks: Task-Conditioned Feature Transformations for Continual Learning** — Yunxiang Fu, Meng Lou, Yizhou Yu. arXiv:2608.31096, Aug 2026 preprint.  
https://arxiv.org/abs/2608.31096

**Native prior/control:** FACET already uses a single shared adapter and a task-conditioned feature consistency scheme. Mirror must beat original conditioning and FiLM.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA430 — MoLoRA: Composable Specialization via Per-Token Adapter Routing

**MoLoRA: Composable Specialization via Per-Token Adapter Routing** — Shrey Shah, Justin Wagle. arXiv:2603.15965, Mar 2026 preprint.  
https://arxiv.org/abs/2603.15965

**Native prior/control:** Native MoLoRA routes per token across independent LoRA adapters; Mirror must reduce the bank without changing routing and without false cache compatibility.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA431 — NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation

**NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adaptation** — Yuxin Yang et al.. arXiv:2603.12378, Mar 2026 preprint.  
https://arxiv.org/abs/2603.12378

**Native prior/control:** Context modulation of frozen sparse random projections and contrastive orthogonality are native NeuroLoRA. Extra Mirror m must outperform same-byte native gate.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA432 — Quantization Dominates Rank Reduction for KV-Cache Compression

**Quantization Dominates Rank Reduction for KV-Cache Compression** — Samuel Salfati. arXiv:2604.11501, Apr 2026 preprint.  
https://arxiv.org/abs/2604.11501

**Native prior/control:** Full-dimensional KV INT4 can outperform low-rank cache compression at equal physical storage on tested models. Require a strong bitpacked control for MA-578.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.


## PA433 — Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving Tradeoffs

**Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving Tradeoffs** — Dushyant Rajput. arXiv:2609.17109, Sep 2026 preprint.  
https://arxiv.org/abs/2609.17109

**Native prior/control:** Reused numerical standard-LoRA cache prefixes can still be physically copied; quality need not be equivalent to the specialist's native prefill. Add to MA-1112 controls.

**Mirror delta:** Demonstrate marginal m utility over native at real serialized bytes, active compute, and held-out quality. Paper results are not Mirror claims.

## PA434 — Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

**Ranggi Hwang et al.** ISCA 2024. DOI: 10.1109/ISCA59077.2024.00078.
https://arxiv.org/abs/2308.12066
https://doi.org/10.1109/ISCA59077.2024.00078

**Established prior:** A pre-gating architecture predicts expert requirements early enough to overlap CPU→GPU expert migration with preceding compute. Early expert transfer and transfer/compute pipelining are **not** a new Mirror invention. Mirror-specific delta must beat this predictor at matched VRAM, latency, and useful quality.

## PA435 — SpecPrefetch: Parameter-Efficient Expert Prefetching for Sparse MoE Foundation Models

**Jinwei Kong et al.** arXiv:2607.24787 (2026 preprint).
https://arxiv.org/abs/2607.24787

**Established prior:** A lightweight shared predictive adapter prefetches possible future experts while the original router still decides executed experts. Prediction accuracy, bytes sent unnecessarily, and real overlap must be used as direct controls for a Mirror grouped scheduler.

## PA436 — SPICE: Speculative Prefetching with Low-Rank Expert Surrogates and Heterogeneous Orchestration for MoE Inference Acceleration

**Yongxiang Lyu, Ning Li, Bonian Jia.** arXiv:2608.21240 (2026 preprint).
https://arxiv.org/abs/2608.21240

**Established prior:** Speculative expert prediction, lightweight low-rank surrogate fallback, and CPU/GPU asynchronous orchestration for unloaded experts. Mirror must surpass an equivalently costed low-rank surrogate, and do not count hidden exact CPU residual compute as free.

## PA437 — MIMMO: Multi-Input Massive Multi-Output Neural Network

**Martin Ferianc, Miguel Rodrigues**. CVPR Workshops (ECV), 2023, pp. 4564–4569.
https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Ferianc_MIMMO_Multi-Input_Massive_Multi-Output_Neural_Network_CVPRW_2023_paper.html

**Native claim:** MIMMO extends MIMO with multiple predictions / early exits within a conventional network; one-forward multiple outputs is not conceptually exclusive to Mirror. Match trained output quality, ensemble diversity, entire serialized head/trunk bytes, and active FLOPs. MIMMO may be a different problem (ensemble outputs rather than addressed specialist functions): match the declared task before quality comparison.

**Mirror-specific question:** Can a very short learned m express K useful, distinct addressed functions at smaller marginal state and comparable inference cost than MIMMO's native output/exit scheme? Do not use repeated full-forward recomputation as the only competing design.

## PA438 — Network Fission Ensembles for low-cost self-ensembles

**Hojung Lee, Jong-Seok Lee**. Pattern Recognition Letters 190 (2025), 22–28; DOI 10.1016/j.patrec.2025.01.032.
https://doi.org/10.1016/j.patrec.2025.01.032
https://arxiv.org/abs/2408.02301

Code: https://github.com/hjdw2/NFE

**Native claim:** Network Fission Ensembles reuses parts of a conventional network to build several output/exit paths, including distillation for accuracy and diversity. Shared-trunk multi-output work must compare native output diversity, trained task fidelity and incremental active compute rather than claiming one forward or a parameter-tied network is new.

**Mirror-specific question:** Output-coded functions must beat same-byte shared multi-exit / regular linear-head controls, with a different role address causing useful functional divergence. Ensemble diversity alone is not proof of independent specialist capacity.

## PA439 — SpecMD: A Comprehensive Study On Speculative Expert Prefetching

**Duc N. M. Hoang, Mohammad Samragh, Ajay Kumar Jaiswal, Minsik Cho**. ICML 2026; PMLR 306:43336–43350.
https://proceedings.mlr.press/v306/hoang26d.html

**Native claim:** SpecMD benchmarks expert caching and prefetch on realistic memory/hardware budgets, including a Least-Stale eviction strategy. This is a direct scheduler/memory-control baseline for MA-1176 after the primary multi-output mechanism is validated. Do not attribute generic prefetch/caching gains to Mirror.

## PA440 — A Unified Sparse Attention via Multi-Granularity Compression

**A Unified Sparse Attention via Multi-Granularity Compression** — Siran Liu, Zane Cao, Yongchao He. ICML 2026, PMLR 306:75532-75548.  
https://proceedings.mlr.press/v306/liu26h.html

**Established method / mandatory native control:** UniSparse builds multi-granularity composite token summaries and a block sparse selector. It is an existing optimized sparse attention algorithm. Mirror must not claim its token-selection or full-attention speedups.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA441 — The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

**The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs** — Piotr Nawrot et al.. Findings of ACL 2026, 38667–38701.  
https://aclanthology.org/2026.findings-acl.1926/

Native implementation/code: https://github.com/PiotrNawrot/sparse-frontier

**Established method / mandatory native control:** Large-scale methodological audit of training-free sparse attention, identifying granularity/routing and GPU-kernel bottlenecks. Requires measured wall-clock sparsity with native kernels and longer-context task splits.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA442 — LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging

**LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging** — Seungeon Lee, Soumi Das, Manish Gupta, Krishna P. Gummadi. ACL 2026 long papers, 39583-39601.  
https://aclanthology.org/2026.acl-long.1837/

**Established method / mandatory native control:** LoGo uses signals from a single pass through LoRA adapters to perform training-free instance-level adapter selection and merging. This native ability, not Mirror, explains adaptive composition.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA443 — RanLoRA: Residual-aware Nonlinear Low-Rank Adaptation

**RanLoRA: Residual-aware Nonlinear Low-Rank Adaptation** — Xu Luo, Yongbin Liu, Chunping Ouyang, Ying Yu. Findings of ACL 2026, 17243-17258.  
https://aclanthology.org/2026.findings-acl.852/

**Established method / mandatory native control:** RanLoRA freezes principal pretrained SVD modes and adapts residual subspaces using nonlinear activation and Hadamard vector modulation. Nonlinearity and factor reuse are native, not Mirror innovations.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA444 — Efficient inference, training, and fine-tuning of protein language models

**Efficient inference, training, and fine-tuning of protein language models** — Muhammed Hasan Çelik, Xiaohui Xie. iScience 28(10):113495 (2025).  
https://doi.org/10.1016/j.isci.2025.113495

Native implementation/code: https://github.com/uci-cbcl/esm-efficient

**Established method / mandatory native control:** ESME optimizes ESM-like protein model execution, packing, quantization and task-specific parameter-efficient adaptation. Protein property heads and native optimized sequence encoding are direct baselines.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA445 — Protein Circuit Tracing via Cross-layer Transcoders

**Protein Circuit Tracing via Cross-layer Transcoders** — Darin Tsui, Kunal Talreja, Daniel Saeedi, Amirali Aghazadeh. ICML 2026, PMLR 306:122371-122402.  
https://proceedings.mlr.press/v306/tsui26a.html

Native implementation/code: https://github.com/amirgroup-codes/ProtoMech

**Established method / mandatory native control:** ProtoMech already learns sparse cross-layer transcoders and retrieves property-specific circuits in ESM2. Compare its native full/windowed CLT and PLT, not a weak independent transcoder.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA446 — CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells

**CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells** — Yuansong Zeng et al.. Nature Communications 16, 4679 (2025).  
https://www.nature.com/articles/s41467-025-59926-5

Native implementation/code: https://github.com/biomed-AI/CellFM

**Established method / mandatory native control:** CellFM trains a RetNet-based single-cell encoder with LoRA and downstream perturbation/function prediction. It is not new to condition a shared expression model on perturbation/task. Check the authors' correction before using figure values.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA447 — RegFormer: a single-cell foundation model powered by gene regulatory hierarchies

**RegFormer: a single-cell foundation model powered by gene regulatory hierarchies** — Luni Hu, Hua Qin et al.. Nature Communications 17, 6432 (2026).  
https://www.nature.com/articles/s41467-026-72198-x

Native implementation/code: https://github.com/BGIResearch/RegFormer

**Established method / mandatory native control:** GRN-guided gene sequence ordering and Mamba-based cell embedding are native RegFormer. Mirror must only test extra factorized perturbation coordinates beyond this biology-specific learned representation.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA448 — Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines

**Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines** — Constantin Ahlmann-Eltze, Wolfgang Huber, Simon Anders. Nature Methods 22, 1657–1661 (2025).  
https://www.nature.com/articles/s41592-025-02772-6

**Established method / mandatory native control:** Empirical negative benchmark finding that single-cell foundation models did not outperform simple mean/linear perturbation predictors on studied settings. Those baselines are mandatory for Mirror.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA449 — Deep learning perturbation models can outperform baselines on calibrated metrics

**Deep learning perturbation models can outperform baselines on calibrated metrics** — Henry E. Miller, Gabriel M. Mejia, Francis J. A. Leblanc et al.. Nature Biotechnology, 1 October 2026.  
https://www.nature.com/articles/s41587-026-03307-w

**Established method / mandatory native control:** Shows evaluation metric calibration/positive controls can change the apparent performance of perturbation predictors. Treat as complementary methodological perspective to PA448, not as a guarantee models always win.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA450 — ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling

**ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling** — Xin He et al.. DAC 2026, DOI 10.1145/3770743.3804292.  
https://doi.org/10.1145/3770743.3804292

Native implementation/code: https://github.com/expertflow-dac/expertflow

**Established method / mandatory native control:** Native predictive routing path, token batch scheduling and expert caching/offload are an end-to-end system control. Credit only additional Mirror grouped physical storage or hot-view coding, not generic prefetch.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA451 — DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge

**DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge** — Yuegui Huang, Zhiyuan Fang, Weiqi Luo, Ruoyu Wu, Wuhui Chen, Zibin Zheng. arXiv:2603.19172 (2026 preprint).  
https://arxiv.org/abs/2603.19172

**Established method / mandatory native control:** Importance/depth-aware mixed-bit expert scheduling with look-ahead prefetch. Required matched-bit edge deployment control for Mirror prefetch, quantization, and resident HBM claims.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA452 — Evaluating the Utilities of Foundation Models in Single-Cell Data Analysis

**Evaluating the Utilities of Foundation Models in Single-Cell Data Analysis** — Liu et al.. Advanced Science (2026), DOI 10.1002/advs.202514490.  
https://doi.org/10.1002/advs.202514490

**Established method / mandatory native control:** Multi-model single-cell evaluation across annotation, embedding, perturbation and other tasks. Hold out perturbation/cell identities and compare native model with strong conventional baselines.

**Mirror delta:** Only a source-calibrated small m giving an incremental heldout task-quality/paid-byte/active-compute benefit beyond this native source is a Mirror-specific positive result. Native results are not registered as Mirror results.

## PA453 — TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling

**TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling** — Yury Gorishniy, Akim Kotelnikov, Artem Babenko. ICLR 2025.  
https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html

Official/author implementation: https://github.com/yandex-research/tabm

**Native method / direct baseline:** Original TabM uses shared MLP weights and BatchEnsemble rank-one member fast weights, with actual parallel member execution. Multiple member predictions and weight sharing are native; an additional Mirror m must beat native TabM/TabM-mini and an equally low-description ordinary r/s code.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA454 — Accurate predictions on small data with a tabular foundation model

**Accurate predictions on small data with a tabular foundation model** — Noah Hollmann, Samuel Müller, Lennart Purucker et al.. Nature 637:319–326 (2025).  
https://www.nature.com/articles/s41586-024-08328-6

Official/author implementation: https://github.com/PriorLabs/TabPFN

**Native method / direct baseline:** TabPFN is a pretrained in-context tabular posterior predictive model. Multi-dataset conditional prediction is native, not evidence that an output-only Mirror adds information. Temperature, vector, Dirichlet calibration and native TabPFN ensemble are direct null baselines.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA455 — TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks

**TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks** — Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, Artem Babenko. ICLR 2025.  
https://research.yandex.com/publications/tabred-analyzing-pitfalls-and-filling-the-gaps-in-tabular-deep-learning-benchmarks

Official/author implementation: https://research.yandex.com/datasets/TabReD

**Native method / direct baseline:** TabReD uses chronological rather than random train-test splits and high-dimensional engineered industrial tables. Native TabM/TabPFN and Mirror claims should be checked under whole time/dataset drift, not only random toy data.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA456 — DINOv3

**DINOv3** — Oriane Siméoni, Huy V. Vo, Maximilian Seitzer et al.. arXiv:2508.10104, August 2025 Meta technical report.  
https://arxiv.org/abs/2508.10104

Official/author implementation: https://github.com/facebookresearch/dinov3

**Native method / direct baseline:** The DINOv3 frozen SSL backbone already supports dense feature maps and native linear/Mask2Former segmentation heads. One encoder with multiple output heads and Gram anchoring are native; Mirror only adds incremental useful readout-code compression beyond these baselines. Official license must be honored.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA457 — Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling

**Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling** — Yair Schiff, Chia Hsiang Kao, Aaron Gokaslan, Tri Dao, Albert Gu, Volodymyr Kuleshov. ICML 2024, PMLR 235:43632–43648.  
https://proceedings.mlr.press/v235/schiff24a.html

Official/author implementation: https://github.com/kuleshov-group/caduceus

**Native method / direct baseline:** Caduceus is naturally reverse-complement equivariant, using native MambaDNA. The cheap reverse-complement orbit is NOT an independent functional-capacity expansion. Mirror parity-coded tasks must improve over native RCPS/augmentation and ordinary same-byte task classifiers.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA458 — Test-time Generalization for Physics through Neural Operator Splitting

**Test-time Generalization for Physics through Neural Operator Splitting** — Louis Serrano, Jiequn Han, Edouard Oyallon, Shirley Ho, Rudy Morel. ICML 2026, PMLR 306:109205–109235.  
https://proceedings.mlr.press/v306/serrano26a.html

Official/author implementation: https://github.com/LouisSerrano/neural-operator-splitting

**Native method / direct baseline:** Native method already searches test-time compositions of pretrained operator dictionaries. An extra Mirror m must reduce dictionary/schedule state or improve rollout quality over native operator splitting and standard Strang splitting; the composition itself is not new.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA459 — Learning Physical Operators using Neural Operators

**Learning Physical Operators using Neural Operators** — Vignesh Gopakumar, Ander Gray, Daniel Giles, Lorenzo Zanisi, Matt J. Kusner et al.. AISTATS 2026, PMLR 300:3223–3231.  
https://proceedings.mlr.press/v300/gopakumar26a.html

**Native method / direct baseline:** Native physical-operator splitting and fixed finite differences already separate linear and learned nonlinear physics operators. Any Mirror-compressed shared operator must match physical conservation and long-horizon quality at real time-step and stored byte cost.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA460 — A foundation model for the Earth system

**A foundation model for the Earth system** — Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic et al.. Nature 641:1180–1187 (2025).  
https://www.nature.com/articles/s41586-025-09005-y

Official/author implementation: https://github.com/microsoft/aurora

**Native method / direct baseline:** Aurora already has shared heterogeneous 3D Perceiver encoder, Swin processor and domain-specific decoder/fine-tunes; forecast horizons are typically autoregressive. One processor step and full long-horizon rollouts must be counted separately, with meteorological variable units and native Perceiver controls.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA461 — Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge

**Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge** — Siying Ma, Mehrdad Momeni Zadeh, Mauricio Soroco, Wuyang Chen, Jiguo Cao, Vijay Ganesh. ICLR 2026.  
https://proceedings.iclr.cc/paper_files/paper/2026/hash/1b6554ab420ad26328d73e3387613f4e-Abstract-Conference.html

**Native method / direct baseline:** Native physics-informed multiphysics training reuses simplified PDE models and fundamental operators. Mirror code experiments must not re-label known physical decomposition or fixed solver kernels as an independent Mirror gain.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA462 — DISCO: learning to DISCover an evolution Operator for multi-physics-agnostic prediction

**DISCO: learning to DISCover an evolution Operator for multi-physics-agnostic prediction** — Rudy Morel, Jiequn Han, Edouard Oyallon. ICML 2025, PMLR 267:44750–44774.  
https://proceedings.mlr.press/v267/morel25a.html

**Native method / direct baseline:** DISCO already uses a hypernetwork to produce a small evolution operator from a short trajectory. The operator generator and its physical-parameter inference are native. Add m only to compact the paid source-trained operator bank or support new ordered useful compositions beyond simple coefficient codes.

**Mirror-specific delta:** Evaluate a small structured m on top of the actual native physical method. Judge heldout useful function quality, whole real serialized bytes and active compute versus the best same-byte linear/diagonal/basis and original-native controls. The published native result is not a Mirror experiment.

## PA463 — LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning

**LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning** — Md Kowsher, Haris Mansoor, Nusrat Jahan Prottasha, Ozlem Garibay, Victor Zhu, Zhengping Ji, Chen Chen. ICML 2026, PMLR 306:60815-60863.  
https://proceedings.mlr.press/v306/kowsher26a.html

Native source/code: https://github.com/Kowsher/LiME

**Native result and honest baseline:** One paid PEFT module, per-expert output modulation p_e and zero-parameter route/Auto Top-K. These ideas are established native; Mirror must beat the original and an ordinary compressed p_e code on quality/bytes/P95, not compare with K full transformer passes.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA464 — M3LoRA: Flexible Task Adaptation via Multiple Low-Rank Matrices With Mixture-of-Subspaces and Minor Singular Components Initialization

**M3LoRA: Flexible Task Adaptation via Multiple Low-Rank Matrices With Mixture-of-Subspaces and Minor Singular Components Initialization** — Xu Luo, Yongbin Liu, Chunping Ouyang, Ying Yu, Yang Yang. CAAI Transactions on Intelligence Technology 11(3):681-694, 2026.  
https://doi.org/10.1049/cit2.70144

**Native result and honest baseline:** Native source mixes multiple low-rank A/B subspaces with a learned matrix and initializes from minor singular vectors of pretrained W. All of those are native. Compress only the additional task mixer/weight-bank state m.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA465 — Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning (LoDA)

**Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning (LoDA)** — Lingfeng He, De Cheng, Huaijie Wang, Xi Yang, Nannan Wang, Xinbo Gao. ICML 2026, PMLR 306:41172-41194.  
https://proceedings.mlr.press/v306/he26f.html

**Native result and honest baseline:** Native LoDA jointly studies shared/private directions, gradient-aligned optimization and closed-form recalibration for retained knowledge. Best direct baseline for Mirror+private residual, not a new Mirror invention.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA466 — FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection

**FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection** — Maxime Fontana, Michael Spratling, Miaojing Shi. CVPR 2026, pp.31135-31144.  
https://openaccess.thecvf.com/content/CVPR2026/html/Fontana_FAAR_Efficient_Frequency-Aware_Multi-Task_Fine-Tuning_via_Automatic_Rank_Selection_CVPR_2026_paper.html

**Native result and honest baseline:** Performance-Driven Rank Shrinking and a Task-Spectral Pyramidal Decoder are both native FAAR. Compare m to the published rank allocation and same-byte task-frequency readout.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA467 — MoEP: Compact and Efficient Sparsity with Modular Expert Paths

**MoEP: Compact and Efficient Sparsity with Modular Expert Paths** — Joonas Tapaninaho and Mourad Oussalah. Neural Networks, September 2026, article 109617.  
https://doi.org/10.1016/j.neunet.2026.109617

**Native result and honest baseline:** Native whole Attention–FFN block path routing at fixed overall parameter budget; native study reports architecture/task-dependent gains and warns that Pythia-1B layer routing may not scale as standard FFN-only MoE. Count whole active block and cache validity.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA468 — Learning Task-Preferred Inference Routes for Gradient De-Conflict in Multi-Output DNNs (DR-MGF)

**Learning Task-Preferred Inference Routes for Gradient De-Conflict in Multi-Output DNNs (DR-MGF)** — Yi Sun, Xiaochang Hu, Xin Xu, Jian Li, Yifei Shi, Ling-Li Zeng. IEEE TPAMI 48(3):3154–3166 (2026).  
https://doi.org/10.1109/TPAMI.2025.3635844

**Native result and honest baseline:** Task-filter importance variables, task-preferred routes and meta-weighted gradient deconfliction are native DR-MGF. Short m must improve on this full route state and strong gradient-only controls PCGrad/CAGrad.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA469 — Gradient Surgery for Multi-Task Learning (PCGrad)

**Gradient Surgery for Multi-Task Learning (PCGrad)** — Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn. NeurIPS 2020.  
https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

**Native result and honest baseline:** PCGrad projects away conflicting per-task gradient components. Any multi-output Mirror training improvements must exceed PCGrad at matched optimization budget; no Mirror storage overhead required for gradient-only controls.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.

## PA470 — Conflict-Averse Gradient Descent for Multi-Task Learning (CAGrad)

**Conflict-Averse Gradient Descent for Multi-Task Learning (CAGrad)** — Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu. NeurIPS 2021.  
https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

**Native result and honest baseline:** CAGrad balances average task loss and worst local descent and is an important gradient optimization control for one-trunk multi-output Mirror training; test actual quality rather than assuming routing is a representation deficiency.

**Mirror-specific frontier:** Add only the smallest paid task/expert m to the native method; require heldout functional utility, full native implementation, real serialized basis/decoder/metadata bytes and latency vs same-byte ordinary task codes. Neither native multi-output nor shared PEFT is Mirror novelty.
