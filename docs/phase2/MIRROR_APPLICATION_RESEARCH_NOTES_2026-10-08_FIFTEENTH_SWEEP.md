# Mirror Application Research — Fifteenth Sweep (2026-10-08 JST)

**Scope:** deeper direct-prior analysis of naturally learned LoRA deltas, shared parameter coordinates, multi-adapter serving and physical KV cache state. This sweep adds **PA351–371** and **MA1096–1115**, and a tested **mathematical intake tool**. **No new natural-task Mirror quality/compression result is claimed.** The existing selected worker candidate MA-255 is unchanged.

## CH. Why this research is more valuable than blindly expanding a registry

The first 47 reported MA experiments provided mostly **aligned synthetic mechanism** evidence (29 PROMISING, 18 FAIL). More synthetic examples from candidate-aligned teachers cannot determine what fraction of *real* independently learned models lives on a cheap common Mirror orbit.

The scientific bottleneck is now **identifiability + natural delta representability + marginal serving economics**. This sweep emphasizes a much smaller set of highly falsifiable tests rather than broad architectural enumeration.

The project target remains `F(x; theta) -> F(x; theta,m)`: one physical shared object and low-description function coordinate `m` that changes useful function at lower bytes or state/routing/runtime cost than the best native method. Extra input codes, nonlinear generators, spectral coordinates, group rotations, low-rank tensor cores, and neutral cache reuse are **all heavily researched; none can be attributed to Mirror as a universal novelty**.

## CI. Four convergent direct-prior results

### CI-1. A whole collection of trained LoRAs can already be compressed

- **PA351 [Compress then Serve](https://proceedings.mlr.press/v267/gabrielsson25a.html)** (ICML 2025) **already uses shared adapter bases + per-LoRA scaling matrices**, with clustering to handle less-related adapters. It reports meaningful serving throughput at up to 1000 LoRA adapters. A plain shared basis plus code does **not** differentiate Mirror from this prior.
- **PA352 [Compress then Merge](https://proceedings.mlr.press/v306/he26h.html)** (ICML 2026) computes common left/right LoRA subspaces and each task's compact `r×r` core **before merging**. Note its *one merged adapter* objective differs from *many switchable task adapters*; compare only where task is matched.
- **PA353 EigenLoRAx** and **PA354 VB-LoRA** also derive or reuse a common low-dimensional adapter bank, while **PA355 MetaTT** already factorizes task/layer/matrix modes of global transformer adaptation.
- **PA369 TC-LoRA** and **PA370 ThanoRA** provide more specialized heterogeneous-task/clustered shared-private controls. **PA361 Share** and **PA362 LoDA** further address continually learned subspaces.

**Mirror-specific test:** After reproducing the native code families, test whether a **more structured/low-description `m`** (e.g. task-specific Givens, phase, sparse core, factorized task×layer code) improves the useful-function-bytes/runtime Pareto relative to the native `r×r` coefficient core, simple diagonal scaling, or private residual. If it loses to CtS/EigenLoRAx/MetaTT, record no marginal Mirror-specific benefit even if it beats standalone LoRA.

### CI-2. The pretrained base spectrum and actual task deltas are distinct

- **PA356 [Pretraining Induces a Reusable Spectral Basis](https://arxiv.org/abs/2605.07302)** reports stable leading singular directions in pretrained weights across downstream fine-tunes, with competitive small-coefficient task adaptation.
- Stability of `W_pretrained` leading singular vectors **does not logically imply** the delta matrices `D_t = W_t - W_pretrained` live in that same basis. Natural task increments may concentrate in a different subspace or orthogonal complement. This distinction is often obscured by experiments where the teacher *was generated* with the candidate View family.

**Required cross-evaluation:** (i) U/V from pretrained `W`, (ii) a basis discovered using only training-task deltas, (iii) independent per-task rank-r LoRA factors, (iv) a native shared basis/TT, (v) structured Mirror `m`, (vi) shared plus private residual. Hold out **whole natural tasks**, not random examples from the same adapter. Fit m using permitted task examples after the oracle representability screen; the oracle uses the hidden held-out LoRA delta and is not deployable new-task adaptation.

### CI-3. LoRA factor directions are gauge dependent

For any invertible `G in GL(r)` and `D = B @ A`:

```text
B' = B @ G
A' = G^(-1) @ A
D' = B' @ A' = D
```

Thus vectorwise cosine of raw `B` factors and raw `A` factors **can vary while the model's weight update does not**. PA357 [Crowded in B-Space](https://arxiv.org/abs/2604.16826) finds shared B-side output directions but interprets them in a factorized setting; PA358 [GLoRA](https://arxiv.org/abs/2605.06733), PA359 GL-equivariant Learning on LoRAs, PA360 canonical W2T and PA367 LoRA-RITE make the nonidentifiability and invariant representations explicit.

**Mirror consequence:** A claim that one functional `m` encodes reusable task directions must be invariant under arbitrary invertible low-rank refactorization. Compute singular spectra, *column and row projectors of D*, principal angles, and downstream predictions. A gauge-only movement counts as **zero functional multiplicity**, even though it might help quantization or optimization.

For `B[d_out,r]`, `A[r,d_in]`, a stable diagnostic avoids dense D using thin QR:

```text
B = Q_B R_B;  A.T = Q_A R_A
R_B R_A.T = Uc S Vc.T
D = (Q_B Uc) S (Q_A Vc).T
```

Singular vectors in degenerate eigenspaces are not unique. Compare projectors, not a choice of signed vector.

**Concrete runnable intake:** `experiments/mirror_applications/research_intake/natural_lora_orbit_20261008/` with a CLI `source/orbit_audit.py` and four unit tests (nonorthogonal gauge/refactorization, related/unrelated matrices, projector edge cases, train/audit manifest disjointness and same-base revision). Tests passed in a CPU container. No third-party model checkpoint was downloaded or measured, and the CLI output is **oracle weight-space representability only**.

### CI-4. LoRA-agent KV state is already split into shared base and private low-rank cache

- **PA364 [LRAgent](https://proceedings.mlr.press/v306/jeon26b.html)** (ICML 2026) decomposes a shared base cache and LoRA-specific low-rank pieces. It provides Flash-LoRA-Attention to avoid fully materializing low-rank components.
- **PA365 [PReCache](https://arxiv.org/abs/2609.34054)** (September 2026) studies precomputed low-rank agent caches and neutral base reconstruction to avoid re-prefill. It is **not** evidence that arbitrary LoRA caches are identical.
- Existing PA154 aLoRA, PA155 standard-LoRA prefix cache tradeoffs, PA156/MA691 exact known-right-View cache algebra, and PA236–239 cross-model cache translators must be retained as independent direct comparators.

**Mirror-specific test:** Does a small structured `m_agent` compress the **remaining adapter-specific low-rank cache state** beyond LRAgent/PReCache at the same prompt/quality and latency? If yes, can the underlying cache be physically aliased, not just numerically copied? Evaluate actual memory allocation, sharing/pointer identity, token provenance, RoPE correctness, read-path fusion, decoder target NLL/QA, TTFT, decode speed and low-rank precomputation cost.

A trained cross-model cache translator is generally **approximate**; exact algebra holds only under its specified compatible-canonical-state conditions.

## CJ. Fifteenth-sweep registry families

| Range | Test family | Strong native control | Primary falsifier |
|---|---|---|---|
| MA1096–1099 | gauge-invariant natural LoRA overlap and structured shared cores | CtS, CtM, EigenLoRAx, rank-r standalone | raw factor coincidence disappears under GL(r); heldout tasks do not fit m |
| MA1100–1104 | thousands of adapter servings, bank/code/tensor factors | CtS clusters, CtM merged LoRA, EigenLoRAx, VB-LoRA, MetaTT | native shared banks have better bytes/quality/throughput |
| MA1105–1109 | continual learning, calibrating task directions, federated/code generation | Share, LoDA, Pico, GLoRA, W2T, Zhyper/HyperLoader | m does not add capacity or simply follows a basis/gauge |
| MA1110–1112 | exact/approximate multi-LoRA cache/serving | LRAgent, PReCache, aLoRA, MA691 | state not physically shared; TTFT/quality/memory loses |
| MA1113–1115 | binary-code, amortization and gauge-integrity checks | LoRDBA, CtS, standalone LoRA, gauge-canonical controls | code is larger/slower; break-even never reached; invariant fails |

All **20 new candidates** are UNTESTED (16 P0, 4 P1). No PROMISING/FAIL statuses were changed.

## CK. Required native quality and byte accounting

### CK-1. Weight-space oracle screening (NOT adoption)

Given *training-task* deltas `D_train`, select fixed shared `U,V`; for an entirely held-out test task delta `D_test`, project:

```text
C_test = U.T @ D_test @ V
D_hat_test = U @ C_test @ V.T
relative_oracle_Frobenius_error = ||D_test - D_hat_test||_F / ||D_test||_F
```

Repeat `C` full, diagonal, sparse q-entry, low-parameter structured Mirror and shared+private forms. For rank `k` bases and `K` tasks, a basic shared-core float-value lower count is:

```text
shared_core_floats = k * (d_out + d_in) + K * (k ** 2)
independent_rank_r_floats = K * r * (d_out + d_in)
```

Both exclude metadata/optimizer states/padding; they do **not** represent actual serialized bytes. Structured `m` must beat the k×k core and cheap diagonal/FiLM controls. If it cannot, it is not a convincing compression gain. Shared-basis calculation and persistent storage cannot be free just because `W` was already stored.

### CK-2. Natural-task adaptive test

Treat the hidden audit task delta as **oracle evaluation-only**; learn the code from allowed training examples, then report true downstream NLL/task score, new task adaptation steps, wall-clock, actual serialized inference/resume bytes and retained prior skills. Use a fresh/audit split selected before testing. A positive oracle subspace projection alone cannot show a trainable Mirror mechanism.

### CK-3. Nonaligned and high-rank boundary

Hold out unrelated tasks from distinct domains but exact same frozen base revision and layer shapes. Sweep rank, basis k, task count K, coefficient code bits, private residual rank and task cluster purity. When a source adapter is measured on a different base model revision, do **not** subtract or pool its factors with the target; separate provenance is mandatory.

### CK-4. Serving and fused kernel gate

Compare total system behavior at realistic concurrent adapters: active code/kernel launches, adapter-load bandwidth, physically allocated KV, prefix alias vs copy, VRAM fragmentation, TTFT/prefill, decode throughput and target QA/NLL. The serving control ladder must include S-LoRA/Punica plus CtS and LRAgent/PReCache where relevant. Report operating points where native methods dominate.

## CL. Worker handoff

- Canonical registry after sweep: **1115** MA, **371** PA, **1068 UNTESTED**, **29 PROMISING**, **18 FAIL**. P0 **597**, P1 **415**, P2 **103**.
- Next worker remains **MA-255 Parameter Superposition**, on its existing frozen order. Do not start an overlapping MA-255 experiment or change its audit seeds.
- The gauge-aware diagnostic research intake is a **supplemental tool**, not a result for MA1096 or a valid change of status.
- If the worker tackling MA-255 can safely import this audit logic **before freezing its protocol**, use it to compare Parameter Superposition against actual adapter-family shared-core controls. If the protocol is already frozen, document as *future* control; do not retroactively add new gates.
- Only source-backed facts belong in PA351–371. The mathematical and deployment hypotheses above are our **untested proposals**.
