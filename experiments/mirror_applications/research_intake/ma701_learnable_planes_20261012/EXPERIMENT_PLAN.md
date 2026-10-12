# MA-701-LP01 — 学習可能な共有Mirror変換面の共同学習 実験計画

> For agentic workers: execute this plan task-by-task, with test-first changes and mandatory verification before scientific claims. This is a plan only; no new measurements have been produced.

**Status:** PLAN_ONLY / NOT STARTED. No new MA number, registry status mutation, fresh audit, or merge into main.
**Date:** 2026-10-12 JST.
**Goal:** 固定されたMirror回転面ではなく、少数の共有ベクトルから生成される回転面・タスク別View座標・共有ニューラル重みを共同学習し、品質/保存量/計算量の優位性を厳密に比較する。
**Primary registry link:** MA-701 (Householder task Views, existing P0 UNTESTED). Related MA-703 (layer×task reflection), MA-956 (learned symmetry), TCSMA MA-1194 staging FAIL; not a newly minted MA-1195.
**Architecture:** Shared non-linear FFN + learnable low-description orthogonal plane generators + per-task code; optional sparse gating and rank-one private residual. The learned axes must be shared globally, not stored per task. Compare exact native operator alias and strong rank/FiLM/LoRA/independent controls.
**Tech Stack:** Python 3.13 / PyTorch 2.10+cpu / NumPy / scikit-learn / pytest, CPU one thread, actual named NPZ serialization.
**Spec:** The objective and evidence contract in docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md and experiments/mirror_applications/EXPERIMENT_CONTRACT.md.

## Prior-research conflict check (verified, do not duplicate)

- MV-JOINT-001 (research/mirror-joint-angular-loss-20261011) jointly trained four task losses and task angles, but on a **fixed two-dimensional rotation plane**. Paired loss was 0.08124 versus native four-head 0.08426; unrelated fixed-view loss 0.31772 versus native 0.06562. Learned planes and generators were not compared.
- TCSMA-001/002 (research/ma-1194-task-sparse-mirror-alignment-20261012) jointly fitted shared dictionary/coefficients/fixed-plane angles/private rank-1; its auxiliary matching gradient uses fixed ridge-derived rows and primarily changes angles, not the shared dictionary. Prospectively preregistered five-seed matching addition failed 0/5.
- MA-574/575 have actual dedicated branch records but concern **discrete Hadamard/sign/permutation code selection for int4 quantization**, not SGD-learned common rotation axes from multiple task losses. MA-574 status is NOT_ESTABLISHED because freeze SHA/provenance diverged; MA-575 development FAIL due exact native BOFT-style alias.
- MA-701 and MA-703 already contain the vector-reflection/factorization idea as **UNTESTED registered proposals**, with no dedicated completed experiment branch observed at planning time. Do not register another ID or promote statuses.
- HRA/Householder, BOFT/OFT and ordinary shared-basis adapters are direct prior art. A mirror-named implementation exactly matching a native parametrization is not a Mirror-specific novelty.

## H — falsifiable hypotheses

H1 *Learnability:* With the same shared backbone and training examples/optimizer updates, learning the shared plane vectors improves held-out task loss over **fixed-plane** task-coordinate learning in families whose useful axes are not coordinate-aligned; target >=10% relative improvement in paired nMSE on the synthetic positive mechanism world.
H2 *Generalization:* Joint axis/coordinate learning produces useful non-identical task functions on both a deliberately aligned teacher and non-complement real-digit classification labels without severe interference.
H3 *Marginal resource utility:* At K=8 and K=16 (if validated), whole-serving-bank serialized bytes and folded/eager latency compete with the best quality-qualified native shared low-rank, FiLM/LoRA and multihead controls. A smaller parameter count alone is not a pass.
H4 *Gradient route:* Unlike the old TCSMA pseudo-alignment auxiliary, a genuinely learned shared-axis model has nonzero gradients from the **task loss** reaching shared feature parameters, axis vectors and per-task coordinates. Alignment auxiliary must be separately ablated.
C0 *Counter-hypothesis:* Improved geometry is only a native HRA/OFT/group-dictionary reparameterization, gauge motion, extra optimizer capacity, task-specific oracle teacher alignment, or poor native baseline optimization.

## Exact candidate operator (first trial)

Let $h_\theta(x)\in\mathbb R^{32}$, $z(x)=W_{\rm up}h_\theta(x)\in\mathbb R^{32}$, shared learned raw axis vectors $a_j,b_j\in\mathbb R^{32}$ for $j=1,\dots,p$, and per-task code $m_t\in\mathbb R^p$.

Normalize $u_j=a_j/(\|a_j\|+\epsilon)$, then $v_j=\widetilde v_j/(\|\widetilde v_j\|+\epsilon)$ with $\widetilde v_j=b_j-(u_j^\top b_j)u_j$. Every generator is $G_j=v_ju_j^\top-u_jv_j^\top$ (skew symmetric). Each exact plane rotation is

$$R_j(\varphi)=I+(\cos\varphi-1)(u_ju_j^\top+v_jv_j^\top)+\sin\varphi\,G_j.$$

Define $Q_t=R_p(m_{tp})\cdots R_1(m_{t1})$ and apply **inside the shared nonlinearity**:

$$F_t(x)=w_{\rm head}^\top\big[(W_{\rm down}+E_t)\,\mathrm{GELU}(Q_tz(x))\big]+b_t.$$

$W_{\rm down}\in\mathbb R^{32\times32}$ is shared, $E_t=P_tS_t^\top$ is optional task-private rank-one with $P_t,S_t\in\mathbb R^{32\times1}$. Task-head bias $b_t$ is paid for every method. The comparison always gives native controls the same shared feature extractor/training examples and at least comparable code budgets.

Use $p=4$ for the initial primary screen. It requires 2pd=256 shared raw axis scalars plus Kp task angles before base/private/state/metadata. Orthogonality is approximate in floating point and must be explicitly unit tested. If a plane becomes numerically degenerate ($\|\widetilde v_j\|$ very small), **log/reinitialize on the permitted development stream** rather than silently returning a non-orthogonal matrix.

Variable/unit/dimensional audit:
| Variable | Meaning | Dimension/type | Unit (normalized network) | Trainability |
|---|---|---|---|---|
| $d,p,K$ | width, generators, tasks | integer scalars; initial $d=32,p=4,K=8$ | 1 | frozen choices |
| $a_j,b_j,u_j,v_j$ | shared plane vectors | $\mathbb R^d$ | 1 | raw $a,b$ learned |
| $m_t$ | task address/code | $\mathbb R^p$ | 1 | learned |
| $G_j,Q_t$ | skew generator, orthogonal View | $d\times d$ | 1 | derived |
| $W_{\rm up},W_{\rm down},E_t$ | common FFN and private delta | $d\times d$ | 1 | common and optional private |
| $F_t(x)$ | task logit | scalar | 1 | output |
| $\ell_t$ | BCE / NLL | nonnegative scalar | nat/label | measured |
| $B_{\rm total}$ | full inference payload | integer | byte | measured |
| $T_{\rm inf}$ | per-batch inference time | positive scalar | ms | measured |

$Q_tz$ is $(d\times d)(d\times1)\to(d\times1)$, then $W_{\rm down}\mathrm{GELU}(Q_tz)\to d$, then $w_{\rm head}^\top\to1$. Orthogonality should keep $\|Q_tz\|_2$ invariant up to dtype error. A fixed invertible $Q$ alone can often be folded into neighboring weights and is not free additional capacity.

## T — benchmark and controls

**T0 / pre-run:** Snapshot and hash source, fixed protocol, baseline code and list of data seeds before training or opening fresh. Compare the same task loss with and without learned axes. Never select a hyperparameter on fresh.

**T1 / synthetic falsification:** 12 task functions, each six output vectors in 24-dimensional latent space, train 128 examples/task and evaluate on 512 truly separate examples/task. Three families: (A) *unknown randomly oriented shared rotation planes* with common sparse rules; (B) same but one of six rows per task private/off-orbit; (C) independent dense unrelated teachers. Generator truth used only to construct data and post hoc audit, **never provided as a fitted initialization/target**. Synthetic teacher-aligned PASS is mechanism evidence only.

**T2 / nonlinear data:** sklearn digits (real 8×8 images) with eight binary prediction roles over the same source image: parity, >=5, prime digit, divisible by 3, digits 0/1/2, digits 3/4/5, digits 6/7/8, digits 1/4/9. All eight losses jointly update shared trunk, shared FFN and axes; image-ID-stratified train/dev/test partitions fixed per seed. The tasks are correlated labels of one small dataset; do NOT call them eight natural independently sourced tasks. Also use a paired-complement four-role fixture only as a positive control.

**Native controls (same example stream and train budget):**
1. Hard shared FFN/no task View.
2. Fixed coordinate-plane rotations with trainable per-task angles and identical architecture.
3. Shared-axis *learned generator* with no matching/private loss: the primary candidate.
4. Native learnable shared-plane/HRA-like parameterization with same degrees of freedom, to detect exact alias (M0).
5. Ordinary rank-matched task-code/low-rank, FiLM/IA3, and rank-1/rank-2 LoRA; sweep their sizes to create a whole-byte Pareto envelope.
6. One shared trunk plus independent task output heads; task-local FFN as expensive quality upper control.
7. Candidate plus paid rank-one private residual; and a sparsity ablation (top-2 gated plane participation vs all generators).
8. Only in a **separate** ablation: differentiable partially-matched functional alignment. Use one-to-one matching with unmatched penalty, gradients through **generated model features and learned axes**, and show $\lambda_{\rm match}=0$ controls. Do not repeat the capped 1-to-many detached-ridge loss as if it were a new mechanism.

**Optimization controls:** Same shared initial trunk tensor and same batch index streams across methods. Fixed primary training: 800 AdamW updates, batch 64 for synthetic and batch 128 for digits; full 2,400-update continuation for the best native and candidate, without post hoc promotion of an 800-step result. One Torch thread and measured training steps/tokens/examples, wall time and active MAC proxy. No training outside container.

## Hyperparameter control and audit firewall

Calibration-only seed 17100 (synthetic) and one disjoint digit development partition choose from a **small fixed predeclared grid**: axis LR $\{3\times10^{-4},10^{-3}\}$, axis warmup $\{0,100\}$ steps, and auxiliary matching weight $\{0,0.003\}$ — 8 settings max; zero auxiliary is the primary analysis. Shared trunk LR fixed at 0.001; AdamW weight decay 0.0001; zero dropout. Keep $p=4$ and no private residual as the primary mechanism screen. A later private rank/sparsity/plane-count sweep must be labeled a separate amendment and must not use audit-world scores.

Three development seeds 17101–17103. Fresh seeds 17111–17115 are pre-reserved and **not accessed unless the development quality/byte/runtime gate passes**. Generate independent data-world random streams per family. If the code or preregistration changes, reserve NEW seeds instead of recycling exposed ones.

**Development gate to open fresh:** learned-axis beats fixed-axis by >=10% relative test nMSE in at least 2/3 planted-axes development worlds, does not lose to matched native non-Mirror method by >0.02 nat/label mean digit BCE, and all serialization/gradient/gauge tests pass. Report dev variability, not a population confidence interval.

**Fresh mechanism H1:** paired positive-world improvements on >=4/5 previously unopened seeds and median relative nMSE improvement >=10% at matched updates.
**Fresh functional H2:** digit non-complement average BCE <= best quality-qualified native +0.02 nat/label; worst-task BCE degradation <=0.05, mean accuracy deficit <=1 percentage point; no role-collapse/near-constant solution.
**Fresh efficiency H3:** at K=8 whole inference bytes <=95% of cheapest native method also satisfying H2, and batch1/batch64 folded forward median latency <=1.25x that native comparator. Report eager and one-off materialization separately; no claim of throughput gain from merely meeting tolerance.
**Fresh specificity H4:** if native HRA/group-shared generators exactly produce the same model outputs/storage, mark **NO MIRROR-SPECIFIC ADVANTAGE** regardless of other gates; otherwise test the incremental difference, never relabel the native technique.
**Unrelated safety:** in synthetic random unrelated worlds report failure severity/private residual fraction, and do not count spurious mathematical separation as independent capability.

All criteria above are **PLAN VALUES, NOT observed experimental results**. Freeze the exact code, generator/teacher/data hashes and any justified modifications in a preregistration commit before first development or fresh run.

## C — primary rival explanations / failure conditions

- A learned shared plane can be a change of coordinates absorbed by shared $W_{\rm up}/W_{\rm down}$, rather than independent task functions. Probe $\partial F_t/\partial m_t$, same-input logit disagreement and functional responses under equivalent gauges.
- Rank-2 ordinary free coefficient code may express the same geometry at equal or lower cost; report an algebraic equivalence/alias as M0.
- More shared parameters and more optimizers/steps can explain improved loss; match real serialized bytes and active training cost, not only nominal codes.
- Sparsity and partial correspondence may introduce gradient noise or incorrect merging. Compare loss terms one at a time and do not assume a matching penalty is helpful.
- Feature-plane collapse (nearly parallel axes), task collapse (identical logits), and private rank consuming the savings each independently FAIL the intended capacity/efficiency claim.
- A positive planted-axis teacher is an aligned construction, not evidence of naturally shared group symmetry.

## U — uncertainty / verification requirements

Report paired seed/task loss differences and their range, per-task curves, code norm/angle histograms, orthogonality error, plane degeneracy frequency, actual whole-file NPZ bytes including axes, codes, private factors, trained trunk and metadata, CPU batch1/64 timings, fit time, and optionally a seed-paired bootstrap explicitly marked **not a population CI** when all digit partitions reuse one source dataset. Do not combine byte/NLL/time scales into one score without defined units.

**Prior TCSMA root-cause regression:** with $\lambda_{\rm match}>0$, take autograd gradients from the auxiliary term alone and explicitly assert nonzero finite gradients into (a) learned axis vectors, (b) shared representations, (c) per-task coordinates, and (d) any private residual the objective actually claims to optimize. When $\lambda_{\rm match}=0$, still require task gradients into axes and codes. If missing, do not interpret an outcome as testing co-adaptation.

## Runnable implementation work breakdown (not executed)

**Target project path:** experiments/mirror_applications/research_intake/ma701_learnable_planes_20261012/.

### Task 1 — frozen protocol + data firewall
- [ ] Write PROTOCOL.json and source/worlds.py with signatures \`make_synthetic_world(seed:int,family:str,cfg:dict)->dict\` and \`make_digit_world(seed:int,cfg:dict)->dict\`; generate only allowed train/calibration/dev worlds.
- [ ] Write tests/test_worlds.py first; assert different train/test image identities, teacher parameters never passed to the learner, task-specific outputs, and reserved fresh disjointness. Run pytest to observe the expected initial missing-implementation failure; then implement and rerun.
- [ ] Record source/protocol SHA256 and branch parent commit in PRE_RUN_LOCK.json before opening any new dev seed. Commit.

### Task 2 — trainable shared-plane generator
- [ ] Write tests/test_planes.py first: identity at zero angles, $Q_t^\top Q_t\approx I$, numerical inverse, finite-difference/autograd gradients for raw $a,b,m$, and non-degenerate fallback. Observe tests fail with missing module.
- [ ] Implement source/planes.py \`orthonormal_planes(raw_a,raw_b,eps:float)->tuple\` and \`apply_plane_views(x,raw_a,raw_b,angles)->Tensor\` using O(pNd) plane updates (avoid materializing dense $d\times d$ per batch); add a dense reference for testing only.
- [ ] Verify fixed-plane equivalence and composition order for noncommuting generators; commit.

### Task 3 — matched task models and optimizer
- [ ] Write tests/test_joint_train.py first: task loss reaches shared trunk, axes, codes; shared plane is *shared across tasks* and task-only residual is correctly paid; no silent task-head copy.
- [ ] Implement source/models.py \`make_method(method:str,cfg:dict)->nn.Module\`; source/train.py \`fit_world(model,world,cfg,seed)->dict\`. All baselines use paired initializations, data batches and comparable updates.
- [ ] Implement training-parameter-count and active MAC reports; 800/2400 step traces. Commit.

### Task 4 — optional sparse/matching ablations
- [ ] Write tests/test_alignment.py first: one-to-one partial assignment, unmatched dummies, sparse top-2 gate, gradient route check for auxiliary loss and no audit-target leakage.
- [ ] Implement source/alignment.py, paying assignment search/training overhead and private rank. Do not include this term in the primary axis comparison unless independently declared; commit.

### Task 5 — paid serialization and independent replay
- [ ] Write tests/test_codec.py first: save/load of every method yields the same outputs before and after serialization, including axes and metadata; exact full NPZ length/hash, K-scaling and no missing reconstruction state.
- [ ] Implement source/codec.py, source/evaluate.py and an isolated eager/folded CPU benchmark. Run pytest for the new tests and the complete repository baseline suite where available; report unrelated failures rather than suppress them. Commit.

### Task 6 — execute prelocked benchmarks / scientific decisions
- [ ] Execute calibration seed, record selected configuration **before development scores**; then 17101–17103.
- [ ] Apply development gate as written. Only if it passes, freeze a separate fresh protocol commit and run 17111–17115 without tuning.
- [ ] Replay every serialized inference checkpoint and at least one full-seed retrain; write RESULTS_CORE.csv, VERIFICATION.json, STATUS.md and README.md with H/T/D/C/U. Mark FAIL/NOT_ESTABLISHED/PROMISING according to evidence; do not revise old MA-701 registry statuses without a complete ID-specific evidence contract.
- [ ] Publish experiment source/tests/protocol/results on a dedicated research branch. Never change main automatically.

## Next decision

This plan is sufficient to start **MA-701-LP01**, a *subexperiment* addressing a missing learnable-axis question within existing registered MA-701/703; it must not appear as a new MA-1195 entry. Its own results are currently **NOT ESTABLISHED / NOT RUN**, and no fresh data have been opened.
