# Real-image distribution-shift Mirror code pilot — frozen protocol

Date: 2026-10-08 JST
State at creation: **preregistered before running numerical pilot**.
Classification: **independent exploratory research intake**, not an MA-1096/1099 result or adopted method.
Relation to prior art: PA351..PA371, [BOLT (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Park_Basis-Oriented_Low-rank_Transfer_for_Few-Shot_and_Test-Time_Adaptation_CVPR_2026_paper.html), [Curvature-Guided LoRA](https://arxiv.org/abs/2603.29824). A shared orthogonal basis plus per-task diagonal code is BOLT-like *prior art*, not claimed Mirror novelty.

## Frozen question

When a base network is trained on **real** handwritten-digit images, and separate LoRA deltas are optimized on five distinct real-image corruption/shift conditions, can a low-description structured View code trained on four **different** adaptation conditions provide a better accuracy/bytes tradeoff than:
- the unchanged base model;
- a single scalar on a pooled shared rank-one adaptation direction;
- **shared orthogonal input/output bases plus diagonal per-task coefficients** (BOLT-like native control);
- a dense `k×k` shared-basis task core (CtS/CtM-style capacity control);
- a native independent rank-four LoRA update?

This is **not** evidence about natural LM/LLM adapters, nor a validation of full BOLT/CG-LoRA. Source and target shift transformations are synthetic perturbations applied to real digit images, not independent natural-domain datasets. Results cannot be promoted beyond this pilot.

## Frozen dataset and computation

- `sklearn.datasets.load_digits()`, 8x8 grayscale handwritten digits, labels 0–9.
- Stratified train/dev/test split: 60% / 20% / 20%, `random_state=seed`; seeds **41 and 42**, run both with identical settings.
- Pretrain a `64→64 ReLU→64 ReLU→10` classifier on clean digits, then freeze it entirely.
- Modify only the second `64×64` linear weight by additive `delta W`. The original base bias and all other parameters remain fixed.
- Independently fit rank-4 LoRA deltas to **source** conditions:
  `rotate(+20°)`, `rotate(-20°)`, `horizontal shift(+1 pixel)`, `contrast×0.7`, `Gaussian blur σ=0.8`.
- Extract shared rank-`k=6` orthogonal column/row bases from **source deltas only**, as leading eigenspaces of sums of `D D^T` and `D^T D`.
- New **target** tasks are:
  `rotate(+33°)`, `rotate(-33°)`, `vertical shift(+1 pixel)`, `Gaussian blur σ=1.2`.
- The target task training images are permitted to fit new codes; target test examples and target fitted full LoRA weights must not be used to select the shared bases or any hyperparameters.
- Each task/method sees exactly the same images/labels, minibatch-size 256, and number of updates. Pretraining 500 updates, source LoRA adaptation 220, target method adaptation 220.
- Optimizer Adam without weight decay, base LR 0.01; rank-4 LoRA LR 0.025; scalar/diag/dense/structured-code LR 0.06. No tuning after inspecting results. `torch.set_num_threads(1)`.

## Candidate and exact control parameterizations

Let `U,V` be `64×6` source-only bases:
- **base:** `D=0`.
- **scalar:** `D=c u_0 v_0^T`, 1 coefficient per task, shared `u_0,v_0`.
- **diagonal (BOLT-like):** `D = U diag(c) V^T`, 6 coefficients/task. This is an already-established native baseline.
- **Mirror candidate:** `D = U R_01(theta) diag(c) R_23(phi)^T V^T`, 8 scalars/task; the two Givens plane rotations are part of a *fixed, preregistered* structured family, not chosen after seeing target deltas.
- **dense core (CtS-style):** `D=U C V^T`, 36 scalars/task.
- **native rank-4 LoRA:** `D = B A` with 512 task-dependent floats.

Every method has the same frozen base and output classifier; no target-private extra head, bias or data-dependent router is allowed. The dense core provides additional freedom at greater task-code size; it does not count as a Mirror-specific method.

## Report

For each seed/target/method, record:
1. held-out digit test cross-entropy and top-1 accuracy;
2. adaptation wall time and average forward time on the same CPU/thread;
3. **actual serialized adaptation-state bytes** for the full four-target *bank*, including shared bases and all task codes, via a deterministic named NPZ record; all methods inherit the same frozen base and report it separately if needed;
4. diagnostic relative weight-delta reconstruction to independently trained target LoRA, and target-output logit difference; neither oracle delta is used for training/tuning the low-description methods;
5. test robustness across both independent seeds and the four qualitatively different shift tasks.

**Scientific decision gate:** do not call Mirror-specific success merely because the structured 8-value code beats the frozen base. The structured code must improve accuracy or cross-entropy beyond the 6-value diagonal code on at least 3/4 target tasks **in each seed**, and beat the dense-core and rank-4 LoRA quality/byte frontier, with no worse actual inference throughput than the diagonal baseline. Otherwise report limited/negative evidence without adjusting the protocol. The outcome remains *exploratory* regardless of the gate.

## Split and claim limitations

- No task held-out full LoRA delta can enter U/V construction or code training.
- Time and serialization should be measured honestly; NPZ metadata overhead is counted and raw tensor floats are only supplemental.
- A positive result remains real-digit image-shift evidence, not a large natural-model compression claim.
- A negative result at these settings does not imply Mirror cannot work with other basis size, rotation family or optimizer; re-tuning after audit requires a new named protocol and new seeds.
- This pilot must not change MA-1096..1115 status, the 47 already verified experiments, or the MA-255 worker queue.
