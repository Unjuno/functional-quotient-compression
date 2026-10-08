# Mirror function-space falsification — natural adapter geometry, real-data pilot and direct controls

Date: 2026-10-08 JST
Type: **research support, negative pilot and falsifiable integration design**.
Status: no new formal MA scientific status, no changes to MA-255 queue, no main-branch edits.

## Core finding

The same data-generating task's **weight delta** can be far from a compressed model in Frobenius norm even when the compressed model predicts well. The converse is also possible. Thus the central question for Mirror's additional parameter `m` should be a **functional loss/bytes/compute frontier**, not simply a geometric orbit discovered in raw weights:

`F(x; theta) -> F(x; theta, m)`

where `m` may be a structured, dynamic, factorized or shared/private code. The specific `m` should be introduced on top of a strong native existing method and compared at **actual task utility**, real codec bytes and inference latency.

A critical antecedent: **BOLT (PA372, CVPR 2026)** extracts orthogonal spectral bases from already-trained task vectors and trains **small diagonal coefficients** for unseen tasks. Therefore a shared basis plus a scalar/diagonal task code is *direct prior art*, not by itself a new Mirror contribution. **Task Vector Bases (PA375)** already uses compact task-basis coefficients for compositional task arithmetic.

## New primary literature and what it changes

| PA | Published method | Mirror-specific implication / nearest control |
|---|---|---|
| PA372 | BOLT (CVPR 2026) | source-task orthogonal basis + tiny **diagonal** new-task code is native; Mirror must beat this baseline |
| PA373 | Curvature-Guided LoRA (2026) | prediction/function-space fit and curvature should outrank weight-delta Frobenius proximity as a metric |
| PA374 | Fora (2026) | retained capabilities live in **activation-derived** subspaces that can differ from weight SVD modes |
| PA375 | Task Vector Bases (2025) | shared task directions plus structured coefficients and arithmetic are established |
| PA376 | SVD + CUR LoRA merge (ACL 2026) | globally shared SVD features and localized CUR task-private features may be complementary |
| PA377 | StructLoRA (ACL 2026) | information bottleneck and coordinated layer selection can improve task fit without an inference-time module |
| PA378 | Training Jacobian geometry (2024) | a weight direction silent in-distribution may be important OOD |
| PA379 | NTK-regime LoRA optimization theory | suboptimal learned code can reflect optimization, not absence of representational capacity |
| PA380 | FuLA functional latent alignment (2025) | functional stitching should be tested with counterfactual information-sensitive probes |
| PA381 | Linearization of LLM fine-tuning (NeurIPS 2025) | identify the range where a local Jacobian of m predicts task behavior and where nonlinear errors dominate |

Read the exact PA entries for original primary-paper links and nuanced controls. Some are **adjacent** to multi-adapter compression, not exact replication of Mirror's architectural code.

## Registered numerical pilot

A genuine numerical experiment was run as independent research intake, with a protocol committed before running:

- [Frozen PROTOCOL](../../experiments/mirror_applications/research_intake/natural_digit_function_20261008/PROTOCOL.md)
- [Runnable source](../../experiments/mirror_applications/research_intake/natural_digit_function_20261008/source/run_pilot.py)
- [All 48 measurement rows](../../experiments/mirror_applications/research_intake/natural_digit_function_20261008/RESULTS_CORE.csv)
- [Results and limitations](../../experiments/mirror_applications/research_intake/natural_digit_function_20261008/RESULTS.md)
- [Replay verification](../../experiments/mirror_applications/research_intake/natural_digit_function_20261008/VERIFICATION.json)

**Dataset:** scikit-learn real handwritten digits, 8×8, frozen 64→64→64→10 base model. Source rank-4 LoRA deltas are separately trained on five input shifts/corruptions and pooled into a rank-6 orthogonal input/output basis. Four distinct **held-out task conditions** adapt only a tiny code, or train a native rank-4 LoRA, at the same 220 updates.

**Important limit:** rotations/blurs/shifts are constructed algorithmically from real images; these are *not independent natural language tasks or natural LoRA adapter banks*.

| Condition (mean of 2 seeds × 4 held-out shifts) | Top-1 accuracy | Test cross entropy | Serialized additional state for 4 logical tasks | Mean eager CPU forward |
|---|---:|---:|---:|---:|
| frozen base, no new state | 37.01% | 7.662 | 22 B NPZ empty archive (base not counted) | 0.114 ms |
| one shared rank-1 direction + task scalar | 40.10% | 6.491 | 2,026 B | 0.126 ms |
| orthogonal shared basis + 6 diagonal scalars/task (**BOLT-like control**) | 49.79% | 3.687 | 4,662 B | 0.124 ms |
| **Mirror**: same basis, 6 scalars + 2 fixed Givens angles/task | **52.19%** | **2.983** | **4,694 B** | **0.197 ms** |
| same basis + dense 6×6 core/task (**CtS-style capacity control**) | 69.97% | 1.195 | 5,142 B | 0.125 ms |
| native separate LoRA rank 4/task | 86.98% | 0.416 | 10,222 B | 0.117 ms |

All methods inherit the same frozen base. The numbers are **actual named NPZ serialized inference additions**. The source-adapter training and basis construction have an offline cost *not included* in these four-target inference bytes. No GPU/system deployment claim can follow.

The Mirror code reduced cross entropy against the diagonal code on all **8/8** held-out measurements, but improved classification **accuracy** on only 4/4 targets under seed 41 and 2/4 under seed 42. Dense core and independent LoRA retained far higher accuracy; Mirror's materialization overhead increased eager CPU time relative to diagonal. **The preregistered Mirror-specific Pareto gate FAILED.** This remains an exploratory pilot, **not MA-1099 FAIL** or MA-1115 completion.

**A scientifically important counterexample:** the dense core has *higher average relative Frobenius delta error* than structured Mirror (1.341 vs 1.302 relative to native LoRA), yet much **better task accuracy** (69.97% vs 52.19%). Weight-space compression error alone misranks these two at least in this experiment; dense core has lower target-logit RMSE (3.719 vs 5.483).

Full replay: **48 rows × 14 deterministic fields exact, max difference 0**, CPU wall/forward timing excluded. Code, raw results and report committed. Small CPU condition and two seeds limit the conclusion.

## Next high-value experiment: function-sensitive versus weight-sensitive m basis

Do **not** simply retry the same held-out test with more Givens rotations and claim a fresh result. A new protocol requires new source/audit splits, fresh seeds, and controls frozen before opening audit.

1. Use at least two pretrained bases or natural task adapters **sharing a verified identical base revision**. A public open-task adapter bank is preferred. Fit bases from **source tasks only**, never audit deltas.
2. Evaluate three *source-only* basis definitions:
   - weight Frobenius / SVD of task deltas;
   - BOLT-like orthogonal task-basis directions;
   - activation- or curvature-weighted, function-space Jacobian basis on source calibration inputs (PA373, PA374, PA381).
3. Fit **new task codes from task training samples**, not from the target's oracle full-LoRA delta. Compare matched update budget versus independent LoRA, BOLT, CtS dense core, simple FiLM/rank-one, CG-LoRA, Fora-protected code, StructLoRA (where feasible), and sparse CUR private residual.
4. Compute source/audit metric separately: weight Frobenius error, output logit/dense prediction error, downstream NLL/accuracy, retain-old-task quality, and OOD data sensitivity. Keep denominators and loss calibration distributions explicit.
5. Hold out source-task identity and optionally task families. Report `K` task bank scaling, common basis size, `m` code size, per-task private residual bytes, serializer metadata, source-adapter pretraining cost, runtime and optimized-kernel feasibility.
6. Formalize `m` function sensitivity: probe `J_m(x)=∂F(x;theta,m)/∂m` and compare its singular directions **on a fixed input calibration distribution** with true source-task function changes. Do not assume local Jacobian sufficiency under nonlinear extrapolation; test linearization error.
7. Require **gauge invariance** of learned conclusions under invertible `B->B G, A->G^{-1} A` reparameterizations and sign/rotation ambiguity of degenerate SVD subspaces.
8. If the structured code needs `k²` freedom or private rank comparable to independent LoRA, or remains slow, mark that family as no Mirror-specific useful capacity; do not make information-theoretic free-capacity claims.

## Worker linkage without queue preemption

Existing registry already has relevant hypotheses:
- **MA-1096:** natural independently learned LoRA subspace and gauge audit;
- **MA-1098/1099:** pretrained singular basis, structured versus dense task-core code;
- **MA-1102:** few-shot code on frozen source-task basis;
- **MA-1105/1114:** shared/private and number-of-adapters break-even;
- **MA-1115:** gauge-falsification of LoRA task-code geometry.

**No new MA ID is created for this pilot or this literature survey** because these IDs already cover the hypotheses. Instead add PA372–381 as direct controls and use this numerical pilot to prioritize a real task-function benchmark. Avoid continuing to inflate the UNTESTED backlog when one of the existing experiments can provide the decisive scientific evidence.

## Interpretation standard

- A smaller coefficient code is not automatically novel (BOLT, Task Vector Bases).
- A good output fit on a narrow probe set is not necessarily informational equivalence (PA380 and PA241).
- A low weight error is not evidence of task utility (this pilot, PA373/374/376).
- A nonzero useful accuracy benefit without a byte/latency win can be a narrow mechanism result, not adoption.
- A failed 2-angle Mirror chart is not proof that every physically realizable chart fails.
- Full MA statuses remain **29 PROMISING / 18 FAIL**; only a formally executed, verified ID-specific experiment can change one.

Current authoritative worker next: **MA-255**, unchanged.
