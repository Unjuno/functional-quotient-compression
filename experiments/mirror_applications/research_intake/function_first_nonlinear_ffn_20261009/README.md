# Function-first Mirror: independent nonlinear FFN functional sharing (2026-10-09)

**Development verdict: FAIL in all three explicitly frozen experiment series.** This research intake is **not an official MA status**, does not overwrite the MA registry, and does not change main. All independent fresh seeds remain sealed.

## Purpose

The earlier MA-403 rotation experiment and function-first linear output head tests constrain their chosen operators rather than falsifying the project's broad low-description functional-coordinate idea. This new study trained 12 separate real-image domain functions at an **internal FFN matrix** (64x96) of a frozen 64-96-ReLU-64-ReLU-10 digit classifier, discovered task-shared bases using *source training only*, then fitted the target codes from 192 labeled new-domain training examples. Four held-out input conditions were evaluated per seed. Real 8x8 sklearn handwritten-digit images with **synthetically generated image shifts/blur/rotation**; not natural-language LLM adaptation.

### Evidence outcomes

| Trial | Seed(s) | Core result | Strong control | Status |
|---|---|---|---|---|
| FF-NL-001 | 6201-6202 | 6-code quadratic m: 63.44% accuracy / 2.1902 CE / 60,960B; free core rank6: 79.21% / 0.7580 / 61,440B | LoRA2: 81.89% / 0.6452 / 62,612B | FAIL |
| FF-NL-002 | 6301-6302 | Family-specific source basis: quadratic rank6 66.05% / 1.1901 / 60,960B; shared free core rank10: 83.95% / 0.5428 / 65,024B | LoRA4: 83.01% / 0.5902 / 67,732B | FAIL Mirror-code gate |
| FF-NL-003 | 6401-6403 | Separate 3-seed development check: supervised shared free core rank10: 83.77% / 0.5311 / 65,024B | Scratch LoRA4: 81.99% / 0.5801 / 67,732B; **source-warm LoRA4: 86.72% / 0.4231 / 67,732B** | FAIL strongest-control gate |

CE is mean categorical cross entropy (natural units per example), accuracy/CE average across development seeds x 4 target conditions, bytes are the actual fully serialized **four-task inference bank** (shared frozen trunk+shared bases+four target codes+all file metadata). These averages are not confidence intervals. Keep training-efficiency differences separate from converged representational capacity.

**Critical new finding:** Ordinary source-trained free rank10 cores can outperform LoRA4 trained from an uninformed initialization at 4.00% lower full bank bytes, but a **native LoRA4 initialized with the source-only mean task delta** substantially outperforms the shared core on all 3 additional development seeds. Thus the value may be information transfer/initialization, not a Mirror-specific operator. Native shared U C V^T is established low-rank sharing, not a newly invented Mirror transformation.

FF-NL-001 independently trained source functions have 30.4–32.4% pairwise prediction disagreement on the same clean development inputs (12 functions), but disagreement alone does not establish independently learned capacity equivalent to 12 models.

## Implementation and verification

- Backend: Python3.13, PyTorch2.10.0+cpu FP32 eager, NumPy2.3, SciPy1.17, scikit-learn1.8; AMD EPYC 9V74 container CPU quota approx 4, torch one inference/training thread; GPU unused.
- Primary training each seed: shared clean trunk 420 updates; twelve distinct source internal FFN task deltas each 210 updates; each source basis family 240 updates; each new target method 210 updates on 192 target-label examples.
- The quadratic code tested is `C(m)=diag(m)+m m^T / rank`. It has only rank-dimensional local degrees of freedom, which is structurally less than a rank x rank unrestricted code. Not a RoPE-specific operation, not a claimed new hypernetwork result.
- Actual inference files saved as individual NPZ archives, source-only basis and target code states included. No source-trained delta used as a target oracle during code fitting. Source-trained full model parameters are not served in the 4-target deployment; source-learning cost is separately recorded.
- The three fully separate test suites ran **16/16 passing**, all **240/240 stored evaluation records** recomputed exactly from **60/60 reloaded serialized inference banks**. No fresh seed opened.
- All three development gates FAIL. No promotion by tuning development thresholds after results.
- Historical repository-wide tests were not rerun; do not infer general LLM or optimized GPU runtime from these CPU results.

## Provenance and artifacts

Conversation attached downloadable full evidence ZIP (source, all 60 stored models, 240 metrics, tests, training/runtimes, frozen protocols, hashes):
`NONLINEAR_FFN_2026-10-09_FULL_EVIDENCE.zip`
SHA256 `5cf348e736f1453e6fa061abad31464990faf637ae3c6d676688bf37f21326cd`.
Lightweight source/reports ZIP: SHA256 `1719a45e1c060fd15f0d0254b8cae3d0b0e352172286d79f70f4d692cde1fabf`.

**Important repository boundary:** Those ZIP files and their runnable sources are distributed as **conversation evidence artifacts**, NOT uploaded into GitHub by this commit. This branch contains the verified summary and hash ledger; it does not claim GitHub reproduction without downloading that evidence archive.

Full Japanese report `NONLINEAR_FFN_REPORT_JA.md` (in the evidence ZIP) includes the variable/unit table, local-rank derivation, all gates, uncertainty and next steps. `EVIDENCE_INDEX.json` holds SHA256 checks of frozen sources, protocols and results.

## Decision and next experiment

Do not run a fourth geometry chosen post hoc against the same audit split. Existing low description codes are not yet demonstrated to add incremental value against a **source-aware** strong native baseline. Next high-information experiment is a new preregistered nonlinear adapter-bank comparison where native source-warm LoRA, native shared core (CtS/BOLT analog), diagonal/FiLM, Mirror m and private residual receive matched pretraining budgets and actual runtime/storage accounting.

Relevant disciplines: low-rank linear algebra/manifold dimension, task transfer/meta-learning, information-theoretic coding, and CPU deployment systems engineering.
