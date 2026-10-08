# Function-first Mirror coordinate discovery — 2026-10-09 CPU research intake

**Scientific status: THREE development-gate FAILs, no fresh-world experiment; no claim of Mirror-specific advantage.** This is a scoped research intake and does **not** change any official MA ID, global registry, or main branch.

## Why this study exists

MA-403 tested a very particular input-conditioned pair rotation, too narrow to falsify the entire additional-functional-coordinate thesis. These experiments first independently learned 12 output-head functions and only then discovered low-description shared coordinates, without presupposing rotations.

Data: `sklearn.datasets.load_digits`, real handwritten 8x8 digit images with deterministic synthetic input corruptions. Shared clean 64→96→64 ReLU encoder is pretrained and frozen; each source domain gets independently trained 10×65 output weights (bias incorporated). Hold out four target domains, fit their task codes from 192 labeled training examples only. Source-only bases have **no access to target checkpoints or target evaluation examples**. Metrics: accuracy, CE nat/example, actually serialized full four-task inference bank bytes, and CPU eager/folded latency.

This is a small CPU classification screen, NOT pretrained LLM model-bank compression or 64x quality-preserving compression.

## Verified aggregate development results

| Experiment | Method | Accuracy mean (2 seed×4 tasks) | CE nat/example | Whole 4-task serving payload |
|---|---|---:|---:|---:|
| FF-001 | weight-SVD shared core rank4 | 73.89% | 1.1302 | 58,804 B |
| FF-001 | source-activation-function-weighted core rank4 | 69.64% | 1.2814 | 58,804 B |
| FF-001 | FiLM | 82.70% | 0.5253 | 57,892 B |
| FF-001 | native LoRA rank2 | 83.67% | 0.5149 | 60,212 B |
| FF-002 | weight-SVD shared core rank4 | 72.08% | 1.1507 | 58,804 B |
| FF-002 | source-CE-supervised shared core rank4 | 78.31% | 0.7627 | 58,804 B |
| FF-002 | native LoRA rank2 | 84.64% | 0.5050 | 60,212 B |
| FF-003 | source-supervised shared core rank2 | 66.64% | 1.3508 | 58,012 B |
| FF-003 | source-supervised shared core rank4 | 73.09% | 0.9955 | 58,804 B |
| FF-003 | source-supervised shared core rank6 | 80.29% | 0.6922 | 59,724 B |
| FF-003 | source-supervised shared core rank8 | 83.77% | 0.5906 | 60,772 B |
| FF-003 | shared rank2 + private rank1 | 79.84% | 0.6601 | 61,180 B |
| FF-003 | native LoRA rank2 | 80.78% | 0.6461 | 60,212 B |

Accuracies are seed/task means; **two seeds are not a confidence interval**. 003 rank8 improves quality relative to LoRA2 but costs **560 B more total**, while rank6 is 488 B smaller but fails to meet the frozen quality gate. Shared-private rank2+1 improves quality but costs 3,168 B over shared rank2.

**Conclusion:** source-supervised basis learning helped relative to unsupervised weight or activation-SVD in this restricted family. Low-description codes have a capacity-storage frontier, and cheap native FiLM/LoRA are strong controls. Ordinary shared U,C,V is already a native tensor factorization (CtS-like) and is **not** a novel Mirror operator. All three strict gates failed; fresh seeds stayed sealed.

## Provenance / verification / caveats

- FF-001 seeds 5201–5202; 80 metric rows; 20 inference files; developer-only post-hoc oracle diagnostic is segregated and never trains an inference candidate.
- FF-002 seeds 5301–5302; 72 metric rows; 18 files. An impossible all-model 10%-saving draft gate was corrected to 2% **before opening either new development seed**; the correction is retained in the local evidence archive.
- FF-003 seeds 5401–5402; 96 metric rows; 24 files. Rank-specific private correction is paid in storage and compute; unfolded and folded CPU execution are measured separately.
- Python 3.13.5, PyTorch 2.10.0+cpu, scikit-learn 1.8.0, NumPy 2.3.5, SciPy 1.17, AMD EPYC 9V74, CPU quota 4, 1 Torch intra-op thread, FP32 eager, no GPU.
- **17/17 local unit tests passed** across the three studies. All 62 NPZ inference files were reread with exact size/hash checks; **248 evaluation rows** were replayed from saved weights (FF-001 and FF-002 exact CE/accuracy, FF-003 max CE difference 1.79e-7, accuracy exact).
- No MA status was changed; fresh worlds 5211–5213, 5311–5313, 5411–5413 remained unopened.
- Complete source, exact full protocols, pretrained inference payloads, CSVs, run logs, and hashes were created in the **2026-10-09 conversation evidence archive**, SHA-256 `ce2e4518adb65056451e469843f7eeb2b530d9efb22fb34828b473f356e81232`. A smaller source/report archive SHA-256 is `cb31d0ac24b6589c7e7100a019a260dbc5440e232d1eb8c35bd60036ae6df01c`. These are conversation artifacts, **not yet GitHub-hosted source archives**; this branch stores the verified summary/provenance, not a complete runnable checkout.

## Next investigation

Before registering any new MA ID, consult MA-1096/1099/1102/1105/1114 natural-adapter hypotheses and existing BOLT/CtS baseline literature. A meaningful follow-up must learn distinct nonlinear adapter functions first, discover shared bases on source-only tasks, train new-task m without oracle deltas, and test full bytes/active compute/eager/folded runtime against native lowrank/FiLM and private residuals. A pure geometry or weight-MSE gain is not a task-quality win.
