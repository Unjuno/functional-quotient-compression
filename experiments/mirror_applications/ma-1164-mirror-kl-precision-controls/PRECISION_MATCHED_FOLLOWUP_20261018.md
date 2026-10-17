# MA-1164: FP16 / FP32 / FP64 paired precision-control follow-up (2026-10-18)

**Result:** The same serialized floating-point coefficient precision for conventional and nonlinear Mirror models does **not** establish a Mirror-specific quality/rate improvement in this tested family. Prior MA-1164 status and earlier negative findings are unchanged.

## Why this comparison matters

Prior comparisons used conventional NN weights serialized as FP16, but nonlinear Mirror physical coefficients as FP64. The resulting storage comparison confounded coefficient count and coefficient bit width. This follow-up controls the serialized dtype **within each paired comparison**, separately for FP16, FP32 and FP64.

Crucial numerical boundary: **all training, Mirror weight-space fitting and inference arithmetic use CPU FP64**, with stored FP16/FP32 values read back and promoted to FP64. This does **not** test FP16/FP32 arithmetic during training, multi-thousand-bit scalar optimization, or fused GPU kernels.

## Fixed experiment

- Model/dataset: sklearn handwritten digits, 64 input features, tanh hidden, 10 classes, 1,797 images total.
- Trained native h16 teacher: 1210 independent coefficients. Native h10 baseline: 760. Native h6: 460. AdamW 900 updates, batch 128; learning rates 0.008 / 0.008 / 0.004 for h16/h10/h6 respectively.
- Nonlinear Mirror virtual h16 (1210 reconstructed weights): m768 physical coordinates, K=8 parallel Mirrors, R=4 nonlinear residual rounds; secondary m512, K=2/R=4. Mirror codes are fixed Givens/Householder/permute transforms separated by tanh.
- Comparators: ordinary independently trained h6/h10/h16, same-m nonMirror orthogonal-decoder fitted to same pretrained teacher.
- All compiled models use pretrained teacher **weights only**, not labels or logits during conversion. Do not compare teacher-assisted conversion training costs directly with small-from-scratch training costs without charging the teacher.
- New held-out evaluation splits 618530, 618531, 618532; init seeds 61, 62 (6 paired runs). These are new stratified splits of the *same* source corpus, not three independent data domains.
- CPU AMD EPYC 9V74 virtual, PyTorch 2.10.0+cpu, 1 thread, eager FP64; real torch.save serialized inference payload includes coefficients, int16 address indices, K/R metadata. Executable program costs are not included; favoring Mirror.

## Accuracy (%) — SAME stored dtype across models

| Stored dtype | Native h10, 760 | Plain nonMirror m768 | Mirror m768 |
|---|---:|---:|---:|
| FP16 | 95.787 | 95.556 | 95.370 |
| FP32 | 95.787 | 95.556 | 95.370 |
| FP64 | 95.787 | 95.556 | 95.370 |

All seven tested families had **identical classification accuracy** at FP16, FP32, and FP64 stored precision across all six samples. NLL did change slightly. This is a finding about coefficient quantization in this small digit task, **not** a general law of numerical precision.

| FP16 stored comparison | Accuracy | Actual inference payload | CPU batch64 median forward |
|---|---:|---:|---:|
| Native h6 (460) | 93.102% | 3229 B | 0.0211 ms |
| Mirror m512 (K2/R4) | 89.306% | 4005 B | 1.009 ms |
| Native h10 (760) | 95.787% | 3869 B | 0.0244 ms |
| Plain m768 | 95.556% | 5029 B | 0.187 ms |
| Mirror m768 (K8/R4) | 95.370% | 5029 B | 3.344 ms |
| Native h16 (1210) | **96.944%** | **4701 B** | 0.0291 ms |

At FP16 Mirror m768 needs **1160 B more than native h10** while scoring **0.417 percentage points worse**. Even full native h16 FP16 scores higher (96.944%) and consumes fewer bytes (4701 B vs 5029 B). At FP32, m768 Mirror/native h10 = 6565/5341 B. At FP64 = 9637/8413 B.

## Paired uncertainty and scientific interpretation

The Mirror m768 minus native h10 per-world mean accuracy differences were +0.556, -1.806, 0.000 percentage points. Overall -0.417 pp. With only three reused-corpus worlds, the nominal two-sided t(2) 95% interval is **[-3.483, +2.650] pp**; it contains zero and does not establish a nonzero population effect. The world-mean sample standard error is 0.713 pp and t multiplier k=4.303; neither is a calibrated metrological combined uncertainty. Mirror m768 minus equal-m plain averaged -0.185 pp.

**H:** At identical serialized precision, Mirror beats native h10 and same-m plain in >=2/3 worlds and uses fewer bytes than native.
**T:** 3 worlds x 2 seeds x 21 model/dtype conditions = 126 rows, frozen K/R and LR, no test tuning.
**D:** Native-accuracy win 1/3, plain win 1/3, storage win 0/3 -> **FAIL** for this family.
**C:** Parameterization restrictions and address/program overhead dominate; stored FP width mismatch was not the main source of the small accuracy gap.
**U:** Same original corpus, small sample, CPU-only FP64 replay, no measured fused-GPU kernels or FP800/8000 high-precision optimizer.

## Verification and incident

- New tests 13/13 PASS; unchanged parent tests 14/14 PASS. 126 fresh-result rows, no duplicate world×seed×model×dtype keys.
- Frozen protocol SHA-256: `c54faf8d4fcdd8770ddca872ed28c8e4107e9a52c2fac90de240b6268c53ece0`
- Experiment driver SHA-256: `a5a9624df110d50d95418b73203c77fdee5c3902a58458befa340463de25fd58`
- Results CSV SHA-256: `57ec1bb001d2a50ca9ea526b19a3d520acef44b8f9dbf6e565b2a153619e7caf`
- A first output attempt finished one world×seed but stopped at CSV writing because the model rows had different columns. Only the CSV column-union writer was repaired; all fixed conditions were rerun with no numerical changes. Original traceback and complete machine-readable results are retained in the research chat artifact bundle.
- Complete frozen protocol, Python driver, tests, 126-row CSV, verification, raw logs and Japanese report exist in the generated `MA1164_FP_precision_matched_evidence.zip` conversation artifact, **not yet committed as individual repository files**.

**Next falsifiable question:** Does a true FP32/FP16 training-compute implementation or a separately designed high-precision Mirror representation improve quality per **total stored bit**, rather than merely altering saved checkpoint dtype? Report optimizer state, precision-induced gradient error, all Mirror programs and GPU kernel work independently.
