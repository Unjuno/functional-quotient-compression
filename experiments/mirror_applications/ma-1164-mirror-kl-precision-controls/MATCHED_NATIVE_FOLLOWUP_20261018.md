# MA-1164 supplement: capacity-matched ordinary MLP versus nonlinear Mirror (2026-10-18)

**Finding: Mirror-specific advantage FAIL in this evaluated configuration.** This is a separately scoped follow-up to the frozen MA-1164 study; older evidence and status are not overwritten.

## Why this experiment

The previous normal MLP was `64→16→10` with 1210 independent trained scalars. It was **not** one-dimensional. The smallest Mirror condition used one physical coordinate, but this comparison extends to m=256,512,768,1024. To address the concern that the normal model was not matched for **independently learned parameter count**, train normal MLP widths 6,7,10,11,13,16, giving 460,535,760,835,985,1210 trainable scalar values respectively, all using the same 64-input/10-class digit task. The Mirror decoder still realizes a **virtual 64→16→10 network** with 1210 logical weights from m physical coefficients and paid coordinate indices.

## Precommitted training and audit

- 8×8 sklearn handwritten digits; 1797 total images from one corpus, re-split, not independently sourced domains.
- Development world splits 618440, 618441, seed 40. Native learning rate selected per width from .004, .008, .016.
- 900 updates, batch128, AdamW no weight decay, float64 CPU, one PyTorch thread, no dropout/augmentation.
- The initial planned fresh world 618450 encountered a NativeMLP/NativeWidth `.h` type mismatch before any test CSV. The abort is preserved in the full evidence archive. Engineering correction only; **new splits** 618460/618461/618462 with seeds 51/52 were sealed in `FREEZE_V2.json` before evaluating them.
- Mirror direction and nonlinear recurrence K/R fixed: m256 K2/R4, m512 K2/R4, m768 K8/R4, m1024 K2/R4; **no selection using replacement fresh labels**.
- Mirror is mechanically fit **from the learned full-width teacher weights only** via teacher-coordinate selection and weight-space L-BFGS, not via label/logit distillation. This gives Mirror a **teacher-pretraining advantage** absent in directly trained narrow native controls; teacher time cannot be excluded from end-to-end compute.
- 36 development rows, 120 unique fresh model/world/seed rows, parent 14 and extension 9 unit tests PASS; development-selection and frozen source SHA check PASS.

## Fresh 3 partitions × 2 independent initializations (6 runs/model)

| Model | Physical trained scalars | Test accuracy | torch.save payload bytes | CPU eager batch64 forward median ms |
| --- | ---: | ---: | ---: | ---: |
| Native width 6, FP16 stored → FP64 replay | 460 | 94.07% | 3229 | 0.0214 |
| Native width 7, FP16 stored → FP64 replay | 535 | 94.07% | 3421 | 0.0224 |
| Mirror, m512 K2/R4 | 512 | 90.42% | 7013 | 1.031 |
| Plain non-Mirror sparse orthogonal, m512 | 512 | 91.44% | 7013 | 0.182 |
| Native width 10, FP16 stored → FP64 replay | 760 | 96.34% | 3869 | 0.0249 |
| Mirror, m768 K8/R4 | 768 | 95.09% | 9573 | 3.338 |
| Plain non-Mirror sparse orthogonal, m768 | 768 | 95.37% | 9573 | 0.183 |
| Native width 13, FP16 stored → FP64 replay | 985 | 97.08% | 4317 | 0.0250 |
| Mirror, m1024 K2/R4 | 1024 | 96.76% | 12133 | 1.014 |
| Native width 16, FP16 stored → FP64 replay | 1210 | 96.90% | 4701 | 0.0273 |

- Additional Mirror m256 accuracy **62.64%**, payload **4453 bytes** versus same-m nonMirror **65.88%**.
- Native width10 has 760 independently learned weights versus Mirror m768's 768 physically stored values: **native +1.25 percentage points**, with the native advantage observed in all three new partitions. The three per-world paired Mirror-minus-native deltas were -1.806, -0.278, -1.667 pp.
- Normal width6 (460) outperforms Mirror m512 (512) by 3.66 pp on average; normal width13 (985) outperforms Mirror m1024 (1024) by 0.32 pp.
- The native width10 FP64 checkpoint is also smaller: 8413 bytes versus 9573 Mirror bytes, even without FP16 storage.
- FP16 here is **serialized representation only**, cast back to FP64 for forward testing; times are not FP16-native GPU kernels.
- K×R generated logical directions do not create K×R independent physical learned coefficients. The solver/fixed Mirror generator source bytes are additional storage (not included in table), so rate comparisons favor the normal model even more when charging them.

## H/T/D/C/U and limitations

- **H:** Mirror nonlinear parallel-recurrent weight program at m physical coordinates should outperform a native ordinary NN with similar independent weights, and preserve an actual inference-byte advantage over the strong native FP16 comparison.
- **T:** Two development partitions and three new fixed replacement evaluation partitions with two seeds; no test-guided LR or K/R retuning; frozen 900 updates and batch128.
- **D:** MIRROR-specific win 0/3 worlds for m768 versus native h10 and same-m plain. Storage and runtime also FAIL. Full one-parameter/80k-bit precision hypotheses remain separately UNTESTED by this FP64 experiment.
- **C:** Useful accuracy may simply come from selecting large teacher-weight coordinates. Similar non-Mirror sparse orthogonal mapping performs as well/better; smaller ordinary NN has enough capacity.
- **U:** The three 'worlds' reuse the same digit corpus and are not independently sampled domains. m768 minus h10 descriptive world-mean accuracy SEM is 0.488 percentage points; calibrated combined uncertainty `u_c` and coverage `k` are not estimated. CPU timing virtualized, clocks not locked. Teacher-pretraining access makes the compression-versus-scratch comparison non-identical; both that fact and teacher training cost are explicitly disclosed.

### Reproduction and evidence location

The full conversation artifact `MA1164_capacity_matched_controls.zip` contains source, tests, frozen v1/v2 protocols, correction chronology, fresh/development CSV, environment, verification and Japanese report. This GitHub note records independently auditable aggregate figures; the full supplemental archive has **not** been committed as individual GitHub source/CSV files here. The complete parent MA-1164 code and original 102-row evidence are present in this branch.

**Conclusion:** the low-m failure cannot be dismissed as 'only one scalar' because both 512 and 768 learned scalar Mirror vectors were compared with 460 and 760 parameter ordinary MLPs. The Mirror configuration tested does not show a capacity, storage or inference-speed advantage over these native controls.
