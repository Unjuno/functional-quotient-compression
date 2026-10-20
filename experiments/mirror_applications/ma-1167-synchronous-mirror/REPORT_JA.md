# MA-1167 Japanese result — Batched synchronous Mirror vs native at equal learned coefficients

**Date:** 2026-10-20 JST. **Result:** H1 mechanism PASS; H2 Mirror-specific quality FAIL; H3 equivalent CPU batching speedup PASS (not GPU). Branch-only; main and prior experimental verdicts unchanged.

## What is actually tested
Five synchronous state nodes, all ordered source→target pairs including self loops and skips, three Mirror addresses per pair: **5×5×3=75 simultaneously executable virtual 6→8→6 MLPs**. Three Loop steps update all five node states simultaneously **within each step**, using only previous-step states. Consecutive steps are serial.

One learned tensor of **289 FP32 physical coefficients**: 64 truly globally shared core floats, 75 independent edge gate logits, and 150 per-edge learned Householder orientation coordinates. These produce **8,250 virtual weight values** using a fixed seeded decoder's `Householder → tanh → inverse Householder`. The generated 8,250 values are **not** 8,250 independent learned degrees of freedom. The source-to-edge addressing remains fixed and does NOT learn free many-to-many physical→virtual correspondence.

**No pruning, no sparsity regularization, no edge sampling.** Train/dev/fresh/OOD: 2048/1024/2048/2048 independent inputs. Worlds: teacher of same Mirror family (`aligned`) and native 6→25→6 MLP (`native`). Seeds 557,563,569. AdamW 450 updates, batch 128, lr .015, CPU FP32 single thread. Dev-best checkpoint selected without fresh tuning. Four methods x2 worlds x3 seeds = 24 completed runs.

## Fresh normalized MSE (smaller is better)

| Model (289 learned scalars each) | Aligned 3-seed median | Native-teacher 3-seed median | All-six median | OOD all-six median |
|---|---:|---:|---:|---:|
| **Mirror synchronous T3** | 0.093389 | 0.047860 | 0.072279 | 0.160837 |
| Mirror synchronous T1 | 0.111583 | 0.040680 | 0.074816 | 0.185113 |
| Direct affine shared decoder T3 | 0.084941 | 0.036494 | 0.062259 | 0.127330 |
| Native MLP289 | **0.026501** | **0.003534** | **0.010209** | **0.037106** |

Mirror T3 vs affine T3: **0 wins / 6 paired fresh trials**, median NMSE ratio Mirror/affine **1.112**. Mirror T3 vs native: **0/6**, median ratio **6.508**. Mirror T3 vs Mirror T1: **4/6 wins**, but threefold active edge-MLP compute.

## CPU latency (median across seeds, measured same trained Mirror weights)

| Inference batch | 75 separate Python virtual-edge MLPs ×3 steps | Same network with batched BMM | Affine batched BMM | Native MLP |
|---|---:|---:|---:|---:|
| B=1 | 15,597 µs | **881 µs** | 511 µs | 24.7 µs |
| B=64 | 18,556 µs | **2,414 µs** | 2,046 µs | 31.0 µs |

**Batched execution is ~17.7× and ~7.7× faster than per-edge execution**, with equivalent outputs/gradients; it is nevertheless much slower than native. 75×3×96=21,600 virtual-MLP matrix MACs per sample, excluding nonlinear decode, vs ~252 native dense MACs plus rank-one side path. GPU hardware **not available/tested**.

**Critical memory caveat:** Learned weight payload 289×4 = 1,156 B; serialized inference blob with 20-byte header = **1,176 B**. Actual runtime fixed seeded buffers: **293,480 B**; temporary fully decoded virtual weights: **33,000 B**. Fixed decoder source/program size must be additionally accounted and cannot be silently called zero storage. This is not yet extreme deployed compression.

## Joint-gradient audit
- 7/7 unit tests PASS: 75 edges, one trainable object, scalar-reference forward/backward equivalence, deterministic reproduction, nonzero gradient in all three physical coefficient groups.
- Individual virtual edge gradients summed into shared physical tensor match batched reverse-mode gradient with relative error ~1.5e-7.
- On all **12 trained Mirror-T3 and affine-T3 checkpoints**, exact full-T=3 VJP decomposition of 75 edges sums to the full task gradient with relative error ~1e-7. Using **8 fresh samples per checkpoint**, no negative shared-core gradient-pair cosines were observed. This is a final-checkpoint small-sample **diagnostic only**: not a claim of no collisions throughout training. No established collision-driven role separation.
- 24/24 saved checkpoints re-evaluated on fresh/OOD and matched. 24/24 serialized inference packets verified by SHA256 and reconstructed. 24 latency records, 12 full-loop gradient diagnostics. All 24 trainings reached 450 optimizer updates.

## Claims and open problems
**Mechanism YES:** parallelize all virtual-weight decoding/edge computations and joint physical-parameter training *within a Loop step* without expanding independent trained weights. **Mirror-specific gain NO** under this protocol. **Deployment efficiency NO** versus native baseline. A fixed 75-edge graph is NOT arbitrary learnable correspondence or discovered sparse attention. Future experiment must optimize physical→virtual address assignment, shrink enormous seeded direction buffers, test gradient conflicts **during** learning, and compare matched active MAC/actual total bytes and real data. Preserve this negative result.

Source SHA-256 `model.py`: `29fa4f15e97ff399d2ade83c303495d46afee1d466aa3e98cd40071015946bc8`. Train source SHA-256: `e778da820f3318d41208db8dcdaf0e4b9ec0a5e819fb597414db3d45e5ef6757`. Canonical verified CSV SHA-256: `22adf83b3627c3543f44d4728103205b493f477fcebeddd76e59d8e1380a3003`. Full self-contained code/weights/evidence archived in the associated conversation artifact (GitHub branch report alone does not yet contain all checkpoint bytes).
