# CPU experiment batch 2026-10-08 — MA-1171, MA-1183 (original mechanism + public data)

**Scientific status:** Three independent preregistered CPU **pilot series** have completed. This is NOT a completion of the original full native-method MA hypotheses. The canonical worker, main, science registry and original source are not changed.

## Study provenance

- Initial protocol freeze: GitHub commit `5002b45a76d27c8b0e1632e70abb5c3c9d989122`, **before dev/fresh**.
- Four source/tests + native one-forward null frozen **before fresh**: `48a29306ad2c8b38c1cd8778aaa4c130b2d34ecb`.
- MA-1183 real-data separate protocol **before accessing outcomes**: `a87010db95e6b90a23648596d8c06d6210b6bcc7`; source frozen at `da65b039faf8af0551f871e5f9795ca4035ecf0e`.
- AMD EPYC 9V74 virtual CPU, PyTorch 2.10.0+cpu + CPU, numpy 2.3.5, one BLAS/torch thread, FP32 training and FP64 codec algebra. Hardware CPU clocks unlocked. CUDA not present.
- Synthetic dev seeds 11,12,13; fresh whole-task seeds 101..105; real stratified split seeds 301..305.
- **14 original unit tests + 3 real-data unit tests PASS.** Full 180/75/50 result rows reproduced at deterministic fields; CPU clock fields excluded from byte-identical equality checks. This is same-session reproduction, not independent verification.

## A. MA-1171 — exact FQC codec: source basis, Mirror code, quantization and private exception

**Experiment:** Each of four 2D signal blocks has three possible method families (native direct, source-trained linear rank-1 factor, Mirror orientation), three packed bit widths (2/4/8) and private correction on/off, for **18 choices/block** and **18^4=104,976** exact combinations per world × task mixture. There are six independently source-fit linear tasks and six separate support-fit logical target tasks; no audit target weights are used for fitting. Every candidate cost is the **length of a genuine decoded/encoded full binary object**, not an unrounded bit proxy; optimization uses the full cross-block calibration Gram matrix. 512 audit observations per target task.

**Aligned orbit, 80% of full native direct 8-bit budget:**

- budget **111 B** versus native 8-bit **139 B**.
- best native *non-Mirror joint allocator* fresh MSE mean: **0.0046431**; best Mirror-enabled joint: **0.0022250**.
- mean **paired relative reduction 48.90%**, median 53.33%, **5/5** fresh worlds improve. True heldout predictive MSE, not surrogate weights.
- decoded bytes: native 109.0 B, Mirror-selected 110.2 B, both at hard budget. Mirror selected 1.20/4 blocks on average.
- **cold codec reconstitution**: observed median of per-world Mirror/native P95 time ratios **1.56×** (larger is worse); benchmarked Python decoder, not a fused kernel or GPU prediction.

**Negative bounds:** At alpha=0.5, 80% budget Mirror-enable is worse in some worlds; at alpha=0 there is no selected Mirror and no gain. At 100% and 115% of the direct budget, all five aligned fresh worlds select native direct codec without Mirror; strict m-specific superiority vanishes. At 65% no feasible state exists among tested family (do not record an improvement or count it as a model failure). Joint search benefits are not necessarily m-specific because the no-Mirror joint solver uses the same exhaustive enumerator. The aligned teacher is favorable to Mirror by construction. This is a narrow source-trained synthetic mechanism result, not established rate-distortion gain for a neural language model.

**Decision:** Aligned low-budget compression is promising as a *mechanism*; original MA-1171 requires general-source task utility and <=1.10x P95, so the full original claim remains **UNTESTED/NOT ADOPTED**. Direct Python decode slower and the nonaligned/large-budget conditions do not support general superiority.

## B. MA-1183 — train native BatchEnsemble/TabM-style fast weights and Mirror coordinate alternatives

**Protocol:** 32 input, 64 GELU hidden, binary target, K=8 ensemble members; one shared trunk and member-specific rank-one input/hidden gating. Train 300 AdamW updates/method with identical data and base weight initialization. Five methods: full fast weight, no-member shared one-pass (optimized native), fixed 4D linear code, learned 4D linear dictionary, structured 4-angle Mirror chart over a paid shared fast-weight template. 3 task types alpha=0/0.5/1, 5 fresh whole-world seeds = **75 trained model conditions**. This is NOT exact TabM or published TabReD reproduction.

| Method | mean fresh ensemble NLL (nat/example) ↓ | ROC AUC | Actual NPZ bytes | mean CPU P95 (ms) ↓ | mean member logit spread |
|---|---:|---:|---:|---:|---:|
| no_member_shared | 0.81928 | 0.6390 | 10396 | 0.0547 | 0.00000 |
| native_full_fast_weights | 0.89235 | 0.6317 | 13954 | 0.2358 | 0.06644 |
| fixed_seed_linear_code4 | 0.89854 | 0.6303 | 11405 | 0.2913 | 0.00170 |
| structured_mirror_rotation4 | 0.89186 | 0.6316 | 11409 | 0.3465 | 0.02019 |
| learned_dictionary_linear_code4 | 1.07503 | 0.6248 | 13201 | 0.2790 | 0.02316 |

**Matched m vs plain linear code:** same 128 learned member-code/template scalars and almost identical NPZ bytes (11409 vs 11405 B). Mirror improves NLL on 14/15 *correlated* beta-world rows, but **5/5** independent world seeds when averaging across beta. The mean absolute NLL difference is -0.006674 nat/example.

**Strong negative:** Optimized no-member shared model has **better mean NLL, lower total bytes and much lower CPU P95** than both Mirror and native full fast weights. Mirror P95/native-full ratio in this toy CPU implementation is **1.47×**. All synthetic NLL are high despite >0.6 AUC, signaling late optimization/calibration weakness of the fixed 300-step protocol. The common native TabM mechanism itself is not enough to establish ensemble gains here.

**Decision:** the strict original MA-1183 Pareto/adoption gate **FAILS in this synthetic fixture** (not a general theorem). The original MA stays UNTESTED because its official native implementation/TabReD benchmark was not run.

## C. MA-1183 real public tabular labels, separate frozen validation-checkpoint study

Two dataset families bundled with sklearn 1.8: original binary breast-cancer diagnosis (569 cases; 30 features) and binary Wine class 0 vs rest (178 cases; 13 features). Per dataset: five overlapping stratified test partitions of 30%; standardize from training only; 20% of the remaining training partition is dev validation; choose the best of steps 25/50/100/150/200 based ONLY on validation ensemble NLL. No test-driven checkpoint tuning. 50 independently trained runs; splits on the same two datasets are **not 10 independent task worlds**.

| Data | Method | Mean test NLL ↓ | Mean ROC AUC ↑ | Mean CPU P95 ms ↓ |
|---|---|---:|---:|---:|
| breast_cancer | no_member_shared | 0.06311 | 0.9973 | 0.0446 |
| breast_cancer | native_full_fast_weights | 0.06086 | 0.9972 | 0.3244 |
| breast_cancer | fixed_seed_linear_code4 | 0.05838 | 0.9977 | 0.3720 |
| breast_cancer | structured_mirror_rotation4 | 0.06031 | 0.9974 | 0.8715 |
| breast_cancer | learned_dictionary_linear_code4 | 0.05964 | 0.9972 | 0.3419 |
| wine_binary | no_member_shared | 0.03432 | 1.0000 | 0.1016 |
| wine_binary | native_full_fast_weights | 0.02833 | 1.0000 | 0.3116 |
| wine_binary | fixed_seed_linear_code4 | 0.04728 | 1.0000 | 0.3829 |
| wine_binary | structured_mirror_rotation4 | 0.04597 | 1.0000 | 0.4237 |
| wine_binary | learned_dictionary_linear_code4 | 0.04247 | 1.0000 | 0.3393 |

On breast cancer, the Mirror-coded ensemble is close to the native-full member NLL but is slower and slightly worse on average than the same-byte fixed-linear-code baseline. On the wine binary task, native full fast weights have better mean NLL than Mirror, and both are slower than the no-member single path. Because the two datasets are small and their five splits overlap, these are **descriptive internal feasibility checks**, not generalization proofs about real tabular tasks.

## D. Gates, ERROR CHECK and interpretation

- **Freeze before fresh:** scripts and protocol committed before fresh synthetic evaluation. A separate real study freezes dataset transformations and code before the first real test result.
- **No test leakage:** the codec fits 6 target codes only from 96 support observations/task; source bases from independent 128-example source tasks; heldout source world and audit X are never used in fitting. Tabular scaler fits only train, checkpoints selected only by validation labels.
- **Physical bytes:** true custom FQCM1 bitpacked + per-block scale/basis/private metadata and decoder roundtrip for MA1171; deterministic NPZ with shared trunk/fast weights/config for TabM-style MA1183. Training source-basis setup cost is reported separately and not claimed free.
- **Strict native controls:** MA1171 exhaustive *no-Mirror joint* and fixed-method optimizer; MA1183 full rank-one fast weight, efficient no-member baseline, learned/fixed linear factor banks. **Neither source reproduces a full author algorithm (original TabM/LoGo/Compress-then-Serve); do not publish such a claim.**
- **Uncertainty:** independent synthetic world count 5; three beta levels within each seed are correlated. Real partitions overlap. No valid 95% population CI from n=5. CPU virtual clock not locked. P95 varies due system jitter; compare paired shape and disclose constraints.
- **No LLM/real GPU/VRAM/production claims.** Historical MA registry/worker queue statuses remain fixed. This intake is separate research evidence; use the original full MA plans for a true promotion.
- **CPU negative results are valuable:** Mirror m is not automatically useful; its value can be limited to aligned low-budget models, and native heads/one-pass computation may dominate.

## Files and reproducibility

- [Frozen MA1171 protocol](MA1171_FROZEN_PROTOCOL.json), [TabM-style synthetic protocol](MA1183_FROZEN_PROTOCOL.json), [separate real-data protocol](MA1183_REAL_TABULAR_FROZEN_PROTOCOL.json).
- [Source/test code](source/) — fully runnable CPU code; unit tests after environment setup.
- [All raw results](results/) — per-world dev/fresh and real stratified rows; three compact results cores; [verification manifest](results/VERIFICATION.json).
- Reproduce deterministic portions with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/ma1171_joint_codec.py --phase fresh` and `python source/ma1183_tabm_style.py --phase fresh`, using separate output directories. Clock fields are expected to vary.
