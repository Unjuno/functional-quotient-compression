# LME01 — LiME-style ONE shared PEFT output plus K role Views (2026-10-09)

**Scope:** rigorous small **sklearn digits** trained multi-output mechanism ONLY. LiME paper is **not** reproduced here (original source and zero-parameter router not run). Original author method: https://proceedings.mlr.press/v306/kowsher26a.html ; published source https://github.com/Kowsher/LiME.

**Frozen before data:** [protocol](../../LME01_FROZEN_PROTOCOL.json), commit `7ee8674e5eb305296a953c66bdab0daf384e8010`; source pinned to commit `ebd7b02c23df8aa7afd3625ae55a4de94fabd97e` after 3 development worlds but **before** fresh worlds; Git blob hashes match local source. No hyperparameter selection from fresh.

## H — single physical forward, useful distinct outputs, marginal short-m value

Given one learned trunk Fθ and a single shared rank-8 PEFT output δ(Fθ(x)), learn K=5 distinct supervised output logits by cheap per-role output operators. Incremental Mirror scientific gate requires improvement >=.01 nat/example over same-four-coefficient native linear modulation in >=4/5 fresh splits, and quality within +.02 nat/example of LiME-style full direct vectors with suitable bytes and P95. One-forward operation itself is already native LiME and normal multihead, NOT novel.

## T — actual execution

- Inputs: each 8×8 normalized handwritten digit image x is shared identically between all five targets; parity, >=5, pixel left-right balance, top-bottom balance and center intensity. The last three thresholds are derived ONLY from training image IDs.
- 64→128 GELU→32 GELU trunk, rank8 shared adapter 32→8 GELU→32, one shared 32→2 classifier where compatible. All 7 methods execute the trunk and adapter **exactly once** per output batch. Ordinary native multi-head has 5 separate 32→2 readouts on the same h+δ, not 5 trunks.
- Methods: shared broadcast, native 5 ordinary output heads, LiME-style direct 32-vector per-role scaling, ordinary fixed 4-code scale, learned 4-code shared dictionary, structured 4-angle rotated shared δ, and structured 4-code shear shared δ. Exact native Givens output is the same function as 'Mirror rotation' (M0).
- Training: 350 AdamW updates, batch128, LR .002, wd .0001, FP32. Same initial trunk, adapter, classifier, same sampled image index stream in each world. Three dev worlds [71, 72, 73]; unseen fresh worlds [701, 702, 703, 704, 705]; no audit tuning. Each world is a 75/25 stratified split of the **same original dataset**, so worlds are correlated, not five independent natural task datasets.
- Real serializer: deterministic named NPZ includes trunk, adapter, role codes/head weights, full dimension, seed-defined basis-reconstruction metadata, normalization and thresholds. Reported bytes are true artifact lengths; no theoretical Shannon compression implied. Single-thread CPU eager P95 at B128 with 20 warmups/80 timed. No GPU/VRAM conclusions.

## D — measured fresh results

| method | mean_test_nll | mean_worst_task_nll | mean_test_accuracy | mean_serialized_bytes | median_cpu_p95_ms | role_trainable_scalars |
| --- | --- | --- | --- | --- | --- | --- |
| shared_broadcast | 0.576445 | 0.751072 | 0.698133 | 55794.600000 | 0.140657 | 0 |
| native_onepass_linearheads | 0.084552 | 0.132069 | 0.970489 | 56852.600000 | 0.184437 | 330 |
| lime_full_diagonal | 0.182913 | 0.286394 | 0.931289 | 56960.600000 | 0.155532 | 170 |
| lime_linear4 | 0.223934 | 0.344162 | 0.911200 | 56394.600000 | 0.162356 | 30 |
| lime_learned_linear4 | 0.223978 | 0.327934 | 0.915022 | 57166.600000 | 0.174038 | 158 |
| mirror_rot4 | 0.228885 | 0.331339 | 0.910489 | 56393.600000 | 0.245976 | 30 |
| mirror_shear4 | 0.276788 | 0.422737 | 0.883822 | 56395.600000 | 0.245172 | 30 |

**Paired rotation minus ordinary linear 4-code test NLL (negative favors rotation):** 701:+0.057720, 702:-0.124118, 703:+0.027560, 704:+0.039427, 705:+0.024164.

Native 5-head is the strongest overall task-quality method at mean NLL 0.084552, not a K-recomputed strawman. Mirror rotation mean 0.228885; native fixed 4-code linear scaling mean 0.223934; native LiME-style full-vector direct scaling mean 0.182913. Mirror rotation beats linear4 by >=.01 only in **1/5** new worlds, rather than preregistered >=4/5. Mirror shears did not compensate. **Scientific gate FAIL/M0.** No MA status change.

## C — negative control and explanation

- The original LiME method already computes one shared PEFT module and cheap expert modulation, plus routing. This pilot strips router because task roles are supervised; it is strictly not the full native method.
- Native regular Givens output head reproduces the Mirror4 mathematical family exactly (unit test), so even a future Mirror4 win against plain scaling would require a second functional/byte mechanism to establish Mirror-specific novelty.
- A 4D coordinate may not span five real binary tasks; a standard native multihead has 330 task-specific scalars and attains much lower loss, while a 32-vector LiME-style p_e provides additional degrees of freedom.
- Fast-kernel effects are different: repeated sin/cos, rank mix and noncontiguous view tensor costs can outweigh a few saved scalars on CPU. Training sample/task correlation and unlocked CPU clock are additional limitations.

## U — uncertainty, validation and provenance

- 5 correlated partition seeds = descriptive evidence, not a 95% population CI. Numerically exact replay of 2 fresh worlds × 7 methods × 11 deterministic fields gave **154 / 154 identical comparisons**, with 0 differing fields. P50/P95 are excluded from bitwise replay due to unlocked CPU clocks. Eight unit tests pass; Givens-mirror output, norm preservation, role-specific variation, all methods' gradient validity, one trunk+one adapter call and true serialization all checked.
- Files: [source](source/run_lme01.py), [tests](source/test_lme01.py), [dev raw](results/dev_raw.csv), [fresh raw](results/fresh_raw.csv), [core results](results/RESULTS_CORE.csv), [verification](results/VERIFICATION.json), [replay verification](results/REPLAY_VERIFICATION.json). Full logs and seeds retained.
- `U_exp=k_cov*u_c` with k_cov2 is not a coverage-certified bound for n5; sources of error are correlated task labels, source data partition, optimizer, clock and task geometry. Clock ms and file bytes cannot be combined or called a real-LM compression ratio.

## Next test, but no fabricated results

Reproduce **native original LiME 47-task MMT-47**, with its parameter-free router and AutoTopK; repeat Mirror m on expert vectors/codebook and compare against **same-byte native LiME linear/vector compression and native one-trunk multiheads**, plus a true holdout of independent task families. Alternatively test M3LoRA mixer codes with its native minor-SV initialization. Do not promote this CPU pilot to MA-1189 verified status.
