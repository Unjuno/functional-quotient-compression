# Direct 2026 Native-Method Research Sweep — 2026-10-09

## Discovery and adverse evidence

**LiME (PA463)** — https://proceedings.mlr.press/v306/kowsher26a.html; https://github.com/Kowsher/LiME. **One shared PEFT** + per-expert output scaling + parameter-free routing and AutoTopK are published mechanisms. Its real MMT-47 study reports up to 4× fewer trained parameters and 29% faster training than their MoE-PEFT controls; those are **author-reported native results**, not tested by us. The important Mirror delta is the value of a short structured m **over native LiME and a same-byte simple code**, not over K independent transformer passes. New MA-1189 with source-only 5-task CPU LME01 ablation negative.

**M3LoRA (PA464)** — https://doi.org/10.1049/cit2.70144. Multiple learned low-rank A/B subspaces, a trainable mixing matrix and minor principal SVD direction initialization are native M3LoRA (2026). Only ask whether task-specific native **mixer L** compresses further with small structured m beyond an ordinary source-SVD/linear mixing bank. New MA-1190; supplement MA-1167 and MA-1178.

**LoDA (PA465)** — https://proceedings.mlr.press/v306/he26f.html. Separates task-shared and genuinely private LoRA update directions; uses source-task-driven projection energy and Gradient-Aligned Optimization plus native shared-component recalibration. This is a stronger M0 baseline for any Mirror+private scheme such as MA-1178, MA-1105 and MA-1190. No new MA solely for rebranding LoDA.

**FAAR (PA466)** — https://openaccess.thecvf.com/content/CVPR2026/html/Fontana_FAAR_Efficient_Frequency-Aware_Multi-Task_Fine-Tuning_via_Automatic_Rank_Selection_CVPR_2026_paper.html. PDRS automatic per-location/per-task rank shrinking, spatial frequency-aware TS-PD task decoder are already native. New MA-1191 on residual task-spectrum m, plus MA-1185 native control update. Native hard comparators include ranked LoRA and cheap task band codes.

**MoEP (PA467)** — https://doi.org/10.1016/j.neunet.2026.109617. Native whole Attention–FFN layer routing, not ordinary sparse FFN MoE; native study notes potential weak scale transfer (Pythia-1B), so record active attention and cache invalidations. New MA-1192 and native MA-1176/241 controls.

**DR-MGF (PA468)** — https://doi.org/10.1109/TPAMI.2025.3635844 and author arxiv https://arxiv.org/abs/2305.19844. Native learns task-preferred per-filter inference paths and optimizes meta-weighted task gradients to reduce sharing conflicts. New MA-1193 with compression m on *already learned* route/importance state, not new optimization claim.

**PCGrad (PA469)** — https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html. Project conflicting gradients per task; direct **optimizer-only M0** before crediting architecture for easier multi-task training.

**CAGrad (PA470)** — https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html. Balances average loss and worst task direction; compare fixed optimization steps/time and task quality, source-only route calibration.

## Observed LME01 — a distinct CPU mechanism study, not LiME native benchmark

Preregistered [protocol](LME01_FROZEN_PROTOCOL.json) before any dev/fresh opening; [source](pilots/lme01/source/run_lme01.py) fixed in GitHub between dev and fresh. Five binary supervised functions from ONE same physical 8x8 image, a common trunk and rank-8 adapter each executed once, simple LiME-style known-role modulation and structured Givens/shear4. Native five-output head **0.084552** mean fresh NLL, native full 32-vector output modulation **0.182913**, ordinary linear4 **0.223934**, Mirror Givens4 **0.228885**, Mirror shear4 **0.276788** (nat/label). Mirror improves at least .01 vs linear4 in **1/5** fresh worlds, not preregistered 4/5. Native one-trunk multiple heads is more accurate; Givens is a perfectly valid native output-head parameterization so an isolated Mirror-vs-linear improvement could still be M0. See [complete result/report](pilots/lme01/REPORT.md).

## Experimental contracting and queue policy

Registered MA-1189..1193 as UNTESTED. They each have native physical object + minimal m insertion, teacher and withheld task generator, source-only learning steps, original paper control, same-code ordinary linear factor, full real serializer and run-time accounting, explicit PASS/FAIL/UNCERTAIN and uncertainty note. Existing MAs receive supplements without status changes. Worker MA-255 is not displaced; branch isolation preserved. No adoption or 46× model compression claim from syntax or synthetic result.
