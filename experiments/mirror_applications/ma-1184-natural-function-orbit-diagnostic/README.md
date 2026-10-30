# MA-1184 — Natural-function Mirror-orbit feasibility (未訓練)

**2026-10-31 / Status: UNTESTED / documentation-only candidate.** This file is **not a frozen experiment protocol** and reports no new model-training results.

## Mirror insertion — what is being tested
This experiment adds a low-description task/view parameter `m` to a **single physical shared nonlinear MLP**. It applies a learned Householder coordinate change before a nonlinearity and the inverse change afterward, rather than storing separate full-width MLPs. The candidate seeks **functional reuse of shared nonlinear structure across independently generated, independently pretrained task functions**. It does not add MoE, Loop, a router, or a private path.

The central task is **diagnosis before expensive joint training**: given several functions *not constructed from Mirror transforms*, estimate whether a shared Mirror orbit is a useful approximation. This is a different question from MA-1179's warm-start distillation optimization, and from MA-1183's Mirror-generated positive-control teacher family.

## Evidence motivating candidate (not evidence of success)
- MA-1183 preregistered partially aligned ReLU gate **0/5 FAIL** against FiLM, coupled-rank1, and small independent MLP; its aligned teacher positive control favored Mirror over FiLM but did not beat the encompassing coupled-rank1.
- A separately frozen **posthoc exploratory** gamma sweep (140 models) found paired geometric-mean Mirror/FiLM Fresh NMSE ratios 0.7629 at gamma 0.1, 0.9412 at 0.2, 1.1267 at 0.3. These numbers are **not a universal transition boundary**.
- MA-1182's absolute CE gain threshold was unachievable because the reference error was near zero. MA-1184 must confirm discriminative baseline error **before** the confirmation evaluation.
- Householder with direction u is a special case of a coupled rank-one conditioner at coefficient beta=-2. The Mirror hypothesis is **better inductive bias / code economy / robustness**, never strict function-class inclusion.

## Exact comparison target
Construct independently pretrained ordinary MLP teachers for **related, partly related, and unrelated** analytic task families. Teachers in these three primary worlds must **not** be produced by Mirror or Householder transforms. A separate synthetic Householder-aligned teacher is permitted only as a labeled positive mechanism control.

For each teacher, fit shared base from training task functions; freeze its weights; then fit **task-specific conditioner codes only** on allowed calibration examples. Measure functional loss on independent evaluation examples and across *new task identities*. Do not pass teacher weights or task-generation coefficients to a student. Follow with a separately labeled shared-core+code joint-training arm only if diagnostic L1 is informative. Compare a baseline with **identical source-task information**.

## Strong non-Mirror controls
1. Shared unconditional nonlinear MLP.
2. Matched-code FiLM (per-view diagonal gate/scale).
3. Coupled-rank1 with a free beta per view (**strictly contains single Householder**, compare oracle-quality and learned-quality separately).
4. BatchEnsemble-style task modulation, with all extra code state counted.
5. Near-byte independent small MLPs, and unrestricted independent full-width upper control (**not equal-byte**).
6. Fixed/random Mirror (learned-direction ablation).

Compare ReLU and SiLU as separate trials. The baseline compute and activation counts must be explicit. No output activation/routing oracle should give Mirror extra information compared with controls.

## Falsifiable hypothesis
**H1 (feasibility):** Some non-Mirror-generated but related task families are well approximated by a small shared-core + Mirror-code representation **in function space**.
**H2 (selectivity):** A diagnostic determined solely from held-out-function-independent source/calibration splits predicts whether Mirror outperforms FiLM on a *new task identity*, and avoids declaring unrelated tasks Mirror-compatible.
**H3 (Mirror-specific practical benefit):** For qualifying natural-function families, the learned Mirror student improves the sealed task-domain quality / actual serialized storage / active runtime frontier against **both** FiLM and enclosing coupled-rank1. H1 does not imply H2 or H3.

## Separation of evidence lanes and stop rules
- **L0 (pilot; no Fresh):** demonstrate correct construction, no teacher leakage, finite gradients, parameter budget, and a non-saturated/non-trivial quality gap. If candidate tasks saturate or a planned threshold is impossible under nonnegative loss, terminate or revise in a **new draft before freeze**.
- **L1 (fixed-core orbit-fit; source-task and DEV-only):** fit codes from allowed function observations, compute error relative to shared-only / independent reference. Do not call a finite-optimizer fit a global mathematical oracle. Check code-multiple-start and additional optimization budget without selecting on Fresh.
- **L2 (new task identities; sealed confirmation):** Only after a separately frozen protocol and L1 preflight, score new teacher functions and OOD. Never reallocate unsuccessful worlds or adjust arm/threshold using L2.
- **L3 (follow-up, not part of MA-1184):** if L2 justifies it, register **a separate MA ID** for Mirror+small Private vs FiLM+Private and coupled-rank1+Private. Keep the original L2 sealed.
- If the model is not compatible with Householder geometry, report negative findings. Do not reinterpret a constructed aligned positive control as a natural-function finding.

## Resources / measurable outputs
Record actual `torch.save` inference bytes including task codes, caches/state, indices, routers and codebooks; parameters and training examples; forward/backward arithmetic proxy and special activations; model quality (macro NMSE per task + tail risk); per-model CPU/GPU latency with hardware/clock/precision/batch; and replays/hashes. All reported percentages must specify denominator and whether ratios are paired-seed geometric means or medians.

## Controls on uncertainty
Use independently generated task identities (not only a new random split of the same teacher), paired seed worlds, separate DEV vs Fresh selection, sealed checkpoint+manifest, and seed-level log-ratio variability. Put uncertainty intervals on an externally held-out task set; a small n=5 developmental estimate is not a definitive 95% CI.

## Key references / nearby results
- [MA-1183 main report](../ma-1183-native-function-sharing/REPORT_JA.md) and [posthoc sweep](../ma-1183-native-function-sharing/exploratory/align_sweep/EXPLORATORY_REPORT_JA.md).
- [First principles and shared/private roadmap](../../../docs/phase2/MIRROR_GEOMETRIC_ORBIT_AND_PRIVATE_NEXT_STEPS_2026-10-30_JA.md).
- Yuan et al., **HRA**, NeurIPS 2024: https://arxiv.org/abs/2405.17484
- Wen et al., **BatchEnsemble**, ICLR 2020: https://arxiv.org/abs/2002.06715
- Fu & Cohen, **Conic Activation Functions**, PMLR 2024: https://proceedings.mlr.press/v285/fu24a.html

**Research integrity:** No training is authorized by this document alone. `PROTOCOL_DRAFT.json` is provisional, and a new frozen `PROTOCOL.json` with exact dataset/source/hashes/decision gates must be committed **before** training. Preserve MA-1179–1183 FAIL verdicts; do not merge this branch into main.
