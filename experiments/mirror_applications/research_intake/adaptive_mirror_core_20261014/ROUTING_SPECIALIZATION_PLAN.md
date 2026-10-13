# AMA-002-R — Mirror View の入力条件付き選択・負荷分散・機能専門化（研究計画）

**状態：PLAN ONLY / NOT RUN。2026-10-14 JST。新規MA番号なし、既存科学判定の変更なし。** これは既存AMA-001の開発結果で判明した「ルータ不在」を埋める独立サブ実験。元AMA-001のFresh seed 5201–5205は未使用のまま維持する。コード・実行・結果のない計画を測定成果としない。

## Background — 実装を読み直して分かったこと

ローカルで監査したAMA-001 \`source/model.py\` では、\`SharedMirrorBank.forward(x)\`が全タスクについてあらかじめ学習した\`angles[task,plane]\`を必ず適用して\`[batch,task,out]\`を出力し、入力×状態に応じたMirror expert選択ルータはない。専用\`private_gate.sigmoid()[task]\`は1タスク1スカラーであり、「複数Viewからの選択」「Top-k実行」「学習中の負荷分散」「物理的なinactive-view削除」のいずれでもない。したがって現在の結果だけではMirror専門化とroutingの価値を測れない。

過去の**SRM001**ではルータ学習は実施済み。単純な均等化はView利用率を高めたが品質を改善しなかった。**MN006**でも強制的な学習初期均等使用で品質改善なし。したがって、ロードバランス自体は目的ではなく、十分な学習機会＋機能分化を両立させる必要がある。

## Scientific objective / hypotheses

H1 — **適切な機能選択**：同じ入力・タスク条件により異なるMirror Viewを使用する場合、入力条件付きのrouterが固定View、均等混合、静的task codeよりもheldout task lossを下げる。

H2 — **学習機会の保証 vs 真の専門化**：初期の負荷均等化がstarved Viewを減らしても、それだけでは機能分化・予測改善を保証しない。usage均等化、route→regime対応、View機能差、worst-task品質を別々に測る。

H3 — **専門化を妨げない負荷配分**：データの状態分布が90/10のように偏る条件では、最終routingを均等に固定するより、初期のみ弱く負荷を均等化し、後半はタスク予測損失を優先する方がよい可能性がある。

H4 — **Mirrorの容量境界を保護**：Orbit外の入力・機能について、routing先を常にMirrorに限定するのではなく、共有のみ（identity/no-view）または有料privateへ退避できる必要がある。品質・保存量・速度の同時評価は必須。

H5 — **Mirror固有性**：通常のshared experts + native FiLM/LoRA、native group-rotation bank、bias-balanced MoE、Expert Choice等の強い代替法より優れない限り、MoE routing一般の利得をMirrorの新規性と呼ばない。

## Architecture / dimensional audit

入力\`x\`とtask id \`t\`をshared backboneでhidden\`h_theta(x) in R^d\`へ変換する。E=4個の共有Mirror作用 \`M_e(h,m_te)\`、同じshared FFN、一つの入力条件付きルータ \`s_e(x,t)\`、任意のprivate fallbackを使用する。初期d=16、task K=8、各Mirrorは学習可能な2つの回転面（角度及びplane axes）、一度の共有trunk forwardを保持。

\`\`\`math
p_e(x,t)=softmax(s_e(h_theta(x),t)/tau),
S(x,t)=TopK_e(p(x,t), k=2).
y_t(x)=F_shared(h_theta(x)) + sum_{e in S(x,t)} w_e(x,t)*DeltaMirror_e(h_theta(x),m_{t,e}) + g_private(x,t) E_t(h_theta(x)).
\`\`\`

\`e=0\`にno-Mirrorのゼロ残差候補を入れる場合は、計算/容量報告でその候補を明示し、fallbackの割当も集計する。\`DeltaMirror_e\`は\`R_e\`を活性化の前段に適用した機能残差で、\`R_e\`が違っても出力が同じなら容量0。バランス制約は**学習中のactive Mirror候補**にだけかけ、no-view/privateへ必要な退避を妨げない。

**重要：usage均等 != 機能専門化。** E=4の利用率がすべて25%でも、4 Viewが同じ出力を出せば独立機能は増えていない。

## Method controls — predefine and attribute the gains

1. **Static AMA-001:** no dynamic route; all tasks have fixed persistent codes, identical trunk.
2. **Router-uniform:** train all views equally, inference equal soft weights; optimization/data-budget control.
3. **Input conditional soft MoE, no balance:** architecture+router primary reference.
4. **Input conditional soft MoE + mild Switch-style load-balance:** \`lambda=0.01\` *relative to normalized auxiliary*; compare \`lambda=0,0.01,0.1\` on calibration only (no fresh tuning).
5. **Warmup-only balance:** same mild loss for first 25% of optimizer steps, then anneal to zero. No forced equilibrium at inference.
6. **Loss-free expert bias (DeepSeek-style):** update expert selection bias from assignment counts, without routing auxiliary gradient to task loss; account for state and selection overhead.
7. **Expert Choice / per-view training quotas:** first-stage capacity-limited balanced exposure; task-dependent route selection is still evaluated on heldout.
8. **Hard uniform every batch** as an explicit potentially harmful negative control.
9. **Native matched-capacity FiLM/LoRA/shared-MoE top2** and task-specific private route (the usual strong attribution controls).

For Top-2, report \`f_e=count(selected expert e)/(N*k)\` and \`P_e=mean_x p_e(x)\`; normalized load-balance \`L_bal=E*sum_e f_e*P_e\`. Non-differentiability of hard\`f\` requires gradients through \`P\`; do not silently treat a hard counter as differentiable. Adaptive balancing bias and fixed quota are separate methodologies. Use physically measured NPZ, including router, Mirror bank, all codes, bias state, masks and private capacity, and measured batch1/64 runtime. Compare same training examples and updates; include all active Mirror forward work performed during soft routing.

## Dataset split and contamination barrier

Use **new synthetic worlds** with known mode/regime assignments used ONLY for diagnostic audit: (i) balanced four modes, (ii) skewed 70/20/8/2 four modes, (iii) incomplete shared modes with truly private out-of-orbit exceptions, (iv) unrelated task functions. The input contains signals that can determine the regime but no explicit regime/oracle expert IDs are passed to the student. Separate labels for forward tasks from task IDs and hidden generator mode IDs; learner only gets permitted (x,t,y).

Calibration seed=7300. Three dev seeds 7311/7312/7313. **Five fresh seeds 7411–7415 remain unopened until a new, code-locked pre-registration passes every development gate.** The original AMA-001 worlds/seeds are NOT reused for tuning. Digits as an optional externally-derived 8-binary-task screen after the synthetic test, using image-ID split and *not* counting as independent datasets. The numbers are proposed allocations, not observed results.

Training: PyTorch CPU 1 thread, FP32, AdamW; initial 500 updates and 1000-update same-budget continuation for promising native/candidate pairs. Calibrate exact learning rate for each method within a matching search budget and temperature \`tau\` schedule; do NOT vary based on dev or fresh. Stepwise route annealing 1→0.2 after warm-up is a predeclared option; the no-anneal control is mandatory. Freeze science source, data-generation hash and all gates BEFORE development data is opened.

## Evaluation metrics and preregistration targets (do not retrofit)

**Quality:** heldout mean + worst-task nMSE or BCE; per-regime quality, unseen combinations, off-orbit preservation vs independent/native private upper controls.

**Utilization:** each Mirror's effective fraction of token/task selections, mean predicted probability, effective expert count \`exp(H(f))\`, zero-use fraction, rare-view training-gradient norm, and per-group exposure counts.

**Specialization:** conditional mutual information between route and hidden true regime (audit only), route/target overlap under permutation-invariant matching, pairwise functional logit differences on *identical* inputs, ablation impact of each View on its routed subgroup, and number of effective independent behaviors. High routing entropy alone is NOT PASS.

**Mechanism/dev gate:** on balanced shared-mode worlds, at least 2/3 dev seeds with input-conditioned router test nMSE < static/soft-uniform baseline, no view starving <5% exposure during warmup, and nontrivial specialization above random regime-routing after label-permutation audit. Load-balancing route must improve on unbalanced/no-balance in >=2/3 without worsening worst-task error >0.02 absolute nMSE. If plain no-balance already performs best, mark balancing itself NOT HELPFUL.

**Skew robustness:** forced uniform may not outperform; the adaptive/warmup-only route must match no-balance within +0.01 nMSE and preserve rare-regime quality. If standard skew naturally requires ≠uniform final load, forcing 25% is NOT required.

**Safety/efficiency:** unrelated/out-of-orbit world worst-task loss ≤native shared+private +0.02 and full lossless inference NPZ bytes ≤native matched quality bank ×1.05; eager batch64 runtime ≤1.25× native. If these are not met, no general efficiency or protected-sharing claims. Independent outputs from genuinely different task functions must not be inferred from combinatorial route counts.

**Gate for Fresh:** Quality, genuine specialisation, worst-task, bytes and compute all pass on dev, plus tests/serialization. Otherwise report \`FAIL_DEV_GATE\` and keep fresh sealed. Any design corrections after seeing dev require a separately versioned follow-up and NEW seeds.

## Implementation/test plan (container only)

- Task 1: Write \`source/conditional_router.py\` with \`MirrorRouter(nn.Module)\` taking hidden+task and returning probs, topk indices, gate weights and telemetry. Write \`tests/test_router.py\` FIRST; fail on missing import, then pass on probability normalization, backward gradient, no-view, hard/soft dispatch equality, selection entropy and top2 conservation.
- Task 2: Write balanced/long-tail mode worlds in \`source/route_worlds.py\` and \`tests/test_route_worlds.py\`; disjoint train/val/test IDs, hidden generator secrets inaccessible at training, independently generated new seed worlds. Monitor load but do not use oracle regime labels as supervision.
- Task 3: Implement all compared routing/gate policies in \`source/route_model.py\` + \`tests/test_routing_models.py\`. Confirm that \`lambda=0\` shares exact initialization/training schedule, load-balance grads flow into router, and annealing sets λ=0 after warmup. Verify no task head copies hide route collapse.
- Task 4: Implement full checkpoint NPZ serializing both route and Mirror bank + \`tests/test_router_codec.py\`; roundtrip exact logits, per-expert counts and no hidden teacher parameters. Report eager/folded batch1 and64 and training MAC/forward count.
- Task 5: Freeze source/protocol hash on a *new research branch*; run ONLY calibration and dev in container. Verify all output metrics by reloading saved payloads, at least one fully deterministic retrain, and publish H/T/D/C/U report. Do not edit canonical registry or main without evidence.

## Relevant prior reports

- \`docs/phase2/SRM001_SHARED_RULE_MOE.md\` — learned route/supervised route and *simple load balancing failed to improve quality*.
- \`docs/phase2/RESEARCH_STATE_THROUGH_MN007.md\` — MN006 forced early balance did not resolve apparent interference.
- \`docs/phase2/RESEARCH_STATE_THROUGH_MN010.md\` — routing must observe task-relevant temporal context.
- AMA-001 local \`source/model.py\`, \`results/DECISION.json\`: all 8 task Views always evaluated; only scalar private gates; unrelated-world quality and byte/runtime failed.
- Zhou et al., Mixture-of-Experts with Expert Choice Routing, NeurIPS 2022, https://arxiv.org/abs/2202.09368
- Hazimeh et al., DSelect-k, NeurIPS 2021, https://arxiv.org/abs/2106.03760
- Wang et al., Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts, 2024, https://arxiv.org/abs/2408.15664

**Scope:** This file is a research protocol plan; no experiment was started by creating it, no new observations and no new official MA ID.
