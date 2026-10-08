# MA-1170 — NeuroLoRA dynamic gate versus structured Mirror online coordinate
**STATUS: UNTESTED, DESIGN ONLY.** Independent branch `research/mirror-isolated-protocols-rebased-20261008`, never current worker queue. No new scientific result is claimed.

## Variables table / unit check
| 記号 | 意味（日本語） | 単位(SI) | 定義・型・範囲 |
|---|---|---|---|
| m | Mirror機能コード | 1 | 実k次元有限ベクトル |
| k | View次元 | 1 | 正整数スカラー |
| E | 専門家数 | 1 | 整数>=2 |
| z_t | token/時刻ごとの選択expert | 1 | カテゴリ値1..E |
| h_t | 過去文脈状態 | 1 | 有限実ベクトル、未来入力禁止 |
| G | native/context変換 | 1 | 正規化実行列または実ベクトル |
| W | frozen共有線形作用素 | 1 | d_out×d_in実行列 |
| B_i | 共有Mirror View基底 | 1 | Wと同形状 |
| S | 推論保存容量 | byte（実用単位） | 全基底/router/code含む非負整数 |
| t | 推論実時間 | s | 非負実数、同期した時計測定 |
| L | token平均NLL | nat/token（SI無次元） | 非負実数 |
| u_seed,u_eval,u_num | 不確かさの成分 | Lと同単位 | 非負実数、共分散確認 |
| u_c | 合成標準不確かさ | Lと同単位 | 非負実数 |
| k_cov | 包含係数 | 1 | 正実数、説明用2 |
| U | 拡張不確かさ | Lと同単位 | k_cov×u_c |
Dimension check: m_i×B_i has the same dimensionless normalized-weight units as W and can only be added when both are conformable. Specialist routing z_t and hidden history h_t are causal per-token data, not extra precomputed oracle task IDs. L is dimensionless nat/token; S is serialized bytes and latency t is SI seconds; never add or conflate those measurements.

## H — falsifiable hypothesis
On unseen hidden-domain switch sequences, a signed/orthogonal m conditioned on only allowed causal history improves fresh-domain cross-entropy at least 1% relative to the best same-byte non-Mirror extension for >=4/5 worlds, while preserving old-domain retention, total state bytes <=1.05 native NeuroLoRA and P95 <=1.10 native.
Related existing MA IDs: MA-288, MA-401, MA-403, MA-849.

## Mirror insertion
> **Mirror insertion:** supply `m` at **native NeuroLoRA frozen sparse random feature projections, original contextual modulator, and contrastive orthogonality loss**, through **additional history-generated small signed/permutation/Givens m after the native gate in a shared frozen projection space**, so temporarily specialized functions without per-domain permanent experts can be realized without a full independent expert/online controller state per task.
- Native method (must preserve): NeuroLoRA context gate + orthogonality, and FlyLoRA-style fixed projection
- Mirror marginal benefit must be measured *against* native behavior, not attributed to the native router/gate.
- Only source-task learned basis and permitted support examples may form m.

## T — independent runnable experiment plan
**Synthetic fixture:** Source six nonidentical synthetic online domains on a 32- or 64-dimensional frozen MLP, 4 source train domains and 2 unseen; transitions at 32/64 events, 5 fresh hidden switch patterns. No test-time oracle domain ID or privileged target label.
**Controls:** 1. fully native NeuroLoRA with original learned dynamic scale and contrastive loss
2. FlyLoRA sparse projection without context
3. native gate plus byte-matched diagonal/IA3 extension
4. small generic low-rank context head
5. independent task LoRAs upper reference
**Exact implementation steps:**
1. Reproduce fixed sparse random feature projection with the original contextual neuromodulation gate and contrastive orthogonality loss.
2. Freeze source/new domain identities and all hidden switches; no audit task identity at inference.
3. Compare native gate, FiLM/IA3 diagonal, generic low-rank conditional head and structured short signed/Givens m at exact byte budgets.
4. Select m dimension k in {2,4,8} on source/dev only, frozen before fresh.
5. Run five fresh switch patterns, log steps to recover CE, old-domain retention, online state write bytes and compute.
6. Report a negative result if native gate or same-byte ordinary extension matches task utility and P95.
**Data and split:** Dev random seeds 11/12/13 may select LR/rank/strength. Fresh seeds 101/102/103/104/105 with heldout whole role/task streams; freeze before opening. No audit labels, target trained delta, future route or task ID may tune m. Execute all five after dev gate, no favorable early stopping.
**Prerequisite / environment:** CPU torch float32 mechanism first; optional CUDA real MMLU/GSM8K/ScienceQA once authors' code/checkpoints confirmed. Record batch, hardware clock, live writes.
**Measurements:** Real serialized resident inference bytes incl basis, router, adapter, code, metadata; active MAC/FLOPs; tokens/examples and optimizer updates; old-task retention, score/CE, CPU/GPU P50/P95 and cache alias/copy logs where relevant.

## D — decision
**PASS (only preliminary):** At least four of five fresh streams show >=1% relative cross-entropy reduction versus best same-byte ordinary dynamic gate and native NeuroLoRA, no retention loss, total resident+writeable bytes <=1.05 native, P95 <=1.10 native, no future domain oracle.
**FAIL:** Native dynamic diagonal gate matches, context/weights leak true task ID, old-task forgetting increases, or generator/online write cost uncounted.
**UNCERTAIN:** native paper cannot be reproduced faithfully or no physical memory/real runtime measure; accidental hidden task-ID/cached future leakage invalidates the run rather than promoting it. Replication on >=10 new task/seed units plus natural workload required for ADOPTED.

## C — plausible opposing hypothesis
Contextual modulation and orthogonal subspace protection are already native NeuroLoRA; rich Mirror transforms could be redundant, less stable or slower.

## U — error budget and stopping
Task-switch severity, online generated state reset, sparse projection collisions, per-domain sample size, optimizer / dtype / RNN stability.
If uncorrelated, `u_c = sqrt(u_seed^2 + u_eval^2 + u_num^2)`; if correlated, include all covariance terms. Illustrative `U=k_cov*u_c` with k_cov=2 does not guarantee 95% interval at n=5; use cluster bootstrap of task/world paired differences. Runtime uncertainty in seconds reported independently from serialized byte count. If dev clearly FAILs, stop before fresh; if fresh is opened, run all five and do not retune.

## Sources
- PA431
- PA429
- PA19
- MoLoRA https://arxiv.org/abs/2603.15965
- NeuroLoRA https://arxiv.org/abs/2603.12378

**Output upon activation:** dedicated source/test fixtures, PROTOCOL.json frozen pre-audit, RESULTS_CORE.csv, VERIFICATION.json, hashes and typed metrics. This plan alone is not execution.
