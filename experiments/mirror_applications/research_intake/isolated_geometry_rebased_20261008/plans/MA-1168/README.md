# MA-1168 — FACET single-adapter dynamic Mirror feature-code boundary
**DESIGN ONLY / UNTESTED.** Branch `research/mirror-isolated-protocols-rebased-20261008`; current worker next MA-255 and frozen queue are unmodified. Native method is NOT a Mirror discovery.

## Variable table (must precede formulas)
| 記号 | 意味（日本語） | 単位(SI) | 定義・定義域・前提 | 型 |
|---|---|---|---|---|
| m | Mirror機能コード | 1 | k次元有限値 | 実ベクトル |
| k | コード次元 | 1 | 正整数 | スカラー |
| W | 共有学習済み重み | 1 | 有限 d_out×d_in | 行列 |
| B_i | 共有View基底 | 1 | Wと同形状 | 行列 |
| R | ネイティブrouter/feature変換 | 1 | task×feature or compatible | 行列・テンソル |
| j | タスクID | 1 | 整数1..J、評価中はoracle取得禁止 | カテゴリ |
| J | タスク数 | 1 | 整数>=2 | スカラー |
| A | 正解率 | 1 | [0,1] 実数、pp差は百分率で報告 | スカラー |
| L | NLL/CE | nat/token (1) | 非負、taskごとに正規化 | スカラー |
| S | 実測保存容量 | byte（実用単位） | セッション利用に必要な全追加状態 | 整数 |
| t | 推論遅延 | s | 同期測定非負 | 実数 |
| u_seed,u_eval,u_num | 不確かさ成分 | Lと同単位 | 0以上、可能な共分散を調査 | 実数 |
| u_c | 合成標準不確かさ | Lと同単位 | 共分散込み、0以上 | 実数 |
| k_cov | 包含係数 | 1 | 正数、暫定値2 | 実数 |
| U | 拡張不確かさ | Lと同単位 | k_cov×u_c | 実数 |
**Dimension check:** `W + sum_i m_i B_i` is shape-correct only for same-shaped real matrices, all coefficients and normalized weights dimensionless; router R may have different native shape and must be operated on according to its documented interface. Accuracy A is a fraction, quality L in nat/token is dimensionless in SI, S uses actual byte payload, t uses seconds; do not combine unlike dimensions.

## H — falsifiable claim
On a frozen 20-task class-incremental sequence with no oracle task ID at inference, replacing native FACET task transformations with structured m preserves final mean accuracy and backward transfer within 1.0 percentage point while reducing total inference task-state bytes >=15% and keeping P95 <=1.10 native, beyond same-byte FiLM.
**Related registered:** MA-401, MA-403, MA-446, MA-849.

## Mirror insertion
> **Mirror insertion:** add `m` at native **one shared FACET adapter, task-conditioned feature transform and replay-free consistency regularizer** by **compact task/history m substituting only the native feature transform coefficients**, without independent full per-task native routing state.
- Required native method: FACET original single-adapter task-conditioned feature consistency with no replay.
- New causal degree of freedom: change task m while same original/native physical base remains.
- Cost: basis, m, decoder, router, per-task metadata, writable state and online generation all paid.

## T — complete minimal implementation (worker need not search papers)
**Fixture:** CPU initially: class-balanced 8×8 handwritten-digit feature vectors or synthetic class-incremental 20 tasks, 32-feature MLP base frozen, train stream disjoint task identities; natural ViT+FACET second stage requires verified official code, no independent paper-gain claim from toy.
**Control hierarchy:** 1. native FACET with full original task conditioning and consistency loss 2. FACET conditioned diagonal/FiLM at matched bytes 3. one shared adapter plus simple task linear embedding 4. independent task-specific adapters as upper bound 5. no-task-conditioning ablation
**Execution order:**
1. Construct disjoint class/task ID stream and five held-out task orders; do not reveal true task ID during inference.
2. Reproduce one shared adapter, native FACET task conditioner and replay-free consistency loss; keep absence of replay buffer explicit.
3. Replace only task transform with diagonal, low-rank ordinary embedding and m signed/Givens chart with k=2/4/8.
4. Use same task-identification policy and support examples for all methods; preserve feature consistency coefficients.
5. Log all prior-class accuracies after each task, forward/backward transfer, task-code bytes and generator FLOPs.
6. Run all five fresh sequences and classify M0 when native conditioner or FiLM matches.
**Frozen worlds:** dev 11/12/13 for hyperparameter search; fresh 101/102/103/104/105 with whole task/order held out. No target oracle delta, true task IDs, audit look-ahead or fresh retuning. Run all five fresh if dev gate proceeds. Normalization, calibration rank, worker CPU/GPU and seeds must be frozen in protocol before reading fresh.
**Measurements:** CIL final mean accuracy, backward transfer and old-task retention, feature consistency loss, task-state actual bytes, hidden task ID leakage, latency. Serialized inference state includes common backbone additions, basis, learned routing, m code bank, optimizer state only when used at inference, generator and metadata. Report train updates/examples, MAC/FLOP proxy, CPU/GPU model and clock, dtype, batch, 20+ warmup and >=100 timed inference calls when P95 is relevant.

## D — PASS / FAIL / UNCERTAIN
**PASS, early mechanism:** 4/5 fresh task orders meet FACET mean and backward transfer within 1.0 pp, >=15% total learned task-state bytes saved with P95 <=1.10, and cheap same-byte FiLM control does not dominate.
**FAIL:** Original FACET/FiLM equal or better at bytes+quality+runtime, task ID leaked or replay enabled, or toy class task not appropriate for trained-model claim.
**UNCERTAIN:** native method unavailable, insufficient independent task identities, kernel/runtime/bytes not measured or calibration leakage; do not call ADOPTED. Independent natural-task replication with >=10 task/seed units required before adoption.

## C — counter-hypothesis
Native FACET already uses one adapter and task-conditioned transformations, so a Mirror chart may offer no new useful function beyond a simpler gate.

## U — uncertainty and sequential stop
Task order, class imbalance, feature representation drift, imperfect task-ID inference, replay leakage and training updates.
For L/nat-token: `u_c=sqrt(u_seed² + u_eval² + u_num²)` **only** for independent components; add covariances otherwise. Expanded `U=k_cov*u_c` with k_cov=2 is indicative for five worlds, not guaranteed 95% coverage. Bootstrap by task identities and report paired task runs; runtime has separate second-based repeat/clock uncertainty. Dev-stage futile screen may stop before fresh; once fresh opened, never stop for favorable outcomes or tune after seeing outcomes.

## Primary sources
- PA429
- PA431
- PA63
- mtLoRA ICLR 2026 https://proceedings.iclr.cc/paper_files/paper/2026/hash/791de7c35bb49cfca56744e67f90eef4-Abstract-Conference.html
- FACET arXiv https://arxiv.org/abs/2608.31096

No results have been generated by this design. Future activated run needs `PROTOCOL.json`, source, unit tests, `RESULTS_CORE.csv`, `VERIFICATION.json` with real serializer and hardware provenance.
