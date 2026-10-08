# MA-1167 — mtLoRA spectral-aware Mirror dimension-router bank
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
On >=12 task source families, Mirror compresses native per-task routing state by at least 10% overall inference bytes while preserving mean task accuracy within 0.5 percentage point and worst-task within 2.0 points of full native mtLoRA, with <=1.10x P95 latency and advantage over a byte-matched linear routing SVD.
**Related registered:** MA-265, MA-268, MA-690, MA-1097.

## Mirror insertion
> **Mirror insertion:** add `m` at native **native mtLoRA block adapters with spectral-aware source basis and task×rank fine-grained routing tensors** by **small task×block m encoding only native routing factors in low-SV task-private components**, without independent full per-task native routing state.
- Required native method: native mtLoRA: source-identical block placement + spectral-aware regularization + dimension-wise router.
- New causal degree of freedom: change task m while same original/native physical base remains.
- Cost: basis, m, decoder, router, per-task metadata, writable state and online generation all paid.

## T — complete minimal implementation (worker need not search papers)
**Fixture:** Frozen 64×64 pretrained block, 15 source tasks plus 5 whole-task heldouts differing high-vs-low singular directions; source spectral decomposition alone selects high-SV shared directions. Native ICLR mtLoRA code/benchmark second stage, not assumed on CPU.
**Control hierarchy:** 1. full native mtLoRA 2. native routing SVD at matched total serialized bytes 3. IA3/scalar gate at equal bytes 4. mtLoRA without spectral regularizer (diagnostic) 5. individual native LoRA upper quality reference
**Execution order:**
1. Freeze base/task identities and source singular decomposition.
2. Train native mtLoRA 15 tasks at identical block placements and original spectral penalty.
3. Compress per-dimension routing using simple SVD/linear low-rank, then m={2,4,8} structured signed/orthogonal coefficient chart.
4. Keep high-SV shared components, optimizer budget, and task identity router identical across baselines.
5. Audit held-out task combinations with all five fresh seeds; report per-task old/new accuracy, worst-task, physical basis and router bytes, learning speed and runtime.
6. Fail if linear native factorization matches m or high-SV preservation is violated.
**Frozen worlds:** dev 11/12/13 for hyperparameter search; fresh 101/102/103/104/105 with whole task/order held out. No target oracle delta, true task IDs, audit look-ahead or fresh retuning. Run all five fresh if dev gate proceeds. Normalization, calibration rank, worker CPU/GPU and seeds must be frozen in protocol before reading fresh.
**Measurements:** balanced mean and worst task accuracy, heldout task cross entropy, routing occupancy, old-task forgetting, real total adapter+router bytes, MACs, CPU/GPU P95. Serialized inference state includes common backbone additions, basis, learned routing, m code bank, optimizer state only when used at inference, generator and metadata. Report train updates/examples, MAC/FLOP proxy, CPU/GPU model and clock, dtype, batch, 20+ warmup and >=100 timed inference calls when P95 is relevant.

## D — PASS / FAIL / UNCERTAIN
**PASS, early mechanism:** 4/5 fresh worlds within native average <=0.5 pp and worst-task <=2.0 pp, >=10% total incremental adapter+router bytes saved, P95 <=1.10 native, and an incremental benefit over same-byte native routing factorization.
**FAIL:** Native simple SVD achieves same Pareto frontier, any task repeatedly collapses, source/native spectral penalty missing or timing/bytes exclude learned router.
**UNCERTAIN:** native method unavailable, insufficient independent task identities, kernel/runtime/bytes not measured or calibration leakage; do not call ADOPTED. Independent natural-task replication with >=10 task/seed units required before adoption.

## C — counter-hypothesis
Fine-grained task router is already an expressive task code, and high-SV/shared directions are native; structured m may merely relabel or overcompress the informative low-SV variation.

## U — uncertainty and sequential stop
Singular-space degeneracy, varying task class balance, seed-specific routing, shared-vs-private interference, optimizer wall-time.
For L/nat-token: `u_c=sqrt(u_seed² + u_eval² + u_num²)` **only** for independent components; add covariances otherwise. Expanded `U=k_cov*u_c` with k_cov=2 is indicative for five worlds, not guaranteed 95% coverage. Bootstrap by task identities and report paired task runs; runtime has separate second-based repeat/clock uncertainty. Dev-stage futile screen may stop before fresh; once fresh opened, never stop for favorable outcomes or tune after seeing outcomes.

## Primary sources
- PA428
- PA427
- PA373
- mtLoRA ICLR 2026 https://proceedings.iclr.cc/paper_files/paper/2026/hash/791de7c35bb49cfca56744e67f90eef4-Abstract-Conference.html
- FACET arXiv https://arxiv.org/abs/2608.31096

No results have been generated by this design. Future activated run needs `PROTOCOL.json`, source, unit tests, `RESULTS_CORE.csv`, `VERIFICATION.json` with real serializer and hardware provenance.
