# MA-1169 — MoLoRA per-token physical specialist-bank compression
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
At fixed original per-token router, a shared specialist m bank preserves at least 99% of native mixed-skill quality or <=1 percentage point drop (use stricter relevant threshold), reduces total specialist+router serialized inference bytes >=15% and P95 decode latency <=1.10 native, while beating a byte-matched ordinary SVD bank and identifying all prefix-cache incompatibilities.
Related existing MA IDs: MA-258, MA-259, MA-403, MA-697, MA-1112.

## Mirror insertion
> **Mirror insertion:** supply `m` at **frozen MoLoRA per-token router and independently trained LoRA specialist bank**, through **small per-expert Mirror m decoding one shared LoRA update basis only on tokens routed to its expert**, so many per-token experts without storing each independent LoRA matrix can be realized without a full independent expert/online controller state per task.
- Native method (must preserve): MoLoRA independent-LoRA per-token routing and native cached-prefix semantics
- Mirror marginal benefit must be measured *against* native behavior, not attributed to the native router/gate.
- Only source-task learned basis and permitted support examples may form m.

## T — independent runnable experiment plan
**Synthetic fixture:** Toy 2-layer d=64 causal Transformer, 4 separate specialists and 4-skill token streams with heldout combinations/switches; native MoLoRA official benchmark later if available. Source banks can be aligned or independently trained, evaluated separately.
**Controls:** 1. original MoLoRA router with independent LoRAs
2. MoLoRA router with SVD-compressed adapter bank at matched total bytes
3. MoLoRA with native task-group shared LoRA
4. FiLM/rank-one cheap expert code
5. reference target-adapter re-prefill and cache-copy controls
**Exact implementation steps:**
1. Freeze base revision, specialist training data, token routing and MoLoRA original top-k semantics.
2. Train four native per-token LoRA experts, split whole task/domain identities into source/dev/fresh.
3. Construct plain shared linear factor bank and signed/orthogonal m bank at matched bytes; do not fit heldout oracle deltas.
4. Run same frozen learned token router on all methods; never give audit task ID or oracle route.
5. Verify target-native fresh prefix on each switched token path; explicit safe alias vs copied or upstream-different invalid state.
6. Measure mixed-domain token loss, per-task quality, physical cache bytes, adapter-bank serialized size, router MAC/FLOP and batch P95.
**Data and split:** Dev random seeds 11/12/13 may select LR/rank/strength. Fresh seeds 101/102/103/104/105 with heldout whole role/task streams; freeze before opening. No audit labels, target trained delta, future route or task ID may tune m. Execute all five after dev gate, no favorable early stopping.
**Prerequisite / environment:** CPU torch float32 synthetic; optionally CUDA GPU with kernel details, true token routing, B=1/8 context=128/512, GPU clock and synchronized P95.
**Measurements:** Real serialized resident inference bytes incl basis, router, adapter, code, metadata; active MAC/FLOPs; tokens/examples and optimizer updates; old-task retention, score/CE, CPU/GPU P50/P95 and cache alias/copy logs where relevant.

## D — decision
**PASS (only preliminary):** At least 4 of 5 fresh worlds quality >=0.99 native and <=1pp absolute decline when native near chance, total incremental bytes <=0.85 native, P95 <=1.10 native, no cache provenance violations, with strict benefit over same-byte ordinary factorization.
**FAIL:** Native compressed bank dominates m, router includes audit labels, incompatible prefix reuse called exact, or hidden re-prefill overhead eliminates runtime savings.
**UNCERTAIN:** native paper cannot be reproduced faithfully or no physical memory/real runtime measure; accidental hidden task-ID/cached future leakage invalidates the run rather than promoting it. Replication on >=10 new task/seed units plus natural workload required for ADOPTED.

## C — plausible opposing hypothesis
Per-token switching and routing benefits originate entirely from native MoLoRA; specialist bank may lack a shared low-dimensional geometry and decoded m can increase kernel launches and cache reconstruction.

## U — error budget and stopping
Few specialist domains, routing entropy and gating confidence, context-dependent role switches, cache-source mismatch, optimizer reproducibility, Eager vs fused GPU runtime.
If uncorrelated, `u_c = sqrt(u_seed^2 + u_eval^2 + u_num^2)`; if correlated, include all covariance terms. Illustrative `U=k_cov*u_c` with k_cov=2 does not guarantee 95% interval at n=5; use cluster bootstrap of task/world paired differences. Runtime uncertainty in seconds reported independently from serialized byte count. If dev clearly FAILs, stop before fresh; if fresh is opened, run all five and do not retune.

## Sources
- PA430
- PA433
- PA364
- MoLoRA https://arxiv.org/abs/2603.15965
- NeuroLoRA https://arxiv.org/abs/2603.12378

**Output upon activation:** dedicated source/test fixtures, PROTOCOL.json frozen pre-audit, RESULTS_CORE.csv, VERIFICATION.json, hashes and typed metrics. This plan alone is not execution.
