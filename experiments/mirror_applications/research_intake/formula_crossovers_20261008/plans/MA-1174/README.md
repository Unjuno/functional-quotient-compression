# MA-1174: Finite-step reachability-adjusted View selection with gauge and retention

**Status: UNTESTED / FROZEN DESIGN. Branch isolated; no worker claim.** The goal is to test a cross-over of already studied formulae, not rediscover those formulae or misattribute native compression.

## Variables before equations / dimension check
| Symbol | 意味（日本語） | SI単位 | 定義域・前提・型 |
|---|---|---|---|
| θ,W | 共有modelパラメータ | 1 | 有限実テンソル、保存対象 |
| m | Mirror機能コード | 1 | k次元実ベクトル、追加生成/保存を課金 |
| k | 1 Viewの座標数 | 1 | 正整数スカラー |
| B_i | 共有View/作用素基底 | 1 | Wとshape整合する実行列 |
| L | tokenまたはtask単位の品質損失 | nat/token（SIでは無次元） | 非負実スカラー、正確な分母を明記 |
| S | 直列化推論容量 | byte（実用単位） | 非負整数、完成ファイルと物理allocatorで計測 |
| t | 推論時間 | s | 非負実スカラー、CPU/GPU別に測定 |
| u_seed,u_eval,u_num | 独立world/標本/数値不確かさ | Lと同単位 | 非負実スカラー |
| u_c | 合成標準不確かさ | Lと同単位 | 非負実スカラー、共分散を含む |
| k_cov | 包含係数 | 1 | 正の実スカラー、説明用2 |
| U | 拡張不確かさ | Lと同単位 | k_cov×u_c、n=5では95%保証ではない |
Additional method-specific symbols:
| Symbol | 意味（日本語） | SI単位 | 定義・範囲・型 |
|---|---|---|---|
| J,B | baselineとMirror出力Jacobian | 1 | compatible real output×parameter matrices |
| Rθ,RM | parameterとView計量 | 1 | SPD matrices |
| λ | 模倣費用の一般化固有値 | 1 | nonnegative scalar |
| η,N | 学習率・更新回数 | 1 | positive scalar and integer ≥0 |

**Unit/shape check:** Any additive View operator must have the same output/input shapes as W and dimensionless normalized coefficients. Code ratios, rank, bits per weight and tensor shape are not seconds. The actual serializer (including metadata, decoder and shared paid atoms) is authoritative. Nat/token CE, byte payload and SI seconds are separate Pareto axes.

## H — falsifiable incremental hypothesis

Among source-trained task families, View directions selected by the *correct* baseline-emulation cost metric **and separately measured task utility** deliver >=2% lower fresh held-out CE than source-only gradient/Fisher ranking and same-byte simple low-rank controls after 100 matched updates in >=4/5 fresh worlds, with no >1% absolute old-task retention regression and <=1.10× P95.

The strongest alternative is not the naive baseline; it is the **native method plus a cheaper same-byte non-Mirror factorization/selection**. Success requires a strict incremental value of small functional m, not success of prior FQC joint optimization, a supplied executor or an exact gauge symmetry.

## Mirror insertion — exact interface

> **Mirror insertion:** add a low-description m to frozen trained base and a task-adapter source bank with task-sensitive Jacobian metrics and a small paid candidate View basis through code coordinates selected by source-only baseline-cost generalized eigenvectors, intersected with task utility and old-task retention constraints, so that new task adaptations from cheap useful View coordinates rather than unreached or redundant parameter directions, without duplicating the corresponding entire physical object.

- Native control: source-only gradient norm / Fisher or curvature directions / native LoRA / simple shared low-rank basis / random equal-byte Views.
- Nearest existing MAs: MA-844, MA-1096, MA-1099, MA-1156, MA-1105, MA-199; these do NOT automatically establish the proposed joint mechanism.
- Internal formula references: IF01, IF02, IF03, IF04, IF11 in [Formula Ledger](../../FORMULA_LEDGER.md).
- External primary baseline: PA373, PA374, PA381, PA414, PA367 in the canonical prior-art map.

## T — runnable autonomous screen, with paper-reading optional

**Minimal fixture, fixed before audit:** First analytic fixture J=diag(1,0.1), Rθ=I, mirror B=I and full observable rank; verify λ=(1,100), unreachable target is infinite and finite-update e_N recursion. Then 64-wide shared MLP with source/target task banks and heldout family identities; fresh training uses 10/50/100 fixed optimizer updates, not near-convergence capacity inference. Natural stage: identical-base rank4 LoRA checkpoints with gauge stress and target code trained only from target support.

**Step-by-step implementation:**
1. Freeze source-only observable witness inputs, source task IDs, Rθ/RM metrics and numerical SVD rank tolerance before any target development.
2. Compute baseline Jacobian J, Mirror Jacobian B, old-task Jacobian and correct R_B/R_M ratio; classify unreachable B directions explicitly.
3. Separately measure task utility/retention on development, select candidate m support by both, and compare simpler gradient/Fisher, shared SVD basis, LoRA and random equal-byte controls.
4. Check invariance under B->BG and A->G^{-1} A factor gauge and exact FFN folding; do not count gauge movement as new function.
5. Lock finite steps 10/50/100, same examples and optimizer initialization; run 5 unseen task-worlds without audit-time eigenbasis updates.
6. Measure CE, old-task failures, Jacobian nonlinearity at finite ρ, serialized basis bytes, CPU/GPU P95, active FLOPs and the continuing baseline reachability envelope.

**Data:** whole task/world source/dev identities disjoint from audit identities. Dev seeds 11,12,13; fresh worlds/seeds 101,102,103,104,105 with frozen generator/checkpoint and role/task sampling. Any high-dimensional / non-CPU hardware requirement is a separate later lane, not an excuse to change Stage0 conditions. If target oracle adapter deltas are used to *fit* m, the experiment is invalid; diagnostics that use an oracle must be separately labeled. No tuning after opening any fresh result.

**Implementation environment:** CPU NumPy float64 for certified local linear algebra; torch FP32 small MLP; only natural/checkpoint stage on CUDA if available. Batch, clock, dtype, rank tolerance fixed.

**Metrics:** held-out task CE / old-task retention / finite-update efficiency / actual bytes / P95; actual serialized inference bytes (including shared basis, codebooks, router, indexes, inverse transforms, private masks, codec headers, decoder dependencies), memory allocation and active compute independently. Calibrate P50/P95 after fixed warmups, declare dtype, batch, clock, alignment and grouped GEMM or fused kernel, and count repeated forward passes.

**First cheap falsification:** run exact finite-dimensional algebra/serialization without model training, including invalid/gauge controls. Then run a source-only training screen, lock and run 5 fresh whole-task worlds. If the simple control wins in development, record that negative mechanism and stop **before opening fresh**; once fresh is open, run all five without favorable early stopping.

## D — predetermined decisions

**PASS (initial mechanism only):** 4/5 fresh world held-out CE >=2% relative better than strongest source-only task-aware native low-rank/Fisher control after 100 updates, old retention drop <=1 percentage point, actual payload <=matched budget and P95 <=1.10, with nonzero functional Jacobian directions beyond gauge.

**FAIL:** E is high but task gradient/retention unfavourable; rank threshold flips selection; native low-rank/LoRA obtains same quality at <=bytes/time; unobservable probe nullspace is miscalled globally unreachable.

**UNCERTAIN:** the native method cannot be reproduced faithfully, hardware/runtime is unavailable, source/audit rank separation is ambiguous, or the confidence interval spans the quality/byte margin. No ADOPTED claim until natural benchmark and separate >=10-world replication with measured real storage and compute.

## C — adversarial alternative / breakage

The local emulation-cost eigenvector can be irrelevant to target task utility, change when the observable set changes, or lose its advantage during full nonlinear optimization; finite-update differences cannot certify capacity.

## U — main uncertainty, combination and sequential rule

Autodiff precision, preconditioner scale, JVP/VJP noise, output witness coverage, optimizer time variation, LoRA GL(r) gauge, Jacobian linearization residual and confidence intervals over task seeds.

On task-quality L, with independent uncertainty components only: `u_c=sqrt(u_seed²+u_eval²+u_num²)`; if components correlate, include covariance cross-terms. Expanded `U=k_cov·u_c` with k_cov=2 is indicative, **not** a guaranteed 95% interval at five worlds. Use paired task-cluster bootstrap and report per-world differences, output logits, and FP32/FP64 replay errors. Runtime clock/repeat uncertainty in seconds must be separate from byte count. Freeze all native controls and decisions prior to fresh.

## Prior sources

- Internal: [FORMULA_LEDGER](../../FORMULA_LEDGER.md), plus the exact repo documents/source linked there.
- Primary method controls: PA373, PA374, PA381, PA414, PA367 from `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`.
- Compare formally with related MAs listed above to avoid duplicating their already-registered hypotheses.

**Status/evidence boundary:** source, tests and output files are created only when this one ID is actively run under its own frozen contract; the current registry addition is UNTESTED.