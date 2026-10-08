# MA-1173: Oracle-free ordered-rule packet Mirror factorial

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
| A,B | 順序のある二つのルール作用 | 1 | square same-state real matrices |
| P | 並列予測token数 | 1 | integer 2 or 4 |
| z | packet潜在分岐 | 1 | finite categorical random variable |
| H | 隠れ状態次元 | 1 | integer positive |

**Unit/shape check:** Any additive View operator must have the same output/input shapes as W and dimensionless normalized coefficients. Code ratios, rank, bits per weight and tensor shape are not seconds. The actual serializer (including metadata, decoder and shared paid atoms) is authoritative. Nat/token CE, byte payload and SI seconds are separate Pareto axes.

## H — falsifiable incremental hypothesis

On unseen order-reversed rule pairs with a hidden packet branch, a non-oracle fixed-depth execution interface with Mirror m attains >=95% exact two-rule and valid-path correctness in >=4/5 fresh worlds at <=1.10× native two-forward executor active compute, and beats a byte-matched signed shared-rule native executor and native PTP in the task-quality/bytes Pareto plane.

The strongest alternative is not the naive baseline; it is the **native method plus a cheaper same-byte non-Mirror factorization/selection**. Success requires a strict incremental value of small functional m, not success of prior FQC joint optimization, a supplied executor or an exact gauge symmetry.

## Mirror insertion — exact interface

> **Mirror insertion:** add a low-description m to one causal Transformer with a common signed shared-rule bank and packet latent interface through compact rule/View address used at two verified causal execution steps, and one sampled packet-level branch shared over future slots, so that ordered noncommuting rule compositions and joint P-token packet futures without independent rule blocks or per-slot branch duplication, without duplicating the corresponding entire physical object.

- Native control: native two-forward predicted-intermediate executor / native PTP packet latent / signed shared basis / Dense / AR with KV.
- Nearest existing MAs: MA-248, MA-131, MA-455, MA-1160, MA-257; these do NOT automatically establish the proposed joint mechanism.
- Internal formula references: IF05, IF08, IF12 in [Formula Ledger](../../FORMULA_LEDGER.md).
- External primary baseline: PA10, PA16, PA420, PA375 in the canonical prior-art map.

## T — runnable autonomous screen, with paper-reading optional

**Minimal fixture, fixed before audit:** Base rules: 16 symbolic states, 8 common signed operator atoms, 4 ordered rule applications; train only atomic and seen ordered pairs; hold out entire unordered pair IDs with both reversed orders in same split. Distinguish revealed/deterministic and hidden shared packet-branch tasks (P={2,4}). No oracle intermediate target during training/inference. Use a fixed 2-step latent-state interface instead of recurrent loop or externally supplied true state.

**Step-by-step implementation:**
1. Implement native Dense, native signed shared operator bank, fixed-depth predicted-intermediate two-step controller, and cached autoregressive/PTP packet baselines; freeze rule corpus.
2. Use 2×2 factorial: external executor absent/present and native signed basis/structured Mirror m; separately cross packet-level branch shared/not shared.
3. Allow only predicted intermediate token/state, never teacher state; verify that all methods receive the same known rule-token positions and context.
4. Train under matched token and optimizer update budgets; measure near-convergence quality separately from fixed update advantage.
5. Audit held-out reversed rule pairs, per-rule exact retention, hidden packet valid joint trajectory, runtime and actual bytes including packet latent/decoder state.
6. Evaluate counterexamples for commutation, parity, changed packet branch, teacher leakage and unobserved rule identity; record all unclosed gates as FAIL or UNCERTAIN.

**Data:** whole task/world source/dev identities disjoint from audit identities. Dev seeds 11,12,13; fresh worlds/seeds 101,102,103,104,105 with frozen generator/checkpoint and role/task sampling. Any high-dimensional / non-CPU hardware requirement is a separate later lane, not an excuse to change Stage0 conditions. If target oracle adapter deltas are used to *fit* m, the experiment is invalid; diagnostics that use an oracle must be separately labeled. No tuning after opening any fresh result.

**Implementation environment:** CPU torch FP32 2-layer width32 Transformer 4 heads, stage0 batch 32/96; optional CUDA with real causal decoding and kernel synchronization. Use total active forwards as explicit resource.

**Metrics:** held-out reversed-pair validity / atomic retention / joint packet NLL / forward count / active FLOPs / wall; actual serialized inference bytes (including shared basis, codebooks, router, indexes, inverse transforms, private masks, codec headers, decoder dependencies), memory allocation and active compute independently. Calibrate P50/P95 after fixed warmups, declare dtype, batch, clock, alignment and grouped GEMM or fused kernel, and count repeated forward passes.

**First cheap falsification:** run exact finite-dimensional algebra/serialization without model training, including invalid/gauge controls. Then run a source-only training screen, lock and run 5 fresh whole-task worlds. If the simple control wins in development, record that negative mechanism and stop **before opening fresh**; once fresh is open, run all five without favorable early stopping.

## D — predetermined decisions

**PASS (initial mechanism only):** At least 4/5 fresh rule worlds reach >=95% ordered pair and hidden joint packet validity, with quality no worse than native two-forward/AR and a strict improvement over the byte-near signed-basis+native packet-latent control at <=1.10× active compute and P95.

**FAIL:** Only an oracle/teacher intermediate state recovers order; native signed executor matches m; hidden branch NLL remains factorized and inconsistent; or method uses extra forward passes claimed free.

**UNCERTAIN:** the native method cannot be reproduced faithfully, hardware/runtime is unavailable, source/audit rank separation is ambiguous, or the confidence interval spans the quality/byte margin. No ADOPTED claim until natural benchmark and separate >=10-world replication with measured real storage and compute.

## C — adversarial alternative / breakage

SRM003's 100% two-forward success is supplied execution structure rather than Mirror capacity; a shared packet latent may be entirely sufficient without richer View geometry.

## U — main uncertainty, combination and sequential rule

High-order credit assignment, hidden branch entropy, source pair leakage, validation task combinatorics, unintended teacher state input, cumulative decoding FLOPs and cache consistency.

On task-quality L, with independent uncertainty components only: `u_c=sqrt(u_seed²+u_eval²+u_num²)`; if components correlate, include covariance cross-terms. Expanded `U=k_cov·u_c` with k_cov=2 is indicative, **not** a guaranteed 95% interval at five worlds. Use paired task-cluster bootstrap and report per-world differences, output logits, and FP32/FP64 replay errors. Runtime clock/repeat uncertainty in seconds must be separate from byte count. Freeze all native controls and decisions prior to fresh.

## Prior sources

- Internal: [FORMULA_LEDGER](../../FORMULA_LEDGER.md), plus the exact repo documents/source linked there.
- Primary method controls: PA10, PA16, PA420, PA375 from `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`.
- Compare formally with related MAs listed above to avoid duplicating their already-registered hypotheses.

**Status/evidence boundary:** source, tests and output files are created only when this one ID is actively run under its own frozen contract; the current registry addition is UNTESTED.