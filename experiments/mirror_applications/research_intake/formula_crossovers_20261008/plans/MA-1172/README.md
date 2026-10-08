# MA-1172: Coupled Fisher/activation allocation of Mirror head-layer state

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
| G | head間結合PSD計量 | 1 | real positive semidefinite 8×8 matrix |
| e | head別の機能誤差 | 1 | 8-vector of real scalars |
| Q,K,V | 注意Q/K/V | 1 | compatible real token×channel matrices |
| r | per-head低ランク | 1 | integer ≥0 |

**Unit/shape check:** Any additive View operator must have the same output/input shapes as W and dimensionless normalized coefficients. Code ratios, rank, bits per weight and tensor shape are not seconds. The actual serializer (including metadata, decoder and shared paid atoms) is authoritative. Nat/token CE, byte payload and SI seconds are separate Pareto axes.

## H — falsifiable incremental hypothesis

At the same physically resident KV+projector budget, allocation using source-only **cross-head coupled** task error avoids all diagonal-estimate false quality admissions and improves held-out NLL >=0.01 nat/token versus independent per-head Mirror rank allocation in >=4/5 fresh worlds, while beating a byte-near coupled *non-Mirror* allocator and keeping P95 <=1.10×.

The strongest alternative is not the naive baseline; it is the **native method plus a cheaper same-byte non-Mirror factorization/selection**. Success requires a strict incremental value of small functional m, not success of prior FQC joint optimization, a supplied executor or an exact gauge symmetry.

## Mirror insertion — exact interface

> **Mirror insertion:** add a low-description m to native xKV/KQ-SVD or GQA physically shared KV/attention base and source-calibrated candidate per-head residual atoms through head×layer Mirror coefficient and selective private residual allocated with the full source-measured cross-head coupling matrix, so that many attention head/layer functions from one paid cache projection/decoder without violating task attention quality, without duplicating the corresponding entire physical object.

- Native control: native KQ-SVD / xKV-SR / same-byte coupled non-Mirror allocator / naive diagonal per-head allocation / packed INT4.
- Nearest existing MAs: MA-1165, MA-1166, MA-879, MA-844, MA-690; these do NOT automatically establish the proposed joint mechanism.
- Internal formula references: IF06, IF07, IF08, IF11 in [Formula Ledger](../../FORMULA_LEDGER.md).
- External primary baseline: PA425, PA426, PA432, PA373 in the canonical prior-art map.

## T — runnable autonomous screen, with paper-reading optional

**Minimal fixture, fixed before audit:** Synthetic 4-head × 2-layer attention, d_head=16, length128, calibrated Q/K/V, coupled source score-error matrix from concatenated output residuals; 4 source prompt families and 2 OOD audit families. Candidate allocations: rank {0,2,4}, code size {0,2,4}, private {0,1}. Native per-role KQ-SVD and true K/V INT4 with scales are mandatory. Second stage use trained tiny LM with same RoPE and frozen tokenizer.

**Step-by-step implementation:**
1. Freeze causal masks, RoPE/cache prefix provenance and query-family splits; source-only estimate PSD cross-head error metric G.
2. Verify G PSD after explicit eigenvalue tolerance; compute exact sum of per-head response errors and majorizer.
3. Enumerate equal serialized-budget private/shared atom choices, once using true coupled score or loss, once using diagonal G, once using native non-Mirror linear code.
4. Train source-only m dictionary, fit fresh role m only from permitted task support; never calibrate on hidden Query audit.
5. Evaluate true softmax attention output, long-context NLL, rare near-tie routing flips and exact physical cache pointer aliases.
6. Report false feasible allocations, NLL, whole-cache bytes incl codebook+indices, FLOPs and synchronized P95; an unattained mathematical upper bound is not a deployment baseline.

**Data:** whole task/world source/dev identities disjoint from audit identities. Dev seeds 11,12,13; fresh worlds/seeds 101,102,103,104,105 with frozen generator/checkpoint and role/task sampling. Any high-dimensional / non-CPU hardware requirement is a separate later lane, not an excuse to change Stage0 conditions. If target oracle adapter deltas are used to *fit* m, the experiment is invalid; diagnostics that use an oracle must be separately labeled. No tuning after opening any fresh result.

**Implementation environment:** Stage0 NumPy FP64/torch FP32 CPU score fixture; Stage1 CUDA optional for native xKV-SR/KQ-SVD and true bitpacked INT4. Record context/batch/clock and code materialization.

**Metrics:** true held-out attention output and NLL / false-pass rate / serialized cache+code bytes / P95 decode; actual serialized inference bytes (including shared basis, codebooks, router, indexes, inverse transforms, private masks, codec headers, decoder dependencies), memory allocation and active compute independently. Calibrate P50/P95 after fixed warmups, declare dtype, batch, clock, alignment and grouped GEMM or fused kernel, and count repeated forward passes.

**First cheap falsification:** run exact finite-dimensional algebra/serialization without model training, including invalid/gauge controls. Then run a source-only training screen, lock and run 5 fresh whole-task worlds. If the simple control wins in development, record that negative mechanism and stop **before opening fresh**; once fresh is open, run all five without favorable early stopping.

## D — predetermined decisions

**PASS (initial mechanism only):** No diagonal-derived false quality PASS after coupling-aware admission in five fresh worlds; >=4/5 fresh worlds meet +0.01 nat/token lower NLL than independently allocated Mirror at matched bytes, and strict Pareto gain over coupled native linear/INT4, P95<=1.10.

**FAIL:** Coupling majorizer too loose and misallocates bits; exact native KQ-SVD or simple coupled linear code matches at equal bytes; learned m changes upstream KV invalidating physical sharing.

**UNCERTAIN:** the native method cannot be reproduced faithfully, hardware/runtime is unavailable, source/audit rank separation is ambiguous, or the confidence interval spans the quality/byte margin. No ADOPTED claim until natural benchmark and separate >=10-world replication with measured real storage and compute.

## C — adversarial alternative / breakage

The true benefit may be from correct cross-head score weighting and task-aware allocation, which a native linear projector and bitpacked KV can already implement without Mirror geometry.

## U — main uncertainty, combination and sequential rule

Source-sample covariance rank, conditional input shift, cross-head correlation uncertainty, RoPE-position leakage, calibration/validation overlap, quantization and GPU materialization details.

On task-quality L, with independent uncertainty components only: `u_c=sqrt(u_seed²+u_eval²+u_num²)`; if components correlate, include covariance cross-terms. Expanded `U=k_cov·u_c` with k_cov=2 is indicative, **not** a guaranteed 95% interval at five worlds. Use paired task-cluster bootstrap and report per-world differences, output logits, and FP32/FP64 replay errors. Runtime clock/repeat uncertainty in seconds must be separate from byte count. Freeze all native controls and decisions prior to fresh.

## Prior sources

- Internal: [FORMULA_LEDGER](../../FORMULA_LEDGER.md), plus the exact repo documents/source linked there.
- Primary method controls: PA425, PA426, PA432, PA373 from `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`.
- Compare formally with related MAs listed above to avoid duplicating their already-registered hypotheses.

**Status/evidence boundary:** source, tests and output files are created only when this one ID is actively run under its own frozen contract; the current registry addition is UNTESTED.