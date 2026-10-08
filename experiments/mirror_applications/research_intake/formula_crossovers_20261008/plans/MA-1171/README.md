# MA-1171: Serializer-aware joint Mirror basis × precision × private allocation

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
| B | 共有辞書basis | 1 | matrix/tensor, compatible with target weight |
| b | 量子化bit幅 | bit | positive integer in {2,4,8}, scalar |
| p | private residual rank | 1 | integer 0,1,2 |
| G | 観測結合計量 | 1 | PSD real 4×4 matrix |

**Unit/shape check:** Any additive View operator must have the same output/input shapes as W and dimensionless normalized coefficients. Code ratios, rank, bits per weight and tensor shape are not seconds. The actual serializer (including metadata, decoder and shared paid atoms) is authoritative. Nat/token CE, byte payload and SI seconds are separate Pareto axes.

## H — falsifiable incremental hypothesis

Across independently trained same-base source task adapters, full joint Mirror+bitwidth+private choice strictly improves held-out task NLL at the same **actual serialized** inference hard budget by >=2% relative to the best simple *non-Mirror* joint codec and one-axis coordinate-descent planner, and does not increase P95 inference more than 10%.

The strongest alternative is not the naive baseline; it is the **native method plus a cheaper same-byte non-Mirror factorization/selection**. Success requires a strict incremental value of small functional m, not success of prior FQC joint optimization, a supplied executor or an exact gauge symmetry.

## Mirror insertion — exact interface

> **Mirror insertion:** add a low-description m to one source-trained task/adapter basis with a paid dictionary/decoder DAG; native quantized independent updates through code m_j selecting a structured basis layout and a quantized task core, with sparse private residual only when needed, so that many held-out task adapters from one physical basis plus small address bank, without duplicating the corresponding entire physical object.

- Native control: native joint codebook + quantization + sparse private allocation, no Mirror / coordinate-only optimizer / native Compress-then-Serve.
- Nearest existing MAs: MA-690, MA-1114, MA-160, MA-1099; these do NOT automatically establish the proposed joint mechanism.
- Internal formula references: IF07, IF08, IF09, IF10, IF11 in [Formula Ledger](../../FORMULA_LEDGER.md).
- External primary baseline: PA351, PA352, PA432 in the canonical prior-art map.

## T — runnable autonomous screen, with paper-reading optional

**Minimal fixture, fixed before audit:** Stage0: K=6 independent linear residual tasks, 4 signal blocks, shared task-delta rank 2/4, two code-basis layouts, 2/4/8-bit quantization and private ranks 0/1/2, with a coupled PSD activation metric; exact enum all legal joint bundles. Stage1: 4 source independently trained rank-4 LoRAs and 4 held-out task adaptations sharing the exact frozen base, no target oracle delta for m fitting. Use actual final serializer for each candidate, not proxy raw bits.

**Step-by-step implementation:**
1. Freeze source/target tasks, base revision, activation calibration split and decoder DAG primitives.
2. Enumerate joint layout × m-family × quant precision × private support; fit all codes on source/support only.
3. For every candidate physically serialize identical inference payload format, including shared roots, private indexes, scalar metadata, and reconstruction state; reject over-budget final file.
4. Evaluate full coupled held-out logits/CE (cross terms), not a sum of independent block distortions; benchmark native non-Mirror joint solver and one-axis coordinate descent.
5. Choose only on 3 development seeds and lock one configuration; run five unseen task families without audit adaptation/selection.
6. Count runtime of dequantization, View decode, private execution, and off-device fetch; report dual axis RAM and on-disk size.

**Data:** whole task/world source/dev identities disjoint from audit identities. Dev seeds 11,12,13; fresh worlds/seeds 101,102,103,104,105 with frozen generator/checkpoint and role/task sampling. Any high-dimensional / non-CPU hardware requirement is a separate later lane, not an excuse to change Stage0 conditions. If target oracle adapter deltas are used to *fit* m, the experiment is invalid; diagnostics that use an oracle must be separately labeled. No tuning after opening any fresh result.

**Implementation environment:** Stage0 CPU NumPy float64 exact enum and actual NPZ serializer; Stage1 torch CPU float32 tiny adapters, optional CUDA for natural LLM throughput. Lock backend, dtype, batch, CPU/GPU clock and serializer version.

**Metrics:** held-out task NLL/accuracy / actual final bytes / decoder DAG bytes / latency; actual serialized inference bytes (including shared basis, codebooks, router, indexes, inverse transforms, private masks, codec headers, decoder dependencies), memory allocation and active compute independently. Calibrate P50/P95 after fixed warmups, declare dtype, batch, clock, alignment and grouped GEMM or fused kernel, and count repeated forward passes.

**First cheap falsification:** run exact finite-dimensional algebra/serialization without model training, including invalid/gauge controls. Then run a source-only training screen, lock and run 5 fresh whole-task worlds. If the simple control wins in development, record that negative mechanism and stop **before opening fresh**; once fresh is open, run all five without favorable early stopping.

## D — predetermined decisions

**PASS (initial mechanism only):** >=4/5 fresh worlds show >=2% relative task NLL improvement over the strongest native non-Mirror joint codec under identical final byte budget, P95<=1.10× native, no dev/audit leakage; superiority persists after eliminating any m code equivalent to a linear core.

**FAIL:** Only the joint search improves but replacing m by a native linear/diagonal core matches the Pareto point; final serializer overflow; private rank approaches unrestricted LoRA; or source-conditioned aligned teacher explains all gains.

**UNCERTAIN:** the native method cannot be reproduced faithfully, hardware/runtime is unavailable, source/audit rank separation is ambiguous, or the confidence interval spans the quality/byte margin. No ADOPTED claim until natural benchmark and separate >=10-world replication with measured real storage and compute.

## C — adversarial alternative / breakage

Optimization layout and quantization synergy is already established by FQC E7; any additional improvement can come from the joint optimizer rather than Mirror's functional parameterization.

## U — main uncertainty, combination and sequential rule

Task covariance/cross-block metric estimation, short-support overfit, quant calibration, integer header overhead, private atom decoder prerequisites, quantization numerical error; stratify worlds and report paired sample CIs.

On task-quality L, with independent uncertainty components only: `u_c=sqrt(u_seed²+u_eval²+u_num²)`; if components correlate, include covariance cross-terms. Expanded `U=k_cov·u_c` with k_cov=2 is indicative, **not** a guaranteed 95% interval at five worlds. Use paired task-cluster bootstrap and report per-world differences, output logits, and FP32/FP64 replay errors. Runtime clock/repeat uncertainty in seconds must be separate from byte count. Freeze all native controls and decisions prior to fresh.

## Prior sources

- Internal: [FORMULA_LEDGER](../../FORMULA_LEDGER.md), plus the exact repo documents/source linked there.
- Primary method controls: PA351, PA352, PA432 from `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`.
- Compare formally with related MAs listed above to avoid duplicating their already-registered hypotheses.

**Status/evidence boundary:** source, tests and output files are created only when this one ID is actively run under its own frozen contract; the current registry addition is UNTESTED.