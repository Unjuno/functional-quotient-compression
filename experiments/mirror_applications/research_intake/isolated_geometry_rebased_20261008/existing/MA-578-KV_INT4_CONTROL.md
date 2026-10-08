# MA-578 — Supplement: INT4 full-dimensional KV versus Mirror rank
**EXISTING-ID CONTROL SUPPLEMENT / UNTESTED.** Branch `research/mirror-isolated-protocols-rebased-20261008` only. This document does NOT preempt active worker experiments or rewrite any precommitted protocol. Before applying to the official MA experiment, freeze a new amendment.

## Variable table and dimension check
| 記号 | 意味 | SI単位 | 定義・範囲・型 |
|---|---|---|---|
| m | Mirror追加コード | 1 | 有限実kベクトル |
| k | コード次元 | 1 | 正整数スカラー |
| W | 共有重み | 1 | 有限実d_out×d_in行列 |
| D | タスク更新差分 | 1 | Wと同型の実行列 |
| Q,K,V | attentionテンソル | 1 | 形状互換実行列 |
| S | 全保存容量 | byte（実用単位） | ゼロ以上整数、実ファイル計測 |
| t | 実行時間 | s | 非負実数 |
| L | 正規化タスク損失 | nat/token (1) | 非負実数 |
| u_seed,u_eval,u_num | 主要不確かさ | Lと同単位 | 非負実数 |
| u_c | 合成標準不確かさ | Lと同単位 | 相関時は共分散込み |
| k_cov | 包含係数 | 1 | 正実数2（暫定） |
| U | 拡張不確かさ | Lと同単位 | k_cov×u_c |
For a task update `D` and fixed `W`, only same-shaped dimensionless operators can be combined. Pair quality L, serialized S in byte, and latency t in SI seconds cannot be arithmetically combined. Gauge equivalence must be checked on full D rather than chosen low-rank factors.

## H — falsifiable marginal Mirror question
At the same whole-KV physical bytes and native token provenance, an m-conditioned compression method preserves target attention/NLL better than true bitpacked full-dimensional INT4 and native KQ-SVD, without >10% P95 overhead. The prior suggests this may FAIL.

## Mirror insertion / closest prior
> **Mirror insertion:** keep the native physical object and introduce low-description m at the existing method's task/head interface, replacing repeated dense task parameters or repeated cached state only if the native control cannot already do so.
**Native required:** true bitpacked INT4/INT8 K/V (with scale and zero-point) and query-aware native KQ-SVD at matched physical bytes.
**Prior-art:** PA432, PA426, PA425.
**Additional evidence axes:** NLL/PPL, rare-token attention argmax-flip rate, attention output error, physically serialized cache bytes and P95.

## T — directly executable protocol
Take a frozen RoPE attention decoder, calibrate source Q distributions only. Compare FP16 uncompressed, real INT8 and packed INT4 K/V with per-channel scales/zero-point, K-only SVD, native query-aware KQ-SVD and Mirror low-rank plus optionally orthogonal rotation m. Sweep equal whole-cache byte budgets, not same rank. Include adversarial near-tie attention logits and OOD queries, and direct softmax outputs. Save bitpacked cache bytes and metadata physically, never estimate bytes using float32 array length. Report NLL, PPL, rare-token argmax flip, attention score RMSE and kernel P95 (with GPU model, clock, dtype, context and batch).
Mandatory baseline hierarchy: native existing method, ordinary byte-matched rank/diagonal sharing, Mirror code, unrestricted upper bound when practical. Use dev seeds 11/12/13 for source-only configuration; fresh seeds 101/102/103/104/105 for independent task identities and all frozen workload shapes. If the original official protocol uses different seeds, an amended protocol must be committed BEFORE accessing new audit data. Never use natural target oracle adapter deltas or labels from an audit split to learn m. Preserve official native source code without editing vendor directory. Report train examples/optimizer updates, all additional inference bytes, active FLOPs, CPU/GPU P50/P95 with dtype/clock/batch/context and warm/cold conditions.

## D — PASS/FAIL/UNCERTAIN
**PASS (narrow, exploratory):** the original registered Mirror benefit survives this added native control in at least 4/5 fresh task-worlds, and gives a strict quality/actual bytes/active compute Pareto improvement over the simplest competitor.
**FAIL:** FAIL for every native INT4 Pareto dominance, bitpacking absent, inappropriate quantization scales, or any ungrounded exact function equivalence claim.
**UNCERTAIN:** native paper baseline or its required checkpoint/hardware cannot be reproduced, task split leaks, or uncertainty spans the predeclared superiority margin. Do not mutate existing scientific MA status without a full MA-specific report.

## C — alternative and failure
Rank reduction removes a rare but high-attention dimension and flips softmax selected tokens. Full-dimensional small-precision quantization preserves token ordering and wins without structured m.

## U — uncertainty / stopping
Calibrate numerical replay in float64 and deployment dtype; estimate per-task/seed paired variance, rare-case risk and serializer/version variability. If independent, `u_c=sqrt(u_seed^2+u_eval^2+u_num^2)`; otherwise add cross-covariance terms. `U=2u_c` is an indicative expansion for n=5, NOT exact 95% coverage; bootstrap by task identities. Report runtime repeat/clock error in seconds separately from serialization. Freeze dev, then all fresh worlds or stop before opening fresh for dev-stage futility. Never tune on audit.

## Source links
- PA432: https://arxiv.org/abs/2604.11501
- PA426: see prior-art map
- PA425: see prior-art map
