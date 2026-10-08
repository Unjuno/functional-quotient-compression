# MA-1112 — Supplement: Standard LoRA base-prefix reuse versus physical alias
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
For ordinary independently trained LoRA Q/K/V/O specialists, physical aliasing and task-native quality are preserved by a source-calibrated Mirror cache View better than direct base-prefix reuse or established multi-LoRA cache sharing; numerical reuse via copies alone is insufficient.

## Mirror insertion / closest prior
> **Mirror insertion:** keep the native physical object and introduce low-description m at the existing method's task/head interface, replacing repeated dense task parameters or repeated cached state only if the native control cannot already do so.
**Native required:** native per-specialist prefilling, standard-LoRA base-prefix reuse, aLoRA-compatible cache, LRAgent and PReCache base-plus-low-rank cache.
**Prior-art:** PA433, PA364, PA365.
**Additional evidence axes:** true peak VRAM and resident cache bytes, data_ptr alias, copies/forks, target NLL/EM, TTFT warm/cold and P95.

## T — directly executable protocol
Use one identical frozen Qwen-style tiny shared backbone and two independently trained LoRA specialists, matched tokenization/RoPE, 128/512 and long-context 2k+ token prefixes. Freeze base and adapters. Route prefix using (A) native full prefilling, (B) direct base-cache reuse, (C) PReCache/LRAgent if native code executable, (D) exact readout-only compatible Mirror View, and (E) approximate learned translator learned only on source calibration. Measure target logit NLL and exact target-prefill output, actual pointer alias (data_ptr, storage ID), peak allocated memory including independent branch caches, copy-on-write count, warm/cold TTFT and decode P95. Include an incompatible upstream LoRA change counterexample and reject false equivalence. For quality, use heldout prompt pairs and stop boundary search on dev only.
Mandatory baseline hierarchy: native existing method, ordinary byte-matched rank/diagonal sharing, Mirror code, unrestricted upper bound when practical. Use dev seeds 11/12/13 for source-only configuration; fresh seeds 101/102/103/104/105 for independent task identities and all frozen workload shapes. If the original official protocol uses different seeds, an amended protocol must be committed BEFORE accessing new audit data. Never use natural target oracle adapter deltas or labels from an audit split to learn m. Preserve official native source code without editing vendor directory. Report train examples/optimizer updates, all additional inference bytes, active FLOPs, CPU/GPU P50/P95 with dtype/clock/batch/context and warm/cold conditions.

## D — PASS/FAIL/UNCERTAIN
**PASS (narrow, exploratory):** the original registered Mirror benefit survives this added native control in at least 4/5 fresh task-worlds, and gives a strict quality/actual bytes/active compute Pareto improvement over the simplest competitor.
**FAIL:** FAIL when alias is only values copied to different buffers, when native spec quality drifts, or when simpler native cache correction saves at least as much at lower compute.
**UNCERTAIN:** native paper baseline or its required checkpoint/hardware cannot be reproduced, task split leaks, or uncertainty spans the predeclared superiority margin. Do not mutate existing scientific MA status without a full MA-specific report.

## C — alternative and failure
Prefix KV values may be reused numerically but physically copied; independently trained specialist prefix activation trajectories are not functionally compatible with base cache. Native PReCache already provides cheaper correction.

## U — uncertainty / stopping
Calibrate numerical replay in float64 and deployment dtype; estimate per-task/seed paired variance, rare-case risk and serializer/version variability. If independent, `u_c=sqrt(u_seed^2+u_eval^2+u_num^2)`; otherwise add cross-covariance terms. `U=2u_c` is an indicative expansion for n=5, NOT exact 95% coverage; bootstrap by task identities. Report runtime repeat/clock error in seconds separately from serialization. Freeze dev, then all fresh worlds or stop before opening fresh for dev-stage futility. Never tune on audit.

## Source links
- PA433: https://arxiv.org/abs/2609.17109
- PA364: see prior-art map
- PA365: see prior-art map
