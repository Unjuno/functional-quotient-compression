# MA-1097 — Supplement: ALoRA output-side B-sharing correction
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
On natural independently trained LoRA adapter banks with identical frozen base, a small Mirror m improves heldout task quality per actual saved byte beyond native ALoRA shared-B and byte-near A-code bank; if native B-sharing already suffices, there is no Mirror-specific gain.

## Mirror insertion / closest prior
> **Mirror insertion:** keep the native physical object and introduce low-description m at the existing method's task/head interface, replacing repeated dense task parameters or repeated cached state only if the native control cannot already do so.
**Native required:** ALoRA single output matrix B and task-specific input matrices A, official source https://github.com/OptMN-Lab/ALoRA; shared-A and independent LoRA controls.
**Prior-art:** PA427, PA367, PA373.
**Additional evidence axes:** heldout NLL, task-mean and worst, gauge delta max error, serialized bank bytes, bytes/task, CPU/GPU P95, training FLOPs.

## T — directly executable protocol
Freeze same base checkpoint and target layer. For each task, train rank-4/8 independent LoRA with truly different initial A, train official native ALoRA shared B with per-task A, shared-A baseline, linear A-code factor bank and structured m input-side rotations. Use 8 source tasks for basis, disjoint 4 target tasks with 32/128 support examples and heldout evaluation. Do not access target LoRA oracle update while learning m. Audit gauge by B->BG, A->G^-1 A for random well-conditioned invertible G and match the full delta, not A/B component cosine. Sweep number of tasks K={4,8,16}; source-only select rank. Count all B, A, m, metadata and router real bytes. Record target NLL, old-task retention, tasks/s, optimizer updates.
Mandatory baseline hierarchy: native existing method, ordinary byte-matched rank/diagonal sharing, Mirror code, unrestricted upper bound when practical. Use dev seeds 11/12/13 for source-only configuration; fresh seeds 101/102/103/104/105 for independent task identities and all frozen workload shapes. If the original official protocol uses different seeds, an amended protocol must be committed BEFORE accessing new audit data. Never use natural target oracle adapter deltas or labels from an audit split to learn m. Preserve official native source code without editing vendor directory. Report train examples/optimizer updates, all additional inference bytes, active FLOPs, CPU/GPU P50/P95 with dtype/clock/batch/context and warm/cold conditions.

## D — PASS/FAIL/UNCERTAIN
**PASS (narrow, exploratory):** the original registered Mirror benefit survives this added native control in at least 4/5 fresh task-worlds, and gives a strict quality/actual bytes/active compute Pareto improvement over the simplest competitor.
**FAIL:** FAIL if any target oracle delta leaks, random rank-gauge changes scientific conclusion, or native ALoRA/simple A code Pareto-dominates m.
**UNCERTAIN:** native paper baseline or its required checkpoint/hardware cannot be reproduced, task split leaks, or uncertainty spans the predeclared superiority margin. Do not mutate existing scientific MA status without a full MA-specific report.

## C — alternative and failure
Shared A apparent alignment is identical initialization, not task semantics; native B-sharing and a simple linear A code consume fewer bytes without loss, leaving zero special value for structured Mirror.

## U — uncertainty / stopping
Calibrate numerical replay in float64 and deployment dtype; estimate per-task/seed paired variance, rare-case risk and serializer/version variability. If independent, `u_c=sqrt(u_seed^2+u_eval^2+u_num^2)`; otherwise add cross-covariance terms. `U=2u_c` is an indicative expansion for n=5, NOT exact 95% coverage; bootstrap by task identities. Report runtime repeat/clock error in seconds separately from serialization. Freeze dev, then all fresh worlds or stop before opening fresh for dev-stage futility. Never tune on audit.

## Source links
- PA427: https://aclanthology.org/2026.findings-acl.625/
- PA367: see prior-art map
- PA373: see prior-art map
