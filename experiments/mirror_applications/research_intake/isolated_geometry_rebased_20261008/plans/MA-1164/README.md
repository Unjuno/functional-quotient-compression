# MA-1164 — LRKV residual head basis with Mirror codes

**Stage:** DESIGNED / UNTESTED; P0; isolated branch only.  
**Prior-art:** PA423 (native LRKV), PA424 (attention equivalence), PA115 (MLA), PA385 (GVA), PA156 (LazyAttention).

## H — Rebuttable, marginal hypothesis

In a pretrained LRKV attention mechanism, replacing per-head independently stored low-rank residuals by one shared residual basis and a small structured head View code `m` saves >=15% of **actual full serialized incremental inference adapter bytes** at head count H>=8, with <=0.02 nats/token heldout NLL degradation versus the best byte-near native LRKV control and <=10% P95 decode latency overhead. These are predeclared experimental thresholds, not published achievements.

## Mirror insertion

> **Mirror insertion:** add compact head-specific `m_h` to the **native LRKV per-head residual key/value projection** to express multiple logical head operators without physically storing the full individual residual factors.

- Physical shared object: frozen common full-rank LRKV K/V maps and a compact common residual dictionary.
- Native method: shared full-rank base projection plus rank-r per-head LRKV residual; preserve original cache layout.
- Small View: `m_h` selects signed/diagonal scales and optionally two Givens rotations within rank-r residual space. The complete residual dictionary, code decoder, metadata, head router and any per-head private exception are paid bytes.
- Non-Mirror controls: ordinary LRKV rank sweep; one shared residual; low-rank basis with simple diagonal coefficient or per-head rank-one; GQA/MLA and GVA are architectural context but not substitutes for native LRKV.
- Claim boundaries: LRKV itself already shares KV; Mirror can claim only marginal resident per-head residual-map compression at same quality/latency. Attention equivalence alone is not independent head capacity.

## T — Implementation plan / independent experiment

1. Algebra screen: construct baseline LRKV with `n_heads=8`, `d_model=64`, `d_head=8`, rank `r` in {2,4}. Hold actual shared K/V tensors fixed; train rank-specific native LRKV head residuals.
2. Stage-one aligned synthetic teacher: eight teacher head residuals exactly in a known shared dictionary, plus off-dictionary stress residuals from {0,0.1,0.3} relative amplitude; use 8 source heads and 4 heldout head/role combinations on a separately instantiated source task. No heldout teacher matrix is available to the fitting algorithm; fit codes from support input/target pairs.
3. Natural screen: use independently learned native LRKV head residuals from frozen shared-base checkpoints. Learn a shared source dictionary from development/source tasks only, then fit `m_h` on target support examples at a fixed update budget. Source task identity and target heldout task identity disjoint; target trained LoRA/LRKV oracle deltas are prohibited from m-fitting.
4. Source/dev seeds 11/12/13 choose dictionary size, rank, Givens locations and LR. Fresh seeds 101..105; fix all hyperparameters, evaluation prompts, source-task/train/dev/audit partitions before audit. No hyperparameter recovery from fresh.
5. Compare: (A) native LRKV independent rank-r residual per head, (B) shared-only LRKV, (C) cheapest rank-one or diagonal native coefficient bank, (D) low-rank linear code at **same bytes as Mirror**, (E) structured Mirror `m`, (F) independent dense MHA upper bound. Compute head diversity using function-output probes; count any private residual rank.
6. First measure active FLOPs/MACs, NLL, independent head usefulness and head ablation. Count serialized NPZ/Safetensors inference bytes for all saved persistent tensors and metadata, and physically allocated cache bytes separately. Inferential quality under frozen and matched optimizer budgets is distinct from converged capacity.
7. Measure CPU float32 and, only if available, CUDA bf16/float16+fp32 comparisons with batch=1/8 and context=128/512; after 30 warmups run >=100 decode iterations with explicit synchronization, log GPU clocks if accessible and report P50/P95.
8. Preserve aligned vs natural, training-efficiency vs converged-capacity, eager-PyTorch vs fused-decode results as different evidence lanes. The native LRKV source is an arXiv preprint, so describe reproduction status accurately.

**Cheap quick falsification:** before any language training, test fixed source residual maps and heldout synthetic bilinear logits with actual serialized bank. If same-byte linear coefficients perform equally, classify M0 immediately.

## D — Decision

- **PASS (preliminary only):** at least 4 of 5 fresh worlds meet the <=+0.02 nats/token NLL margin, the median full incremental serialized persistent payload is <=0.85 the byte-near native LRKV residual control and P95 decode <=1.10 baseline, and head-function diversity does not collapse.
- **FAIL:** native LRKV rank/linear-code controls match or beat the Mirror frontier on all axes, >1 fresh world violates the quality margin, or physical cache + residual dictionary overhead consumes the claimed savings.
- **UNCERTAIN:** strong native reproduction unavailable, numerical algebra mismatches, insufficient true head/task variety or device/kernel timing unmeasured. A synthetic aligned PASS does not establish natural-task capacity.
- Fresh-world interpretation: five trials are a falsification screen. No ADOPTED claim without separate independent replication, stronger natural LMs, checkpoint provenance and near-convergence.

## C — Alternative explanation / failure path

Native LRKV may already provide all useful head diversity at far lower overhead than any extra code. A low-rank linear code can explain the same gain as a structured rotation. Even an exact shared cache does not imply the residual head maps can be compressed without accuracy loss.

## U — Uncertainty and stopping

Report per-world paired NLL differences and task-cluster bootstrap intervals. Primary `Q` is CE (nat/token); standard uncertainty components `u_seed`, `u_eval`, `u_num` (each nat/token) combine as `u_c = sqrt(u_seed^2 + u_eval^2 + u_num^2)` if independent; else include covariances. `k_cov=2` and `U=k_cov*u_c` are indicative only at n=5. Memory bytes are exact for the specified serializer, while runtime `u_t` is in seconds and reported separately. The time standard uncertainty combines clock and repetition terms in seconds, never bytes. All 3 dev and all 5 fresh runs are predeclared; no favorable audit early stopping or retuning.

## Variables and dimension audit

| Symbol | 日本語の意味 | Unit | 定義 / 範囲 / 型 |
|---|---|---|---|
| `H` | 注意ヘッド数 | 1 | integer >=2; scalar |
| `r` | ヘッド残差のランク | 1 | integer >=1; scalar |
| `m_h` | ヘッドView座標 | 1 | real length-k vector, finite |
| `k` | Viewコード次元 | 1 | integer >=1 |
| `W` | 共有KV射影重み | 1 | real d_in × d_out matrix; normalized activations |
| `B_i` | 共有残差基底 | 1 | real d_in × d_out matrix, same shape as W |
| `S` | 直列化容量 | byte (non-SI practical) | integer >=0; measured payload |
| `L` | token平均CE | nat/token (dimensionless) | finite scalar >=0 |
| `t` | decode遅延 | s | finite scalar >=0 |
| `u_c` | 品質合成標準不確かさ | nat/token | real >=0, covariance-aware |
| `k_cov` | 包含係数 | 1 | real positive, default illustrative 2 |

**Dimensional check:** `m_i * B_i` and `W` are both dimensionless weight matrices of identical shape, so their sum is conformable. The angle entries in m have radians, treated dimensionless in SI, and `cos(angle)` is dimensionless. A ratio of serialized byte counts is dimensionless. Never add s to byte.

## Required output layout on experiment activation

`README.md`, immutable `PROTOCOL.json`, `STATUS.md`, `source/`, `tests/`, `RESULTS_CORE.csv`, `VERIFICATION.json` with all dataset/seed/hardware/clock/hash provenance. No result is claimed here.

## Primary sources

- O'Neill et al., [Low-Rank Key Value Attention](https://arxiv.org/abs/2601.11471), arXiv:2601.11471.
- Tran et al., [Functional Equivalence in Attention](https://proceedings.mlr.press/v306/tran26d.html), ICML 2026.
- Tripathi et al., [Grouped Value Attention](https://arxiv.org/abs/2609.13285), arXiv:2609.13285.
