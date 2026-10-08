# MA-1122 — GVA physical cache alias on Mirror role switches

**Stage:** DESIGNED / UNTESTED (isolated research branch; not a worker claim).  
**Priority:** P0; **Evidence lane:** MECHANISM first; **Prior art:** PA385;PA364;PA365;PA156.  
**Frozen research baseline:** `407ca7e2047326d1e4b753e55e05c4730f26f32b` (the cited canonical branch as of staging).  
**Closest registry:** MA-691 canonical lazy KV, MA-692 RoPE commuting View, MA-1112 switch-state truthfulness; GVA native is strongest new control.

## H — Falsifiable hypothesis

For a verified identical prefill trajectory with readout-only role m, 2 or 4 logical role switches preserve target logits within float32 atol=1e-4 and reuse a single physical cache allocation, while reducing combined resident-prefix bytes >=25% versus native per-role copies with P95 switch <=1.10 native GVA readout-switch control.

## Mirror insertion: exact location and marginal claim

> **Mirror insertion:** this experiment adds a compact functional code `m` to **one physically aliased grouped-V and positional-key prefix cache** at **Change only the query-side reconstruction/readout code m after a frozen prefix; physically alias the exact V_cache/Kpos buffers** so that **multiple logical role/readout functions reusing the same prefix tensors** can be expressed without a corresponding full physical object for every logical case.

- **Native method:** Build native GVA prefix once. Compare zero-switch, per-role copied caches, native per-role full GVA M_h without m, lazy algebraic transform, re-prefill, and proposed structured readout m. Do not compare to a purposefully inefficient memcpy-only strawman as the main control.
- **What is stored physically:** one physically aliased grouped-V and positional-key prefix cache.
- **What changes causally when m changes:** small readout m switched between compatible GVA attention roles without rebuilding prefix.
- **Nearest non-Mirror control:** native GVA role switch / eager copies / target re-prefill / MA-691 alias.
- **Native prior vs proposed delta:** Separate algebraic query-side readout switching from physically shared cache and show precisely when a changed upstream hidden trajectory invalidates the prefix.

## T — Executable minimal test

**Implementation recipe.** In 4-head 2-group GVA, construct three switch conditions: (A) identical prefix content/weights with readout-only m (exactly safe), (B) compatible invertible readout changes with explicit conjugation/commutant check, (C) upstream attention weights changed during prefill (unsafe). Trace data_ptr and allocator counts, tensor lifetime/refcounts and copy-on-write; compare each target to true target re-prefill. Include batch fork/cold/warm cache and RoPE offset changes.

**Dataset/harness.** Frozen tiny decoder with deterministic 128/256 token prefixes and switch at 32/64/128 tokens, 2 and 4 roles, heldout source prompts. No training is needed for algebraic case; a separately pretrained tiny GVA is required for learned-role quality.

**Training/data partition.** Development random seeds `11,12,13` select rank, strength, learning rate and stopping rules. Fresh seeds `101..105` and all heldout task identities/pairs are locked before observing their losses. Within each task, support/dev/audit examples are independent; only support labels may update m. Five fresh trials constitute a mechanism screen, not a statistical proof of capacity. Freeze exact checkpoint hashes, task generation and preprocessing.

**Required controls:** native GVA role switch / eager copies / target re-prefill / MA-691 alias; independent unrestricted upper benchmark if scientifically meaningful. A native method must be implemented faithfully or marked UNREPRODUCED. No comparison with deliberately crippled baselines.

**Environment/implementation prerequisites:** Stage 1 CPU PyTorch contiguous tensor pointers and byte storage; optional CUDA GPU for actual VRAM `max_memory_allocated` with explicit synchronize, report device, dtype, batch and context.

**Measured quantities:** Exact target logits/NLL, heap/VRAM allocation and alias pointers, prefix bytes, TTFT, 50/95/99% switching time, synchronization, additional FLOPs, prefill copies, buffer fork counts.

## D — Predeclared decision rule

**PASS (preliminary mechanism gate):** All five fresh prompt seeds case A and B meet target-logit atol=1e-4 float32 and actual alias equality with no prefix copies; vs real independent per-role caches physical bytes <=0.75 and measured P95 switch <=1.10 native GVA role-read baseline. Case C MUST be detected as unsafe, not counted as shared.

**FAIL:** Case C incorrectly declared safe, any prefix copy called alias, quality drift exceeds tolerance in case A, or native GVA readout handles switches equally well at lower memory/latency.

**UNCERTAIN:** numerical/gauge tests pass but a paper-native control, unbiased heldout replication, correct physical-memory counter, or stable confidence interval is unavailable. Do not silently loosen the thresholds to declare success.

The thresholds are **design choices for this experiment**; they are not reported improvements in any cited paper. These results alone can be called *mechanism PASS*, never ADOPTED. Confirmed deployment Pareto value requires a separate larger replication and the strongest available native baseline.

## C — How the claim can fail (counter-hypothesis)

Exact algebra at an attention readout does not imply cache equivalence after different upstream hidden states; native GVA with shared heads may already provide the same alias.

## U — Uncertainty and error budget

Allocator caching and tensor view alias semantics, async GPU timing synchronization, RoPE offset and token provenance, copy-on-write, model training differences; log device allocator behavior and confidence interval over prompts.

For each dimensionless quality metric, report paired differences over task/seed units, a bootstrap 95% interval, and per-seed values (avoid interpreting 5 runs as definitive). For numeric invariance/precision, report max absolute and relative error in float64 and the deployed dtype. For latency, report actual clock source, clock/GPU frequency when available, 30+ warm iterations and P50/P95, with asynchronous device synchronization.

## Minimal worker-free runbook

1. Implement A/B/C compatibility proof at the exact cached tensor interface.
2. Write independent reference prefill for each target role.
3. Inspect true buffers (data_ptr, storage nbytes), memory snapshots and explicit copies.
4. Time cold and warm switches with synchronization and frozen lengths.
5. Compare against native GVA not only naive per-role caches.
6. Document incompatible prefill counterexample as first-class negative result.

**Hard stop:** Never extrapolate from identical-prefix readout-only compatibility to cross-model or independently generated prefixes.

**Required experiment outputs when later promoted:** `README.md`, frozen `PROTOCOL.json`, `STATUS.md`, implementation under `source/`, tests, `RESULTS_CORE.csv`, `VERIFICATION.json` containing dataset/hash/seed/byte/compute provenance. None of these runs have been executed by this design document.

### Quantified uncertainty and sequential stopping

For quality observable `Q`, combine same-unit standard uncertainties `u_seed` (independent seed/world mean standard error), `u_eval` (task-cluster bootstrap sample error), and `u_num` (numerical replay effect expressed in Q units). The independence model is `u_c = sqrt(u_seed^2 + u_eval^2 + u_num^2)`, expanded `U = k_cov * u_c` with indicative `k_cov = 2`. For only five fresh worlds, this **does not guarantee 95% coverage**; report paired per-seed outcomes and a task-bootstrap 95% interval. If terms correlate, use full covariance. Runtime has separate `u_t = sqrt(u_repeat^2 + u_clock^2)` (s); storage S is measured exactly as serialized bytes. Never combine uncertainties of different dimensions.

| Symbol | Meaning (Japanese) | Unit (SI/practical) | Definition, domain, type |
|---|---|---|---|
| `Q` | 主評価量 | nat/token or 1 | finite real scalar, preselected |
| `u_seed` | 独立試行による標準不確かさ | Qと同単位 | nonnegative real scalar, paired std/sqrt(n), n=5 |
| `u_eval` | 評価標本の標準不確かさ | Qと同単位 | nonnegative real scalar, task bootstrap |
| `u_num` | 数値誤差成分 | Qと同単位 | nonnegative real scalar, dtype/replay |
| `u_c` | 合成標準不確かさ | Qと同単位 | nonnegative scalar, covariance-aware |
| `k_cov` | 包含係数 | 1 | dimensionless positive scalar; indicative 2 |
| `U` | 拡張不確かさ | Qと同単位 | nonnegative scalar, `k_cov * u_c` |
| `u_t,u_repeat,u_clock` | 実行時間の標準不確かさ | s | real nonnegative scalars, runtime only |

**Dimension check:** every squared term in `u_c` has unit Q², so square-root returns unit Q; `k_cov` has unit 1. Runtime uncertainty remains seconds, not bytes. **Sequential stop:** tune only on 3 dev seeds, run all 5 frozen fresh worlds without selective early stopping, and never use fresh outputs to retune. Separate independent replication is mandatory for ADOPTED.

## Variable table / dimension check

| Symbol | Meaning (Japanese) | Unit (SI / practical) | Domain / type |
|---|---|---|---|
| `m` | Mirror機能座標 | 1 | real vector of length k |
| `k` | 機能座標の数 | 1 | positive integer scalar |
| `W`, `B_i` | 対象重み・共有基底 | 1 | same-shaped real matrices, finite entries |
| `Q,K,V` | 注意機構の活性値 | 1 | real token x channel matrices, compatible attention shape |
| `S` | 追加推論状態の容量 | byte (non-SI byte unit) | nonnegative integer; actual serialized payload |
| `L` | 正規化交差エントロピー | nat/token (dimensionless) | finite nonnegative scalar |
| `t` | 推論実時間 | second (s) | nonnegative scalar |

Dimension check: `m_i * B_i` has the same dimensionless learned-weight units as `W`; `W + sum_i m_i B_i` is conformable by matrix shape, and logit error/CE is not a byte or second. A byte ratio `S_m/S_control` is dimensionless; compare wall-clock and bytes on distinct axes.

## Primary literature and verified venue

- **PA385:** Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction — https://arxiv.org/abs/2609.13285 (arXiv:2609.13285 (2026 preprint)); Reconstructs head content keys from grouped values with an absorbable query-side map and separately cached position key. Mirror can only claim extra gain from compact per-head/readout m or safe role sharing; paper itself does not establish end-to-end kernel throughput gains.

**Evidence status:** FACT = paper exists and offers described native method; HYPOTHESIS = incremental Mirror m benefit; UNKNOWN = whether Pareto improvement exists. No results may be inferred from this plan.
