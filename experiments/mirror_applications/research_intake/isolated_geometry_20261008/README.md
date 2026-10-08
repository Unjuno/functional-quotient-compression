# Isolated Mirror research protocols — 2026-10-08

**QUARANTINE / DO NOT AUTO-CLAIM.** This directory and its staged MA-1116..1123 proposals live only on `research/mirror-isolated-gauge-cache-protocols-20261008`. The authoritative worker-ready branch and current next MA-255 are untouched. Do not edit WORKER_START_HERE, CONTEXT_ROUTER, WORKER_QUEUE or active experiment files, and do not ask the current worker to scan the plans. These MA IDs are provisional until reconciled with the latest canonical registry at promotion.

## Purpose

Eight literature-derived, non-duplicative **marginal Mirror** hypotheses (not 8 proved improvements); three additional detailed plans for existing MA IDs where a new ID would duplicate an existing question. PA382..PA390 contain the source titles, links and mandatory native controls.

A plan is self-contained for a cheap mechanism screen: exact insertion, native method, code shape, controls, train/dev/fresh split, pass/fail/uncertain criteria, gauge and byte audit, implementation order, hardware and expected failure modes. Full native reproduction still requires reading/checking the original paper or official implementation before publication, but no external reading is required to start the synthetic screen.

## Status / handoff

- Plan status: **DESIGNED / UNTESTED**, not SCREENING, PROMISING, FAIL or ADOPTED.
- Existing status unchanged: 29 PROMISING / 18 FAIL; verified experiment directories remain 47.
- Current worker next remains MA-255 on the canonical branch; no work is claimed for it.
- No code, weights or fresh data are fetched by these notes. No trained-model evidence is produced.
- Promotion workflow: re-fetch canonical max MA ID, reconcile any collision and renumber on promotion; recheck PA numbers; open exactly one MA-specific experiment branch and move that plan into template; freeze first; only then run.
- Use `python experiments/mirror_applications/check_registry_integrity.py` on a full checkout. This staging branch locally updates registry/PA/status only for internal consistency.

## New proposals

- **MA-1116** [Horizontal-gauge Mirror task codes](plans/MA-1116/README.md) — P0; controls: unprojected Mirror code / direct low-rank code / native gauge-invariant optimization.
- **MA-1117** [Holonomy-robust Mirror code transport](plans/MA-1117/README.md) — P1; controls: no transport / orthogonal Procrustes / direct native re-fit / gauge-aware optimizer.
- **MA-1118** [Invariant fingerprint selected Mirror bank](plans/MA-1118/README.md) — P1; controls: output-probe clustering / weight-SVD clustering / bispectral-only invariant test / independent adapters.
- **MA-1119** [Symmetry-equivariant functional Mirror code generator](plans/MA-1119/README.md) — P0; controls: ordinary MLP hypernetwork / native UNF or NFN / direct learned LoRA task code.
- **MA-1120** [Lie-bracket controlled noncommuting Mirror composition](plans/MA-1120/README.md) — P0; controls: additive Task Arithmetic / native sequential gradient updates / full matrix products / MA-257.
- **MA-1121** [GVA head-reconstruction Mirror code bank](plans/MA-1121/README.md) — P0; controls: native GVA full M_h / GVA low-rank M_h / GQA / MLA.
- **MA-1122** [GVA physical cache alias on Mirror role switches](plans/MA-1122/README.md) — P0; controls: native GVA role switch / eager copies / target re-prefill / MA-691 alias.
- **MA-1123** [Cross-architecture functional descriptor Mirror code](plans/MA-1123/README.md) — P1; controls: native Universal Task Descriptors / per-model least-squares edits / LoRA task edits / generic factorized descriptor.

## Strengthen existing rows, not add duplicates

- **MA-338** [Stabilizer/quotient audit extension](existing/MA-338.md): MA-332/333/338, PA382, PA383.
- **MA-692** [RoPE commutant and exact cache transport audit](existing/MA-692.md): MA-691/692/693, PA382, PA385.
- **MA-1115** [Gauge and function-space falsification supplement](existing/MA-1115.md): MA-1096/1099/1115, PA382/383/384.

## Strict scientific boundaries

1. **Function:** equivalence under a gauge change has zero new independent task function count; local probe agreement is not a global guarantee.
2. **Bytes:** compare actual serialized inference state, including codebooks, bases, routers, maps and normalization; show physical cache aliasing instead of identical numeric copies.
3. **Compute:** active MACs/FLOPs, training updates, adaptation examples, P50/P95 wall latency and cold/warm switch measurements are separate axes. CPU/GPU results are not interchangeable.
4. **Baseline:** compare native method, cheapest byte-near ordinary low-rank/diagonal/shared-basis control, and independent upper reference when available.
5. **Evidence:** 5 fresh seeds are sufficient only for initial falsification. No ADOPTED conclusion without separate replication and real-task evidence.
6. **Uncertainty:** report task/seed variance and numerical conditioning; prefer UNCERTAIN to changing the gate after seeing audit.

### Variable and unit glossary (used by every plan)

| symbol | Japanese meaning | SI unit | definition / domain / type |
|---|---|---|---|
| m | Mirror機能座標 | 1 | m in real^k; dimensionless vector; static per-task unless specified |
| k | 座標次元 | 1 | integer >= 1; scalar |
| K | タスク／役割数 | 1 | integer >= 1; scalar |
| B_i | 共有変換基底 | 1 | same weight-space shape as target map; real matrix/tensor |
| W | 実行される線形重み | 1 | real d_in x d_out (depending row convention) |
| Q/K/V | Query・Key・Value行列 | 1 | token x head-dim real matrices; activation-scaled dimensionless |
| t | 実行時間 | s | monotonic wall clock; nonnegative scalar |
| S | 直列化容量 | byte = 8 bit | nonnegative integer (byte is practical non-SI unit); scalar |
| L | 交差エントロピー | nat/token | dimensionless normalized per token; scalar |

All learned weights/activations in these neural models are dimensionless after normalization; a weighted sum of B_i by dimensionless m therefore has the same units as W. Compute latency in seconds and storage in bytes (not equivalent to either latency or FLOPs).
