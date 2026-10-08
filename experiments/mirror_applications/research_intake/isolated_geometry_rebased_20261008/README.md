# Isolated Mirror research protocols — 2026-10-08

**QUARANTINE / DO NOT AUTO-CLAIM.** This directory and its staged MA-1156..1170 proposals live only on `research/mirror-isolated-protocols-rebased-20261008`. The authoritative worker-ready branch and current next MA-255 are untouched. Do not edit WORKER_START_HERE, CONTEXT_ROUTER, WORKER_QUEUE or active experiment files, and do not ask the current worker to scan the plans. These MA IDs are provisional until reconciled with the latest canonical registry at promotion.

## Purpose

15 literature-derived marginal Mirror hypotheses (9 earlier + 6 new; none is presumed to work); three additional detailed plans for existing MA IDs where a new ID would duplicate an existing question. PA414..PA433 contain the source titles, links and mandatory native controls.

A plan is self-contained for a cheap mechanism screen: exact insertion, native method, code shape, controls, train/dev/fresh split, pass/fail/uncertain criteria, gauge and byte audit, implementation order, hardware and expected failure modes. Full native reproduction still requires reading/checking the original paper or official implementation before publication, but no external reading is required to start the synthetic screen.

## Status / handoff

- Plan status: **DESIGNED / UNTESTED**, not SCREENING, PROMISING, FAIL or ADOPTED.
- Existing status unchanged: 29 PROMISING / 18 FAIL; verified experiment directories remain 47.
- Current worker next remains MA-255 on the canonical branch; no work is claimed for it.
- Only a read-only stdlib integrity checker and four-case pure-algebra smoke test are provided; no model experiments, pretrained weights or fresh datasets have been run. No trained-model evidence is produced.
- Promotion workflow: re-fetch canonical max MA ID, reconcile any collision and renumber on promotion; recheck PA numbers; open exactly one MA-specific experiment branch and move that plan into template; freeze first; only then run.
- Use `python experiments/mirror_applications/check_registry_integrity.py` on a full checkout. This staging branch locally updates registry/PA/status only for internal consistency.

## New proposals

- **MA-1156** [Horizontal-gauge Mirror task codes](plans/MA-1156/README.md) — P0; controls: unprojected Mirror code / direct low-rank code / native gauge-invariant optimization.
- **MA-1157** [Holonomy-robust Mirror code transport](plans/MA-1157/README.md) — P1; controls: no transport / orthogonal Procrustes / direct native re-fit / gauge-aware optimizer.
- **MA-1158** [Invariant fingerprint selected Mirror bank](plans/MA-1158/README.md) — P1; controls: output-probe clustering / weight-SVD clustering / bispectral-only invariant test / independent adapters.
- **MA-1159** [Symmetry-equivariant functional Mirror code generator](plans/MA-1159/README.md) — P0; controls: ordinary MLP hypernetwork / native UNF or NFN / direct learned LoRA task code.
- **MA-1160** [Lie-bracket controlled noncommuting Mirror composition](plans/MA-1160/README.md) — P0; controls: additive Task Arithmetic / native sequential gradient updates / full matrix products / MA-257.
- **MA-1161** [GVA head-reconstruction Mirror code bank](plans/MA-1161/README.md) — P0; controls: native GVA full M_h / GVA low-rank M_h / GQA / MLA.
- **MA-1162** [GVA physical cache alias on Mirror role switches](plans/MA-1162/README.md) — P0; controls: native GVA role switch / eager copies / target re-prefill / MA-691 alias.
- **MA-1163** [Cross-architecture functional descriptor Mirror code](plans/MA-1163/README.md) — P1; controls: native Universal Task Descriptors / per-model least-squares edits / LoRA task edits / generic factorized descriptor.

## Strengthen existing rows, not add duplicates

- **MA-338** [Stabilizer/quotient audit extension](existing/MA-338.md): MA-332/333/338, PA414, PA415.
- **MA-692** [RoPE commutant and exact cache transport audit](existing/MA-692.md): MA-691/692/693, PA414, PA417.
- **MA-1155** [Gauge and function-space falsification supplement](existing/MA-1155.md): MA-1096/1099/1155, PA414/383/384.

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

## Read-only preflight scripts

- `python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/selfcheck.py` checks the 15 plans and JSON schemas, frozen seeds, PA refs and staged registry/board. Needs a complete local checkout. No network or GPU.
- `python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/algebra_smoke.py` checks exact Q/K gauge identity, GVA query absorption, valid/invalid RoPE commutation and noncommuting matrix products. Pure algebra is **not** trained-model or efficiency evidence.
- The branch has been remotely cross-checked for registry/claim consistency; Python script execution requires a local checkout and should not be claimed without an interpreter exit status.

- **MA-1164** [LRKV head residual Mirror code](plans/MA-1164/README.md) — P0; native LRKV mandatory baseline; source PA423 and exact attention equivalence PA424.

**Collision fix:** Previous unrelated preliminary IDs MA-1116..1123 / PA382..390 are reserved by the live worker on its canonical branch. This isolated branch allocates MA-1156..1164 / PA414..424 instead. Do NOT cherry-pick old registry from the previous experimental branch.

## Recorded exploratory negative result: GVA role switching

The frozen [MA-1162 GVA Stage-0 protocol](pilots/gva_alias_stage0/PROTOCOL.json) and [five-world result](pilots/gva_alias_stage0/REPORT.md) confirm exact cache alias under identical prefix/readout-only changes, and a failure counterexample when upstream prefix states differ. The native same-basis factorized control matches Mirror's 4,936-B serialized code state, while Mirror recomposition increases eager CPU P95 latency (median ratio 1.248); therefore **Mirror-specific M0/FAIL at this narrow synthetic scope**. This is **not** MA-1162 completion and does not change its UNTESTED status. Full executed source is now checked into `pilots/gva_alias_stage0/source/run_stage0.py` and also preserved in a companion conversation ZIP; inspect SHA256 in verification.

## Recorded gauge rank audit: MA-1156 Stage-0

[Precommitted protocol](pilots/gauge_rank_stage0/PROTOCOL.json) and [fresh five-world result](pilots/gauge_rank_stage0/REPORT.md): a generic 4x2 Q/K bare-score Jacobian has rank 12 (four gauge-only null directions); adding one nontrivial RoPE rotation yields joint rank 14 (two commuting gauge directions). Both predictions held in 5/5 fresh worlds with exact invalid-shear counterexamples. This is a **negative control for false logical functional multiplicity**, not evidence of improved Mirror task quality/bytes; MA-1156 stays UNTESTED for its planned learned-code hypothesis.

## Independent runbook

Open [SELF_CONTAINED_RUNBOOK.md](SELF_CONTAINED_RUNBOOK.md) for exact isolated worktree commands, per-MA launch interface, CPU algebra pilots, and the data/audit firewall. It intentionally does not alter the active worker context.

## Seventeenth independent paper sweep

Six new UNTESTED MA-1165..1170, PA425..433 controls; [implementation and deduplication](SEVENTEENTH_SWEEP.md). Original worker and MA-255 unchanged.
- **MA-1165** [xKV cross-layer shared-basis Mirror reconstruction codes](plans/MA-1165/README.md) — P0; PA425;PA426;PA432.
- **MA-1166** [KQ-SVD multi-context Mirror query-conditioned cache View](plans/MA-1166/README.md) — P0; PA426;PA425;PA432.
- **MA-1167** [mtLoRA spectral-aware Mirror dimension-router bank](plans/MA-1167/README.md) — P0; PA428;PA427;PA373.
- **MA-1168** [FACET single-adapter dynamic Mirror feature-code boundary](plans/MA-1168/README.md) — P1; PA429;PA431;PA63.
- **MA-1169** [MoLoRA per-token physical specialist-bank compression](plans/MA-1169/README.md) — P0; PA430;PA433;PA364.
- **MA-1170** [NeuroLoRA dynamic gate versus structured Mirror online coordinate](plans/MA-1170/README.md) — P1; PA431;PA429;PA19.

### Independent current-state preflight

`python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/selfcheck.py` verifies 1170 MA rows, 433 PA entries, all 15 frozen designs, claim/status agreement and isolation markers on a complete local checkout. This script does not change the worker and needs no GPU/network. No synthetic result counts as native-paper reproduction.

## KQ-SVD audit-algebra Stage-0 negative-control evidence

The [preregistered MA-1166 control study](pilots/kqsvd_controls_stage0/REPORT.md), its [complete source](pilots/kqsvd_controls_stage0/source/run_stage0.py), [four unit tests](pilots/kqsvd_controls_stage0/source/test_stage0.py), [development/fresh CSV](pilots/kqsvd_controls_stage0/results/fresh_raw.csv) and [verification](pilots/kqsvd_controls_stage0/results/VERIFICATION.json) are preserved here. Native optimal score-aware rank-4 oracle mean error was 0.416 vs 0.902 key-only SVD on five fresh synthetic worlds; ordinary linear code bank 12,580 B vs full projected bank 15,886 B. **This does NOT certify any structured Mirror benefit** and leaves all MA statuses unchanged.

## Internal historical-formula crossovers (new isolated branch only)

Four extra UNTESTED MA-1171..MA-1174 and 12 existing-MA source-controlled protocol extensions are documented in [FORMULA CROSSOVER INDEX](../formula_crossovers_20261008/README.md). Original FQC E1–E7, SRM, TM and RA-Mirror formulas and their evidence boundaries are in the [audited formula ledger](../formula_crossovers_20261008/FORMULA_LEDGER.md). The newly extended read-only selfcheck validates 1174 staged rows and 19 independent plans. Current worker and scientific claims unaffected.
