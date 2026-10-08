# 2026-10-09 LiME-based one-pass/multi-output and native-method crossovers

**ISOLATED RESEARCH**, not the authoritative worker queue. Canonical worker remains `research/mirror-application-worker-ready-20261007` with its current MA-255 next. Main and scientific statuses unchanged; MA-1175 has **no special priority**.

## Current results / standalone reproducibility

- [LME01 frozen protocol](LME01_FROZEN_PROTOCOL.json) — 3 dev [71,72,73] and 5 fresh [701..705], fixed source/model/optimizer before fresh.
- [Source freeze](pilots/lme01/SOURCE_FREEZE.json) — exact source and tests frozen after dev and before fresh. [model](pilots/lme01/source/run_lme01.py) and [8 tests](pilots/lme01/source/test_lme01.py).
- [Detailed LME01 report](pilots/lme01/REPORT.md) — **native onepass 5-head strongest, Mirror4 m fails same-byte ordinary code gate**, not a full native LiME paper replication.
- [Full dev CSV](pilots/lme01/results/dev_raw.csv), [fresh CSV](pilots/lme01/results/fresh_raw.csv), [seven-method core](pilots/lme01/results/RESULTS_CORE.csv), [verification](pilots/lme01/results/VERIFICATION.json), [154-field replay](pilots/lme01/results/REPLAY_VERIFICATION.json).
- [Source/replay scripts](pilots/lme01/source/replay_check.py), [report generation](pilots/lme01/source/make_report.py).

## New method-specific isolated MA experiments (all UNTESTED)

- [MA-1189 LiME shared adapter Mirror m](plans/MA-1189/README.md): native LiME shared adapter and zero-param router is mandatory strong baseline.
- [MA-1190 M3LoRA native mixer code](plans/MA-1190/README.md): multi A/B subspace and minor singular initialization native.
- [MA-1191 FAAR dense multi-task frequency/rank](plans/MA-1191/README.md): original PDRS + Task-Spectral Pyramidal Decoder remain native.
- [MA-1192 MoEP whole routed block families](plans/MA-1192/README.md): whole Attention–FFN route active compute/native FFN-MoE controls.
- [MA-1193 DR-MGF task-preferred paths](plans/MA-1193/README.md): native gradient deconflict, PCGrad/CAGrad mandatory.

Every new plan is fully standalone H/T/D/C/U, source method, exact Mirror insertion, synthetic dimensions, native hard controls, source/fresh firewall, realistic byte/runtime metrics and numerically honest rejection gates. No MA completion or paper-benchmark score is claimed by these design entries.

## Source research / reuse

- [Literature and comparisons](NATIVE_METHOD_SWEEP.md) — PA463..PA470 and LME01 observation boundaries.
- [Read-only integrity checks](selfcheck.py) — verify 1193 MA, 470 PA, 5 plans, 7 supplements, prior 47 claim records on a complete checkout.
- [Existing ID supplements](existing/) — native upgrades to MA-1175, 1178, 1167, 1105, 1185, 1176, 1173; those are NOT new candidate statuses.

**Exact metric boundary:** K different heads vs K independent capability, shared adapter vs shared compute, metadata+decoder+code bytes vs logical tensor values, CPU eager vs production CUDA. If prior LiME already attains native output multiplicity, changing name to Mirror is M0, not novelty.
