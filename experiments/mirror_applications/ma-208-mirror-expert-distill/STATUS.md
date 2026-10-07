# MA-208 status

- Status: FAIL at the registered development quality gate.
- Branch: `research/ma-208-mirror-expert-distill-20261007`
- Base commit: `621dec2ffdf55c3f0503dd75695ea27fdc68544c`
- Initial protocol freeze: `faad9ec1a1af5c474e7bdff39829931b8e221972`
- Measurement amendment: `d421074e08a8f83e8e24eccf37bfc143ea7aa88f`
- Development: complete; LR 0.003 selected by the registered aggregate metric.
- Development gate: FAIL in both aligned seeds on relative KL versus multi-head (38.8x and 1,027.7x; limit 1.10x).
- Fresh/audit opened: yes, in error after overlooking the relative-KL subgate. Preserve all 630 rows as exploratory only; do not use them as confirmatory evidence.
- Fresh split integrity: false.
- Result report and complete raw rows: `README.md`, `RESULTS_CORE.csv`.
- Verification: exact fresh rerun for quality, bytes and compute proxy; 3 tests pass.

## H — hypothesis

Specialist teachers in one hidden Givens orbit can be distilled into a shared student and task angles with lower actual bytes than ordinary per-task student heads, while independent teachers require private weights.

## T — setup

Synthetic five-task four-class MLP. Compare hard tie, sequential KD, private output heads, rank-2 LoRA, Mirror hidden views, rank-2 hypernetwork, independent full students, and teacher upper references. Development seeds 20801/20802; fresh seeds 20811–20813 were run despite the development gate and are exploratory only.

## D — decision

FAIL at development. Mirror passed absolute KL, top-1, ECE and actual byte subgates, but failed the preregistered relative-KL condition against a near-exact multi-head control in both development worlds. The separate fresh-access mistake is disclosed and represented as a false split-integrity check.

## C — strongest counter-hypothesis

The low-dimensional Givens orbit was deliberately built into the aligned teachers; ordinary multi-head KD fit more exactly and rank-2 LoRA provides a simple equivalent low-rank output update.

## U — unresolved

Near-convergence behavior, generic byte-matched basis comparison, natural MoE distillation and language-model quality remain untested.
