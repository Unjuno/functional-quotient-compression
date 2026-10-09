# MA-457 — Continual path reuse before module birth

Status: **FAIL** (development screen; fresh seeds sealed)  
Evidence lane: MECHANISM/STORAGE/CONTINUAL_TRANSFER  
Branch: `research/ma-457-continual-path-reuse-20261008`  
Base commit: `2264c54`  
Prior art: PA80 (PathNet)

## H — Hypothesis

A shared 4×4 module plus a one-scalar task coordinate over a shared rank-one basis could absorb six related tasks before birthing private full modules, with at most three total modules, low held-out error, retained early-task quality and at least 25% fewer actual payload bytes than a PathNet-style reuse-then-birth policy. A direct native rank-one parameterization was frozen to test attribution.

## T — What ran

Eight seeded synthetic 4D linear tasks arrive in sequence: six share a base matrix plus scalar multiples of one rank-one update; two are independent outliers. Each task has 256 support and 256 query examples with output noise σ=0.01. The shared basis was trained jointly on tasks 0–3 (1,600 Adam updates × 32 examples); tasks 4–7 received 300 scalar-code updates each. A full private matrix was born when support RMSE exceeded 0.05. Development seeds: 45701 and 45702. Controls: frozen PathNet-style route reuse and module birth, one pooled shared module, a direct native `W + m u vᵀ` rank-one control, and independent full task matrices. Fresh seeds 45711–45713 stayed sealed after frozen gates failed.

The PathNet-style screen greedily reuses the lowest-support-error frozen module if it is below threshold; otherwise it fits and stores a new matrix. This is a small allocation mechanism screen, not a reproduction of PathNet's evolutionary training. The independent upper control fits one matrix per task by least squares. All task routing indices, code, shared basis, private modules and schema metadata are included in actual uncompressed `.npz` payloads.

## Results — facts

| Seed | Method | Mean query RMSE | Payload bytes | Physical modules | New births | Train wall (s) |
|---:|---|---:|---:|---:|---:|---:|
| 45701 | PathNet-style | 0.01012 | 1,442 | 8 | 7 | 0.0168 |
| 45701 | Mirror | 0.02149 | 2,222 | 3 | 2 | 2.737 |
| 45701 | native rank-one | 0.02149 | 2,222 | 3 | 2 | 2.737 |
| 45701 | independent | 0.01012 | 1,180 | 8 | 7 | 0.0029 |
| 45701 | shared only | 0.62419 | 735 | 1 | 0 | 0.0026 |
| 45702 | PathNet-style | 0.00996 | 1,442 | 8 | 7 | 0.0226 |
| 45702 | Mirror | 0.01469 | 2,222 | 3 | 2 | 2.899 |
| 45702 | native rank-one | 0.01469 | 2,222 | 3 | 2 | 2.899 |
| 45702 | independent | 0.00996 | 1,180 | 8 | 7 | 0.0029 |
| 45702 | shared only | 0.67868 | 735 | 1 | 0 | 0.0026 |

Mirror reduced full-module births from seven to two and retained the early functions (measured maximum RMSE increase 0 in both seeds because accepted codes/modules were frozen after assignment). But Mirror's payload was 2,222 bytes versus 1,442 for PathNet (+54.1%) and 1,180 for independent matrices (+88.3%). The actual `.npz` count matters here: the Mirror state stores seven named arrays, whose per-array serialization headers consume substantial space. Mirror's mean query error also exceeded PathNet/independent, and task 1 in seed 45701 had RMSE 0.037, above the aligned-task 0.035 gate. Native rank-one conditioning has identical payload hash, bytes and outputs in both seeds.

The Mirror fit used 2,800 Adam updates over 89,600 sampled examples and 2.74–2.90 s on this CPU. PathNet's closed-form sequential fits took 0.017–0.023 s; independent closed-form fits took about 0.003 s. The rough inference proxy was 24 operations/example for Mirror and 16 for PathNet/independent. Query batches were small and timings are not reliable deployment-latency evidence. The training method and count differ across conditions, so these wall and proxy values describe the screen, not a matched-compute capacity result.

Three incomplete software preflights and an initial run with unmeasured closed-form control times were preserved under `runs/invalid_preflight_*` and `runs/superseded_initial_compute_reporting_*`; none was used for tuning or included above. The corrected full runs are in `runs/dev_45701` and `runs/dev_45702`. `verification_report.json` confirms payload hash/byte checks, metric replay and exact native alias.

## D — Decision

**FAIL.** The mechanism reduces the number of private modules but misses the storage gate: serialized payload is larger than both PathNet-style growth and the independent bank. It also uses more training time, has worse mean query error than those controls, misses the aligned-task RMSE bound on one seed, and is exactly the same function as a standard rank-one shared-basis update. Early-task retention is preserved by freezing, but this alone does not provide a useful Pareto point. Fresh tasks remain sealed.

## C — Strongest counter-hypothesis

The effect is ordinary rank-one task-vector adaptation plus a birth threshold. The native rank-one control is an exact alias, while direct task-specific modules give better quality and smaller serialized payload in this tiny setting. Module count alone hides the code/basis/metadata storage and optimization cost.

## U — Unconfirmed

This toy stream does not establish natural continual transfer, PathNet-scale routing, near-convergence capacity, or robust inference latency. The six aligned task functions were generated from the same rank-one family. Fresh seeds were not opened under the frozen gate.
