# MA-502 — Many task codes over a shared LoReFT subspace

Status: **FAIL for Mirror-specific attribution** (aligned quality/storage screen passes)  
Branch: `research/ma-502-loreft-many-task-codes-20261008`  
Prior art: PA96 (ReFT / LoReFT)

## H — Hypothesis

For exactly aligned rank-four hidden-state interventions, one shared LoReFT-style basis plus one four-value code per task will preserve all tested functions as task count grows from 1 to 256 and use at most 0.30x independent rank-four bytes by eight tasks.

## T — Executed protocol

Two frozen development seeds (50201/50202), task counts {1,2,4,8,16,32,64,128,256}, hidden dimension 24 and rank four. Functions share one U/V basis and each explicit task has an independently generated four-value coordinate. Each function receives 64 calibration examples and 128 heldout inputs. Basis fitting uses at most the first eight task IDs. Compare shared codes, exact native shared coefficients, independent per-task rank-four factors and full per-task matrices. All bases, codes, factors, matrices and metadata are included in uncompressed NPZ byte counts. Fresh seeds 50211–50213 remain sealed.

## D — FAIL for Mirror-specific attribution

Every tested task count had distinct stored codes and functions, and the full-matrix upper replayed. Shared relative output RMSE ranged from 1.1e-6 to 3.6e-6 across both seeds and all counts. Actual shared payload was 1,923 B for one task (slightly larger than independent rank four at 1,915 B), 2,035 B for eight tasks versus 7,403 B independent (27.49%), and 6,005 B for 256 tasks versus 201,837 B (2.98%). The storage ratio first falls below 0.30 at eight tasks. Shared and independent rank-four application both cost 196 operation-proxy units per example; full-matrix application costs 576.

The native shared-subspace coefficient payload and outputs exactly match Mirror at every count and seed. Therefore this is a useful aligned standard shared-low-rank storage curve, but not Mirror-specific evidence. Fresh seeds were not opened.

## C — Strongest counter-hypothesis

The complete storage and function result is the ordinary parameter accounting of a shared basis plus one coefficient vector per task. The `Mirror coordinate` adds no distinct function or frontier over native shared LoReFT coefficients.

## U — Still unconfirmed

Only exactly aligned synthetic rank-four functions were tested. MA-501 showed that quality deteriorates when private directions are introduced. This experiment does not establish natural task capacity, learned-language behavior, or the storage frontier when task functions require private residuals.

## Evidence classification

**Facts:** actual per-count NPZ sizes, exact function outputs, per-count code uniqueness and replay are in `runs/`, `RESULTS_CORE.csv` and `VERIFICATION.json`.  
**Interpretation:** 256 actual functions fit in 6,005 B in this aligned toy family, with the ordinary shared-coefficient parameterization fully explaining it.  
**Hypothesis:** a natural task family could yield a useful shared/private frontier; this screen does not test that hypothesis.
