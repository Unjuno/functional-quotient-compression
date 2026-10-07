# MS005 — Gaussian ES × fixed Mirror ensemble

Date: 2026-10-07 JST. Evidence: small synthetic world-model MLP optimization study. No loop recurrence, Transformer/LLM/real-sensor/compression claim.

## Question
Can many fixed Mirror views reduce Gaussian ES estimator noise when all views update one shared world-model parameter set?

## Protocol
Base dynamics were learned for 800 AdamW updates. ES then adapted the shared model to a changed dynamics law. Sixteen fixed views were formed from eight low-rank nilpotent shear generators with +/-rho=0.1. Fresh world/init pairs were (54201,701), (54202,702), (54203,703). ES used sigma=0.01, normalized step=0.05, batch192, 120 generations and antithetic pairs. Development world 54200 was excluded from final medians.

Two noise modes were separated:
- `shared_eps`: the same Gaussian direction is reused by all Mirrors.
- `independent_eps`: each Mirror receives independent Gaussian directions; per-Mirror ES gradient estimates are averaged.

Two budgets were separated:
- fixed 128 candidate×Mirror branches per estimator/generation;
- fixed 8 antithetic pairs per Mirror, so total branch count grows 16,64,128,256 for K=1,4,8,16.

## Core result
With 8 pairs per Mirror, independent-noise normalized estimator variance had fresh-world medians:

|K|branches|variance|gradient cosine|
|---:|---:|---:|---:|
|1|16|170.40|0.074|
|4|64|43.32|0.149|
|8|128|20.72|0.203|
|16|256|10.68|0.287|

The K=16/K=1 variance ratio was 0.0627, close to 1/16. Shared noise did not show this effect: 170.03,179.42,180.63,178.31 for K=1,4,8,16.

Under a fixed total 128-branch budget, independent noise maintained roughly constant variance (21.74,21.78,20.72,21.05) while shared noise became much worse as K increased (21.75,86.61,180.63,328.15), because the number of independent Gaussian pairs fell from64 to4.

Fixed-budget final full-16-view mean NMSE medians:

|method|K|NMSE|
|---|---:|---:|
|ordinary Gaussian ES, no Mirror|1|0.010657|
|shared-epsilon Mirror ES|4|0.020258|
|shared-epsilon Mirror ES|8|0.072761|
|shared-epsilon Mirror ES|16|0.145641|
|independent-epsilon Mirror ES|4|0.011265|
|independent-epsilon Mirror ES|8|0.011211|
|independent-epsilon Mirror ES|16|0.009963|

The K=16 independent result was not a stable paired win over ordinary Gaussian ES: paired mean-NMSE differences were -6.51%, +3.27%, +1.09% (positive is worse), median +1.09%.

## Decision
- PASS: independent Gaussian populations across Mirror views average estimator variance down when extra independent samples/compute are added.
- FAIL: copying the same Gaussian direction across many Mirrors does not average away ES direction noise.
- FAIL: increasing K gives no free estimator improvement at fixed total branch budget.
- UNCERTAIN/negative so far: no stable fixed-compute learning advantage over ordinary Gaussian ES.

Interpretation: Mirror should be treated as a parallel population/view axis while Gaussian ES remains the exploration distribution. The observed 1/K noise reduction is consistent with averaging independent estimators; it is not evidence that Mirror views themselves create extra information for free.

## Parallel benchmark
K=16, 8 pairs/Mirror, 256 branches, batch192, AMD EPYC 9V74 virtual CPU, PyTorch2.10.0+cpu, 1 thread: packed evaluation 35.83 ms median vs Python loop 43.03 ms, 1.20x speedup, exact loss equality on the fixture. GPU scaling is untested.

## Audit
6 unit tests passed. 45 train/robust models plus 3 parents were reloaded; all 36 stored training metrics reproduced with maximum absolute difference 0.0. Three fresh diagnostic files contain 48 estimator rows total.