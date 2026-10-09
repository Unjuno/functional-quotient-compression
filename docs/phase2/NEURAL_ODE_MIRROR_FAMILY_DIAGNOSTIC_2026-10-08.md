# Neural ODE / continuous-depth Mirror family diagnostic — 2026-10-08

## Decision
Pause the current continuous-depth candidates MA-424..433 after the MA-424 screen. MA-427 and MA-424 are two completed candidates in this family with the same demonstrated structural cause: a small Mirror code is an ordinary native conditioning parameterization with no incremental Mirror-specific function. Resume after redesigning a candidate around a view operation that is not exactly the ordinary conditioned baseline.

## Evidence

| MA | Candidate | Measured result | Strongest native control | Outcome |
|---|---|---|---|---|
| 427 | Rank-1 task-conditioned DEQ map | Aligned quality/stability matched full bias, but payload was 1,450 B vs 1,064 B independent full bias; boundary tasks degraded | Byte-identical native rank-1 task-bias conditioner with identical outputs and iteration counts | FAIL; all fresh worlds in its protocol failed Mirror-specific Pareto gate |
| 424 | Givens coordinate views of fixed neural ODE field | 2,566/2,562 B vs 16,557/16,612 B independent fields; relative trajectory RMSE 3.82e-6/3.90e-6. Throughput was 0.474x/0.889x native generated weights. Mean mode diversity RMS was only 7.39e-4/5.55e-4. | Native generated-weight field had byte-identical payload and replay-identical trajectories | FAIL; fresh sealed |

MA-424 has two preserved initial runs with invalid timing setup: one retained autograd graphs, and a second unnecessarily materialized generated weights during Mirror deserialization. The canonical run uses no-grad inference and method-specific one-time loading. Protocol settings and seeds were unchanged; only corrected implementation results are authoritative. See the experiment README and `runs/initial_*` provenance directories.

## Shared structural cause
The proposals condition a shared dynamical map through ordinary task biases or coordinate-dependent weight generation. When the native control is supplied the same code and shared base, it reconstructs the same function and paid state exactly. MA-424 additionally finds that its tiny random contractive field yields very small absolute differences across modes, and that coordinate rotations can reduce CPU throughput.

## Next design requirements
Do not continue MA-425/426/428..433 unchanged. A new continuous-depth proposal must specify a state-dependent View that cannot be rewritten as the tested native code/bias/weight generator, demonstrate useful trajectory diversity at an application-relevant scale, and include solver NFEs, error, stiffness/stability, bytes, and runtime. This pause does not cover different neural sequence families such as state-space models.

## Evidence labels
- **Fact:** MA-427 and MA-424 each have exact native-control function aliases; MA-424's corrected serialized payload replay passed.
- **Interpretation:** the current continuous-depth View proposals do not establish incremental Mirror-specific functionality, and MA-424's mode differences are small on this weak field.
- **Hypothesis:** a nonlinear state-dependent or adaptive solver View may still produce useful dynamics, but would need a new non-aliased design.
