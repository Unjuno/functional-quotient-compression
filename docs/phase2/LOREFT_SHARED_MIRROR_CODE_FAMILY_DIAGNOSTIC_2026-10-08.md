# Shared LoReFT subspace plus task-code family diagnostic — 2026-10-08

## Scope and decision

MA-501 and MA-502 test static task coordinates over a shared low-rank representation-intervention basis. Both are **FAIL for Mirror-specific attribution** for the same structural reason: the candidate is exactly the native shared-subspace coefficient parameterization. Pause unchanged variants of this insertion. MA-503 factorizes layer × task codes and remains a separate hypothesis only if it includes held-out combinations and direct native factorized-parameter controls.

## Evidence

- **MA-501:** at private ratio rho=0, shared rank four gives exact synthetic outputs at 2,138 B versus 13,668 B for per-task rank four. At rho=.25, shared heldout relative RMSE .226/.173 misses its .10 gate; independent per-task rank four is .178/.161. The native shared-coefficient implementation exactly matches.
- **MA-502:** for aligned rank-four functions, eight tasks occupy 2,035 B versus 7,403 B independent (27.49%); 256 tasks occupy 6,005 B versus 201,837 B independent (2.98%), with relative output RMSE <=3.6e-6 and all explicit codes unique. Native shared LoReFT coefficients exactly match bytes and outputs at all counts.

## Interpretation

Shared basis plus a small vector of per-task coefficients can have a strong storage frontier when the task deltas lie in the same low-rank representation subspace. The physical-to-logical count in MA-502 is a measured synthetic aligned result, not independent capacity, and it is already provided by ordinary multi-task LoReFT coefficient storage. MA-501 shows the aligned benefit degrades when task-private directions appear; rank-four private task interventions can preserve more quality at higher storage.

## Boundary

The two results do not test trained Transformer behavior, NLL, natural ReFT interventions, task interference during adapter training, or private residual break-even beyond rho=.25. They support pausing duplicate static shared-subspace coefficient experiments. A materially different factorization/dynamic-coordinate hypothesis requires a matched native parameterization and held-out combinations.
