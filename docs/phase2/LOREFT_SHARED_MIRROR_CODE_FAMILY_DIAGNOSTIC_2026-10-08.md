# Shared LoReFT subspace and factorized task-code family diagnostic — 2026-10-08

## Scope and decision

MA-501 and MA-502 tested static task coordinates over a shared low-rank representation-intervention basis. Both are **FAIL for Mirror-specific attribution** because the candidate exactly is the native shared-subspace coefficient parameterization. MA-503 extended the insertion to layer×task bilinear factors and again exactly matched its native bilinear parameterization. Pause unchanged shared-subspace coefficient and layer/task-factor code variants. A future representation experiment must add a materially different dynamic/behavioral function beyond standard coefficients and show a margin against its native ReFT control.

MA-508 activation-addition basis is a separate insertion and remains eligible.

## Evidence

- **MA-501:** at private ratio rho=0, shared rank four gives exact synthetic outputs at 2,138 B versus 13,668 B for per-task rank four. At rho=.25, shared heldout relative RMSE .226/.173 misses its .10 gate; independent per-task rank four is .178/.161. The native shared-coefficient implementation exactly matches.
- **MA-502:** for aligned rank-four functions, eight tasks occupy 2,035 B versus 7,403 B independent (27.49%); 256 tasks occupy 6,005 B versus 201,837 B independent (2.98%), with relative output RMSE <=3.6e-6 and all task codes unique. Native shared LoReFT coefficients exactly match bytes and outputs at all counts.
- **MA-503:** a rank-two bilinear layer×task factor achieved heldout rho=0 relative RMSE .00215 in one world and 6.75411 in the other; rho=.25 results were .563/22.515. Its 2,568 B payload was 95.6% of a dense shared-basis code table (2,685 B), missing the frozen <=.50x threshold. Native bilinear factors exactly match Mirror. The full-matrix oracle was exact. Two initial runner attempts stopped before metrics due cache bookkeeping; these partial outputs were preserved, then the cache-only Amendment 1 reran the same seeds and unchanged protocol.

## Interpretation

Shared basis plus task coefficient vectors can have a strong aligned storage frontier at high logical task counts. The crossed factorization can in principle extrapolate to heldout layer/task pairs, but MA-503 did not establish robust generalization across its two development worlds and had little actual byte advantage over the dense coefficient table at this grid size. All three screens are fully explained by ordinary low-rank/shared coefficients or native bilinear factorization. These are useful parameter/storage results, not Mirror-specific gains.

## Boundary

These are synthetic hidden-state interventions. They do not test trained Transformer behavior, NLL, natural task transfer, full LoReFT optimization, or production latency. MA-503 does not distinguish matrix-completion identifiability from single-initialization optimizer failure; alternate masks or restarts were not opened after the frozen screen.
