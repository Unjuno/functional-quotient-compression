# BOFT Mirror-family diagnostic (MA-273 and MA-274)

Date: 2026-10-08 JST
Scope: current compact shared butterfly-angle atom + small task/expert code pattern. This does not indict native BOFT or all orthogonal Views.

## Evidence

- MA-273 tested a shared butterfly angle atom with one scalar code per task against independent per-task BOFT angle tables on an adapter-regression fixture. After removing target-angle initialization leakage, the development result saved only about 5% serialized bytes versus BOFT while its MSE was higher; the independent upper was much better. Fresh data stayed sealed.
- MA-274 tested the same factorized angle-address structure at a nonlinear expert activation interface, with hard tying, IA3, rank-one BatchEnsemble, native per-expert BOFT, and independent FFNs. At the selected LR, Mirror used 5,430 B versus BOFT 5,624 B (3.4% fewer) and rank-one 6,703 B (19.0% fewer). Mirror beat rank-one in both development worlds (MSE 0.0214/0.0449 vs 0.0405/0.0723), but did not meet its independent-quality gate (independent MSE 0.0060/0.0040) and was worse than BOFT (0.0121/0.0358). Eager throughput was 0.33–0.36M examples/s, versus 3.58–4.32M rank-one and 6.35–7.20M IA3.

## Interpretation

The repeated shortfall is the *marginal frontier against native BOFT*: the compact address reduces bytes only slightly and does not retain BOFT quality in these fixed-budget screens. The expert screen did find a useful comparison against rank-one modulation, but its stronger BOFT and independent controls prevent a broad success claim.

Both experiments use fixed-update optimization and do not prove representational impossibility. The shared angle/code factorization may be poorly conditioned from neutral initialization; alternatively, practical recovery may require more private angle state, eliminating much of the small byte margin.

## Family decision

Pause the current task/adapter/expert scalar-code shared-angle BOFT line after MA-273 and MA-274. Resume that insertion family only with a materially redesigned factorization or initialization protocol preregistered before fresh evaluation. This pause does not relabel other BOFT insertion points. In particular, a separate MA-276 depth-view branch reports a PROMISING aligned storage result with a distinct depth interface and a larger runtime workspace; that is counterevidence against generalizing this diagnostic to all BOFT views. Continue the global queue in another family.


## Reconciliation scope note (2026-10-08)

MA-274 was selected from a random worker pool and selectively imported from `research/ma-274-boft-logical-experts-20261008` at `d9df358`. The related MA-273 evidence remains on `research/ma-273-boft-mirror-bank-screen-20261008`. MA-276 depth-view results remain on their own research branch and are explicitly outside the task/expert angle-code pause.
