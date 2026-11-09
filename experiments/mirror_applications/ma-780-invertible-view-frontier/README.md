# MA-780 — Invertible-view evidence frontier

Status: **NOT ESTABLISHED** for the requested cross-family Pareto claim.

## Fact

The selected source reports cover MA-771 (synthetic 2D RealNVP steering), MA-772 (synthetic condition×behavior RealNVP codes), and MA-775 (synthetic 4D contractive residual blocks). Each reports exact serialized inference payloads and its own quality, inverse, compute, and timing metrics. Mirror exactly aliases the native conditional RealNVP in MA-771, the native product code in MA-772, and the native rank-2 residual basis in MA-775. MA-771's linear control has heldout function nRMSE .289 at 693 B versus exact RealNVP at 701 B. MA-775's basis uses 970 B vs 1,927 B independent and doubles its MAC proxy. The source task banks, dimensions, metrics, and compute conventions differ.

## Interpretation

The evidence supports a repeated attribution result: the tested invertible Views are expressible by their ordinary native invertible/factorized controls. Across these selected synthetic screens, invertibility itself can preserve exact recovery while the encoding may add storage or compute. It does not establish a single cross-family frontier because comparisons are not paired on a common representation or task bank. OFT and trained INNSteer rows are absent.

## Hypothesis result

**NOT ESTABLISHED.** The frozen common-benchmark gate was not met: there is no common frozen backbone/task bank, no matched mapper/code budgets across families, and no fresh whole-task identities. The listed values remain source-level observations and must not be pooled or ranked as one experiment.

## H / T / D / C / U

- **H:** An invertible View family improves the quality/bytes/compute frontier over linear steering and native invertible controls on a common activation benchmark.
- **T:** Selected source evidence from MA-771 seed 77101, MA-772 seed 77201, and MA-775 seed 77501; source rows copied with task and measurement boundaries intact. Tests validate row completeness and disallow pooled comparisons.
- **D:** **NOT ESTABLISHED** for cross-family frontier. Three exact native aliases are observed; a matched common benchmark was not available in the source artifacts.
- **C:** The apparent benefits are ordinary RealNVP conditioning or low-rank residual factorization, and source-level timing differences are implementation noise.
- **U:** Common frozen LM activation tasks; trained OFT and INNSteer controls; matched code/mapping budgets; fresh task identities; comparable latency hardware and inverse accounting.

## Provenance

The source rows are copied from the committed `RESULTS_CORE.csv` files for MA-771, MA-772 and MA-775. `FRONTIER_SCOPE.csv` identifies which comparisons are prohibited. No new model training, fresh data access, or pooled score was used.
