# MA-809 — shared action basis and private skill residual

## H / T / D / C / U

**H:** A shared invertible action basis and small per-skill Views represent common skills; a sparse rank-1 private residual restores one off-orbit skill while retaining heldout OOD quality and reducing full bytes versus independent policies.

**T:** One synthetic 8D linear action world, four skills, three shared Givens-orbit maps and one rank-1 perturbed exception. Seed 80901; 256 fitting examples per skill, 512 heldout examples per skill, with a fixed orthogonal input-plane shift. No optimizer updates. Controls: independent full matrices, shared-only, native coefficient table, Givens views, and Mirror/native shared-plus-private rank-1 residual.

**D: FAIL.** Independent policies serialize to 1,075 B with zero error. Shared-only/View payloads are 884–886 B but miss quality (aggregate OOD nRMSE .03271; exceptional skill .06432; success .812). Adding the private residual recovers quality (nRMSE 8.36e-8; success 1.0) but costs 1,748 B for the View and 1,750 B for the native control, both larger than independent. Mirror and native Givens/low-rank controls are functionally equivalent; the private payloads differ by only 2 B and use the same 80 MAC proxy.

**C:** The observed common subspace is real in this synthetic world, but charging all shared basis, code, private vectors, selectors and NPZ metadata removes any useful storage gain once the exception is repaired. Standard additive Givens plus a rank-1 residual accounts for the whole result.

**U:** Fresh seeds, learned policies, nonlinear dynamics, robot rollouts, safety and hardware latency. Fresh remained sealed under the frozen release rule.

## Evidence separation

- **Fact:** The serialized payload hashes, byte sizes, heldout errors, success fractions, compute proxies, and zero payload replay error are in `RESULTS_CORE.csv`.
- **Interpretation:** The shared-only code is too restrictive for the off-orbit skill; one private residual repairs quality but pushes full storage past independent.
- **Hypothesis:** A private exception can be worthwhile only when its residual and metadata amortize over more skills or have exploitable low-description structure. This experiment does not show that regime.
