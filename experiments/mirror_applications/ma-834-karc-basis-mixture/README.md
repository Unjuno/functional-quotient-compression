# MA-834 — task-selected KARC basis family

## H / T / D / C / U

**H:** Task-specific Mirror codes select Fourier, spline or Chebyshev KARC features from one heterogeneous basis bank, preserving heldout function quality with fewer total bytes and active features than a fixed/full bank, beyond native family selection.

**T:** Nine analytic scalar tasks, three from each family; 64 support and 256 test points per task; seed 83401; closed-form least-squares fit selects family from support; no optimizer updates. Compared independent sampled curves, full heterogeneous coefficient bank, native family-ID selection, Mirror family Views and fixed Fourier.

**D: FAIL for Mirror-specific attribution.** Native family selection uses 1,020 B with heldout nRMSE 4.88e-8 and 11,520 active feature evaluations. Mirror uses 1,022 B with identical selection accuracy/error/evaluations. Full heterogeneous bank uses 1,268 B at zero error; sampled curves use 5,363 B at .000428; fixed Fourier uses 624 B but misses quality (.953 nRMSE). Native family selection is 2 B smaller than Mirror.

**C:** Standard KARC basis-family selection and per-task coefficients fully explain the result. The basis-view code differs only in serialization key layout.

**U:** Learned KARC, real dynamics, family mismatch/noisy support, deployment-time family inference, and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Bytes, hashes, selection accuracy, heldout error, active features and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** Adaptive basis choice improves on one fixed family and compactly represents this bank, but native family selection is slightly smaller and functionally equivalent.
- **Hypothesis:** Mirror value would require composition across families or better robustness/latency than native basis selection.
