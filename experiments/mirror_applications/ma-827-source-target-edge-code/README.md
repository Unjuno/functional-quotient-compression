# MA-827 — source × target edge-function codes

## H / T / D / C / U

**H:** Factoring edge function identity into source and target node codes reconstructs heldout edge functions with lower complete bytes and better than native GS-KAN coefficient storage.

**T:** Analytic parent `sin(x)+0.2x^3`; 32 edges across 10 nodes; rank-2 source/target scales and additive biases; eight edges separately scored; 8,192 test samples; seed 82701; no updates. Compared independent sampled edge functions, native per-edge coefficients, source×target codes and source-only codes.

**D: FAIL.** Source×target Mirror codes achieve zero heldout error but use 1,708 B, versus 1,285 B for native edge coefficients and 1,189 B for source-only (which misses quality at .157 nRMSE). Independent sampled functions use 7,012 B at nRMSE 7.0e-5. All methods perform the same number of parent evaluations. The rank-2 factor state is larger than direct edge coefficients at this scale.

**C:** Direct GS-KAN per-edge affine coefficients are cheaper than storing both endpoint factors and reconstructing their products. Independent sampled curves are a weak storage control here because they store 64-point tables.

**U:** Learned KAN, larger sparse graphs, factor amortization across many graph instances, natural edge functions and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Payload bytes/hashes, heldout errors, evaluation counts and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** The source-target factorization is an overparameterized coordinate system for this small graph; it loses to native direct coefficients.
- **Hypothesis:** Endpoint factors might amortize across many graph instances, but that needs a multi-graph full-byte test against shared GS-KAN controls.
