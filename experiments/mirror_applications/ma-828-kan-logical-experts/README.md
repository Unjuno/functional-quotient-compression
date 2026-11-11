# MA-828 — logical KAN experts from shared edge functions

## H / T / D / C / U

**H:** Shared KAN edge-function Views provide multiple nonlinear experts at lower complete bytes and useful quality than dense MoE and native GS-KAN at equal physical basis.

**T:** Synthetic analytic nonlinear experts, 8 experts over an 8D scalar-edge basis, two parent-function evaluations per sample, balanced fixed router IDs; seed 82801; 4,096 examples; no training. Compared stored independent outputs, dense expert matrices, native GS-KAN coefficient codes and Mirror codes.

**D: FAIL for Mirror-specific attribution.** Mirror and native GS-KAN each use 1,197 B and reproduce their analytic teacher exactly with the same 8,192 parent evaluations. Independent sampled outputs use 123,900 B. The dense matrix control uses 3,575 B but has nRMSE .998 because its weights were not trained to the analytic teacher; it is not a valid quality-matched MoE comparison and cannot support a dense-MoE frontier claim. The Mirror hypothesis fails because the required native GS-KAN control exactly matches it.

**C:** The tested expert Views are the ordinary GS-KAN shared-parent coefficient parameterization. The apparent compression does not isolate a Mirror-specific mechanism.

**U:** A trained KAN/MoE comparison, learned router, natural task metrics, quality-matched dense experts, and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Bytes, hashes, outputs, evaluation counts and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** Shared analytic parents compactly produce experts, but native GS-KAN accounts for the complete result. The dense baseline is invalid for quality comparison.
- **Hypothesis:** Any further expert claim needs a quality-matched trained dense MoE and a mechanism beyond GS-KAN edge coefficients.
