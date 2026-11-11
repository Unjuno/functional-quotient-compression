# MA-829 — shared parent functions for KAN adapter banks

## H / T / D / C / U

**H:** Mirror task Views over shared Group-Rational KAN parents preserve heldout adapter output with fewer bytes than independent KanAdapter modules and improve on native shared-parent coefficients.

**T:** Synthetic 8D, eight-task analytic adapter bank; four shared rational basis functions; 2,048 support and 2,048 evaluation examples; seed 82901; zero optimizer updates. Compared independent sampled adapter functions, native shared coefficients, Mirror task Views and shared parent without task code.

**D: FAIL for Mirror-specific attribution.** Mirror task Views use 1,861 B and achieve zero heldout error, versus 15,991 B independent sampled adapter payload at nRMSE .00050. Native shared rational coefficients use 1,869 B with identical outputs and parent evaluations. The 8 B difference is NPZ field layout. Shared parent without task code uses 1,002 B but has heldout nRMSE .903.

**C:** Native shared-parent task coefficients fully explain the result; no functional or compute difference is established.

**U:** Trained KanAdapter, pretrained speech encoders, continual forgetting, real audio metrics and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Full serialized bytes, hashes, output errors, evaluation count and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** A shared rational parent greatly compresses this synthetic adapter bank, but it is the native GS-KAN-style coefficient method.
- **Hypothesis:** Mirror-specific gains would require learned task Views that beat native shared-parent adaptation on real downstream metrics and full bytes.
