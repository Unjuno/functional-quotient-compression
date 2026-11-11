# MA-832 — shared KAN parent plus sparse private residuals

## H / T / D / C / U

**H:** A shared parent plus private rational bumps on exceptional edges recovers all edge functions at fewer complete bytes than independent tables and beats native GS-KAN plus rational residuals.

**T:** Synthetic 32-edge analytic bank; shared `sin(x)+0.2x^3` parent; four exceptional edges carry one rational bump each; eight edges separately scored; 8,192 test samples; seed 83201; zero updates. Compared independent sampled tables, shared parent only, Mirror plus private residual, and native GS-KAN plus rational residual.

**D: FAIL for Mirror-specific attribution.** Independent samples use 3,545 B with heldout nRMSE 7.13e-5. Shared-only uses 438 B but misses quality at .0386. Sparse private Mirror uses 1,495 B with zero error (57.8% fewer bytes than independent); native residual uses 1,505 B with the same function and evaluation counts. The 10 B difference is NPZ representation overhead, not a functional advantage.

**C:** Ordinary GS-KAN plus sparse rational residuals recover the same quality/byte frontier; private exception parameters are the necessary capacity boundary in this constructed example.

**U:** Learned exception selection, trained KAN, natural data, fresh seeds and physical runtime. Fresh remained sealed.

## Evidence separation

- **Fact:** Actual payload bytes, heldout errors, private-edge counts/evaluations and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** Sparse private functions restore the exceptional edges; this is a useful shared/private frontier observation but not Mirror-specific.
- **Hypothesis:** Private edge functions may be worthwhile when exceptions are sparse and amortized; natural KAN tasks are needed to establish prevalence.
