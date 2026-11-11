# MA-834 status

- Status: FAIL for Mirror-specific attribution; adaptive basis selection frontier observed synthetically.
- Branch: `research/ma-834-karc-basis-mixture-20261111`
- Base: `115061cf`
- Development: nine analytic tasks; Fourier/spline/Chebyshev; 64 support, 256 test points each.
- Fresh/audit: sealed after native family-selection alias.
- Verification: focused test passed; support-only selection and payload replay checked.

## H / T / D / C / U

- **H:** Sparse family Views outperform fixed/full KARC banks and native family selection.
- **T:** Fit family and coefficients from support; compare sampled curves, full bank, native selection, Mirror and fixed Fourier.
- **D:** **FAIL attribution.** Native selection 1,020 B, nRMSE 4.88e-8; Mirror 1,022 B with identical outputs/features. Full bank 1,268 B; fixed Fourier 624 B but .953 error.
- **C:** Ordinary KARC family ID plus coefficients already implements the View.
- **U:** Learned KARC, real dynamics, noisy or unknown families and fresh data.
