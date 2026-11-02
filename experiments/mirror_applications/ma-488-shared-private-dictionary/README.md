# MA-488 — Shared/private dictionary + Mirror coefficients

Status: **FAIL** for the all-heterogeneity byte gate; useful private-state frontier observed on this synthetic task.

**H:** Shared coordinates can store aligned functions compactly; private residuals should be allocated only where needed and beat dense functions at <=80% bytes.

**T:** 64 synthetic 32D functions, rank-4 shared basis; 25/50/75% functions received private residuals. Development worlds selected tau=.05; fresh worlds 48810-48812 × seeds 0-2. Dense, shared-only, all-private residual, and adaptive shared/private payloads were serialized with all state charged.

**D:** FAIL to preregistered every-heterogeneity gate. Adaptive residuals restored exact outputs, using 6,169B (62.7% dense) at 25% heterogeneity, 8,217B (83.6%) at 50%, and 10,265B (104.4%) at 75%. Shared-only used 3,681B but NRMSE rose .480/.605/.667. The data identify a clear private-state boundary: savings disappear as private fraction rises.

**C:** The private vectors are full 32D residuals, and the fixed shared basis plus per-function coefficients already costs state; no learned shared/private dictionary or quantized residual was tested.

**U:** Natural task functions, optimized low-rank private residuals, larger amortization banks, and learned dictionary adaptation.
