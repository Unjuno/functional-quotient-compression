# MA-827 status

- Status: FAIL for frozen complete-byte frontier.
- Branch: `research/ma-827-source-target-edge-code-20261111`
- Base: `aab89aeb`
- Development: synthetic 10-node, 32-edge analytic function bank; eight heldout-scored edges.
- Fresh/audit: sealed after native direct coefficient storage was smaller.
- Verification: focused test passed; payload reconstruction replay exact.

## H / T / D / C / U

- **H:** Source×target factors beat native edge coefficients while preserving heldout functions.
- **T:** Analytic parent; compare independent sampled functions, native coefficients, endpoint factors and source-only.
- **D:** **FAIL.** Mirror zero-error code uses 1,708 B; native direct coefficients use 1,285 B. Source-only is 1,189 B but has .157 error. Independent sampled functions use 7,012 B.
- **C:** Native edge coefficients avoid duplicated endpoint factors and remain the cheaper representation.
- **U:** Larger graph amortization, trained KAN, fresh data and natural tasks.
