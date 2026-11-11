# MA-826 status

- Status: FAIL for Mirror-specific attribution.
- Branch: `research/ma-826-gskan-parent-edge-views-20261111`
- Base: `5f3a7ec4`
- Development: synthetic 32-edge analytic function bank; 8 edges separately scored.
- Fresh/audit: sealed after native GS-KAN exact alias.
- Verification: focused test passed; payload replay exact.

## H / T / D / C / U

- **H:** Mirror edge Views beat native GS-KAN shared-parent transforms.
- **T:** Analytic scalar edge functions; independent samples, native affine transforms, Mirror affine and modulation codes.
- **D:** **FAIL.** Native and Mirror affine each use 1,096 B with zero error and same parent evaluation count; modulation uses 1,295 B. Independent sampled bank uses 8,438 B at nRMSE 8.85e-5.
- **C:** GS-KAN is an exact native control for the tested affine orbit.
- **U:** Trained KAN, spline parents, fused runtime and fresh seeds.
