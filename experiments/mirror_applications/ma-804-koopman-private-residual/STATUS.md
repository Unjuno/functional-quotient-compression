# MA-804 status

- Status: FAIL for Mirror-specific attribution and 25% storage gate; synthetic private-residual need observed.
- Branch: `research/ma-804-koopman-private-residual-20261111`
- Base: `d59dd48e`
- Development: seed 80401; two exceptional nonlinear regimes.
- Fresh/audit: sealed after byte miss and exact native control alias.
- Verification: one focused test passes; all serialized payloads roundtrip exactly.

## H / T / D / C / U

- **H:** Shared operator plus sparse residuals preserves exceptional-regime forecasts with lower storage than independent nonlinear dynamics.
- **T:** Synthetic 8D latent dynamics, two exceptional of nine regimes, horizon 40, independent/shared-only/Mirror/native controls.
- **D:** **FAIL** Mirror attribution and storage gate. Shared-only errors .116 on exceptions; two private codes restore nRMSE to 1.04e-7, but save only 14.3% vs independent, cost 1.89× MAC, and exactly alias native shared/private dynamics.
- **C:** Ordinary sparse per-regime scaling of a shared nonlinear feature is sufficient.
- **U:** Learned nonlinear feature bases, natural systems, fresh regimes, longer horizons, optimization and hardware kernels.

## Next action

Verify and record this negative/threshold result, then proceed to MA-805.
