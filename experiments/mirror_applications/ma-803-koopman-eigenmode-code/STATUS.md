# MA-803 status

- Status: FAIL for Mirror-specific value and registered storage frontier.
- Branch: `research/ma-803-koopman-eigenmode-code-20261111`
- Base: `a181098d`
- Development: seed 80301; two diagonal regimes held out.
- Fresh/audit: sealed after byte-gate miss and exact native spectral alias.
- Verification: focused test passes; payload replay is exact.

## H / T / D / C / U

- **H:** Eigenmode task codes preserve stable heldout forecasts and reduce storage/compute.
- **T:** Synthetic 8D stable modal system, horizon 48; dense, diagonal, shared and rank-2 spectral controls.
- **D:** **FAIL.** Horizon error 1.01e-7 and rho .680, but only 1.6% bytes saved vs dense; native diagonal is 28 B smaller at equal MACs; native spectral exactly aliases.
- **C:** The teacher is diagonal in a known shared basis, so a native eigenvalue table is already compact and efficient.
- **U:** Learned eigenmodes, non-normal systems, natural nonlinear data and accelerator runtime.

## Next action

Verify and record the scoped FAIL; proceed to MA-804's shared-plus-private residual test.
