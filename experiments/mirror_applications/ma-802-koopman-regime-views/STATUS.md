# MA-802 status

- Status: FAIL for Mirror-specific attribution; synthetic storage/quality/stability screen passes vs independent operators.
- Branch: `research/ma-802-koopman-regime-views-20261111`
- Base: `177b234d`
- Development: seed 80201; two diagonal regimes held out.
- Fresh/audit: sealed after exact native low-rank alias.
- Verification: pending focused test and serialization replay.

## H / T / D / C / U

- **H:** Regime Views over shared latent Koopman dynamics preserve stable heldout forecasts with fewer complete bytes.
- **T:** Synthetic 8D nonlinear observation conjugacy, rank-2 regime operator, horizon 40; independent, shared-only, View and native conditional controls.
- **D:** **FAIL** Mirror attribution. View uses 2,607 B vs 3,565 B independent (26.9% savings), nRMSE 1.06e-7, spectral radius .536, but active MACs are 1.67× and native control aliases exactly.
- **C:** Ordinary native low-rank conditional latent dynamics is exactly the same operator family.
- **U:** Learned Koopman encoding, real nonlinear systems, fresh regimes, control quality and deployment runtime.

## Next action

Verify exact payload replay, record the scoped failure, and proceed to MA-803.
