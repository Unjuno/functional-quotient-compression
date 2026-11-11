# MA-833 status

- Status: FAIL for Mirror-specific attribution; synthetic shared/private frontier observed.
- Branch: `research/ma-833-karc-dynamics-code-20261111`
- Base: `82eca699`
- Development: synthetic 4D nonlinear Fourier-feature dynamics; eight systems; horizon 8.
- Fresh/audit: sealed after native low-rank KARC alias.
- Verification: focused test passed; matrix replay and finite rollouts checked.

## H / T / D / C / U

- **H:** Shared KARC basis plus Mirror readout codes beats independent and native readouts.
- **T:** Synthetic closed-form Fourier feature dynamics; independent, shared-only, native low-rank and Mirror readout controls.
- **D:** **FAIL attribution.** Mirror 6,259 B/nRMSE 3.19e-8; native low-rank 6,267 B with same error/features; independent 8,420 B. Shared-only 1,693 B misses at .0355.
- **C:** Native low-rank KARC readout fully accounts for storage/quality frontier.
- **U:** Trained KARC, real time series, unseen systems and fresh data.
