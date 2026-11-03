# MA-624 status

- Status: FAIL (development screen)
- Branch: `research/ma-624-example-context-code-20261103`
- Base commit: `b2bcc634`
- Protocol frozen before development: yes
- Development complete: yes (62401, 62402)
- Fresh/audit opened: no; all methods met quality but Mirror was dominated
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** Inferring a factor-addressed code from support examples improves the storage/compute/quality frontier over generic conditional context aggregation and direct CAVIA.

**T:** Synthetic 16-input, 8-context linear family; 4×4 compositional factors; diagonal tasks held out; 16 support and 128 audit examples; development seeds 62401 and 62402. Compared direct CAVIA least squares, pooled residual sufficient statistics, support-matched Mirror codebook inference, and independent codebook. Factor identity was not supplied. Actual `.npz` bytes measured; fresh seeds sealed.

**D — FAIL.** All methods achieved numerical-zero held-out nRMSE. Direct CAVIA payload was 1,771–1,773 B with active compute proxy 1,024. Generic pooled context was 1,781–1,783 B with the same proxy. Mirror code inference used 2,540 B and proxy 32,768, including 256 candidate evaluations for each held-out task. Independent codebook was 3,044 B. Mirror was strictly dominated by direct CAVIA on bytes and compute at equal quality in both seeds; fresh remained sealed.

**C:** Direct linear CAVIA context fitting is a sufficient statistic for this task family. Mirror's factor codebook adds persistent state and candidate-search computation without improving task quality.

**U:** No noisy-support, nonlinear task, trained encoder, or natural few-shot test. The result does not determine whether learned context encoders help where support inference is underdetermined.

## Fact / Interpretation / Hypothesis

**Fact:** each method reached numerical-zero held-out nRMSE. Direct CAVIA used fewer bytes and lower compute than Mirror in both development seeds.

**Interpretation:** for this identifiable linear task family, example-derived factor code inference is unnecessary and dominated by generic adaptation.

**Hypothesis:** codes inferred by a trained encoder could help when support is noisy or insufficient, but must beat generic CNP-style aggregation on fresh tasks.
