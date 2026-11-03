# MA-624 status

- Status: SCREENING
- Branch: `research/ma-624-example-context-code-20261103`
- Base commit: `b2bcc634`
- Protocol frozen: yes (`freeze.json`)
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no

## H / T / D / C / U

**H:** Infer a compact factor-addressed context code from support examples and improve the storage/compute/quality frontier over generic context aggregation and direct CAVIA.

**T:** Frozen synthetic 16-input / 8-context linear task family, 4×4 factors, four diagonal pairs held out, 16 support and 128 audit examples, development seeds 62401/62402. Factor identity is not supplied to the encoder. Controls are zero context, direct CAVIA least squares, generic pooled residual aggregation, inferred factorized code, and independent codebook upper control. Fresh seeds remain sealed.

**D:** Pending.

**C:** Direct CAVIA support fitting may exactly solve the linear problem with less codebook state and lower compute.

**U:** Natural few-shot and meta-learning performance are outside this screen.
