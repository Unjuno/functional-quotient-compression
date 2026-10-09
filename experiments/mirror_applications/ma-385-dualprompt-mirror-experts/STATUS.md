# MA-385 status

- Status: **FAIL** on frozen quality and total-payload gates; fresh remained sealed.
- Branch: `research/ma-385-dualprompt-mirror-experts-20261009`
- Protocol frozen before implementation/development: `0e099ad8`
- Development worlds: 38501, 38502
- Fresh worlds 38511–38513: not opened
- Tests: 3 passed
- Stored inference payloads: 10; byte/hash/final metrics and all sequential retention stages replayed

## H / T / D / C / U

- **H:** A shared expert-prompt basis plus one task Mirror angle preserves DualPrompt task quality/retention at ≤60% independent bytes.
- **T:** Fixed common general prompt; eight sequential task experts; independent/tied/scalar/generic/Mirror controls; task ID supplied; 1,200 updates per task.
- **D:** FAIL. Mirror final mean NRMSE .0509/.2692 vs independent <1.5e-7 and generic <2.7e-7; payload 2,238/2,241B (64.9%/65.2% of independent); seed-to-seed Mirror fitting was unstable.
- **C:** The aligned orbit is representable, but basis fitting from the first two tasks may be optimization-sensitive; generic coefficients fit it reliably.
- **U:** Pretrained DualPrompt, natural task stream, private residuals and deployment latency.

See README for fact/interpretation/hypothesis separation and limits.
