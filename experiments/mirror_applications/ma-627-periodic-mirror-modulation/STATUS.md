# MA-627 status

- Status: FAIL for Mirror-specific attribution; strong scoped compression vs independent sampled curves
- Branch: `research/ma-627-periodic-mirror-modulation-20261103`
- Base commit: `d5c1b59e`
- Protocol frozen before development: yes
- Development complete: yes (62701, 62702)
- Fresh/audit opened: no; native activation alias gate failed
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** A shared periodic decoder with compact function codes recovers held-out signals at lower bytes than independent curves; Mirror-specific value must beat native periodic activation modulation.

**T:** 4×4 harmonic/phase family, diagonal pairs held out, 64 support and 512 audit points, development seeds 62701 and 62702. Frequency inferred by support residual over harmonics 1–4; amplitude and phase fit by least squares. Compared shared decoder, native periodic activation with same code, and independent 512-point curves. Fresh remained sealed.

**D — FAIL for Mirror-specific attribution.** On both seeds, Mirror and native activation produced identical serialized code payloads (879 and 883 B) and identical mean held-out nRMSE (6.1–7.0e-16). Compute proxies were also identical (25,600). Independent sampled curves cost 24,788–24,892 B and had mean nRMSE about 1.03e-4. The shared code is 3.5% of the independent curve payload, but the exact native activation alias blocks Mirror-specific credit. Fresh stayed sealed.

**C:** This is ordinary modulated periodic activation with amplitude, phase and frequency coordinates. The storage reduction comes from a shared analytic sine decoder versus per-signal sampled curves.

**U:** No learned SIREN, multiharmonic mixtures, non-periodic functions or natural signal task was tested.

## Fact / Interpretation / Hypothesis

**Fact:** shared periodic code payload is 879–883 B against 24,788–24,892 B for independent curves; the native activation control is exactly equal in arrays, bytes, compute proxy and output.

**Interpretation:** parameterized periodic functions compress this narrow family effectively, but Mirror adds no demonstrated advantage beyond standard activation modulation.

**Hypothesis:** richer structured phase/frequency geometry could help broader signal families only if it improves over native modulation and trained latent codes on fresh tasks.
