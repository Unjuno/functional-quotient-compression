# MA-273 — BOFT Mirror adapter bank

## H — Hypothesis

A shared butterfly generator plus task coordinates might reconstruct an aligned orthogonal task family with less state than independent BOFT factors. Independent transforms probe a private-factor boundary.

## T — Conditions and amendments

Frozen 32×32 linear map, eight tasks, 256 support and 1,024 audit examples. Fresh worlds 27310–27312 × seeds 0–2, 576 rows. Controls: shared identity, full OFT oracle upper, support-fit independent BOFT, aligned independent BOFT, shared Mirror scalar and generic scalar. Prior to fresh, the source was amended after development audits: dense oracle mislabeled as BOFT was replaced with angle fitting; slow loops were interrupted; disjoint rotations were optimized; fitting budgets were reduced. The final fixed-budget fitting still left substantial residual. Fresh outputs were not used to adjust anything. Canonical inference payloads charge shared W, factor/code tensors and metadata. CPU only.

## D — FAIL / NOT ESTABLISHED

| Stratum | Method | Mean NRMSE | Max NRMSE | Payload B | Apply μs |
|---|---|---:|---:|---:|---:|
| aligned_shared_generator | oft_oracle | 0 | 0 | 8,821 | 37.9 |
| aligned_shared_generator | boft_independent | 0 | 0 | 5,021 | 36.5 |
| aligned_shared_generator | mirror_shared_boft | 0.04674045 | 0.1365916 | 4,734 | 37.8 |
| aligned_shared_generator | generic_scalar_boft | 0.04674045 | 0.1365916 | 4,735 | 36.3 |
| aligned_shared_generator | shared_identity | 0.09275002 | 0.2035892 | 4,727 | 36.1 |
| independent_orthogonal | oft_oracle | 0 | 0 | 8,821 | 40.4 |
| independent_orthogonal | boft_independent | 1.394858 | 1.435614 | 5,021 | 37.3 |
| independent_orthogonal | shared_identity | 1.418077 | 1.460121 | 4,727 | 37.0 |

Fact: fresh aligned shared Mirror/generic scalar mean NRMSE was 0.04674 and max 0.13659, failing the 1e-4 success gate. Payload 4,734 B was only 5.7% below independent BOFT at 5,021 B; the stronger quality requirement was not met. The generic scalar result was exactly the same quality, so no Mirror-specific improvement appears. Aligned independent BOFT used the teacher's generated factor values and reached zero reconstruction error.

Fact: independent support-fitted BOFT mean NRMSE was 1.3949, close to shared identity at 1.4181, whereas the dense OFT oracle was exact. Thus the fixed-budget factor fitting did not solve these independent tasks.

Interpretation: this is an implementation/optimization-limited result. It does not establish that butterfly factors lack capacity; rather the fixed two-sweep support-only search was inadequate. The aligned Mirror code also failed despite the teacher lying near its intended shared-generator family, indicating a mismatch in generator construction/decoding or fit procedure. Fresh worlds are closed and results are preserved without retuning.

Hypothesis: a differentiable, batched BOFT factor learner with verified encode/decode reconstruction is needed before meaningful compression claims. The existing evidence supports no deployment or Mirror claim.

## C — Strongest counter-hypothesis

The apparent failure is caused by a malformed butterfly pairing schedule and weak factor fitting: the schedule is not a validated canonical BOFT implementation, and the shared scalar generator is not a robust factor-bank encoder. The source must be audited before treating this as evidence about the BOFT family.

## U — Unknown

No canonical BOFT training, natural image task, larger dimension, optimized independent-factor fit, or GPU kernel was measured. Registered quality/storage gates failed; general capacity is not established.
