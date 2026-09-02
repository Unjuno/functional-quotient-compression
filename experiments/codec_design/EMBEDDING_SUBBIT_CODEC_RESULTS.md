# Sub-bit embedding codec design results

This note records high-value **synthetic codec-design evidence** produced in the local container. It is not a real SmolLM2 embedding-quality result and it does not establish 64x feasibility.

## 1. Serializer authority at ~0.25 bit/scalar

For the SmolLM2 embedding shape (`49,152 × 576`), the proportional 0.25 bit/scalar embedding budget is `884,736` bytes.

A `PQ M=16, K=256` layout with 128 token-specific index bits, a 3-bit codebook, FP32 scale + U8 zero per codeword, and a 4 KiB header serializes to:

- `866,304` bytes;
- `0.2447917` bit/scalar;
- `18,432` bytes headroom.

The 4-bit-codebook variant is exactly `884,736` bytes and has zero headroom. Record alignment can therefore change PASS/FAIL. For example, the corresponding 4-bit-codebook layout fails if 23-byte codeword records are rounded to 24 bytes; planar packing fits exactly.

Low-rank representations have the same issue. `rank-35 / 4-bit` fits at `875,838` bytes only with global or grouped bit-packing. A naive per-token 18-byte record becomes `900,414` bytes and fails the 0.25 bit/scalar component budget.

## 2. Invertible alignment is useful only under a restricted codec and only if its description is paid

A full data-adaptive PCA rotation can greatly improve PQ on correlated synthetic data, but an arbitrary `576 × 576` rotation is not free and cannot be treated as a fundamental compression gain.

A low-description alternative was tested: a sequence of quantized Householder reflections. Each reflector is a decoder-visible unit vector. With fixed-range 6-bit entries, one 576-dimensional reflector uses 432 bytes before optional record alignment.

The practical candidate tested here is:

- base codec: `PQ M=16, K=256`, 3-bit codebook;
- transform: 40 Householder reflections;
- reflector precision: 6 bit/entry;
- rotation payload: `17,280` bytes;
- total embedding payload: `883,584` bytes;
- headroom vs 0.25 bit/scalar embedding budget: `1,152` bytes.

A 6-bit reflector record is 432 bytes, so it is already 16-byte aligned. If each reflector is padded to 448 bytes for 32/64-byte alignment, the q=40 configuration still fits with 512 bytes headroom.

## 3. Synthetic correlated-Gaussian results

For dense correlated Gaussian embeddings (`D=576`, additive noise 0.25), the q=40 / 6-bit Householder transform improved 3-bit-codebook PQ NMSE as follows, with two independent seeds per latent rank:

| latent rank | plain PQ NMSE | Householder+PQ NMSE | relative improvement |
| ---: | ---: | ---: | ---: |
| 64 | ~0.7246 | ~0.4210 | **~41.9%** |
| 128 | ~0.7707 | ~0.6069 | **~21.3%** |
| 256 | ~0.8003 | ~0.7143 | **~10.75%** |

An isotropic-Gaussian negative control showed the opposite behavior: the same Householder procedure worsened NMSE by roughly `0.4–0.7%` across three seeds. Therefore the transform must be gated by real structure; it is not a universal improvement.

A full free PCA/variance-balanced rotation can produce much larger synthetic gains, but that result is **not codec evidence** unless the rotation description is included. The paid q=40 Householder result above is the relevant finite-description candidate.

## 4. Spectrum-dependent codec family

Synthetic experiments also compared three embedding codec families at approximately the same 0.25 bit/scalar storage scale:

- plain product quantization;
- q=40 Householder-aligned product quantization;
- rank-35 / 4-bit low-rank coding.

The preferred family changes with covariance concentration. In the tested correlated-Gaussian family:

- near-isotropic (`top-35 explained variance ~0.106`): plain PQ is best;
- moderate anisotropy (`~0.23–0.40`): Householder+PQ is best;
- strong low-rank structure (`~0.42+`): rank-35 coding becomes competitive/best;
- the crossover near `0.41–0.42` is a synthetic-family observation, not a universal threshold.

This motivates measuring the actual embedding spectrum before choosing the codec family rather than committing to one representation in advance.

## 5. Boundaries

Do not infer from this note that:

- SmolLM2's actual embedding is compressible to acceptable quality at these rates;
- the synthetic spectrum thresholds transfer to real embeddings;
- invertible alignment lowers the fundamental rate-distortion function;
- the embedding result by itself makes 64x model compression feasible.

The next real-model gate is to measure the pinned embedding matrix spectrum and evaluate the candidate codecs on actual embedding/logit behavior under exact serialized accounting.
