# MA-628 status

- Status: FAIL for Mirror-specific attribution; strong scoped rate-distortion result vs independent curves
- Branch: `research/ma-628-coinpp-mirror-codec-20261103`
- Base commit: `8ff5351d`
- Protocol frozen before development: yes
- Development complete: yes (62801, 62802)
- Fresh/audit opened: no; native code alias gate failed
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** Quantized/entropy-coded periodic function codes preserve held-out quality with fewer actual payload bytes than independent sampled curves; Mirror-specific value must beat a native COIN++-style code.

**T:** Frozen 4×4 harmonic/phase family, diagonal pairs held out, 64 support and 512 audit samples, seeds 62801/62802. Amplitude/phase quantized at 4/6/8/10/12 bits, harmonic at 2 bits. Full bitstream includes ranges, schema, task IDs and zlib-coded packed values. Compared native modulation with the exact same stream and 8-bit independent sampled curves. Fresh stayed sealed.

**D — FAIL for Mirror-specific attribution.** At 12 bits, mean held-out nRMSE was 0.00038–0.00039 and complete compressed payload was 218–219 B, against 4,962–5,272 B for independent 8-bit curves (about 4.1%). At 4/6/8/10 bits, nRMSE was about 0.107/0.026/0.0063/0.0015 with 183/191/200/210 B. Native COIN++-style modulation used the exact same stream and matched every quality/byte point; fresh stayed sealed.

**C:** The measured gain comes from ordinary quantization and entropy coding of a shared periodic decoder's modulation vector. The Mirror code is exactly the native activation parameter vector.

**U:** No learned decoder, noisy code fitting, multiharmonic mixtures, or natural signals were tested. The codec screen does not establish a gain over actual trained COIN++ models.

## Fact / Interpretation / Hypothesis

**Fact:** 12-bit codes reached nRMSE below 0.001 at ~4.1% of independent sampled-curve bytes. The native control was byte-for-byte the same stream with identical reconstruction.

**Interpretation:** compact functional codes provide a strong storage point for this narrow periodic family; Mirror-specific geometry contributes no gain over native modulation coding.

**Hypothesis:** code geometry could matter in richer signal families if it improves the actual entropy-coded rate-distortion frontier over trained native modulation codes.
