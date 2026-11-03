# MA-578 results

## Fact

The FP16-cache next-token NLL was 5.1871 and 4.3043 across the two valid-text development worlds. Identity int4 cache worsened it by 1.3640 and 1.6288 nat/token. The fitted four-code rotation bank worsened it by 0.7313 and 1.5503 nat/token, far beyond the frozen 0.05 nat/token bound. The fitted method is exactly equal in serialized code/cache arrays and NLL to the native shared-codebook control.

Independent 16-code rotation had deltas +0.7823/+1.4779; one global QuaRot code had +0.6763/+1.7722; random four-code banks had +0.7205/+1.1216. Random four-code NLL is 0.4287 nat better than fitted in seed 57802. These results vary by prompt draw, but all miss the FP16 quality gate by large margins.

Measured serialized one-prefix cache bytes were 787,746 B FP16; 222,762 B identity int4; 235,902 B independent16; 224,126 B fitted4/native4/random4; and 223,742 B global QuaRot. The shared bank saves 11,776 B (4.99%) versus independent16, while int4 state is about 71.5% smaller than FP16. The serialized rotation-code array members alone are 992 B for codebook4 versus 12,768 B independent16; remaining payload differences include quantized cache values, scales and NPZ metadata. The shared bank does not improve next-token quality over native or random code selection. File sizes and SHA-256s for each NPZ and metrics file are listed in `ARTIFACT_PROVENANCE.json`.

CPU time per evaluated 64-token cache was about 0.044–0.053 s for rotated codes versus 0.010–0.013 s identity int4; per-query model time was about 0.011–0.014 s. Calibration candidate error took 2.78–2.93 s per world. Added online query/value view proxy is 30,720 operations/token. Query timing reconstructs original-coordinate K/V tensors and is not an optimized rotated-cache kernel measurement.

## Interpretation

MA-578 FAILS the preregistered quality gate in both dev worlds and exactly aliases the native shared-codebook control. The bank reduces metadata against independent rotations, but that storage saving does not recover KV-cache quality. Identity int4 is smaller and, although poor, is less damaging than rotated candidates in these runs. Fresh WikiText test data stayed sealed.

## H / T / D / C / U

**H:** a four-entry shared rotation bank can preserve next-token quality near FP16 while reducing paid int4 KV-cache state versus independent role rotations.

**T:** pinned Pythia-70M, WikiText-2 raw; four 64-token train prefixes fit the 16-candidate code set, and 16 one-token continuation queries on valid text measured cache NLL. K/V state spans 6 layers × 8 heads × 2 roles. Controls: FP16, identity int4, global QuaRot, independent16, learned codebook4, random4 and native codebook4. Actual serialized cache bytes and CPU timing were measured. Fresh seeds 57811–13 remain sealed.

**D:** FAIL. Both dev NLL deltas exceed 0.05 nat/token; fitted codebook exactly aliases native selection. The random codebook beats fitted quality in one world.

**C:** int4 cache quantization itself is too destructive at group size 32 for these K/V tensors, and rotation codebook selection does not restore the lost next-token likelihood.

**U:** full-sequence perplexity, larger models/context lengths, 2/3/8-bit caches, an optimized online rotation/fused attention kernel, and fresh test-set transfer.
