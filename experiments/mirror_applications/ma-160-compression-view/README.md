# MA-160 — Mirror residual quantization

## Question

Can a shared int4 residual payload, viewed through small role-specific Mirror coordinates, recover role-specific quantized matrices at lower serialized size than independent quantization?

## Literature context

The targeted search surfaced AWSRC (Activation-Weighted Seeded Residual Coding, arXiv:2608.23144), which selects seeded residual tiles by activation-weighted error reduction per serialized byte and compares sparse, low-rank, and vector-quantized alternatives. PRQuant was also surfaced as a quantization-plus-permutation/residual-compensation approach that reports additional runtime cost. QER/SRR (arXiv:2602.02001) motivates explicit low-rank residual controls. These methods make actual sidecar bytes and runtime part of the comparison; this screen does not claim to reproduce their algorithms.

## Protocol

The fixed screen uses four 16×16 matrices. In the aligned condition, a shared base and shared residual are conjugated by four Givens angles, with a role coefficient and small private noise. The independent condition uses arbitrary unrelated role matrices. All methods receive the same fixed int4 quantizer, and no optimizer updates occur. The Mirror shared-residual method stores one int4 base, one int4 residual, four float32 angles, and four float32 coefficients. The full serialized payload includes tensor names, dtypes, shapes, scales, and method metadata.

Controls are independent int4, residual-free Mirror, private int4 residuals, private rank-2 residuals, independent int4 plus rank-2 QER residual, and untied fp32. Metrics are held-out Gaussian activation-output MSE, relative Frobenius error, exact bytes, a decode/reconstruction MAC proxy, and isolated CPU encode/decode wall time. This is a synthetic post-training reconstruction screen, not a language-model result.

## Reproduction

From this directory, run `PYTHONPATH=source python source/run.py --split development --seeds 16001 16002 --out /tmp/ma160-dev.csv`. The fresh split is locked by `PROTOCOL.json` and should only be opened after the development gate passes.

## Results

### Fact

- The shared-residual method serialized to **393 bytes**, versus **607 bytes** for independent int4 (0.647×, 35.3% fewer). It stored one quantized base, one quantized residual, four float32 angles and four float32 role coefficients, plus the charged headers and scales.
- Development activation-MSE ratios to independent int4 were 1.031 and 1.040; both development byte and quality gates passed.
- Fresh aligned MSE ratios were **1.106, 0.982, 0.976** for seeds 16011, 16012, 16013. Thus the all-seed quality gate of 1.10 missed on seed 16011; the byte gate passed on all seeds.
- Fresh aligned mean MSE was 0.02103 for shared-residual Mirror and 0.02031 for independent int4 (ratio 1.035). Residual-free Mirror averaged 0.06847 (3.37× independent int4). The private int4 residual used 788 bytes and averaged 1.033× independent-int4 MSE. Private rank-2 residual used 825 bytes and averaged 2.48× MSE. Independent int4 plus rank-2 QER used 1,228 bytes and averaged 0.614× independent-int4 MSE.
- On unrelated independent role matrices, shared-residual Mirror averaged 34.6× the independent-int4 MSE. Private int4 residual returned near independent-int4 quality at 1.30× the bytes.
- Shared-residual Mirror's reconstruction MAC proxy was 34,304 versus 1,024 for independent int4 (33.5×). Average measured CPU encode/decode time was 1.09 ms versus 0.89 ms in this small screen. No optimizer updates were used; 512 held-out Gaussian activation examples were scored per row.
- The custom serialized payload was deserialized before scoring. All 70 rows replayed with exact payload/compute fields and zero activation-MSE or relative-Frobenius replay difference. The three experiment tests passed.

### Interpretation

Within this deliberately aligned matrix family, a single quantized residual with role coefficients recovers most of the independent-int4 quality using about 35% fewer actual bytes. It improves substantially over residual-free Mirror. The method does not dominate a higher-byte independent QER residual control on quality, and its dense view reconstruction has a large compute proxy. Arbitrary role functions need private state. This is a promising narrow storage tradeoff; the preregistered fresh quality gate was not fully met, so it is not a gate pass or general compression result.

### Hypothesis

The shared residual likely captures variation intentionally placed in one shared residual direction, while private perturbations and unrelated functions need private residual capacity. The 393-byte result is also close to the 0.65 byte-ratio boundary and should be treated as a small synthetic screen, not evidence of a broad deployment advantage.

### Counter-hypothesis and unknowns

**Strongest counter-hypothesis:** ordinary shared residual coding, rather than Mirror itself, explains the gain; the gain is specific to the known conjugation family and may disappear under learned or imperfect coordinates. A byte-near non-Mirror shared-basis residual control has not been optimized here.

**Still unconfirmed:** behavior on real quantized neural-network layers, calibration data, optimized kernels, larger shapes, learned addresses, realistic activation distributions, and whether a simple low-rank shared residual reaches the same frontier at comparable bytes and runtime.

### Decision

**PROMISING, with the preregistered fresh quality gate missed.** The all-seed success gate was not satisfied because seed 16011 measured 1.106× the independent-int4 activation MSE against a 1.10× limit. Retaining PROMISING reflects the repeated storage improvement and near-threshold fresh quality result, not a full PASS.
