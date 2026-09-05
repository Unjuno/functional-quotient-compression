# MLP tail rate/support boundary — T215–T220, corrected by T221–T224

## Critical correction (2026-09-06)

The earlier instruction to treat exact restoration of selected neurons as a task-quality upper bound, and prune a candidate when that restoration loses to uniform K128, is **withdrawn**. Partial restoration is a useful control, not a certified optimum: errors in the remaining compressed parameters can be compensated by an inexact selected correction. No candidate may be hard-pruned from restoration quality alone.

The earlier support ratio also pooled two separately trained residual codebooks. The correct support must be counted **per codebook**, not across both FC and PROJ.

All functional experiments in T221–T224 are constructed or synthetic. No TinyStories weights were available and no new real-model or whole-model 64x quality claim is made.

## Historical exact candidate layouts (unchanged)

Under the 28M layout assumptions, 32 KiB metadata reserve, and 1,624,624-byte ceiling:

| Tail codewords | Protected bundles | Whole placeholder bytes | Headroom, bytes |
|---:|---:|---:|---:|
| 256 | 1,976 | 1,624,599 | 25 |
| 512 | 1,542 | 1,624,624 | 0 |
| 1024 | 959 | 1,624,600 | 24 |

These measured layouts contain deterministic placeholder payloads, not real encoded weights. The 4,352-byte directory is included in the 32,768-byte reservation. Neither file size nor checksums establish functional quality.

## Corrected per-codebook training support

A target bundle has one 512-coordinate FC row and one 512-coordinate PROJ column. There are 32 total 32-coordinate blocks per bundle, but **16 go to each of two independent codebooks** in the audited T217 implementation.

| Tail codewords | Bundles | Training blocks per codebook | Samples per codeword |
|---:|---:|---:|---:|
| 256 | 1,976 | 31,616 | 123.5 |
| 512 | 1,542 | 24,672 | 48.1875 |
| 1024 | 959 | 15,344 | 14.984375 |
| 2048 | 59 | 944 | 0.4609375 |

K1024 remains the largest tested power-of-two candidate satisfying both the historical budget and the current selected-only KMeans fitter's sample-count prerequisite. This is not an information-theoretic codec impossibility result. Other training pools or unused/duplicate centroids change the training assumption; an external training pool need not add decoder bytes when the resulting codebook was already fully paid. K4096 still exceeds the historical tail budget in codebook metadata alone.

## T221: explicit counterexample to restoration pruning

Consider a two-class model whose logits are the sum of two weights and zero. All quantities are dimensionless; natural logarithms are used for KL. The original weights are 1 and 1, giving logits [2,0]. The second compressed weight stays fixed at 0.5.

| Selected first weight | Interpretation | KL from original distribution |
|---:|---|---:|
| 1.5 | Compressed base; errors cancel | 0 |
| 1.0 | Exact restoration of selected weight | 0.014883805929 |
| 1.2 | Comparison candidate | 0.005097141737 |
| 1.25 | Inexact restoration | 0.003495408890 |

Exact restoration loses to the comparison candidate, yet the inexact correction beats it. This is enough to refute the general pruning rule. The same construction was checked with two identical GELU neurons on 4,097 inputs. It does not say that restoring *all* original parameters is suboptimal; it concerns a selected subset while other parameters remain compressed.

## T222: decoded INT4 nonlinear control

A fixed test used 24 random GELU networks: 12 independent-input-weight instances and 12 correlated-pair-input instances. Dimensions were input 64, hidden 128, output 16. Thirteen coupled neurons were selected on 96 calibration inputs; a global gain from five fixed candidates was selected on a different 96 inputs; task KL was measured on 384 untouched audit inputs.

Each tail was packed and decoded before evaluation: symmetric INT4 codes, FP16 scales, 7-bit IDs, FP16 gain, header, and SHA-256 trailer. Every selected tail was 962 bytes. This is a tail-only synthetic format, not the target VQ codec.

- Unit-gain decoded INT4 corrections beat exact partial restoration in **2/24** instances.
- Across all 24 instances, mean KL was **0.05176176** for exact restoration and **0.05252787** for the INT4 candidate: exact restoration was better on average.
- The five-value gain selection selected unit gain in all 24 cases; this test does not demonstrate a gain-tuning improvement.
- Against the separately defined uniform three-bit control, no actual false-prune event occurred in these 24 instances. The universal pruning rule is refuted by T221 and by the two restoration-order reversals, not by claiming a nonzero empirical false-prune rate against that particular control.

## T223: why the old high-K proxy ordering is not a target-quality result

T217 preserved protected-neuron fractions but not codebook occupancy. Its input/hidden dimensions were both 768, with only 45 selected neurons for K1024. Thus each residual codebook saw 1,080 blocks: **1.0546875 samples per codeword**, versus **14.984375** in the intended 28M layout.

The corresponding proxy/target occupancies are:

| Codewords | Small proxy | Intended target |
|---:|---:|---:|
| 256 | 8.71875 | 123.5 |
| 512 | 3.375 | 48.1875 |
| 1024 | 1.0546875 | 14.984375 |

A controlled IID Gaussian residual experiment held block length and quantization fixed, ran 18 fits (three codeword counts, two support sizes, three seeds), and decoded affine INT4 centroids with FP16 offsets/scales. For K1024:

| Support condition | Training-block NMSE, median [range] | Independent-block NMSE, median [range] |
|---|---:|---:|
| Proxy-like | 0.023564 [0.023563, 0.024124] | 0.859891 [0.858134, 0.864603] |
| Target-like | 0.634888 [0.634202, 0.638603] | 0.750352 [0.749750, 0.752326] |

Near-interpolation is therefore a concrete transfer risk. Fitting a codebook to the very weights being compressed is legitimate; the issue is that the tiny proxy and intended target have materially different representational resources per training vector. These NMSE values are **not task KL** and do not establish which tail wins on TinyStories.

The audited T217 base centroids and tail quantizer scales/offsets also remained FP32; its functional forward was not a replay of the T215 FP16-scale binary. The old K1024-first quality ordering is downgraded to a weak proposal, not an established target ranking.

## T224: evidence-gate regression tests

The archived T212 gate accepted six invalid or inadequate fixture cases: NaN metrics, negative bytes, synthetic evidence labelled with joint scope, placeholder payloads, one improved metric with catastrophic regression in the others, and a mismatched replayed-file hash.

The replacement local gate passed **33 regression tests**. It checks finite metrics, ranges, actual supplied artifact size/hash, checkpoint and audit/backend matching, real versus synthetic/placeholder evidence, a predeclared quality policy, and all quality guardrails. Restoration quality cannot trigger pruning. Its positive status is **PASS_FIXED_SAMPLE**, not a generalization or intrinsic-mirror certificate. Input validation cannot independently attest that a trusted runner executed a forward pass.

The replacement gate's numeric acceptance limits in unit tests are demonstration fixtures only; no TinyStories acceptance thresholds were retroactively chosen.

## Updated real-model run protocol

1. Restore the original checkpoint and verify hash, shapes, and a baseline forward.
2. Separate calibration, development selection, and final audit passages before fitting.
3. Build uniform K128 and the K64-base candidates using the declared, fully charged serializer format.
4. Freeze calibration-selected supports. Evaluate exact restoration only as a diagnostic control; do not prune from it.
5. Train and serialize actual K256/K512/K1024 candidates as resources permit. Select on development data, not final audit.
6. Evaluate the exact decoded artifact jointly. Do not add component KLs.
7. Keep RATE, decoding fidelity, fixed-sample quality, uncertainty, and intrinsic-scaling claims separate.
8. Without a verified real checkpoint, serialized candidate, and predeclared quality policy, real quality remains BLOCKED/PENDING.

## Execution boundary

T221–T224 ran locally on CPU (Intel Xeon Platinum 8370C, nominal 2.80 GHz; clock not locked; two numerical threads), Python 3.13.5 and PyTorch 2.10.0+cpu. This is not a speed benchmark. The original T221 development assertion used the wrong comparison logit; it failed and was corrected to 1.7 before recording results. Raw scripts, fixed protocols, per-seed JSON, decoded tails, source snapshots and failure notes are retained in the local T221–T224 result bundle.
