# Semantic MLP tail codec and candidate selection — T225–T238

This note records durable conclusions from local T225–T238 experiments. No verified TinyStories checkpoint weights were available; all functional results are synthetic. Exact serializer/decoder claims are software evidence only unless explicitly stated otherwise.

## 1. Historical rate skeleton versus executable codec

The T133/T135/T215 MLP-tail layout paid for two FP16 per-bundle tail scales, INT4 32-D codewords with FP16 affine metadata, packed bundle IDs and fixed-bit indices. The byte accounting was exact, but the historical executable scripts did not define a decoder that used the stored per-bundle scale bytes to reconstruct residual weights. T217 functional experiments also operated on raw residual blocks rather than the paid bundle-scale representation.

T231 therefore defines an explicit versioned semantic tail contract while preserving the historical byte counts:

1. one stored FP16 scale for the FC side and one for the PROJ side of every protected bundle;
2. residual blocks normalized by the rounded/stored bundle scale;
3. separate global FC and PROJ codebooks;
4. each 32-D centroid stored as INT4 symbols plus FP16 affine zero/scale;
5. indices reassigned to the serialized centroids;
6. bundle-major index order: 16 FC blocks followed by 16 PROJ blocks;
7. quality is evaluated only after decoding the emitted bytes.

RMS bundle scaling was better than max-abs scaling in every tested synthetic scenario/K. The exact tail sizes remain unchanged:

| tail K | protected bundles | tail bytes |
|---:|---:|---:|
| 256 | 1,976 | 84,898 |
| 512 | 1,542 | 84,923 |
| 1024 | 959 | 84,899 |

T233 embedded these semantic tail payloads into the inherited 64x-sized whole placeholder container and successfully decoded the tail from the emitted whole file. Whole sizes remained 1,624,599 / 1,624,624 / 1,624,600 bytes for K256/K512/K1024. The non-tail payload was still placeholder data, so this is not a whole-model codec-quality result.

## 2. Fixed K1024-first ordering is withdrawn

Once target protected counts, per-codebook occupancy and executable RMS semantics are matched, the best K depends on functional-importance concentration and residual geometry.

Representative RMS-semantic phase map from T232 (rows: importance sigma 0.25, 0.6, 1.0, 1.4, 1.8; columns: residual cluster alpha 0, 0.4, 0.7, 0.9):

```
256  256  256  256
256  256  256  256
512  512  256  256
512  512  256  256
1024 1024 256  256
```

Boundary reseeds showed K512/K1024 competition near sigma=1.4 with weak clustering. Therefore K1024 is not a universal first choice; K256/K512/K1024 must remain feasible candidate representations.

## 3. Actual TinyStories-28M MLP dimensions: synthetic end-to-end rehearsal

T234/T235 used the intended MLP dimensions (8 layers, hidden 512, intermediate 2048, 16,384 coupled neurons) with emitted/decoded K64 and K128 base codebooks plus the semantic RMS tails.

Gaussian-weight runs showed examples where all hierarchy candidates beat uniform K128. Deliberately structured weight geometries provided a stronger falsification test:

- low functional-importance concentration (sigma=0.6): strong structured geometry lost to K128 for all K in 3/3 seeds; moderate structure lost in 2/3 seeds;
- high concentration (sigma=1.4): all tested K256/K512/K1024 tails beat K128 in all six structured setting/seed runs.

Thus the hierarchy is not universally superior to K128 even at target dimensions and target byte semantics.

## 4. Exact partial restoration remains diagnostic only

T236 extended the T221 correction to target MLP dimensions. Across 19 synthetic worlds / 57 candidate comparisons, the emitted inexact semantic tail had lower KL than exact partial restoration in 14/57 cases. No crossing relative to K128 happened in those sampled worlds, but that does not restore a general upper-bound property. Partial restoration must not hard-prune a candidate.

## 5. Structural break-even proxy failed

T237 tested a cheap proposal score:

`positive first-order score share * selected structural residual recovery - (D64-D128)/D64`.

Across 24 candidate cases it predicted success 24/24 times while only 14 actually beat K128, giving 10 false-optimistic predictions and 58.3% accuracy. This proxy is rejected as a hard or proposal gate. Structural reconstruction error remains an inadequate substitute for task-functional replay.

## 6. Current selection protocol: K128-inclusive development tournament

T238 rehearsed a leakage-safe pipeline on 12 structured actual-shape synthetic worlds:

1. calibration inputs select first-order support;
2. K128/K256/K512/K1024 are emitted and decoded;
3. development inputs choose one candidate;
4. final audit inputs are untouched until the choice is frozen.

Results:

- all six low-concentration runs selected K128 on development, and K128 was the audit oracle in all six;
- all six high-concentration runs selected K1024 on development; audit oracle was K1024 in five and K512 in one;
- audit-oracle match: 11/12;
- selected candidate beat or equaled audit K128 in 12/12;
- maximum relative audit KL regret: 0.573%.

This is synthetic pipeline evidence, not a TinyStories quality result. It supports the following real-run protocol:

1. verify real checkpoint provenance and baseline forward;
2. pre-freeze calibration, development and final-audit passages;
3. build emitted/decoded semantic K64 and K128 MLP bases;
4. use multiple independent calibration groups for support estimation;
5. build all feasible K256/K512/K1024 RMS-semantic tails as resources allow;
6. keep exact partial restoration as a diagnostic only;
7. do not use structural break-even gating;
8. include uniform K128 in the development tournament;
9. freeze the development winner and evaluate final audit once;
10. require whole-model joint decoded replay before any 64x quality claim.

## Evidence boundary

No TinyStories real-model quality claim is created by T225–T238. The current high-information blocker remains access to the verified 28M checkpoint (and ideally the full 1M/3M/8M/28M family for mirror-science diagnostics).