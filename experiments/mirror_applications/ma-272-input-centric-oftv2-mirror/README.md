# MA-272 — Input-centric OFTv2 Mirror views

## H — Hypothesis

Applying an orthogonal task transform to activations should reproduce weight-centric OFT without materializing transformed weights. An aligned one-angle view may compress task state; independent rotations should require full transform codes.

## T — Conditions

Shared frozen 32×32 linear map, eight tasks, 256 support and 1,024 audit examples. Fresh worlds 27210–27212 × seeds 0–2, 576 rows. Compared shared baseline, input-centric OFTv2, materialized-weight OFT, aligned Mirror angle and generic fixed-plane scalar. Canonical payload charges shared W and task transform/code; WQ is deterministic from paid W,Q and is not double charged. Fresh source/protocol commit `92a396b1` preceded fresh. CPU only.

## D — Function equivalence passes; Mirror-specific and runtime gates fail

| Stratum | Method | Mean relative error | Max relative error | Payload B | Materialize μs | Forward μs | Extra MACs/batch |
|---|---|---:|---:|---:|---:|---:|---:|
| aligned_plane | oftv2_input | 0 | 0 | 8,270 | 0.0 | 36.3 | 1,048,576 |
| aligned_plane | oft_materialized | 2.81e-08 | 3.27e-08 | 8,275 | 15.9 | 18.4 | 0 |
| aligned_plane | mirror_input_angle | 0 | 0 | 4,181 | 0.0 | 35.5 | 1,048,576 |
| aligned_plane | generic_plane_scalar | 0 | 0 | 4,183 | 0.0 | 35.6 | 1,048,576 |
| aligned_plane | shared_baseline | 0.0996 | 0.202 | 4,174 | 0.0 | 17.7 | 0 |
| independent_orthogonal | oftv2_input | 0 | 0 | 8,270 | 0.0 | 35.4 | 1,048,576 |
| independent_orthogonal | oft_materialized | 2.05e-07 | 2.13e-07 | 8,275 | 15.7 | 17.6 | 0 |
| independent_orthogonal | shared_baseline | 1.41 | 1.48 | 4,174 | 0.0 | 17.4 | 0 |

Fact: input-centric and materialized OFT outputs agree within 2.13e-7 maximum relative error. At batch size 1024, activation-side multiplication took ~35–36 μs vs ~17–18 μs for a pre-materialized weight, and it adds 1,048,576 MACs per batch. Materialization itself took ~16 μs once per task. Activation-side execution avoids that setup but keeps the full 32×32 task transform payload (8,270 B vs 8,275 B materialized serializer).

Fact: the aligned one-angle Mirror payload was 4,181 B vs 8,270 B for full input-centric OFT (~49.4% lower). The generic plane scalar had the same functions, compute, fit and nearly identical payload (4,183 B), so the saving is not Mirror-specific. Shared baseline was ~0.10 relative error on aligned and ~1.41 on independent tasks.

Interpretation: input-centric placement is an equivalent compute/storage trade: it removes the need to construct a transformed weight for each task, but computes an extra matrix product at inference. It is attractive when task switching/setup matters and batch-level latency is secondary. Mirror compression is possible for an aligned orbit, but ordinary scalar coordinates fully explain it.

Hypothesis: at small batches or frequent task switching, a fused input transform may reduce setup latency; at large batches weight materialization amortizes. Fused kernels and GPU measurements could change this frontier.

## C — Strongest counter-hypothesis

The algebra is a standard associativity identity `(XQ)W = X(QW)`; no new functional freedom is introduced by input-centric execution. The Mirror result is a restricted one-plane task family and is exactly matched by a generic scalar control.

## U — Unknown

No GPU, fused kernel, end-to-end neural task, learned Q adaptation, multi-task setup amortization curve, or real OFTv2 training was measured. No Mirror-specific claim is established.
