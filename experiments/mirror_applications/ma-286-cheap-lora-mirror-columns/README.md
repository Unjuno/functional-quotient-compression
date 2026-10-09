# MA-286 — Cheap-LoRA Mirror column-subspace views

## H — Hypothesis

Shared column bases with compact task views may improve the quality/byte frontier over fixed Cheap-LoRA columns for aligned tasks; independent subspaces may require private factors.

## T — Conditions

Frozen 32×32 linear map, rank 4, eight tasks. Fresh worlds 28610–28612 × seeds 0–2; 108 rows. Compared no adapter, fixed identity-column cLA, per-task Cheap-LoRA, shared-basis Mirror address, generic coefficient over the same basis, and full rank-4 LoRA. Aligned targets use one shared orthonormal left subspace and shared right factor with task coefficients. Inference payload charges W, bases and all task codes/factors. CPU only.

## D — FAIL for Mirror-specific gain; strong aligned shared-subspace signal

| Stratum | Method | Mean NRMSE | Payload B | Fit seconds |
|---|---|---:|---:|---:|
| shared_subspace | none | 1 | 4,160 | 0.000 |
| shared_subspace | fixed_cla | 0.9899013 | 5,337 | 0.460 |
| shared_subspace | cheap_lora | 0.01246363 | 12,376 | 0.242 |
| shared_subspace | shared_coeff | 2.991173e-08 | 5,340 | 0.324 |
| shared_subspace | mirror_address | 2.991173e-08 | 5,342 | 0.324 |
| shared_subspace | full_lora | 0.009437419 | 12,375 | 0.229 |
| independent | none | 1 | 4,160 | 0.000 |
| independent | fixed_cla | 0.9982309 | 5,337 | 0.318 |
| independent | cheap_lora | 0.0002894497 | 12,376 | 0.241 |
| independent | shared_coeff | 0.9979512 | 5,340 | 0.342 |
| independent | mirror_address | 0.9979512 | 5,342 | 0.341 |
| independent | full_lora | 0.0001592127 | 12,375 | 0.255 |

Fact: aligned tasks: Mirror and generic coefficient have identical NRMSE 2.99e-8 at 5,342/5,340 B, versus per-task Cheap-LoRA NRMSE 0.01246 at 12,376 B. Full LoRA reaches 0.00944 at 12,375 B. Fixed identity columns fail at 0.990 because the teacher subspace is randomly oriented.

Fact: independent tasks: Mirror/generic NRMSE ~0.998, near no adapter 1.0; full/cheap LoRA reach ~1.6–2.9e-4 at 12,375–12,376 B.

Interpretation: a shared learned column subspace and task coefficient bank can encode aligned task updates with ~57% fewer payload bytes than per-task Cheap-LoRA at high quality. The generic coefficient control exactly matches Mirror; this is not Mirror-specific. Independent task subspaces need private factors.

## C — Strongest counter-hypothesis

The aligned teacher is generated directly from the same shared left/right factors used by the compact code, while fixed cLA uses identity columns. This measures orbit compression, not general adaptation performance. The generic control is mathematically identical.

## U — Unknown

No real model/task, feature-space learning of a shared basis, heterogeneous task distribution, GPU runtime, or adaptive private-residual frontier was tested.
