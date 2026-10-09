# MA-282 — Monarch Mirror FFN transform

## H — Hypothesis

A small task coordinate over a shared Monarch factor direction can yield multiple useful logical FFN functions at lower storage than independent structured maps; generic coefficient control tests whether this is Mirror-specific.

## T — Conditions

Frozen 16×16 linear FFN, eight tasks, block size 4. Fresh worlds 28210–28212 × seeds 0–2, 108 rows. Compared shared baseline, per-task Monarch factors, one-scalar Mirror/generic control, rank-2 LoRA and dense upper. Development showed the common Monarch factor belongs to the shared physical map; it was included in paid W before fresh. The one-dimensional code is fit by closed-form support-only least squares. Actual serialized payload includes W, slow/fast factors, task codes and metadata once. CPU only.

## D — FAIL for Mirror-specific claim; aligned structured result

| Stratum | Method | Mean NRMSE | Payload B | Fit s | Apply μs |
|---|---|---:|---:|---:|---:|
| aligned_monarch | shared | 0.1309296 | 1,136 | 0.0000 | 78.0 |
| aligned_monarch | monarch_native | 5.810354e-08 | 3,202 | 0.5972 | 93.4 |
| aligned_monarch | mirror_scalar | 4.819806e-08 | 1,971 | 0.0002 | 100.0 |
| aligned_monarch | generic_scalar | 4.819806e-08 | 1,972 | 0.0002 | 100.6 |
| aligned_monarch | lora2 | 0.100194 | 3,201 | 0.2040 | 103.7 |
| aligned_monarch | dense_upper | 0 | 9,343 | 0.0000 | 85.6 |
| independent | shared | 0.3612355 | 1,136 | 0.0000 | 85.7 |
| independent | monarch_native | 0.3156757 | 3,202 | 0.4972 | 101.3 |
| independent | mirror_scalar | 0.3604846 | 1,971 | 0.0002 | 86.4 |
| independent | generic_scalar | 0.3604846 | 1,972 | 0.0002 | 84.9 |
| independent | lora2 | 0.2867345 | 3,201 | 0.1865 | 74.1 |
| independent | dense_upper | 0 | 9,343 | 0.0000 | 68.1 |

Fact: aligned tasks: Mirror and generic scalar NRMSE are both 4.82e-8, at 1,971 and 1,972 B. Native per-task Monarch factors reach 5.81e-8 at 3,202 B, so the shared scalar representation uses ~38% fewer bytes. Shared baseline NRMSE is 0.131; rank-2 LoRA 0.100.

Fact: independent tasks: Mirror/generic NRMSE 0.360, near shared baseline 0.361; native per-task Monarch 0.316 and rank-2 LoRA 0.287; dense upper exact.

Interpretation: one physical shared map plus task scalars compresses the aligned orbit, but generic scalar coefficients are functionally identical. Independent task maps require private transform state. No Mirror-specific gain is established.

## C — Strongest counter-hypothesis

The teacher uses the same Monarch factor direction and scalar coordinate as the tested adapter, so this is an exactly aligned function family. The modest LoRA/Monarch residual on the aligned targets is optimizer-dependent; it does not change the matched scalar control result.

## U — Unknown

No natural Transformer FFN, larger dimension, alternative block sizes, GPU kernels, or end-to-end task quality was measured.
