# MA-586 — Quantized MLA latent with Mirror gauge

Status: **FAIL against Hadamard/direct controls**  
Branch: `research/ma-586-quantized-mla-mirror-gauge-20261009`  
Base commit: `99351d0b`  
Prior art: PA112 QuaRot; PA115 MLA

## H

A compact Mirror gauge selected on development caches can lower int4 MLA latent cache attention distortion at equal actual bytes versus identity and randomized Hadamard gauges, beyond a matched direct rotation code.

## Frozen screen

Generate heavy-tailed latent K/V caches for eight roles, length 4 and dimension 16. Quantize with symmetric int4 and per-role scale. Compare identity latent quantization, fixed Hadamard rotation, block-Givens Mirror gauge selected by dev query-output distortion, and direct block-Givens control with the same search budget. Reconstruct caches and measure attention output nMSE on fixed query probes. Two development seeds, 64 angles on a preregistered grid; all scales, packed values, rotation indices and metadata are charged. Also measure bytes and reconstruction/attention MAC proxies. No fresh split opens if direct angle control matches.

PASS requires Mirror to lower attention-output nMSE by ≥10% vs identity and Hadamard at no more than 5% byte overhead, and beat the direct rotation control. FAIL if no frontier gain or direct control matches.

## H / T / D / C / U

- **H:** Low-description gauge selection before quantization protects attention function quality.
- **T:** Eight roles, int4 K/V latent, train/eval queries split 32/32, two seeds, 64 angle evaluations; eight rows.
- **D:** FAIL. Givens Mirror reduces identity attention nMSE .0606→.0381 and payload 650→647B, but Hadamard is better at .0194/650B and direct Givens exactly matches Mirror.
- **C:** QuaRot-style Hadamard and direct Givens angle sweep.
- **U:** GPU kernels, actual MLA checkpoints, long-context serving and end-to-end latency.
