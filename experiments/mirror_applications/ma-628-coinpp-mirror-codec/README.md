# MA-628 — Quantized periodic Mirror code codec

Status: SCREENING; prior art PA133 COIN++. Dedicated branch: `research/ma-628-coinpp-mirror-codec-20261103`.

## H

A quantized/entropy-coded periodic Mirror code can preserve held-out signal quality at fewer actual payload bits than independent sampled functions. Mirror-specific credit requires outperforming ordinary COIN++-style quantized modulation vectors using the same decoder.

## T

Reuse the frozen 4×4 harmonic/phase signal family, diagonal pairs held out, 64 support and 512 audit points; development seeds 62801/62802. Fit amplitude/phase/frequency from support, quantize amplitude and phase at 4, 6, 8, 10, 12 bits with frequency at 2 bits, bit-pack and zlib-compress the full inference payload with schema/ranges/indices. Compare Mirror codes, identical native periodic modulation vectors, and independent 512-point 8-bit sampled curves. Measure actual file bytes, encode/decode time, compute and audit nRMSE. Fresh 62811–62813 sealed on native alias.

## D

Pending.

## C / U

Counter-hypothesis: code quantization and entropy coding are ordinary COIN++ practice; Mirror uses the same code as native periodic modulation. This is a synthetic single-layer harmonic screen, not a learned implicit network.
