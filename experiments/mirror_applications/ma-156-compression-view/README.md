# MA-156 — one int4 weight payload to multiple Mirror decodes

Status: PROMISING with a missed secondary gate
Evidence lane: STORAGE
Base commit: `35f6bb0646d566011554222851e3b4473306ca57`

## Hypothesis

H: One int4 payload plus charged Mirror coordinates can recover four useful role-specific matrices at similar activation quality to independently quantized matrices, using substantially fewer serialized inference bytes.

## Physical-to-logical claim

- Physical object: one symmetric int4-quantized 16×16 matrix.
- Mirror coordinate: one float32 Givens angle per role.
- Logical multiplicity: four 16×16 role-specific matrices.
- Failure modes: quantization error may grow after the view; independent or QER reconstructions may have a better quality/byte/compute frontier; metadata can consume the savings.

## Prior-art delta

The targeted paper search found no direct method for one quantized neural weight payload decoding to several logical matrices. PA11/PA12 cover VSA binding and HRR, not neural weight reconstruction. Preserve-Then-Quantize / SRR (arXiv:2602.02001) combines quantized weights with low-rank error reconstruction, and qFHRR (arXiv:2604.25939) quantizes holographic phases while retaining binding algebra. This screen therefore includes individual int4, per-matrix int4+rank-2 QER, and shared int4+rank-2 residual controls. The Givens-aligned teacher is deliberately favorable; no general quantization claim follows.

## Fresh results

There were no optimizer updates. A fixed int4 quantizer was applied and all paid bytes were written to a custom binary inference format. The decoder reconstructs matrices exactly from those bytes.

| Method | Median aligned activation MSE | Payload | Decode MAC proxy |
|---|---:|---:|---:|
| independent int4 matrices | 0.01577 | 585B | 1,024 |
| hard tied int4 | 0.06441 | 161B | 256 |
| Mirror shared int4 + four angles | 0.01516 | 187B | 33,024 |
| shared int4 + rank-2 residual/role | 0.02424 | 745B | 4,352 |
| independent int4 + rank-2 QER | 0.00952 | 1,159B | 5,120 |
| untied fp32 | 0 | 4,128B | 0 |

In all three fresh aligned worlds, Mirror MSE was within 1.01× of independent int4 and used 68.0% fewer actual serialized bytes. It improved hard tying by about 4.2× in median activation MSE, but missed the preregistered stricter threshold requiring ≤0.10× the tied MSE (observed 0.235×). Thus the core independent-int4 storage/quality frontier is promising; the full registered pass gate did not pass.

The shared rank-2 residual control was larger and less accurate than Mirror, while independent QER was more accurate at a larger payload. Those methods occupy different points on the storage/quality frontier. On the independent-matrix teacher, Mirror error rose to median MSE 0.768 versus 0.01499 for independent int4; role-specific private reconstruction was required.

Mirror's decode proxy was 32× the independent-int4 dequantization proxy due to dense Givens conjugation. Measured CPU decode/application wall was similar to independent int4 in this tiny case, but the proxy shows the arithmetic cost that a production kernel must remove.

## Verification

Sixty rows (development and fresh, aligned and independent) were replayed. All actual payload lengths matched exactly and deserialization reproduced every in-memory decode with zero tensor difference. Tests passed 3/3.

## Decision

**FACT:** Fresh aligned quality and byte gates versus independent int4 passed 3/3; Mirror payload was 187B vs 585B. The secondary hard-tie margin gate missed. Independent matrices were not represented well by Mirror. QER was more accurate but used 1,159B.

**INTERPRETATION:** A quantized shared physical matrix can support several useful logical weights when they lie on the same Givens orbit. It does not replace private parameters for unrelated matrices. The current realization trades storage for extra reconstruction arithmetic.

**HYPOTHESIS:** Better view kernels or structured low-cost decode may improve the compute frontier; broader quantized weight families may need small private residuals.

**BOUNDARY:** Synthetic post-training 16×16 matrices; fixed int4 symmetric quantizer; teacher generated from the tested view; no transformer-wide or language-model quality claim.
