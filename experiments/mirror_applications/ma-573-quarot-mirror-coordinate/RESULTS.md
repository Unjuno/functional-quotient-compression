# MA-573 checked results

## H / T / D / C / U

**H:** Exact symmetry views might improve output-projection int4 quality at bounded bytes/compute.
**T:** Pinned Pythia-70M `embed_out.weight`, int4 group size 32, fixed calibration/held-out output-row partition; two registered development codebook seeds, five serialized methods, zero optimizer updates.
**D:** FAIL; no ≥5% held-out NRMSE gain, identity/SmoothQuant are better, and the Mirror sweep exactly aliases the native QuaRot sweep. Fresh sealed.
**C:** Standard QuaRot/SpinQuant coordinate search explains the entire method; the chosen group quantizer favors identity.
**U:** Natural language NLL/perplexity, end-to-end inference, other quantization layouts and learned SpinQuant optimization remain unknown.

## Results

| Method | Payload bytes | Dev 57301 heldout NRMSE | Dev 57302 heldout NRMSE | Transform ops/token | Encode/search seconds |
|---|---:|---:|---:|---:|---:|
| Identity int4 | 14,490,256 | 0.055981 | 0.055981 | 0 | not timed |
| Single randomized QuaRot | 14,491,783 | 0.101579 | 0.101579 | 3,072 | 2.11 / 1.71 |
| Best of 16 Hadamard views | 14,491,782 | 0.101579 | 0.101579 | 3,072 | 28.80 / 28.76 |
| Native QuaRot sweep | 14,491,789 | 0.101579 | 0.101579 | 3,072 | same sweep |
| SmoothQuant diagonal scale | 14,491,551 | 0.056563 | 0.056563 | 512 | 0.76 / 0.74 |

The fixed row partition uses 20% calibration rows and the complementary held-out rows. Both dev seeds select different candidate indices, but the best and first random views have numerically indistinguishable held-out error. Both dev worlds pass the full-precision symmetry check at max absolute output delta ≤1.27e-6.

The 16-bit source matrix is 51,511,296 B. Int4 payloads are about 28.1% of that size, or about 71.9% fewer bytes. This improvement is from ordinary int4 quantization; identity achieves it too. Relative to identity, the selected View adds about 1.5 KB and does not improve the error. The transform FLOP proxy is 0.006% of the quantized matrix-vector projection.

All 10 actual NPZ payloads remain in the external artifact workspace; byte counts and SHA-256 hashes are committed in `ARTIFACT_PROVENANCE.json`. Replay of seed 57301 reproduced all five payload hashes. No fresh codebook seed was opened.

**FACT:** Quantized matrix errors, actual payload sizes, encode times, hashes and symmetry checks are in the per-seed metrics.
**INTERPRETATION:** No added value beyond int4 plus existing QuaRot/SmoothQuant coordinate methods is established.
**HYPOTHESIS:** A learned rotation may improve this result under representative activation statistics, but the native SpinQuant baseline would be mandatory.
