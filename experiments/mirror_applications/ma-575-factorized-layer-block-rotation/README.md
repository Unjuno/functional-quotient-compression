# MA-575 — Factorized layer × block rotation code

Status: FAIL
Prior art: PA113 SpinQuant; PA21 BOFT

## H

A layer-indexed signed-permutation factor shared by the two projections in that layer, composed with an input-block-indexed factor shared across layers, may approximate independent int4 quantization gauges with less serialized rotation state.

## T

Use the pinned Pythia-70M checkpoint, attention output and MLP up matrices in layers 0–5, groupwise int4 at group size 32, and 16 fixed Hadamard/signed-permutation candidates. Fit factors on calibration rows from layers 0–3; audit on complementary rows from layers 4–5. Compare identity, independent matrix×block codes, layer-shared codes, random factors, fitted factors, and a native BOFT-style factor-selection control. Exact paid bytes include quantized weights, FP16 scales, factor codebook and all factor IDs.

Development seeds are 57501/57502. Fresh seeds 57511–57513 remain sealed unless both development gates pass and the native control is not an exact alias.

See `PROTOCOL.json` for the frozen protocol and `RESULTS.md` for final evidence.
