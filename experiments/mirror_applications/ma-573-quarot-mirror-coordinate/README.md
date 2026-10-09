# MA-573 — QuaRot Mirror coordinate sweep

Status: **FAIL** (quality gate and native attribution)
Branch: `research/ma-573-quarot-mirror-coordinate-20261009`
Prior art: PA112 QuaRot, PA113 SpinQuant, PA114 SmoothQuant

## H — Hypothesis

Selecting among exact block-Hadamard coordinate views might improve 4-bit output-projection reconstruction over a single randomized QuaRot view, at acceptable serialized bytes and transform compute.

## T — Execution

Used the pinned Pythia-70M output projection (`W`, 50,304 × 512, stored F16), groupwise symmetric int4 quantization (32 columns/group), a fixed 20% output-row calibration split and 80% held-out rows. Compared identity, one QuaRot randomized Hadamard view, the best of 16 randomized Hadamard/sign/permutation views, the exact same native QuaRot sweep, and SmoothQuant-style diagonal equilibration. Development codebook seeds were 57301/57302. Fresh seeds 57311–57313 stayed sealed after both development worlds missed the quality gate.

All int4 weights, FP16 scales, codes, IDs and metadata were serialized to actual NPZ payloads in `/workspace/artifacts/ma573_dev_57301` and `...57302`; their byte counts and hashes are recorded in `ARTIFACT_PROVENANCE.json`. The full-precision symmetry check uses the same weight matrix and input transform before quantization. No model training or optimizer updates were used.

## D — Decision: FAIL

The best of 16 exact rotations did not reduce held-out NRMSE versus a single randomized QuaRot view in either development world (0.101579 vs 0.101579). Both identity int4 (0.055981) and SmoothQuant (0.056563) are more accurate. The best coordinate adds 1,526 B to the identity int4 payload (14,491,782 vs 14,490,256 B) and 3,072 transform operations/token, with a 0.006% FLOP proxy relative to the quantized matrix-vector product. Its calibration search takes about 28.8 s per seed versus about 2 s for one QuaRot encoding.

Full-precision function preservation passes (max absolute logit delta 1.19e-6 / 1.26e-6). The native best-of-16 QuaRot control selects the exact same code and quantized arrays; only descriptive method metadata differs in the NPZ. Fresh stayed sealed by protocol.

The int4 layer payload is about 14.49 MB versus 51.51 MB for the original F16 matrix (about 71.9% fewer bytes), but that compression comes from ordinary int4 quantization and is present in the identity baseline. The tested rotations provide no extra quality benefit.

## C — Strongest counter-hypothesis

This is ordinary randomized Hadamard coordinate search already represented by QuaRot/SpinQuant. In this group-size-32 output-projection setup, random rotations worsen error relative to identity quantization and calibration selection does not recover that loss.

## U — Unknown

Natural-text NLL/perplexity, other layers/models, non-output activations, learned SpinQuant rotations, end-to-end decoding throughput, and other group sizes remain untested. No functional capacity or Mirror-specific quantization improvement is established.

## Fact / Interpretation / Hypothesis

**FACT:** 16-view search ties a single random QuaRot code on held-out rows in both development seeds; identity and SmoothQuant have lower error; exact full-precision symmetry error is below 1.3e-6; the sweep aliases the native control; fresh was sealed.

**INTERPRETATION:** These exact rotations do not improve this int4 Pareto point. The size reduction is the baseline quantizer's gain, while the extra coordinate and calibration sweep add state and setup compute without quality benefit.

**HYPOTHESIS:** Learned rotations may help when trained on representative activation statistics or different group sizes, but that would test SpinQuant-style optimization rather than Mirror-specific logical functions.
