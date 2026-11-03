# MA-574 — SpinQuant rotation codebook

Status: NOT ESTABLISHED
Branch: `research/ma-574-spinquant-rotation-codebook-20261009`
Prior art: PA113 SpinQuant; PA112 QuaRot; PA114 SmoothQuant

## H

A small shared codebook of exact orthogonal quantization Views learned from training layers can represent held-out layer rotation choices with near-independent quality and reduce total serialized multi-layer int4 payload bytes versus storing an independent rotation per layer. Native SpinQuant/QuaRot codebook selection is the strongest attribution control.

## T

Use 12 pinned Pythia-70M linear matrices with a common 512D input: attention output and FFN up-projection matrices from layers 0–5. Layers 0–3 train/select the rotation codebook; layers 4–5 are held-out layers. Quantize each matrix to symmetric int4 with 32-value groups. Compare identity, one global QuaRot view, independent per-matrix selection among 16 candidate views, a learned shared codebook of four views, a random four-view codebook, the exact native codebook control, and SmoothQuant. Calibration output rows select each matrix's code; complementary rows evaluate quantized reconstruction. Fresh codebook seeds 57411–57413 remain sealed until development gates pass.

The codebook is learned by greedily choosing four of 16 random block-Hadamard/sign/permutation candidates to minimize average calibration NRMSE over training layers. No optimizer or model training is used. Charge actual serialized int4 matrices, FP16 per-row scales, all codebook or independent transform signs/permutations, layer IDs, SmoothQuant channel scales and metadata. Report total payload bytes as well as the rotation-code-only subtotal. Each exact full-precision rotation is audited before quantization.

## D gates

PASS for shared-codebook storage requires both development seeds to keep held-out-layer NRMSE within 5% of independent per-matrix selection, reduce rotation-code bytes by at least 50%, and reduce the complete serialized multi-layer payload by at least 0.1%. Native codebook controls must be reported separately. FAIL if the quality gate fails, full-payload byte savings miss 0.1%, or a simpler native control exactly matches the selected codes and outputs.

## Evidence boundary

This is a real-weight quantization reconstruction screen, not language NLL/perplexity or end-to-end inference. Output-row reconstruction error is the primary quality measure. Codebook selection uses training layers only; held-out layers remain separate.
