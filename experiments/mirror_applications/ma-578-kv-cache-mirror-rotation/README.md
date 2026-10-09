# MA-578 — KV-cache Mirror rotation code

Status: SCREENING — amended freeze; registered dev pending
Prior art: PA112 QuaRot; PA113 SpinQuant

## H

A small shared rotation-code bank may keep int4 KV-cache next-token quality near FP16 while storing fewer bytes than independent layer/head/K/V transforms.

## T

Run pinned Pythia-70M on WikiText-2. Calibrate 16 exact orthogonal block-Hadamard/sign/permutation candidates from four train prefixes, fit a four-entry codebook, and evaluate one-token NLL after 64-token cached prefixes on valid text. Compare FP16, identity int4, global QuaRot, independent per-role search, fitted and random codebooks, and native codebook selection. Fresh test text remains sealed until the frozen dev gates pass.

The script reconstructs quantized caches into the original coordinate basis for NLL scoring and reports cache encode/query timing and the online view-operation proxy. This is not a full-sequence perplexity or optimized serving-kernel result.
