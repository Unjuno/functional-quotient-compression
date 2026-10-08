# MA-417 — Mirror codebook for neural functions

## H — falsifiable hypothesis

A shared decoder with a K-entry codebook of two-angle Mirror views will represent 16 functions at K=16 within 10% of continuous latent2 reconstruction error using fewer actual bytes, and the K=4/8/16 sweep will expose the quality/storage frontier for reduced logical code multiplicity.

## T — protocol

Use the same synthetic Fourier function family as MA-416, but use a learned vector quantization assignment: K two-angle Mirror codes or K ordinary two-dimensional latent codes, with a per-function stored index. Compare continuous latent2, per-function Mirror2, codebooks K=4/8/16, and independent decoders. Straight-through hard assignment is used during training; inference payload excludes assignment logits and stores only selected indices. Three fresh worlds report function-wise held-out reconstruction, unique code use, and actual bytes.

This is a codebook mechanism screen, not 3D shape or DeepSDF evidence.

