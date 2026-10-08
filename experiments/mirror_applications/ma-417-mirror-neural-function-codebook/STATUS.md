# MA-417 status: FAIL

## H — falsifiable hypothesis

A K=16 two-angle Mirror codebook would approach continuous latent2 reconstruction quality with fewer actual bytes, while K=4/8/16 would reveal the quality/storage/multiplicity frontier.

## T — executed

Sixteen random weighted Fourier functions (frequencies 1, 2, 3); 64 training queries/function and 128 held-out queries/function. Compared continuous latent2, per-function Mirror2, Mirror and ordinary latent codebooks K=4/8/16 with straight-through hard assignment and inference indices, and independent decoders. 500 AdamW updates × batch 256; 2 development worlds × 3 seeds selected LR per method; 3 fresh worlds × 3 seeds. Inference payloads contain decoder, codebook, and selected function indices; training logits were excluded.

## D — FAIL

Fresh normalized RMSE / actual bytes / mean unique code assignments:
- Continuous latent2: 0.7099 / 3,161B / 16
- Per-function Mirror2: 0.8620 / 2,905B / 16
- Mirror codebook K=4: 0.9348 / 3,093B / 4
- Mirror codebook K=8: 0.9007 / 3,093B / 7.1
- Mirror codebook K=16: 0.9032 / 3,157B / 9.6
- Latent codebook K=4: 0.8657 / 3,349B / 4
- Latent codebook K=8: 0.8646 / 3,349B / 6.6
- Latent codebook K=16: 0.7994 / 3,413B / 10
- Independent decoders: 0.5569 / 8,349B / 16

Mirror K=16 has 7.5% fewer bytes than latent K=16 but 13.0% higher normalized RMSE, missing the <=10% gate. At every matched K the ordinary latent codebook had lower error. K=16 Mirror used only about 9.6 distinct entries for 16 functions, so nominal K does not equal useful logical multiplicity.

## C — strongest counter-hypothesis

The arbitrary Fourier family does not align with hidden rotations. Straight-through assignment may also collapse code use. The small byte advantage at K=16 may reflect the decoder input expansion required by ordinary latent codes, while quality loss comes from the Mirror parameterization.

## U — unresolved

Structured phase/amplitude functions, a learned assignment schedule, larger models, 3D shape reconstruction, and near-convergence remain untested. This experiment does not establish a general codebook capacity bound.

