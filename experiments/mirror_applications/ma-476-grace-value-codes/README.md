# MA-476 — GRACE codebook value compression through a shared value basis

Status: SCREENING  
Branch: `research/ma-476-grace-value-codes-20261008`  
Base commit: `38b263fcb3c6af5b8dd681b2a6814252d546b114`  
Prior art: PA90 (GRACE), PA91 (VQ-VAE)

## H — Hypothesis

GRACE-style discrete retrieval may store counterfactual values more compactly as indices into a shared latent codebook decoded through a shared value basis. Fixed keys/routing separate retrieval from value storage. Continuous PCA and latent int8 show whether VQ adds a useful quality/bytes point.

## T — Frozen protocol

A 256-edit memory contains 64-dimensional values on a seeded rank-eight subspace; 192 train the basis/codebooks and 64 are held out. K-means codebooks use K=16, 64 and 128. All methods pay identical key/radius state. Compare explicit float32, continuous basis codes, native PCA, latent int8 and no-edit. Fresh seeds 47611–47613 remain sealed unless all frozen gates pass. See `PROTOCOL.json` for exact accounting.

PA90 stores key-value edits with local retrieval; PA91 supplies the discrete-codebook control. MA-476 changes only value representation and uses fixed keys and routing.

## Results

Pending frozen development runs.
