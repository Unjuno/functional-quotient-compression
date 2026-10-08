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

### Facts

| Seed | Representation | Heldout value RMSE | False trigger | Actual bytes |
|---:|---|---:|---:|---:|
| 47601 | Explicit values | 0.00000 | 0 | 99,506 |
| 47601 | Continuous Mirror/PCA | 0.00000 | 0 | 44,482 |
| 47601 | Latent int8 | 0.01191 | 0 | 38,623 |
| 47601 | VQ16 | 0.14198 | 0 | 37,298 |
| 47601 | VQ64 | 0.12736 | 0 | 38,834 |
| 47601 | VQ128 | 0.11517 | 0 | 40,883 |
| 47602 | Explicit values | 0.00000 | 0 | 99,506 |
| 47602 | Continuous Mirror/PCA | 0.00000 | 0 | 44,482 |
| 47602 | Latent int8 | 0.00694 | 0 | 38,623 |
| 47602 | VQ16 | 0.13410 | 0 | 37,298 |
| 47602 | VQ64 | 0.11916 | 0 | 38,834 |
| 47602 | VQ128 | 0.11112 | 0 | 40,883 |

All methods had zero wrong routes and false triggers. VQ16 used 3.4% fewer bytes than latent int8 but its value RMSE was over 0.13; VQ64/128 used more bytes than int8 and also had much larger error. Continuous PCA stored values at 44,482 bytes with near-zero error and exactly aliased the native PCA bank. The key/router table is paid in every payload and dominates absolute storage.

### T — Execution

Two frozen development worlds, 192 training values and 64 heldout; VQ sizes 16/64/128 fit only on training latent values; fixed retrieval keys/radius; explicit values, continuous PCA, latent int8 and no-edit controls. Actual NPZ bytes, basis/codebook fit wall time, query wall and operations proxies are retained in `RESULTS_CORE.csv`. Serialization and metric replay were exact. Fresh seeds 47611–47613 were not accessed.

### D — Decision

**FAIL.** The discrete codebooks do not meet the frozen heldout quality threshold of 0.02. Latent int8 is a smaller and much more accurate simple control than VQ64/128. Continuous basis coding retains quality but exactly matches native PCA, so there is no Mirror-specific benefit. Fresh remains sealed.

### C — Strongest counter-hypothesis

The shared basis already removes the eight-dimensional subspace redundancy. Scalar int8 quantization preserves that geometry more efficiently than a small vector codebook; with only 256 entries, K-means centers need substantial codebook storage and still incur large quantization error.

### U — Unconfirmed

Larger edit counts, residual VQ, private/out-of-subspace values, natural GRACE lifetimes, production-scale retrieval throughput and any benefit beyond native PCA/int8 remain unconfirmed.

