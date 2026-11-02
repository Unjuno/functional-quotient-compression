# MA-502 — Shared LoReFT basis + many Mirror task codes

Status: **FAIL** for <=.05 NRMSE.

**H:** A shared subspace plus discrete task-mode codes can compress clustered interventions while preserving task identity and function quality.

**T:** Synthetic 64D frozen representation interventions, 64 tasks, rank-4 shared PCA basis, 8 latent modes, tiny within-mode noise. Basis and codebook fitted on development worlds 50200-50201. Fresh worlds 50210-50212 × seeds 0-2 compared dense vectors, FP16 shared task coordinates, and VQ Mirror codes.

**D:** FAIL. FP16 shared coordinates averaged 3,809B and NRMSE .00018; dense was 18,025B exact. Mirror VQ averaged 3,677B but NRMSE .0927, above the .05 gate. The mode-retrieval metric is not interpretable because unsupervised codebook IDs have arbitrary permutation; reconstruction NRMSE is the authoritative quality measure. Only 132B were saved relative to FP16 coordinates while distortion increased substantially.

**C:** With only 64 tasks, codebook/index overhead offers little additional storage reduction; FP16 coordinates are already compact. The fitted VQ centroids did not recover the underlying modes accurately enough.

**U:** Pretrained Transformer and downstream task quality, label-aware mode codebooks, larger task banks, and optimized entropy-coded indices remain untested.
