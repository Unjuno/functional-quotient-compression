# MA-1030 status

- Status: **COMPLETE — FAIL**, synthetic SAM2.1 Hiera-tiny CPU mechanism test
- Branch: `research/ma-1030-sam2-rare-object-private-fallback-20261115`
- Base commit: `fc7c2640fef0da03f7064d3d41e4b20f3afc2545`
- Protocol/source pre-data freeze: `ec729acfe`
- Development threshold/data freeze before audit generation: `2f3d44166b663c8c551d552640b2345a15dd444e`
- Fresh audit seeds 103001–103003: complete
- Results/verification: complete

**H:** A causal one-slot private-memory fallback selected from object-to-shared residual energy should protect the rare appearance-shifted track while retaining quality and reducing object-memory bytes.

**T:** Four-object, six-frame synthetic SAM2.1 Hiera-tiny video; object 99 changes appearance under occlusion. Independent, hard-shared, two-angle Givens, rank-1 additive and Givens-plus-private-fallback controls; threshold selected on dev seed 103000, then frozen for three fresh seeds. No future error or mask is used by the selector.

**D:** FAIL. The selector chose object 42 on every yielded frame and never selected shifted object 99. Fallback J&F averages 0.17772 vs 0.60567 independent and 0.28987 rank-1 additive. Object-memory tensors are 50.0% smaller than independent, but full-session savings are only 0.240%; rank-1 is 144 B smaller in memory state and 702 B smaller overall, with better quality and FPS.

**C:** This synthetic tracking scene is hard even for independent SAM2 (mean J&F 0.606); conclusions apply to the tested selector and scene only.

**U:** Natural videos, longer sequences, benchmark datasets, larger backbones, GPU, multiple private slots, learned codes, and temporal-change-aware fallback.

Development and audit images, hashes, predictions, serialized state and metric replay are preserved. Four tests pass. See `RESULTS.md`.
