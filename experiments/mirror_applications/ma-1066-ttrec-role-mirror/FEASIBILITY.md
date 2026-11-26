# MA-1066 feasibility checkpoint

**Registry status remains UNTESTED. No MA-1066 model was trained and no audit data was opened.**

## Required evidence

MA-1066 compares a role-specific Mirror view over TT-Rec tensor-train embedding cores with optimized native `TTEmbeddingBag`. Its registered first metrics include AUC, real serialized table bytes, and GPU QPS. CPU timing or published kernel numbers cannot replace a same-device, same-cache comparison.

## Environment facts (2026-11-26)

- The available PyTorch build is `2.14.1+cpu`; `torch.cuda.is_available()` is false and device count is zero. `nvidia-smi` is absent.
- No TT-Rec / `TTEmbeddingBag` implementation is present in the worker repository.
- The official upstream implementation was located at `facebookresearch/FBTT-Embedding`, pinned source commit `b95947ccd32cf9ab851242cc606562f2a74d9efa`. Its README documents CUDA kernels and software caching; the upstream package reports testing with PyTorch 1.6+ and CUDA 10.1.
- GPU Control exposes a GitHub repository+commit source-registration and review route. Its current training-source list contains only an unrelated `unjuno/model-as-a-paper` source in `pending_review` state; there is no reviewed MA-1066 profile available to execute now. Fixed GPU canary workloads are not the TT-Rec research workload.

## Reopen conditions

1. Implement an experiment-local adapter and runner against the pinned upstream TT-EmbeddingBag, with the exact TT shapes/ranks, cache policy, Mirror, matched diagonal/FiLM controls, independent TT, and dense upper control fixed before fresh audit access.
2. Commit and push the source on this dedicated research branch; submit the exact repository commit through GPU Control source review and confirm an executable profile.
3. Confirm the usable GPU type and cost ceiling, then freeze the data split, training budget, lookup/cache policy, success gates, serialization format, and throughput protocol before any audit run.
4. Run every method on that same GPU and cache setting. If source review or an appropriate GPU remains unavailable, preserve MA-1066 as UNTESTED and report NOT ESTABLISHED; do not substitute CPU latency for GPU QPS.

The consultation with Astra confirmed that GPU QPS and the optimized native TT control are essential for judging the registered hypothesis. It also identified CPU-side adapter, equality, and byte-accounting work as preparatory tasks that do not resolve the GPU claim.
