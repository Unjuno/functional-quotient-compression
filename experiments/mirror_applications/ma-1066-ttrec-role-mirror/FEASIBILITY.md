# MA-1066 feasibility checkpoint

**Registry status remains UNTESTED. No MA-1066 model was trained and no audit data was opened.**

## Required evidence

MA-1066 compares a role-specific Mirror view over TT-Rec tensor-train embedding cores with optimized native `TTEmbeddingBag`. Its registered first metrics include AUC, real serialized table bytes, and GPU QPS. CPU timing or published kernel numbers cannot replace a same-device, same-cache comparison.

## Environment facts (2026-11-26)

- The available PyTorch build is `2.14.1+cpu`; `torch.cuda.is_available()` is false and device count is zero. `nvidia-smi` is absent.
- The official upstream implementation was located at `facebookresearch/FBTT-Embedding`, pinned source commit `b95947ccd32cf9ab851242cc606562f2a74d9efa`, and its unmodified MIT-licensed CUDA sources are now vendored in `source/vendor/FBTT-Embedding`. Its README documents CUDA kernels and software caching; the upstream package reports testing with PyTorch 1.6+ and CUDA 10.1.
- An experiment-local runner and protocol now cover independent TT, shared TT, matched diagonal/FiLM, Givens Mirror, and dense controls. The runner refuses to train unless CUDA is available; it has not been trained or audited.
- GPU Control source registration does not provision an execution environment. Its current training service only has an execution profile for the configured MAP-44M (`unjuno/model-as-a-paper`) pinned commit; other listed registrations are non-executable. The public tool surface has no operation to configure a new repository/commit profile. Fixed GPU canary workloads are not the TT-Rec research workload.
- The exact MA-1066 branch commit `8bf1d91c8ef34ec6e7b7c511d362b6edfde34ce3` was registered as source `d328e6b4cea94db4897e7f8cac07219d`; it is `pending_review`, `executable=false`. Preview returned `TRAINING_PROFILE_UNAVAILABLE` (“no validated training image configured for this commit”).
- After source-freeze hardening, exact commit `ac5957c0fc1cb497e66e4e1be05ef07419171cf8` was registered as `cd662491288542b08c03bf103d9bf8ec`; it is also `pending_review`, `executable=false`. Its $0.000001 preview returned `TRAINING_PROFILE_UNAVAILABLE`. No training submission was made; use this latest registration when review resumes.

## Reopen conditions

1. Build the pinned upstream TT-EmbeddingBag in a reviewed CUDA image and test the adapter on that GPU.
2. Obtain an operator-provisioned, reviewed GPU execution profile for the exact TT-EmbeddingBag experiment source, or use another GPU environment that can build and run the pinned optimized control. The public GPU Control tools cannot provision this profile.
3. Confirm the usable GPU type and cost ceiling, run development selection, then execute `source/freeze.py` to freeze data split, training budget, backend/cache policy, success gates, serialization format, and throughput protocol before any audit access.
4. Run every method on that same GPU and cache setting. If source review or an appropriate GPU remains unavailable, preserve MA-1066 as UNTESTED and report NOT ESTABLISHED; do not substitute CPU latency for GPU QPS.

The consultation with Astra confirmed that GPU QPS and the optimized native TT control are essential for judging the registered hypothesis. It also identified CPU-side adapter, equality, and byte-accounting work as preparatory tasks that do not resolve the GPU claim.
