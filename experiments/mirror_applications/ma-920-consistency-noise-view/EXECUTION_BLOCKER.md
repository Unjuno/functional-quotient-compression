# MA-920 execution blocker

## Fact

- Official Torch repository: `openai/consistency_models`, commit `e32b69ee436d518377db86fb2127a3972d0d8716`, covers ImageNet64/LSUN; official CIFAR implementation is a separate JAX repository.
- Official CIFAR repository: `openai/consistency_models_cifar10`, commit `d086e77dfdc30b51671685f9eec90c90d4f4eaa6`; README checkpoint URLs point to `openaipublic.blob.core.windows.net`.
- A HEAD request to the official checkpoint host failed with proxy `403 Forbidden`.
- Runtime is CPU-only, with no JAX or Flax installed. The official JAX repository documents JAX 0.4.10/CUDA 11 assumptions.
- Hugging Face hosts an official native ImageNet64 checkpoint, but it is ~1.18 GB. It supplies no trained Mirror counterpart; running native inference alone or adding an untrained coordinate shim would not test MA-920.

## Decision

Keep UNTESTED and advance. Do not use a synthetic flow map as a substitute because MA-919 already contains a scoped analytic flow screen and MA-920's registered claim is consistency-model image quality/FID at matched NFE.

## Resume condition

Make a matched official baseline checkpoint accessible and provide an adequate accelerator/evaluation budget. Freeze train/fresh data, steps, noise schedules, actual serialized state accounting, FID/quality metrics and NFE controls before training the Mirror variant.
