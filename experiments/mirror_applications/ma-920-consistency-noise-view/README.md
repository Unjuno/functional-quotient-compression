# MA-920 — consistency-model noise-level Mirror view

Status: UNTESTED
Evidence lane: GENERATIVE_QUALITY / NFE / STORAGE / RUNTIME

## H — falsifiable hypothesis

A small Mirror noise-level/condition coordinate on a fixed consistency model improves FID or quality-per-byte at matched inference steps and compute over the native one/few-step consistency map and ordinary time-conditioning controls.

## T — source and runtime feasibility

Inspected PA261 and official OpenAI repositories `consistency_models` (Torch, ImageNet64/LSUN) and `consistency_models_cifar10` (JAX). The CIFAR-10 repository's released checkpoints are hosted on `openaipublic.blob.core.windows.net`, which returns proxy 403 from this environment. The CIFAR implementation assumes a JAX/CUDA stack; local runtime is CPU-only and has no JAX/Flax. Hugging Face mirrors of the official ImageNet64 native model exist, but a native-only inference or an untrained Mirror shim cannot establish the registered comparison; full model is ~1.18 GB and CPU FID adaptation/evaluation is unavailable.

## D — UNTESTED

No Mirror training, quality, FID, NFE, actual payload-byte or latency comparison was run. This is not a FAIL.

## C — strongest counter-hypothesis

The native noise/time conditioning and consistency distillation already span useful noise-level behavior; an extra view may only add conditioning parameters and worsen quality or compute.

## U — unresolved

Accessible matched baseline checkpoint, GPU/JAX runtime or feasible full evaluation budget, and a preregistered Mirror-versus-native training/evaluation protocol. See [EXECUTION_BLOCKER.md](EXECUTION_BLOCKER.md).
