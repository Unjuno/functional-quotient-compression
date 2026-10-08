# MA-998 — MACE element × local-environment Mirror coordinates

Status: NOT ESTABLISHED — BLOCKED before a valid run
Evidence lane: MECHANISM / QUALITY / STORAGE
Base commit: `16807a7` (`research/mirror-application-worker-ready-20261007`)

## H — hypothesis

A compact factorized Mirror code over chemical element and local environment could share part of an equivariant interatomic potential while preserving energy invariance, force equivariance and conservative forces on held-out element–environment combinations.

## T — attempted

Read PA297 (MACE-MP) and PA298 (NequIP). The current container has NumPy/SciPy but no PyTorch, MACE, e3nn, ASE or visible CUDA/GPU tooling. No model weights, atomistic dataset or training/fresh outcomes were opened. The required equivariant energy/force controls cannot be reproduced in this environment, so no toy scalar surrogate was substituted.

## D — NOT ESTABLISHED

This is an execution blocker, not a scientific FAIL. Registry status remains UNTESTED. Resume only in an environment with the native MACE/NequIP stack and suitable force-evaluation compute.

## C — strongest counter-hypothesis

MACE/NequIP's native element embeddings and equivariant message-passing layers already capture element–environment variation; a factorized `m` may add no quality per byte and can break conservative/equivariant force structure.

## U — unknown

All scientific outcomes remain unknown: held-out chemistry quality, force/energy consistency, actual bytes, inference/training compute, and whether any Mirror code beats native element embeddings or independent element-specific heads.

## Fact / Interpretation / Hypothesis

FACT: package capability probe found no PyTorch, MACE, e3nn, ASE, CUDA runtime or visible GPU.
INTERPRETATION: native MACE/NequIP hypothesis cannot be evaluated in this container; the ID remains UNTESTED.
HYPOTHESIS: structured element and local-environment coordinates may permit reusable logical force laws, subject to strict energy/force consistency audits.
BOUNDARY: no scientific experiment was run.
