# MA-891 execution blocker

## Fact

- The selected baseline is Continual Neural Graphics Primitives (C-NGP), which builds on Instant-NGP hash-grid rendering.
- The official repository README requires `tinycudann` and compilation of its CUDA extension; the training code places camera data on `cuda`.
- The official README states that model checkpoints were not released.
- This container has PyTorch 2.14.1 CPU-only, `torch.cuda.is_available() == False`, and no `tinycudann`, `nerfacc`, or Instant-NGP package. The workspace contains no C-NGP checkpoint or scene dataset.

Evidence sources:

- PA243: `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`.
- Official C-NGP source: https://github.com/prajwalsingh/C-NGP/ (README, `continual-ngp/train.py`, `conditional-ngp/train.py`).
- Local capability probe: recorded in `VERIFICATION.json`.

## Interpretation

The required C-NGP control and its CUDA field renderer cannot be reproduced in this container. Substituting a hand-built CPU MLP or a synthetic 2D task would answer a different question and would not measure the registered held-out radiance PSNR, scene storage, or continual forgetting comparison.

## Hypothesis

The MA-891 hypothesis remains untested. No positive or negative Mirror result is inferred from missing hardware or assets.

## Requirements to resume

- CUDA GPU and compatible compiler/toolchain for the official C-NGP CUDA extension and `tinycudann`.
- The exact C-NGP code revision, configuration, scene training/evaluation assets, and baseline checkpoints or a reproducible baseline training budget.
- Then preregister the scene split, replay memory, native scene-code control, Mirror coordinate, per-scene PSNR/forgetting gates, actual serialized payload accounting, and held-out evaluation before training.
