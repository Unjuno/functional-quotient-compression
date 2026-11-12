# MA-893 execution blocker

## Fact

- Registry entry requires the Instant-NGP hash-grid model, including the full learned hash-table payload and fused CUDA runtime (PA245); C-NGP is also named as an adjacent control (PA243).
- The available runtime reports PyTorch `2.14.1+cpu`, `torch.cuda.is_available() == False`, and `torch.version.cuda is None`.
- `nvidia-smi` is unavailable. Python package probes found no `tinycudann`, `torch_ngp`, or `nerfacc`.
- No native Instant-NGP checkpoint or scene assets were present in the experiment workspace at this review.

## Interpretation

The faithful native baseline and runtime comparison cannot be reproduced here. A CPU hash-grid reimplementation or synthetic field would test a different model and would not establish MA-893's PSNR, complete grid-byte, or rendering-speed frontier.

## Hypothesis

The scene-View hypothesis remains untested. Missing hardware and assets are not negative model evidence.

## Requirements to resume

- CUDA GPU with a compatible Instant-NGP fused implementation and `tinycudann` build.
- Exact implementation revision/configuration and accessible scene datasets/checkpoints or a fixed native training recipe.
- Before training: lock scene splits and fresh seeds; compare independent per-scene grids, shared grid plus ordinary scene code/FiLM, and Mirror Views; count all hash tables, decoder weights, codes, metadata and reconstruction state in actual serialized payload bytes; measure PSNR, collision interference, fit/render wall-clock and throughput.
